"""
Integration helpers for NVIDIA PhysicsNeMo (formerly Modulus).

This module exposes a light-weight adapter that uses PhysicsNeMo's fully
connected PINN blocks to generate synthetic diffusion–advection data,
train a surrogate, and export the model to PyTorch or ONNX formats.

The adapter is intentionally implemented without depending on the broader
Hydra/launch tooling that ships with PhysicsNeMo—everything is orchestrated
programmatically so that the FastAPI/CLI layer can call into it directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import logging
import torch
import torch.nn.functional as F


class PhysicsNemoUnavailable(RuntimeError):
    """Raised when nvidia-physicsnemo is missing from the environment."""


def _get_fc_cls():
    """Import the PhysicsNeMo fully connected model lazily."""

    try:
        from physicsnemo.models.mlp import FullyConnected  # type: ignore
    except ModuleNotFoundError as exc:
        raise PhysicsNemoUnavailable(
            "nvidia-physicsnemo is not installed. Install it via "
            "`pip install nvidia-physicsnemo` (and ensure torch>=2.4.0).",
        ) from exc
    return FullyConnected


def physics_nemo_available() -> bool:
    """Utility to check availability without raising."""

    try:
        _get_fc_cls()
    except PhysicsNemoUnavailable:
        return False
    return True


@dataclass
class NemoSimulationConfig:
    """Spatial/temporal configuration for the synthetic pipeline."""

    length: float = 1.0
    x_points: int = 64
    profile_time: float = 0.6
    gaussian_center: float = 0.35
    gaussian_width: float = 0.08
    diffusion_range: Tuple[float, float] = (0.01, 0.2)
    velocity_range: Tuple[float, float] = (0.05, 1.2)


@dataclass
class NemoTrainingConfig(NemoSimulationConfig):
    """Training hyper-parameters for the PhysicsNeMo surrogate."""

    epochs: int = 400
    batch_size: int = 2048
    collocation_points: int = 8192
    boundary_points: int = 1536
    initial_points: int = 1024
    learning_rate: float = 5e-4
    pde_weight: float = 1.0
    ic_weight: float = 0.4
    bc_weight: float = 0.2
    hidden_dim: int = 128
    hidden_layers: int = 6
    activation: str = "tanh"
    dataset_profiles: int = 8
    export_onnx: bool = True
    export_torchscript: bool = True
    seed: int | None = None
    device: str = "auto"


@dataclass
class NemoRunResult:
    """Metadata describing the latest PhysicsNeMo pipeline execution."""

    dataset_path: Path
    model_path: Path
    onnx_path: Path | None
    torchscript_path: Path | None
    training_history: List[Dict[str, float]] = field(default_factory=list)
    profiles: List[Dict[str, List[float]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "dataset_path": str(self.dataset_path),
            "model_path": str(self.model_path),
            "onnx_path": str(self.onnx_path) if self.onnx_path else None,
            "torchscript_path": str(self.torchscript_path) if self.torchscript_path else None,
            "training_history": self.training_history,
            "profiles": self.profiles,
        }
        return payload


logger = logging.getLogger(__name__)


class PhysicsNemoAdapter:
    """High-level helper that owns the PhysicsNeMo model lifecycle."""

    def __init__(
        self,
        data_dir: Path,
        artifact_dir: Path,
    ) -> None:
        self.data_dir = data_dir
        self.artifact_dir = artifact_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.artifact_dir / "physics_nemo_meta.json"

    def run_pipeline(self, config: NemoTrainingConfig) -> NemoRunResult:
        """Train the PhysicsNeMo surrogate and export artifacts."""

        if not physics_nemo_available():
            raise PhysicsNemoUnavailable(
                "nvidia-physicsnemo is required. Install via `pip install nvidia-physicsnemo`.",
            )

        torch.manual_seed(config.seed or torch.randint(0, 10_000, (1,)).item())

        model, history = self._train_model(config)
        model.eval()
        dataset_path, profiles = self._sample_dataset(model, config)
        model_path = self.artifact_dir / "physics_nemo_surrogate.pt"
        torch.save(model.state_dict(), model_path)

        onnx_path = None
        if config.export_onnx:
            onnx_path = self._export_onnx(model, config)

        ts_path = None
        if config.export_torchscript:
            ts_path = self._export_torchscript(model, config)

        result = NemoRunResult(
            dataset_path=dataset_path,
            model_path=model_path,
            onnx_path=onnx_path,
            torchscript_path=ts_path,
            training_history=history,
            profiles=profiles,
        )
        self._write_meta(config, result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, config: NemoTrainingConfig) -> torch.nn.Module:
        fc_cls = _get_fc_cls()
        model = fc_cls(
            in_features=4,
            out_features=1,
            num_layers=config.hidden_layers,
            layer_size=config.hidden_dim,
            activation_fn=config.activation,
        )
        return model

    def _train_model(self, config: NemoTrainingConfig) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
        device_str = config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA device requested but unavailable. Falling back to CPU.")
            device_str = "cpu"
        device = torch.device(device_str)
        model = self._build_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        history: List[Dict[str, float]] = []

        for epoch in range(1, config.epochs + 1):
            optimizer.zero_grad()

            collocation = self._sample_collocation(config.collocation_points, config).to(device)
            collocation.requires_grad_(True)
            pde_loss = self._pde_residual(model, collocation)

            init_inputs = self._sample_initial(config.initial_points, config).to(device)
            init_inputs.requires_grad_(True)
            u_init = model(init_inputs)
            target_init = self._gaussian(init_inputs[:, :1], config)
            ic_loss = F.mse_loss(u_init, target_init)

            boundary_inputs = self._sample_boundary(config.boundary_points, config).to(device)
            boundary_inputs.requires_grad_(True)
            bc_loss = self._boundary_loss(model, boundary_inputs)

            total_loss = (
                config.pde_weight * pde_loss
                + config.ic_weight * ic_loss
                + config.bc_weight * bc_loss
            )
            total_loss.backward()
            optimizer.step()

            history.append(
                {
                    "epoch": float(epoch),
                    "pde_loss": float(pde_loss.detach().cpu().item()),
                    "ic_loss": float(ic_loss.detach().cpu().item()),
                    "bc_loss": float(bc_loss.detach().cpu().item()),
                    "total_loss": float(total_loss.detach().cpu().item()),
                }
            )

        return model, history

    def _pde_residual(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        u = model(inputs)
        ones = torch.ones_like(u)
        grads = torch.autograd.grad(u, inputs, grad_outputs=ones, create_graph=True)[0]
        du_dx = grads[:, 0:1]
        du_dt = grads[:, 1:2]
        second = torch.autograd.grad(du_dx, inputs, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        du_dxx = second[:, 0:1]
        D = inputs[:, 2:3]
        v = inputs[:, 3:4]
        residual = du_dt - D * du_dxx + v * du_dx
        return (residual**2).mean()

    def _boundary_loss(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        prediction = model(inputs)
        grad = torch.autograd.grad(
            prediction,
            inputs,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
        )[0]
        du_dx = grad[:, 0:1]
        return (du_dx**2).mean()

    def _sample_collocation(self, n: int, config: NemoTrainingConfig) -> torch.Tensor:
        x = torch.rand(n, 1) * config.length
        t = torch.rand(n, 1) * config.profile_time
        D = torch.rand(n, 1) * (config.diffusion_range[1] - config.diffusion_range[0]) + config.diffusion_range[0]
        v = torch.rand(n, 1) * (config.velocity_range[1] - config.velocity_range[0]) + config.velocity_range[0]
        return torch.cat([x, t, D, v], dim=1)

    def _sample_initial(self, n: int, config: NemoTrainingConfig) -> torch.Tensor:
        x = torch.rand(n, 1) * config.length
        t = torch.zeros_like(x)
        D = torch.rand(n, 1) * (config.diffusion_range[1] - config.diffusion_range[0]) + config.diffusion_range[0]
        v = torch.rand(n, 1) * (config.velocity_range[1] - config.velocity_range[0]) + config.velocity_range[0]
        return torch.cat([x, t, D, v], dim=1)

    def _sample_boundary(self, n: int, config: NemoTrainingConfig) -> torch.Tensor:
        half = n // 2
        x_left = torch.zeros(half, 1)
        x_right = torch.full((n - half, 1), config.length)
        x = torch.cat([x_left, x_right], dim=0)
        t = torch.rand(n, 1) * config.profile_time
        D = torch.rand(n, 1) * (config.diffusion_range[1] - config.diffusion_range[0]) + config.diffusion_range[0]
        v = torch.rand(n, 1) * (config.velocity_range[1] - config.velocity_range[0]) + config.velocity_range[0]
        return torch.cat([x, t, D, v], dim=1)

    def _gaussian(self, x: torch.Tensor, config: NemoSimulationConfig) -> torch.Tensor:
        return torch.exp(-0.5 * ((x - config.gaussian_center) / config.gaussian_width) ** 2)

    def _sample_dataset(
        self,
        model: torch.nn.Module,
        config: NemoTrainingConfig,
    ) -> Tuple[Path, List[Dict[str, List[float]]]]:
        model.eval()
        device = next(model.parameters()).device
        x = torch.linspace(0.0, config.length, config.x_points, device=device)
        dataset_inputs: List[torch.Tensor] = []
        dataset_targets: List[torch.Tensor] = []
        profile_records: List[Dict[str, List[float]]] = []

        for _ in range(config.dataset_profiles):
            D = torch.rand(1).item() * (config.diffusion_range[1] - config.diffusion_range[0]) + config.diffusion_range[0]
            v = torch.rand(1).item() * (config.velocity_range[1] - config.velocity_range[0]) + config.velocity_range[0]
            t = torch.full_like(x, config.profile_time)
            inputs = torch.stack(
                [
                    x,
                    t,
                    torch.full_like(x, D),
                    torch.full_like(x, v),
                ],
                dim=-1,
            )
            with torch.no_grad():
                profile = model(inputs.to(device)).cpu()

            dataset_inputs.append(inputs.cpu())
            dataset_targets.append(profile)
            profile_records.append(
                {
                    "D": [float(D)],
                    "v": [float(v)],
                    "x": inputs[:, 0].tolist(),
                    "u": profile.squeeze(-1).tolist(),
                }
            )

        all_inputs = torch.cat(dataset_inputs, dim=0).numpy().astype("float32")
        all_targets = torch.cat(dataset_targets, dim=0).numpy().astype("float32")
        dataset = {
            "inputs": all_inputs.tolist(),
            "targets": all_targets.tolist(),
            "profile_time": config.profile_time,
        }
        dataset_path = self.data_dir / "surrogate_nemo_latest.json"
        with dataset_path.open("w", encoding="utf-8") as f:
            json.dump(dataset, f)

        return dataset_path, profile_records

    def _export_onnx(self, model: torch.nn.Module, config: NemoTrainingConfig) -> Path:
        onnx_path = self.artifact_dir / "physics_nemo_surrogate.onnx"
        dummy = torch.zeros(1, 4, device=next(model.parameters()).device)
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["features"],
            output_names=["u"],
            dynamic_axes={"features": {0: "batch"}, "u": {0: "batch"}},
            opset_version=17,
        )
        return onnx_path

    def _export_torchscript(self, model: torch.nn.Module, config: NemoTrainingConfig) -> Path:
        ts_path = self.artifact_dir / "physics_nemo_surrogate.ts"
        traced = torch.jit.trace(model, torch.zeros(1, 4, device=next(model.parameters()).device))
        traced.save(ts_path)
        return ts_path

    def _write_meta(self, config: NemoTrainingConfig, result: NemoRunResult) -> None:
        summary = {
            "config": asdict(config),
            "result": result.to_dict(),
        }
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def load_meta(self) -> Dict[str, object] | None:
        if not self.meta_path.exists():
            return None
        with self.meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
