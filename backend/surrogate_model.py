"""
Dummy surrogate (PINN-like) network used by the optimization demonstrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SurrogatePINN(nn.Module):
    """Simple fully-connected network that mimics a physics-informed surrogate."""

    def __init__(self, hidden_sizes: Iterable[int] = (64, 64, 64)) -> None:
        super().__init__()
        layers = []
        input_dim = 4
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.Tanh())
            input_dim = size
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


@dataclass
class SurrogateTrainingConfig:
    epochs: int = 200
    batch_size: int = 512
    learning_rate: float = 1e-3
    pde_weight: float = 0.1
    val_split: float = 0.2
    device: str = "cpu"


def _pde_residual(
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the diffusion–advection residual for a batch of inputs.
    """

    inputs.requires_grad_(True)
    u_hat = model(inputs)

    grad = torch.autograd.grad(
        outputs=u_hat,
        inputs=inputs,
        grad_outputs=torch.ones_like(u_hat),
        retain_graph=True,
        create_graph=True,
    )[0]

    du_dx = grad[:, 0:1]
    du_dt = grad[:, 1:2]

    grad2 = torch.autograd.grad(
        outputs=du_dx,
        inputs=inputs,
        grad_outputs=torch.ones_like(du_dx),
        retain_graph=True,
        create_graph=True,
    )[0]

    d2u_dx2 = grad2[:, 0:1]

    D = inputs[:, 2:3]
    v = inputs[:, 3:4]

    residual = du_dt - D * d2u_dx2 + v * du_dx
    return residual


def train_surrogate(
    dataset: Dict[str, np.ndarray],
    training_cfg: SurrogateTrainingConfig,
    hidden_sizes: Tuple[int, ...] = (64, 64, 64),
) -> Tuple[SurrogatePINN, List[Dict[str, float]]]:
    """
    Train the surrogate network on synthetic data with a PDE residual penalty.
    Returns the trained model and a history of loss components per epoch.
    """

    device = torch.device(training_cfg.device)
    model = SurrogatePINN(hidden_sizes).to(device)

    inputs = torch.from_numpy(dataset["inputs"])
    targets = torch.from_numpy(dataset["targets"])

    num_samples = inputs.shape[0]
    val_size = int(num_samples * training_cfg.val_split)
    if training_cfg.val_split > 0.0 and val_size == 0:
        val_size = 1
    if val_size >= num_samples:
        val_size = max(num_samples - 1, 0)

    permutation = torch.randperm(num_samples)
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    if val_size > 0:
        val_inputs = inputs[val_indices]
        val_targets = targets[val_indices]
    else:
        val_inputs = None
        val_targets = None

    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]

    tensor_ds = TensorDataset(train_inputs, train_targets)
    loader = DataLoader(tensor_ds, batch_size=training_cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    mse = nn.MSELoss()
    history: List[Dict[str, float]] = []

    for epoch in range(training_cfg.epochs):
        epoch_losses = []
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            preds = model(batch_inputs)
            data_loss = mse(preds, batch_targets)
            pde_loss = _pde_residual(model, batch_inputs).pow(2).mean()
            loss = data_loss + training_cfg.pde_weight * pde_loss
            loss.backward()
            optimizer.step()

            epoch_losses.append(
                (
                    float(data_loss.item()),
                    float(pde_loss.item()),
                    float(loss.item()),
                )
            )

        if epoch_losses:
            data_avg = float(np.mean([item[0] for item in epoch_losses]))
            pde_avg = float(np.mean([item[1] for item in epoch_losses]))
            total_avg = float(np.mean([item[2] for item in epoch_losses]))
        else:
            data_avg = pde_avg = total_avg = 0.0

        val_metrics: Dict[str, float] = {
            "val_data_loss": None,
            "val_pde_loss": None,
            "val_total_loss": None,
        }
        if val_inputs is not None and val_targets is not None and val_inputs.numel() > 0:
            val_in = val_inputs.to(device)
            val_tgt = val_targets.to(device)
            val_in = val_in.clone().detach().requires_grad_(True)
            preds = model(val_in)
            val_data = mse(preds, val_tgt).item()
            val_pde = _pde_residual(model, val_in).pow(2).mean().item()
            val_total = val_data + training_cfg.pde_weight * val_pde
            val_metrics = {
                "val_data_loss": float(val_data),
                "val_pde_loss": float(val_pde),
                "val_total_loss": float(val_total),
            }

        history.append(
            {
                "epoch": epoch,
                "train_data_loss": data_avg,
                "train_pde_loss": pde_avg,
                "train_total_loss": total_avg,
                **val_metrics,
            }
        )

    return model, history


def evaluate_profile(
    model: nn.Module,
    x: np.ndarray,
    t: float,
    D: float,
    v: float,
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict the spatial profile at time ``t`` for a given (D, v).
    """

    with torch.no_grad():
        grid = np.stack(
            [x, np.full_like(x, t), np.full_like(x, D), np.full_like(x, v)],
            axis=-1,
        ).astype(np.float32)
        tensor = torch.from_numpy(grid).to(device)
        preds = model(tensor).cpu().numpy().squeeze(-1)
    return preds
