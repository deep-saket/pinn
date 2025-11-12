"""Synthetic facility-stage surrogate model and dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FACILITY_FEATURE_NAMES = [
    "pipe_outlet_pressure",  # bar
    "pipe_outlet_temp",  # C
    "throughput",  # kg/s
    "separator_pressure",  # bar
    "separator_temp",  # C
    "mixture_gas_fraction",  # 0-1
]

FACILITY_TARGET_NAMES = [
    "vapor_fraction",
    "gas_flow_rate",
    "liquid_flow_rate",
]


def generate_facility_dataset(num_samples: int, seed: int | None = None) -> Dict[str, np.ndarray]:
    """Generate synthetic facility data given pipe outputs and separator setpoints."""

    rng = np.random.default_rng(seed)
    pipe_pressure = rng.uniform(120.0, 250.0, size=num_samples)
    pipe_temp = rng.uniform(30.0, 90.0, size=num_samples)
    throughput = rng.uniform(5.0, 40.0, size=num_samples)
    sep_pressure = rng.uniform(50.0, 150.0, size=num_samples)
    sep_temp = rng.uniform(20.0, 80.0, size=num_samples)
    gas_frac = rng.uniform(0.1, 0.9, size=num_samples)

    features = np.stack(
        [pipe_pressure, pipe_temp, throughput, sep_pressure, sep_temp, gas_frac],
        axis=1,
    ).astype(np.float32)

    efficiency = np.clip(1.0 - (sep_pressure / np.maximum(pipe_pressure, 1.0)) * 0.4, 0.2, 0.95)
    temp_factor = np.clip(1.0 - np.abs(sep_temp - 45.0) / 100.0, 0.2, 1.0)
    vapor_fraction = np.clip(gas_frac * efficiency * temp_factor, 0.05, 0.95)
    gas_flow = throughput * vapor_fraction * rng.uniform(0.9, 1.1, size=num_samples)
    liquid_flow = throughput - gas_flow + rng.normal(0.0, 0.3, size=num_samples)

    targets = np.stack([vapor_fraction, gas_flow, liquid_flow], axis=1).astype(np.float32)
    return {"inputs": features, "targets": targets}


class FacilitySurrogate(nn.Module):
    def __init__(self, hidden_sizes: Iterable[int] = (64, 64)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = len(FACILITY_FEATURE_NAMES)
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.ReLU())
            input_dim = size
        layers.append(nn.Linear(input_dim, len(FACILITY_TARGET_NAMES)))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


@dataclass
class FacilityTrainingConfig:
    epochs: int = 150
    batch_size: int = 512
    learning_rate: float = 1e-3
    val_split: float = 0.2
    device: str = "cpu"


def train_facility_surrogate(
    dataset: Dict[str, np.ndarray],
    config: FacilityTrainingConfig,
    hidden_sizes: Tuple[int, ...] = (64, 64),
) -> Tuple[FacilitySurrogate, List[Dict[str, float]]]:
    device = torch.device(config.device)
    model = FacilitySurrogate(hidden_sizes).to(device)

    inputs = torch.from_numpy(dataset["inputs"])
    targets = torch.from_numpy(dataset["targets"])

    num_samples = inputs.shape[0]
    val_size = int(num_samples * config.val_split)
    if config.val_split > 0 and val_size == 0:
        val_size = 1
    if val_size >= num_samples:
        val_size = max(num_samples - 1, 0)

    permutation = torch.randperm(num_samples)
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    val_inputs = inputs[val_indices] if val_size > 0 else None
    val_targets = targets[val_indices] if val_size > 0 else None

    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]

    loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=config.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mse = nn.MSELoss()
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        batch_losses: List[float] = []
        for batch_in, batch_tgt in loader:
            batch_in = batch_in.to(device)
            batch_tgt = batch_tgt.to(device)
            optimizer.zero_grad()
            preds = model(batch_in)
            loss = mse(preds, batch_tgt)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        val_loss = None
        if val_inputs is not None and val_targets is not None:
            with torch.no_grad():
                preds = model(val_inputs.to(device))
                val_loss = mse(preds, val_targets.to(device)).item()

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses) if batch_losses else 0.0),
                "val_loss": float(val_loss) if val_loss is not None else None,
            }
        )

    return model, history


def evaluate_facility(model: nn.Module, inputs: np.ndarray, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(inputs.astype(np.float32)).to(device)
        preds = model(tensor).cpu().numpy()
    return preds
