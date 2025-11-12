"""Synthetic financial-stage surrogate model and dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FINANCIAL_FEATURE_NAMES = [
    "vapor_fraction",
    "gas_flow_rate",
    "liquid_flow_rate",
    "price_per_unit",
    "hedge_ratio",
    "opex_multiplier",
]

FINANCIAL_TARGET_NAMES = [
    "net_profit",
    "risk_index",
]


def generate_financial_dataset(num_samples: int, seed: int | None = None) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vapor_fraction = rng.uniform(0.05, 0.95, size=num_samples)
    gas_flow = rng.uniform(2.0, 30.0, size=num_samples)
    liquid_flow = rng.uniform(2.0, 35.0, size=num_samples)
    price = rng.uniform(40.0, 110.0, size=num_samples)
    hedge_ratio = rng.uniform(0.0, 1.0, size=num_samples)
    opex_multiplier = rng.uniform(0.6, 1.4, size=num_samples)

    features = np.stack(
        [vapor_fraction, gas_flow, liquid_flow, price, hedge_ratio, opex_multiplier],
        axis=1,
    ).astype(np.float32)

    throughput = gas_flow + liquid_flow
    revenue = throughput * price * (0.4 + 0.6 * vapor_fraction)
    hedge_penalty = (1 - hedge_ratio) * price * 5.0
    opex = throughput * 12.0 * opex_multiplier
    net_profit = revenue - opex - hedge_penalty

    volatility = np.abs(vapor_fraction - 0.5) + 0.3 * hedge_ratio + 0.2 * opex_multiplier
    risk_index = np.clip(volatility * 10.0 + rng.normal(0, 0.5, size=num_samples), 0.5, 15.0)

    targets = np.stack([net_profit, risk_index], axis=1).astype(np.float32)
    return {"inputs": features, "targets": targets}


class FinancialSurrogate(nn.Module):
    def __init__(self, hidden_sizes: Iterable[int] = (64, 64)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = len(FINANCIAL_FEATURE_NAMES)
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.ReLU())
            input_dim = size
        layers.append(nn.Linear(input_dim, len(FINANCIAL_TARGET_NAMES)))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


@dataclass
class FinancialTrainingConfig:
    epochs: int = 150
    batch_size: int = 512
    learning_rate: float = 1e-3
    val_split: float = 0.2
    device: str = "cpu"


def train_financial_surrogate(
    dataset: Dict[str, np.ndarray],
    config: FinancialTrainingConfig,
    hidden_sizes: Tuple[int, ...] = (64, 64),
) -> Tuple[FinancialSurrogate, List[Dict[str, float]]]:
    device = torch.device(config.device)
    model = FinancialSurrogate(hidden_sizes).to(device)

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


def evaluate_financial(model: nn.Module, inputs: np.ndarray, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(inputs.astype(np.float32)).to(device)
        preds = model(tensor).cpu().numpy()
    return preds
