"""
Gaussian-search optimizer for the facility surrogate stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class FacilityOptimizationTargets:
    vapor_fraction: Optional[float] = 0.65
    gas_flow_rate: Optional[float] = None
    liquid_flow_rate: Optional[float] = None
    weight_vapor: float = 1.0
    weight_gas: float = 0.3
    weight_liquid: float = 0.2


@dataclass
class FacilityOptimizationConfig:
    max_iters: int = 20
    population: int = 20
    elite_fraction: float = 0.3
    init_mean: Tuple[float, float, float] = (100.0, 45.0, 0.5)
    init_std: Tuple[float, float, float] = (12.0, 8.0, 0.15)
    std_floor: Tuple[float, float, float] = (2.0, 2.0, 0.05)
    pressure_bounds: Tuple[float, float] = (50.0, 150.0)
    temp_bounds: Tuple[float, float] = (20.0, 90.0)
    gas_fraction_bounds: Tuple[float, float] = (0.1, 0.9)
    lambda_reg: float = 1e-3
    device: str = "cpu"


@dataclass
class FacilityIterationRecord:
    iteration: int
    separator_pressure: float
    separator_temp: float
    gas_fraction: float
    cost: float
    metrics: Dict[str, float]
    gradients: Dict[str, float]


@dataclass
class FacilityOptimizationResult:
    best_controls: Dict[str, float]
    best_cost: float
    best_metrics: Dict[str, float]
    history: List[FacilityIterationRecord] = field(default_factory=list)


class FacilityGaussianOptimizer:
    """Gaussian sampling optimizer for facility-stage knobs."""

    def __init__(
        self,
        facility_model: nn.Module,
        pipe_context: Dict[str, float],
        config: FacilityOptimizationConfig | None = None,
        targets: FacilityOptimizationTargets | None = None,
    ) -> None:
        self.model = facility_model
        self.model.eval()
        self.pipe_context = pipe_context
        self.config = config or FacilityOptimizationConfig()
        self.targets = targets or FacilityOptimizationTargets()
        self.device = torch.device(self.config.device)

    def _evaluate_candidate(
        self,
        separator_pressure: float,
        separator_temp: float,
        gas_fraction: float,
    ) -> Tuple[Dict[str, float], float, Dict[str, float]]:
        self.model.to(self.device)
        context = torch.tensor(
            [
                self.pipe_context["outlet_pressure"],
                self.pipe_context["temperature"],
                self.pipe_context["throughput"],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        p_tensor = torch.tensor(separator_pressure, dtype=torch.float32, device=self.device, requires_grad=True)
        t_tensor = torch.tensor(separator_temp, dtype=torch.float32, device=self.device, requires_grad=True)
        g_tensor = torch.tensor(gas_fraction, dtype=torch.float32, device=self.device, requires_grad=True)
        inputs = torch.cat([context, torch.stack([p_tensor, t_tensor, g_tensor])]).unsqueeze(0)
        preds = self.model(inputs).squeeze(0)
        metrics = {
            "vapor_fraction": float(preds[0].item()),
            "gas_flow_rate": float(preds[1].item()),
            "liquid_flow_rate": float(preds[2].item()),
        }

        cost = torch.tensor(0.0, device=self.device)
        if self.targets.vapor_fraction is not None:
            diff = preds[0] - self.targets.vapor_fraction
            cost = cost + self.targets.weight_vapor * diff * diff
        if self.targets.gas_flow_rate is not None:
            diff = preds[1] - self.targets.gas_flow_rate
            cost = cost + self.targets.weight_gas * diff * diff
        if self.targets.liquid_flow_rate is not None:
            diff = preds[2] - self.targets.liquid_flow_rate
            cost = cost + self.targets.weight_liquid * diff * diff

        mid_pressure = sum(self.config.pressure_bounds) / 2.0
        mid_temp = sum(self.config.temp_bounds) / 2.0
        mid_gas = sum(self.config.gas_fraction_bounds) / 2.0
        cost = cost + self.config.lambda_reg * (
            ((p_tensor - mid_pressure) / 50.0) ** 2
            + ((t_tensor - mid_temp) / 40.0) ** 2
            + ((g_tensor - mid_gas) / 0.4) ** 2
        )

        grads = torch.autograd.grad(cost, (p_tensor, t_tensor, g_tensor))
        gradients = {
            "dJ_dP": float(grads[0].item()),
            "dJ_dT": float(grads[1].item()),
            "dJ_dGas": float(grads[2].item()),
        }
        return metrics, float(cost.item()), gradients

    def run(self) -> FacilityOptimizationResult:
        cfg = self.config
        rng = np.random.default_rng()
        mu = np.array(cfg.init_mean, dtype=np.float32)
        sigma = np.array(cfg.init_std, dtype=np.float32)

        history: List[FacilityIterationRecord] = []
        best_cost = float("inf")
        best_controls = {
            "separator_pressure": float(mu[0]),
            "separator_temp": float(mu[1]),
            "gas_fraction": float(mu[2]),
        }
        best_metrics: Dict[str, float] = {}

        for iteration in range(cfg.max_iters):
            samples = rng.normal(mu, sigma, size=(cfg.population, 3))
            samples[:, 0] = np.clip(samples[:, 0], *cfg.pressure_bounds)
            samples[:, 1] = np.clip(samples[:, 1], *cfg.temp_bounds)
            samples[:, 2] = np.clip(samples[:, 2], *cfg.gas_fraction_bounds)

            evaluations: List[Tuple[float, Tuple[float, float, float], Dict[str, float], Dict[str, float]]] = []
            for cand in samples:
                pressure, temp, gas = map(float, cand)
                metrics, cost, gradients = self._evaluate_candidate(pressure, temp, gas)
                evaluations.append((cost, (pressure, temp, gas), metrics, gradients))

            evaluations.sort(key=lambda item: item[0])
            elite_count = max(1, int(cfg.population * cfg.elite_fraction))
            elites = evaluations[:elite_count]
            elite_params = np.array([elite[1] for elite in elites], dtype=np.float32)
            mu = elite_params.mean(axis=0)
            sigma = np.maximum(elite_params.std(axis=0), np.array(cfg.std_floor, dtype=np.float32))

            best_eval = elites[0]
            record = FacilityIterationRecord(
                iteration=iteration,
                separator_pressure=float(best_eval[1][0]),
                separator_temp=float(best_eval[1][1]),
                gas_fraction=float(best_eval[1][2]),
                cost=float(best_eval[0]),
                metrics=best_eval[2],
                gradients=best_eval[3],
            )
            history.append(record)

            if record.cost < best_cost:
                best_cost = record.cost
                best_controls = {
                    "separator_pressure": record.separator_pressure,
                    "separator_temp": record.separator_temp,
                    "gas_fraction": record.gas_fraction,
                }
                best_metrics = record.metrics

        return FacilityOptimizationResult(
            best_controls=best_controls,
            best_cost=best_cost,
            best_metrics=best_metrics,
            history=history,
        )
