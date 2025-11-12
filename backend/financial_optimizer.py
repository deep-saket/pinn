"""Gaussian-search optimizer for the financial stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class FinancialTargets:
    net_profit: Optional[float] = None
    risk_index: Optional[float] = None
    weight_profit: float = 1.0
    weight_risk: float = 0.5


@dataclass
class FinancialOptimizationConfig:
    max_iters: int = 20
    population: int = 20
    elite_fraction: float = 0.3
    init_mean: Tuple[float, float, float] = (80.0, 0.5, 1.0)
    init_std: Tuple[float, float, float] = (10.0, 0.2, 0.3)
    std_floor: Tuple[float, float, float] = (2.0, 0.05, 0.1)
    price_bounds: Tuple[float, float] = (40.0, 120.0)
    hedge_bounds: Tuple[float, float] = (0.0, 1.0)
    opex_bounds: Tuple[float, float] = (0.5, 1.6)
    lambda_reg: float = 1e-3
    device: str = "cpu"


@dataclass
class FinancialIterationRecord:
    iteration: int
    price_per_unit: float
    hedge_ratio: float
    opex_multiplier: float
    cost: float
    targets: Dict[str, float]
    gradients: Dict[str, float]


@dataclass
class FinancialOptimizationResult:
    best_controls: Dict[str, float]
    best_cost: float
    best_targets: Dict[str, float]
    history: List[FinancialIterationRecord] = field(default_factory=list)


class FinancialGaussianOptimizer:
    def __init__(
        self,
        model: nn.Module,
        facility_context: Dict[str, float],
        config: FinancialOptimizationConfig | None = None,
        targets: FinancialTargets | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.context = facility_context
        self.config = config or FinancialOptimizationConfig()
        self.targets = targets or FinancialTargets()
        self.device = torch.device(self.config.device)

    def _evaluate_candidate(
        self,
        price: float,
        hedge: float,
        opex: float,
    ) -> Tuple[Dict[str, float], float, Dict[str, float]]:
        self.model.to(self.device)
        ctx = torch.tensor(
            [
                self.context["vapor_fraction"],
                self.context["gas_flow_rate"],
                self.context["liquid_flow_rate"],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        price_tensor = torch.tensor(price, dtype=torch.float32, device=self.device, requires_grad=True)
        hedge_tensor = torch.tensor(hedge, dtype=torch.float32, device=self.device, requires_grad=True)
        opex_tensor = torch.tensor(opex, dtype=torch.float32, device=self.device, requires_grad=True)
        inputs = torch.cat([ctx, torch.stack([price_tensor, hedge_tensor, opex_tensor])]).unsqueeze(0)
        preds = self.model(inputs).squeeze(0)
        net_profit = preds[0]
        risk_index = preds[1]

        cost = torch.tensor(0.0, device=self.device)
        if self.targets.net_profit is not None:
            diff = net_profit - self.targets.net_profit
            cost = cost + self.targets.weight_profit * diff * diff
        if self.targets.risk_index is not None:
            diff = risk_index - self.targets.risk_index
            cost = cost + self.targets.weight_risk * diff * diff

        midpoint_price = sum(self.config.price_bounds) / 2.0
        midpoint_hedge = sum(self.config.hedge_bounds) / 2.0
        midpoint_opex = sum(self.config.opex_bounds) / 2.0
        cost = cost + self.config.lambda_reg * (
            ((price_tensor - midpoint_price) / 30.0) ** 2
            + ((hedge_tensor - midpoint_hedge) / 0.3) ** 2
            + ((opex_tensor - midpoint_opex) / 0.4) ** 2
        )

        grads = torch.autograd.grad(cost, (price_tensor, hedge_tensor, opex_tensor))
        metrics = {
            "net_profit": float(net_profit.detach().cpu().item()),
            "risk_index": float(risk_index.detach().cpu().item()),
        }
        gradients = {
            "dJ_dPrice": float(grads[0].item()),
            "dJ_dHedge": float(grads[1].item()),
            "dJ_dOpex": float(grads[2].item()),
        }
        return metrics, float(cost.item()), gradients

    def run(self) -> FinancialOptimizationResult:
        cfg = self.config
        rng = np.random.default_rng()
        mu = np.array(cfg.init_mean, dtype=np.float32)
        sigma = np.array(cfg.init_std, dtype=np.float32)

        history: List[FinancialIterationRecord] = []
        best_cost = float("inf")
        best_controls = {
            "price_per_unit": float(mu[0]),
            "hedge_ratio": float(mu[1]),
            "opex_multiplier": float(mu[2]),
        }
        best_targets = {}

        for iteration in range(cfg.max_iters):
            samples = rng.normal(mu, sigma, size=(cfg.population, 3))
            samples[:, 0] = np.clip(samples[:, 0], *cfg.price_bounds)
            samples[:, 1] = np.clip(samples[:, 1], *cfg.hedge_bounds)
            samples[:, 2] = np.clip(samples[:, 2], *cfg.opex_bounds)

            evaluations: List[Tuple[float, Tuple[float, float, float], Dict[str, float], Dict[str, float]]] = []
            for price, hedge, opex in samples:
                metrics, cost, gradients = self._evaluate_candidate(float(price), float(hedge), float(opex))
                evaluations.append((cost, (price, hedge, opex), metrics, gradients))

            evaluations.sort(key=lambda item: item[0])
            elite_count = max(1, int(cfg.population * cfg.elite_fraction))
            elites = evaluations[:elite_count]
            elite_params = np.array([elite[1] for elite in elites], dtype=np.float32)
            mu = elite_params.mean(axis=0)
            sigma = np.maximum(elite_params.std(axis=0), np.array(cfg.std_floor, dtype=np.float32))

            best_eval = elites[0]
            record = FinancialIterationRecord(
                iteration=iteration,
                price_per_unit=float(best_eval[1][0]),
                hedge_ratio=float(best_eval[1][1]),
                opex_multiplier=float(best_eval[1][2]),
                cost=float(best_eval[0]),
                targets=best_eval[2],
                gradients=best_eval[3],
            )
            history.append(record)

            if record.cost < best_cost:
                best_cost = record.cost
                best_controls = {
                    "price_per_unit": record.price_per_unit,
                    "hedge_ratio": record.hedge_ratio,
                    "opex_multiplier": record.opex_multiplier,
                }
                best_targets = record.targets

        return FinancialOptimizationResult(
            best_controls=best_controls,
            best_cost=best_cost,
            best_targets=best_targets,
            history=history,
        )
