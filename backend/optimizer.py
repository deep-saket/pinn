"""
Stochastic Gaussian-search optimizer that operates on top of the surrogate model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class OptimizationConfig:
    max_iters: int = 30
    population: int = 24
    elite_fraction: float = 0.25
    lambda_reg: float = 0.01
    init_mean: Tuple[float, float] = (0.08, 0.5)
    init_std: Tuple[float, float] = (0.02, 0.2)
    std_floor: Tuple[float, float] = (0.005, 0.05)
    diffusion_bounds: Tuple[float, float] = (0.01, 0.2)
    velocity_bounds: Tuple[float, float] = (0.05, 1.5)
    device: str = "cpu"


@dataclass
class IterationRecord:
    iteration: int
    D: float
    v: float
    cost: float
    u_profile: List[float]
    grad_D: float
    grad_V: float
    grad_norm: float


@dataclass
class OptimizationResult:
    best_params: Tuple[float, float]
    best_cost: float
    best_profile: List[float]
    history: List[IterationRecord] = field(default_factory=list)


class GaussianOptimizer:
    """Gaussian sampling optimizer with mean/std adaptation."""

    def __init__(
        self,
        surrogate: torch.nn.Module,
        x_grid: np.ndarray,
        target_profile: np.ndarray,
        target_time: float,
        config: OptimizationConfig | None = None,
    ) -> None:
        self.config = config or OptimizationConfig()
        self.device = torch.device(self.config.device)
        self.surrogate = surrogate.to(self.device)
        self.x_grid = x_grid
        self.target_profile = target_profile
        self.target_time = target_time
        self.x_tensor = torch.from_numpy(x_grid).float().to(self.device)
        self.t_tensor = torch.full_like(self.x_tensor, target_time)
        self.target_tensor = torch.from_numpy(target_profile).float().to(self.device)
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _evaluate_candidate(self, D: float, v: float) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        ones = torch.ones_like(self.x_tensor)
        D_tensor = torch.tensor(D, dtype=torch.float32, device=self.device, requires_grad=True)
        v_tensor = torch.tensor(v, dtype=torch.float32, device=self.device, requires_grad=True)
        inputs = torch.stack(
            [
                self.x_tensor,
                self.t_tensor,
                ones * D_tensor,
                ones * v_tensor,
            ],
            dim=1,
        )
        preds = self.surrogate(inputs).squeeze(-1)
        diff = preds - self.target_tensor
        mse = torch.mean(diff**2)
        reg = self.config.lambda_reg * (D_tensor**2 + v_tensor**2)
        cost_tensor = mse + reg
        grad_D, grad_V = torch.autograd.grad(cost_tensor, (D_tensor, v_tensor))
        profile = preds.detach().cpu().numpy()
        cost = float(cost_tensor.item())
        gradients = (float(grad_D.item()), float(grad_V.item()))
        return profile, cost, gradients

    def run(self, callback: Optional[Callable[[IterationRecord], None]] = None) -> OptimizationResult:
        cfg = self.config
        rng = np.random.default_rng()

        mu = np.array(cfg.init_mean, dtype=np.float32)
        sigma = np.array(cfg.init_std, dtype=np.float32)

        history: List[IterationRecord] = []
        best_cost = float("inf")
        best_params = (float(mu[0]), float(mu[1]))
        best_profile: List[float] = []

        for iteration in range(cfg.max_iters):
            if self._stop_requested:
                break

            samples = rng.normal(mu, sigma, size=(cfg.population, 2))
            samples[:, 0] = np.clip(samples[:, 0], *cfg.diffusion_bounds)
            samples[:, 1] = np.clip(samples[:, 1], *cfg.velocity_bounds)

            evals: List[Tuple[float, np.ndarray, float, float, Tuple[float, float]]] = []
            for cand in samples:
                D, v = float(cand[0]), float(cand[1])
                profile, cost, gradients = self._evaluate_candidate(D, v)
                evals.append((cost, profile, D, v, gradients))

            evals.sort(key=lambda item: item[0])
            elite_count = max(1, int(cfg.population * cfg.elite_fraction))
            elites = evals[:elite_count]

            elite_params = np.array([[e[2], e[3]] for e in elites], dtype=np.float32)
            mu = elite_params.mean(axis=0)
            sigma = np.maximum(elite_params.std(axis=0), np.array(cfg.std_floor, dtype=np.float32))

            record = IterationRecord(
                iteration=iteration,
                D=float(elites[0][2]),
                v=float(elites[0][3]),
                cost=float(elites[0][0]),
                u_profile=elites[0][1].tolist(),
                grad_D=float(elites[0][4][0]),
                grad_V=float(elites[0][4][1]),
                grad_norm=float(math.sqrt(elites[0][4][0] ** 2 + elites[0][4][1] ** 2)),
            )
            history.append(record)

            if record.cost < best_cost:
                best_cost = record.cost
                best_params = (record.D, record.v)
                best_profile = record.u_profile

            if callback is not None:
                callback(record)

        return OptimizationResult(
            best_params=best_params,
            best_cost=best_cost,
            best_profile=best_profile,
            history=history,
        )
