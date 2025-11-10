"""
Simulation utilities for the dummy Geminus-style pipeline demonstrator.

The module exposes a light-weight finite-difference solver for the 1-D
diffusion–advection equation together with helper routines that synthesize
training data for the surrogate model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for the 1-D diffusion–advection solver."""

    length: float = 1.0
    n_points: int = 64
    dt: float = 0.01
    steps: int = 60
    source_center: float = 0.35
    source_width: float = 0.08
    noise_std: float = 0.002
    diffusion_range: Tuple[float, float] = (0.01, 0.2)
    velocity_range: Tuple[float, float] = (0.1, 1.0)


def _initial_condition(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """Gaussian bump that mimics a pressure disturbance."""

    return np.exp(-0.5 * ((x - center) / width) ** 2)


def run_simulation(D: float, v: float, config: SimulationConfig) -> Dict[str, np.ndarray]:
    """
    Integrate the diffusion–advection equation with simple finite differences.

    Args:
        D: Diffusion coefficient.
        v: Flow velocity.
        config: Simulation parameters.

    Returns:
        Dictionary containing the spatial grid, time stamps, and the solution history.
    """

    x = np.linspace(0.0, config.length, config.n_points)
    dx = x[1] - x[0]
    u = _initial_condition(x, config.source_center, config.source_width)

    history: List[np.ndarray] = [u.copy()]

    for _ in range(config.steps):
        laplacian = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx**2)
        advection = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
        u = u + config.dt * (D * laplacian - v * advection)

        # Enforce zero-flux boundary conditions.
        u[0] = u[1]
        u[-1] = u[-2]

        history.append(u.copy())

    solution = np.stack(history, axis=0)
    t = np.linspace(0.0, config.dt * config.steps, config.steps + 1)

    if config.noise_std > 0.0:
        solution += np.random.normal(0.0, config.noise_std, size=solution.shape)

    # Ensure the synthetic field stays numerically well-behaved.
    solution = np.nan_to_num(solution, nan=0.0, posinf=5.0, neginf=-5.0)
    np.clip(solution, -5.0, 5.0, out=solution)

    return {"x": x, "t": t, "u": solution}


def generate_training_data(
    num_systems: int,
    config: SimulationConfig,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Build a synthetic dataset by sampling random (D, v) pairs and snapshots.

    Args:
        num_systems: Number of distinct simulations to draw.
        config: Shared simulation configuration.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary with stacked inputs and targets ready for surrogate training.
    """

    rng = np.random.default_rng(seed)
    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _ in range(num_systems):
        D = rng.uniform(*config.diffusion_range)
        v = rng.uniform(*config.velocity_range)
        sim = run_simulation(D, v, config)

        # Sample a few time slices per simulation to keep the dataset small.
        time_indices = rng.choice(len(sim["t"]), size=5, replace=True)
        for idx in time_indices:
            t_val = sim["t"][idx]
            x_vals = sim["x"]
            field = sim["u"][idx]

            stacked = np.stack(
                [
                    x_vals,
                    np.full_like(x_vals, t_val),
                    np.full_like(x_vals, D),
                    np.full_like(x_vals, v),
                ],
                axis=-1,
            )
            inputs.append(stacked)
            targets.append(field[:, None])

    all_inputs = np.concatenate(inputs, axis=0)
    all_targets = np.concatenate(targets, axis=0)

    return {"inputs": all_inputs.astype(np.float32), "targets": all_targets.astype(np.float32)}
