"""Sequential pipeline utilities for pipe → facility stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from surrogate_model import evaluate_profile
from facility_surrogate import evaluate_facility
from financial_surrogate import evaluate_financial


@dataclass
class PipeControls:
    diffusion: float
    velocity: float
    time: float


@dataclass
class FacilityControls:
    separator_pressure: float
    separator_temp: float
    gas_fraction: float


@dataclass
class FinancialControls:
    price_per_unit: float
    hedge_ratio: float
    opex_multiplier: float


@dataclass
class PipeStageOutput:
    x: List[float]
    profile: List[float]
    outlet_pressure: float
    mean_pressure: float
    du_dx: List[float]


@dataclass
class FacilityStageOutput:
    vapor_fraction: float
    gas_flow_rate: float
    liquid_flow_rate: float


@dataclass
class FinancialStageOutput:
    net_profit: float
    risk_index: float


def _finite_difference(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    gradients = np.zeros_like(values)
    if len(values) < 2:
        return gradients
    gradients[0] = (values[1] - values[0]) / max(x[1] - x[0], 1e-6)
    gradients[-1] = (values[-1] - values[-2]) / max(x[-1] - x[-2], 1e-6)
    if len(values) > 2:
        gradients[1:-1] = (values[2:] - values[:-2]) / np.maximum(x[2:] - x[:-2], 1e-6)
    return gradients


def _pipe_to_facility_features(
    pipe_outlet_pressure: float,
    pipe_controls: PipeControls,
    facility_controls: FacilityControls,
) -> np.ndarray:
    pipe_temp = 30.0 + 0.8 * pipe_controls.velocity * 40.0
    throughput = max(5.0, 10.0 * pipe_controls.diffusion + 2.0 * pipe_controls.velocity)
    return np.array(
        [
            pipe_outlet_pressure,
            pipe_temp,
            throughput,
            facility_controls.separator_pressure,
            facility_controls.separator_temp,
            facility_controls.gas_fraction,
        ],
        dtype=np.float32,
    )


def run_pipeline(
    pipe_controls: PipeControls,
    facility_controls: FacilityControls,
    financial_controls: FinancialControls | None,
    pipe_surrogate,
    facility_surrogate,
    financial_surrogate,
    x_grid: np.ndarray,
) -> Dict[str, Dict[str, float | List[float]]]:
    if x_grid is None or len(x_grid) == 0:
        raise ValueError("x_grid is required for pipe evaluation")

    pipe_profile = evaluate_profile(
        pipe_surrogate,
        x=x_grid.astype(np.float32),
        t=pipe_controls.time,
        D=pipe_controls.diffusion,
        v=pipe_controls.velocity,
        device="cpu",
    )
    du_dx = _finite_difference(x_grid, pipe_profile)
    outlet_pressure = float(pipe_profile[-1])
    mean_pressure = float(np.mean(pipe_profile))

    facility_features = _pipe_to_facility_features(outlet_pressure, pipe_controls, facility_controls)
    facility_preds = evaluate_facility(
        facility_surrogate,
        facility_features[None, :],
        device="cpu",
    )[0]

    pipe_stage = PipeStageOutput(
        x=x_grid.tolist(),
        profile=pipe_profile.tolist(),
        outlet_pressure=outlet_pressure,
        mean_pressure=mean_pressure,
        du_dx=du_dx.tolist(),
    )

    facility_stage = FacilityStageOutput(
        vapor_fraction=float(facility_preds[0]),
        gas_flow_rate=float(facility_preds[1]),
        liquid_flow_rate=float(facility_preds[2]),
    )

    result: Dict[str, Dict[str, float | List[float]]] = {
        "pipe": pipe_stage.__dict__,
        "facility": facility_stage.__dict__,
        "facility_features": facility_features.tolist(),
    }

    if financial_surrogate is not None and financial_controls is not None:
        fin_features = np.array(
            [
                facility_stage.vapor_fraction,
                facility_stage.gas_flow_rate,
                facility_stage.liquid_flow_rate,
                financial_controls.price_per_unit,
                financial_controls.hedge_ratio,
                financial_controls.opex_multiplier,
            ],
            dtype=np.float32,
        )
        financial_preds = evaluate_financial(
            financial_surrogate,
            fin_features[None, :],
            device="cpu",
        )[0]
        financial_stage = FinancialStageOutput(
            net_profit=float(financial_preds[0]),
            risk_index=float(financial_preds[1]),
        )
        result["financial"] = financial_stage.__dict__
        result["financial_features"] = fin_features.tolist()

    return result
