"""Utility CLI for exporting datasets, training the surrogate, and running offline optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from optimizer import GaussianOptimizer, OptimizationConfig
from simulator import SimulationConfig, generate_training_data, run_simulation
from surrogate_model import SurrogatePINN, SurrogateTrainingConfig, train_surrogate


def _dataset_to_dataframe(dataset: Dict[str, np.ndarray]) -> pd.DataFrame:
    inputs = dataset["inputs"]
    targets = dataset["targets"].squeeze(-1)
    data = np.concatenate([inputs, targets[:, None]], axis=1)
    df = pd.DataFrame(data, columns=["x", "t", "D", "v", "u"])
    return df


def _save_dataset(dataset: Dict[str, np.ndarray], fmt: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {
            "inputs": dataset["inputs"].tolist(),
            "targets": dataset["targets"].tolist(),
        }
        output.write_text(json.dumps(payload))
    elif fmt == "excel":
        df = _dataset_to_dataframe(dataset)
        df.to_excel(output, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def export_data(args: argparse.Namespace) -> None:
    cfg = SimulationConfig()
    dataset = generate_training_data(args.num_systems, cfg, seed=args.seed)
    _save_dataset(dataset, args.format, Path(args.output))
    print(f"Exported dataset with {len(dataset['inputs'])} samples to {args.output}")


def _load_dataset(path: Path, fmt: str) -> Dict[str, np.ndarray]:
    if fmt == "json":
        payload = json.loads(path.read_text())
        inputs = np.array(payload["inputs"], dtype=np.float32)
        targets = np.array(payload["targets"], dtype=np.float32)
    elif fmt == "excel":
        df = pd.read_excel(path)
        inputs = df[["x", "t", "D", "v"]].to_numpy(dtype=np.float32)
        targets = df[["u"]].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return {"inputs": inputs, "targets": targets}


def train_offline(args: argparse.Namespace) -> None:
    dataset = _load_dataset(Path(args.data), args.format)
    training_cfg = SurrogateTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pde_weight=args.pde_weight,
        val_split=args.val_split,
    )
    model, history = train_surrogate(dataset, training_cfg)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"Saved surrogate weights to {args.model_out}")
    if args.history_out:
        path = Path(args.history_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=2))
        print(f"Training history written to {args.history_out}")


def _history_to_dataframe(history: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for rec in history:
        rows.append(
            {
                "iteration": rec["iteration"],
                "D": rec["D"],
                "v": rec["v"],
                "cost": rec["cost"],
                "u_profile": json.dumps(rec["u_profile"]),
            }
        )
    return pd.DataFrame(rows)


def _save_history(history: List[Dict[str, object]], result: Dict[str, object], fmt: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {"history": history, "result": result}
        output.write_text(json.dumps(payload, indent=2))
    elif fmt == "excel":
        df = _history_to_dataframe(history)
        with pd.ExcelWriter(output) as writer:
            df.to_excel(writer, sheet_name="iterations", index=False)
            summary = pd.DataFrame(
                [
                    {
                        "best_D": result["best_params"]["D"],
                        "best_v": result["best_params"]["v"],
                        "best_cost": result["best_cost"],
                    }
                ]
            )
            summary.to_excel(writer, sheet_name="summary", index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def optimize_offline(args: argparse.Namespace) -> None:
    model = SurrogatePINN()
    state_dict = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    cfg = SimulationConfig()
    target_sim = run_simulation(0.12, 0.65, cfg)
    optimizer = GaussianOptimizer(
        surrogate=model,
        x_grid=target_sim["x"],
        target_profile=target_sim["u"][-1],
        target_time=float(target_sim["t"][-1]),
        config=OptimizationConfig(max_iters=args.max_iters, population=args.population),
    )
    result = optimizer.run()
    history = [
        {
            "iteration": rec.iteration,
            "D": rec.D,
            "v": rec.v,
            "cost": rec.cost,
            "u_profile": rec.u_profile,
        }
        for rec in result.history
    ]
    summary = {
        "best_params": {"D": result.best_params[0], "v": result.best_params[1]},
        "best_cost": result.best_cost,
        "best_profile": result.best_profile,
    }
    _save_history(history, summary, args.history_format, Path(args.history_out))
    print(f"Optimization history saved to {args.history_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline utilities for the Geminus demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_cmd = subparsers.add_parser("export-data", help="Generate synthetic data and save as JSON/Excel")
    export_cmd.add_argument("--num-systems", type=int, default=10)
    export_cmd.add_argument("--format", choices=["json", "excel"], default="json")
    export_cmd.add_argument("--output", required=True)
    export_cmd.add_argument("--seed", type=int)
    export_cmd.set_defaults(func=export_data)

    train_cmd = subparsers.add_parser("train", help="Train surrogate offline from a saved dataset")
    train_cmd.add_argument("--data", required=True, help="Path to dataset file")
    train_cmd.add_argument("--format", choices=["json", "excel"], default="json")
    train_cmd.add_argument("--epochs", type=int, default=200)
    train_cmd.add_argument("--batch-size", type=int, default=512)
    train_cmd.add_argument("--lr", type=float, default=1e-3)
    train_cmd.add_argument("--pde-weight", type=float, default=0.1)
    train_cmd.add_argument("--val-split", type=float, default=0.2, help="Fraction of data reserved for validation loss tracking")
    train_cmd.add_argument("--model-out", required=True)
    train_cmd.add_argument("--history-out", help="Optional path to save training loss history as JSON")
    train_cmd.set_defaults(func=train_offline)

    opt_cmd = subparsers.add_parser("optimize", help="Run optimization offline using a trained model")
    opt_cmd.add_argument("--model", required=True, help="Path to surrogate weights (.pt)")
    opt_cmd.add_argument("--history-out", required=True, help="Output path for iteration history")
    opt_cmd.add_argument("--history-format", choices=["json", "excel"], default="json")
    opt_cmd.add_argument("--max-iters", type=int, default=30)
    opt_cmd.add_argument("--population", type=int, default=24)
    opt_cmd.set_defaults(func=optimize_offline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
