from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np

from optimizer import GaussianOptimizer, OptimizationConfig
from simulator import SimulationConfig, generate_training_data, run_simulation
from surrogate_model import (
    SurrogatePINN,
    SurrogateTrainingConfig,
    evaluate_profile,
    train_surrogate,
)


logger = logging.getLogger("geminus.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_JSON = DATA_DIR / "surrogate_latest.json"
DATA_XLSX = DATA_DIR / "surrogate_latest.xlsx"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "surrogate_latest.pt"
TRAINING_LOG_PATH = ARTIFACT_DIR / "surrogate_training_history.json"
EXPORT_DIR = Path(__file__).resolve().parent / "exports"
OPT_JSON = EXPORT_DIR / "optimization_latest.json"
OPT_XLSX = EXPORT_DIR / "optimization_latest.xlsx"
TARGET_DIFFUSION = 0.12
TARGET_VELOCITY = 0.65


def _dataset_to_dataframe(dataset: Dict[str, np.ndarray]) -> pd.DataFrame:
    data = np.concatenate([dataset["inputs"], dataset["targets"]], axis=1)
    return pd.DataFrame(data, columns=["x", "t", "D", "v", "u"])


def _persist_dataset(dataset: Dict[str, np.ndarray]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "inputs": dataset["inputs"].tolist(),
        "targets": dataset["targets"].tolist(),
    }
    DATA_JSON.write_text(json.dumps(payload))
    df = _dataset_to_dataframe(dataset)
    df.to_excel(DATA_XLSX, index=False)
    logger.info("Dataset persisted to %s and %s", DATA_JSON, DATA_XLSX)


def _persist_model(model: SurrogatePINN) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Surrogate weights saved to %s", MODEL_PATH)


def _persist_training_history(history: List[Dict[str, float]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_PATH.write_text(json.dumps(history, indent=2))
    logger.info("Training history saved to %s (%d epochs)", TRAINING_LOG_PATH, len(history))


def _persist_optimization_history(history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    OPT_JSON.write_text(json.dumps({"history": history, "result": result}, indent=2))
    df = pd.DataFrame(
        [
            {
                "iteration": rec["iteration"],
                "D": rec["D"],
                "v": rec["v"],
                "cost": rec["cost"],
                "grad_D": rec.get("grad_D"),
                "grad_V": rec.get("grad_V"),
                "grad_norm": rec.get("grad_norm"),
                "u_profile": json.dumps(rec["u_profile"]),
            }
            for rec in history
        ]
    )


def _build_reference_state(sim_cfg: SimulationConfig) -> Dict[str, Any]:
    target_sim = run_simulation(TARGET_DIFFUSION, TARGET_VELOCITY, sim_cfg)
    return {
        "target_profile": target_sim["u"][-1],
        "target_time": float(target_sim["t"][-1]),
        "x_grid": target_sim["x"],
        "target_params": {"D": TARGET_DIFFUSION, "v": TARGET_VELOCITY},
    }


def _apply_reference_state(reference: Dict[str, Any]) -> None:
    state["target_profile"] = reference["target_profile"]
    state["target_time"] = reference["target_time"]
    state["x_grid"] = reference["x_grid"]
    state["target_params"] = reference["target_params"]


def _ensure_surrogate_ready() -> bool:
    try:
        _get_surrogate_or_load()
    except HTTPException:
        return False

    if (
        state.get("target_profile") is None
        or state.get("target_time") is None
        or state.get("x_grid") is None
        or state.get("target_params") is None
    ):
        reference = _build_reference_state(state["simulation_config"])
        _apply_reference_state(reference)
    return True
    summary = pd.DataFrame(
        [
            {
                "best_D": result["best_params"]["D"],
                "best_v": result["best_params"]["v"],
                "best_cost": result["best_cost"],
            }
        ]
    )
    with pd.ExcelWriter(OPT_XLSX) as writer:
        df.to_excel(writer, sheet_name="iterations", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    logger.info(
        "Optimization history saved to %s and %s (%d iterations)",
        OPT_JSON,
        OPT_XLSX,
        len(history),
    )


def _load_persisted_dataset() -> Dict[str, np.ndarray]:
    if not DATA_JSON.exists():
        raise HTTPException(status_code=404, detail="No persisted dataset found. Run /simulate or provide dataset payload.")
    payload = json.loads(DATA_JSON.read_text())
    return _dataset_from_payload(payload)


def _dataset_from_payload(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    try:
        inputs = np.array(payload["inputs"], dtype=np.float32)
        targets = np.array(payload["targets"], dtype=np.float32)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Dataset missing key: {exc}") from exc
    if inputs.ndim != 2 or inputs.shape[1] != 4:
        raise HTTPException(status_code=400, detail="Inputs must be shape (N, 4) with columns [x,t,D,v].")
    if targets.ndim != 2 or targets.shape[1] != 1:
        raise HTTPException(status_code=400, detail="Targets must be shape (N, 1) with scalar u values.")
    return {"inputs": inputs, "targets": targets}


def _load_dataset_from_path(file_path: str) -> Dict[str, np.ndarray]:
    base_dir = Path(__file__).resolve().parent
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        return _dataset_from_payload(payload)
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
        try:
            inputs = df[["x", "t", "D", "v"]].to_numpy(dtype=np.float32)
            targets = df[["u"]].to_numpy(dtype=np.float32)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail="Excel file must contain columns x,t,D,v,u") from exc
        return {"inputs": inputs, "targets": targets}
    raise HTTPException(status_code=400, detail="Unsupported dataset format. Use .json or .xlsx")


def _get_training_history() -> Optional[List[Dict[str, float]]]:
    history = state.get("training_history")
    if history is None and TRAINING_LOG_PATH.exists():
        history = json.loads(TRAINING_LOG_PATH.read_text())
        state["training_history"] = history
    return history


def _train_surrogate_with_dataset(
    dataset: Dict[str, np.ndarray],
    training_cfg: SurrogateTrainingConfig,
    sim_cfg: SimulationConfig,
) -> Dict[str, Any]:
    model, history = train_surrogate(dataset, training_cfg)
    _persist_model(model)
    _persist_training_history(history)

    timestamp = datetime.now(timezone.utc).isoformat()
    dataset_size = len(dataset["inputs"])

    state["training_data"] = dataset
    state["surrogate"] = model
    reference = _build_reference_state(sim_cfg)
    _apply_reference_state(reference)
    state["training_history"] = history
    state["dataset_size"] = dataset_size
    state["last_trained_at"] = timestamp

    logger.info(
        "Surrogate ready: %s samples, target D=%.3f v=%.3f",
        len(dataset["inputs"]),
        reference["target_params"]["D"],
        reference["target_params"]["v"],
    )

    return {
        "dataset_size": dataset_size,
        "target_time": reference["target_time"],
        "x_grid": reference["x_grid"].tolist(),
        "target_profile": reference["target_profile"].tolist(),
        "training_history": history,
        "last_trained_at": timestamp,
        "last_trained_epochs": len(history),
    }


def _get_surrogate_or_load() -> SurrogatePINN:
    surrogate = state.get("surrogate")
    if surrogate is not None:
        return surrogate
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No trained surrogate available. Run /simulate or /surrogate/train first.")
    model = SurrogatePINN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    state["surrogate"] = model
    logger.info("Loaded surrogate weights from %s", MODEL_PATH)
    return model


class StreamBroker:
    """Minimal pub/sub helper for WebSocket clients."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()

    async def register(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._queues.add(queue)
        return queue

    async def unregister(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        for queue in list(self._queues):
            await queue.put(message)

    def broadcast_threadsafe(self, message: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)


class SimulateRequest(BaseModel):
    num_systems: int = 10
    seed: Optional[int] = None


class OptimizeRequest(BaseModel):
    max_iters: int = 30
    population: int = 24
    initial_D: Optional[float] = None
    initial_V: Optional[float] = None


class TrainSurrogateRequest(BaseModel):
    dataset: Optional[Dict[str, Any]] = None
    dataset_path: Optional[str] = None
    epochs: int = 120
    batch_size: int = 256
    learning_rate: float = 1e-3
    pde_weight: float = 0.1


class PredictRequest(BaseModel):
    x: list[float]
    t: float
    D: float
    v: float


class SurrogateStatusResponse(BaseModel):
    has_model: bool
    model_path: Optional[str]
    dataset_size: Optional[int]
    last_trained_epochs: Optional[int]
    last_trained_at: Optional[str]
    training_history: Optional[List[Dict[str, Any]]]
    sample_prediction: Optional[Dict[str, Any]]


app = FastAPI(title="Dummy Geminus Optimization API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

broker = StreamBroker()

state: Dict[str, Any] = {
    "simulation_config": SimulationConfig(),
    "training_data": None,
    "surrogate": None,
    "target_profile": None,
    "target_time": None,
    "x_grid": None,
    "target_params": None,
    "optimizer_future": None,
    "optimizer": None,
    "running": False,
    "history": [],
    "result": None,
    "training_history": None,
    "dataset_size": None,
    "last_trained_at": None,
}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/simulate")
@app.post("/simulate-train")
async def simulate_and_train(request: SimulateRequest) -> Dict[str, Any]:
    cfg: SimulationConfig = state["simulation_config"]
    logger.info("/simulate requested: num_systems=%s seed=%s", request.num_systems, request.seed)
    dataset = generate_training_data(request.num_systems, cfg, seed=request.seed)
    _persist_dataset(dataset)
    response = _train_surrogate_with_dataset(
        dataset,
        SurrogateTrainingConfig(epochs=120, batch_size=256),
        cfg,
    )

    return {
        "message": "surrogate trained",
        **response,
    }


def _ensure_ready() -> None:
    if not _ensure_surrogate_ready():
        raise HTTPException(
            status_code=400,
            detail="No trained surrogate available. Run /simulate-train or /surrogate/train first.",
        )


@app.post("/optimize")
async def start_optimization(request: OptimizeRequest) -> Dict[str, str]:
    _ensure_ready()
    if state["running"]:
        raise HTTPException(status_code=400, detail="Optimization already running.")

    loop = asyncio.get_running_loop()
    surrogate: SurrogatePINN = state["surrogate"]
    surrogate.eval()

    defaults = OptimizationConfig()
    init_mean = (
        request.initial_D if request.initial_D is not None else defaults.init_mean[0],
        request.initial_V if request.initial_V is not None else defaults.init_mean[1],
    )

    opt_cfg = OptimizationConfig(
        max_iters=request.max_iters,
        population=request.population,
        init_mean=init_mean,
        device="cpu" if not torch.cuda.is_available() else "cuda",
    )
    optimizer = GaussianOptimizer(
        surrogate=surrogate,
        x_grid=state["x_grid"],
        target_profile=state["target_profile"],
        target_time=state["target_time"],
        config=opt_cfg,
    )

    state["optimizer"] = optimizer
    state["running"] = True
    state["history"] = []
    state["result"] = None

    logger.info(
        "Optimization started: max_iters=%s population=%s init_mean=(%.3f, %.3f)",
        opt_cfg.max_iters,
        opt_cfg.population,
        opt_cfg.init_mean[0],
        opt_cfg.init_mean[1],
    )

    def _callback(record) -> None:
        payload = {
            "iteration": record.iteration,
            "D": record.D,
            "v": record.v,
            "cost": record.cost,
            "u_profile": record.u_profile,
            "grad_D": record.grad_D,
            "grad_V": record.grad_V,
            "grad_norm": record.grad_norm,
        }
        state["history"].append(payload)
        broker.broadcast_threadsafe({"type": "iteration", "payload": payload}, loop)
        logger.info(
            "Iter %s: cost=%.4f D=%.3f v=%.3f",
            record.iteration,
            record.cost,
            record.D,
            record.v,
        )

    def _run():
        return optimizer.run(callback=_callback)

    future = loop.run_in_executor(None, _run)
    state["optimizer_future"] = future

    async def _monitor() -> None:
        result = await future
        state["running"] = False
        state["result"] = {
            "best_params": {"D": result.best_params[0], "v": result.best_params[1]},
            "best_cost": result.best_cost,
            "best_profile": result.best_profile,
        }
        _persist_optimization_history(state["history"], state["result"])
        broker.broadcast_threadsafe({"type": "complete", "payload": state["result"]}, loop)
        logger.info(
            "Optimization finished: best_cost=%.4f D=%.3f v=%.3f",
            result.best_cost,
            result.best_params[0],
            result.best_params[1],
        )

    asyncio.create_task(_monitor())
    return {"status": "started"}


@app.post("/optimize/stop")
async def stop_optimization() -> Dict[str, str]:
    if not state["running"]:
        return {"status": "idle"}
    optimizer: GaussianOptimizer = state["optimizer"]
    if optimizer:
        optimizer.request_stop()
        logger.info("Stop requested for optimization loop")
    return {"status": "stopping"}


@app.get("/dataset")
async def latest_dataset() -> Dict[str, Any]:
    dataset = state.get("training_data")
    if dataset is None:
        raise HTTPException(status_code=404, detail="No dataset generated yet. Run /simulate first.")
    logger.info("/dataset requested: returning %s samples", len(dataset["inputs"]))
    return {
        "inputs": dataset["inputs"].tolist(),
        "targets": dataset["targets"].tolist(),
    }


@app.get("/surrogate/history")
async def surrogate_history() -> Dict[str, Any]:
    history = _get_training_history()
    if history is None:
        raise HTTPException(status_code=404, detail="No training history available. Train the surrogate first.")
    return {"training_history": history}


@app.post("/surrogate/train")
async def train_surrogate_api(request: TrainSurrogateRequest) -> Dict[str, Any]:
    cfg: SimulationConfig = state["simulation_config"]
    if request.dataset is not None:
        dataset = _dataset_from_payload(request.dataset)
        logger.info("/surrogate/train using inline dataset (%s samples)", len(dataset["inputs"]))
    elif request.dataset_path is not None:
        dataset = _load_dataset_from_path(request.dataset_path)
        logger.info("/surrogate/train loading dataset from %s", request.dataset_path)
    else:
        dataset = _load_persisted_dataset()
        logger.info("/surrogate/train loading persisted dataset %s", DATA_JSON)

    training_cfg = SurrogateTrainingConfig(
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        pde_weight=request.pde_weight,
    )

    response = _train_surrogate_with_dataset(dataset, training_cfg, cfg)
    return {
        "message": "surrogate trained",
        **response,
    }


@app.post("/surrogate/predict")
async def surrogate_predict(request: PredictRequest) -> Dict[str, Any]:
    if not request.x:
        raise HTTPException(status_code=400, detail="Field 'x' must contain at least one spatial coordinate.")
    surrogate = _get_surrogate_or_load()
    x = np.array(request.x, dtype=np.float32)
    predictions = evaluate_profile(
        surrogate,
        x=x,
        t=float(request.t),
        D=float(request.D),
        v=float(request.v),
        device="cpu",
    )
    logger.info(
        "/surrogate/predict: len(x)=%s t=%.3f D=%.3f v=%.3f",
        len(request.x),
        request.t,
        request.D,
        request.v,
    )
    return {
        "x": request.x,
        "u_hat": predictions.tolist(),
    }


@app.get("/surrogate/status", response_model=SurrogateStatusResponse)
async def surrogate_status() -> SurrogateStatusResponse:
    history = _get_training_history()
    dataset_size = state.get("dataset_size")
    last_trained_at = state.get("last_trained_at")
    last_epochs = len(history) if history else None
    if last_trained_at is None and TRAINING_LOG_PATH.exists():
        last_trained_at = datetime.fromtimestamp(TRAINING_LOG_PATH.stat().st_mtime, tz=timezone.utc).isoformat()

    sample_prediction = None
    has_model = MODEL_PATH.exists() or state.get("surrogate") is not None
    if has_model:
        try:
            surrogate = _get_surrogate_or_load()
            x_grid = state.get("x_grid")
            if x_grid is not None and len(x_grid) > 0:
                sample_x = np.array(x_grid, dtype=np.float32)
            else:
                sim_cfg: SimulationConfig = state["simulation_config"]
                sample_x = np.linspace(0.0, sim_cfg.length, 64, dtype=np.float32)

            if sample_x.size > 64:
                sample_x = sample_x[:: int(np.ceil(sample_x.size / 64))]

            t_val = state.get("target_time") or 0.0
            params = state.get("target_params") or {
                "D": state["simulation_config"].diffusion_range[0],
                "v": state["simulation_config"].velocity_range[0],
            }
            preds = evaluate_profile(
                surrogate,
                x=sample_x,
                t=float(t_val),
                D=float(params["D"]),
                v=float(params["v"]),
                device="cpu",
            )
            sample_prediction = {
                "x": sample_x.tolist(),
                "u_hat": preds.tolist(),
                "t": float(t_val),
                "D": float(params["D"]),
                "v": float(params["v"]),
            }
        except HTTPException:
            pass
        except Exception as exc:
            logger.warning("Failed to compute surrogate status prediction: %s", exc)

    return SurrogateStatusResponse(
        has_model=has_model,
        model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None,
        dataset_size=dataset_size,
        last_trained_epochs=last_epochs,
        last_trained_at=last_trained_at,
        training_history=history,
        sample_prediction=sample_prediction,
    )


@app.get("/optimize/history")
async def optimization_history() -> Dict[str, Any]:
    history = state.get("history") or []
    result = state.get("result")
    if not history and OPT_JSON.exists():
        payload = json.loads(OPT_JSON.read_text())
        history = payload.get("history", [])
        result = payload.get("result")
    if not history:
        raise HTTPException(status_code=404, detail="No optimization history recorded yet.")
    return {"history": history, "result": result}


@app.get("/status")
async def status() -> Dict[str, Any]:
    return {
        "running": state["running"],
        "iterations": state["history"],
        "result": state["result"],
    }


@app.get("/result")
async def result() -> Dict[str, Any]:
    if state["result"] is None:
        raise HTTPException(status_code=404, detail="No result yet.")
    return state["result"]


@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = await broker.register()
    try:
        # Send existing history so late subscribers catch up.
        for item in state["history"]:
            await websocket.send_json({"type": "iteration", "payload": item})
        if state["result"]:
            await websocket.send_json({"type": "complete", "payload": state["result"]})

        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    finally:
        await broker.unregister(queue)
