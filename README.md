# Dummy Geminus-Style Pipeline Optimization Demonstrator

This repository contains a lightweight end-to-end mock-up of a Geminus-style pipeline optimizer. The backend simulates a 1‑D diffusion–advection process, trains a PINN-inspired surrogate, and runs a Gaussian sampling optimizer. The Angular dashboard provides live controls, charts, and result summaries.

## Repository layout

```
backend/            FastAPI + PyTorch service (simulation, surrogate, optimizer, API)
frontend/           Angular dashboard (controls, charts, status panels)
```

Key backend modules:
- `simulator.py` – finite-difference solver & synthetic dataset generator
- `surrogate_model.py` – simple PINN-like network with PDE residual loss
- `optimizer.py` – stochastic Gaussian sampler with elite selection
- `api.py` – FastAPI endpoints (`/simulate`, `/optimize`, `/optimize/stop`, `/status`, `/result`, `/ws/updates`)
- `utils/plotting.py` – helper for optional PNG plots

Key frontend pieces:
- `DashboardComponent` – controls, sliders, start/stop buttons, Chart.js plots, activity log
- `ResultPanelComponent` – live D/v/cost panel and final summary
- `BackendService` – REST + WebSocket client for the FastAPI service

## Mathematical core & tracking

- **Physics model** – The synthetic data generator integrates the 1‑D diffusion–advection equation

  $$
  \frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} - v \frac{\partial u}{\partial x}
  $$

  with random $D$ and $v$ samples. All spatial grids, time stamps, and fields are saved so dashboards can replay or export them.

- **Surrogate loss** – The PINN-like network minimizes

  $$
  \mathcal{L} = \text{MSE}\big(u_\theta, u_\text{data}\big) + \lambda \left\lVert \frac{\partial u_\theta}{\partial t} - D \frac{\partial^2 u_\theta}{\partial x^2} + v \frac{\partial u_\theta}{\partial x} \right\rVert_2^2,
  $$

  tracked per epoch for both training and validation splits (`val_split` defaults to 0.2). Metrics live in `backend/artifacts/surrogate_training_history.json` and are exposed via `/simulate`, `/surrogate/train`, `/surrogate/history`, and `/surrogate/status`.

- **Optimization objective** – The Gaussian search minimizes

  $$
  J(D,v) = \sum_i \left(u_\theta(x_i, t_f; D, v) - u_\text{target}(x_i, t_f)\right)^2 + \lambda \big(D^2 + v^2\big),
  $$

  logging every iteration (params, cost, spatial profile) and persisting the history to `backend/exports/optimization_latest.{json,xlsx}`. Use `/ws/updates`, `/status`, `/result`, or `/optimize/history` to visualize convergence.

## Backend: local development (no Docker required)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

The optimizer requires `/simulate` to be called once (POST) before `/optimize` so that (a) a fresh dataset exists and (b) the surrogate weights are trained/loaded. Without a ready surrogate, the optimizer returns a 400 directing you to generate/train first. A WebSocket stream is available at `ws://localhost:8000/ws/updates`. The backend now emits structured INFO logs (shown directly in the terminal where you run `uvicorn`) for every simulate request, surrogate training run, inference call, optimization iteration, completion, and stop request. Each `/simulate` or `/surrogate/train` call also writes artifacts: datasets under `backend/data/` and surrogate weights under `backend/artifacts/surrogate_latest.pt`.

### REST/WS API overview

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | GET | Lightweight readiness probe (`{"status":"ok"}`). |
| `/simulate` (`/simulate-train`) | POST | Generates synthetic diffusion–advection data, trains the surrogate, stores the dataset in memory, and persists it to `backend/data/surrogate_latest.{json,xlsx}`. Body: `{"num_systems": int, "seed": optional}`. |
| `/dataset` | GET | Returns the most recent dataset (`inputs`, `targets`) that `/simulate` produced—handy for Swagger inspection. |
| `/surrogate/train` | POST | Retrains the surrogate using one of three sources (priority order): inline `dataset.inputs/targets`, a file path specified via `dataset_path` (JSON or Excel), or the latest persisted dataset. Optional hyperparameters: `epochs`, `batch_size`, `learning_rate`, `pde_weight`, `val_split`. Saves weights to `backend/artifacts/surrogate_latest.pt` and logs train/validation losses to `backend/artifacts/surrogate_training_history.json`. |
| `/surrogate/history` | GET | Returns the stored training loss history (epoch vs. train/validation losses). Useful for plotting convergence. |
| `/surrogate/predict` | POST | Runs inference with the latest surrogate. Body: `{ "x": [...], "t": float, "D": float, "v": float }`. Returns `{ "x": [...], "u_hat": [...] }`. |
| `/surrogate/status` | GET | Summarizes surrogate health: whether a model is available, dataset size, last training timestamp/epochs, latest training loss history, and a sample prediction for plotting dashboards. |
| `/physics-nemo/run` | POST | Executes the NVIDIA PhysicsNeMo PINN pipeline (synthetic simulation → training → Torch/ONNX export). Body mirrors `PhysicsNemoTrainRequest` (epochs, collocation points, diffusion/velocity ranges, etc.). Requires `nvidia-physicsnemo`. |
| `/physics-nemo/status` | GET | Returns the latest PhysicsNeMo configuration, metrics, and artifact paths plus a boolean `engine_available`. |
| `/physics-nemo/dataset` | GET | Streams the dataset generated by the most recent PhysicsNeMo run (`surrogate_nemo_latest.json`). |
| `/omniverse/export` | POST | Writes a USD scene (e.g., `exports/omniverse_pressure.usda`) so you can visualize target/surrogate/optimization curves inside NVIDIA Omniverse or any USD viewer. Flags toggle which profiles are included. |
| `/optimize/history` | GET | Returns the most recent optimization iteration history and summary (memory or `backend/exports/optimization_latest.json`). |
| `/optimize` | POST | Launches the Gaussian sampling optimizer. Body: `{"max_iters": int, "population": int, "initial_D": optional, "initial_V": optional}`. Streams iteration updates via WebSocket. |
| `/optimize/stop` | POST | Requests early termination of the optimizer loop. |
| `/status` | GET | Snapshot of the running flag, iteration history, and final result (if available). |
| `/result` | GET | Final optimized parameters/profile once the optimizer completes; 404 otherwise. |
| `/ws/updates` | WebSocket | Push channel delivering `{"type":"iteration"}` and `{"type":"complete"}` messages for live dashboards. |

These endpoints are all documented in FastAPI’s Swagger UI at `http://localhost:8000/docs`.

#### Endpoint details

All log output (INFO level) is printed directly to the terminal where you run `uvicorn api:app --reload`. Each call reports its parameters, artifact paths, and overall status so you can follow along while testing.

- **POST /simulate** (alias `/simulate-train`)
  - Input body: `{ "num_systems": int (default 10), "seed": optional int }`.
  - Response: `{ "message": "surrogate trained", "dataset_size": N, "target_time": float, "x_grid": [...], "target_profile": [...], "training_history": [...] }`.
  - Side effects: saves dataset to `backend/data/`, retrains surrogate, saves weights to `backend/artifacts/surrogate_latest.pt`, logs per-epoch losses to `backend/artifacts/surrogate_training_history.json`, logs progress.

- **GET /dataset**
  - Input: none.
  - Response: `{ "inputs": [[x,t,D,v], ...], "targets": [[u], ...] }` from the most recent `/simulate`.

- **POST /surrogate/train**
  - Input: optionally `dataset` (inline arrays) or `dataset_path` (JSON/Excel). When supplying inline data, ignore Swagger’s placeholder `additionalProp1` object and instead send `{ "inputs": [[x,t,D,v], ...], "targets": [[u], ...] }`. Hyperparameters: `epochs`, `batch_size`, `learning_rate`, `pde_weight`, `val_split`.
  - Response matches `/simulate` (dataset size, target profile metadata, plus `training_history` array).
  - Side effects: retrains surrogate, saves weights to `backend/artifacts/surrogate_latest.pt`, writes per-epoch losses to `backend/artifacts/surrogate_training_history.json`, logs source of data.

- **GET /surrogate/history**
  - Input: none.
  - Response: `{ "training_history": [{ "epoch": int, "train_total_loss": float, "val_total_loss": float | null, ... }, ...] }`.

- **POST /surrogate/predict**
  - Input: `{ "x": [x0, ...], "t": float, "D": float, "v": float }`.
  - Response: `{ "x": [...], "u_hat": [...] }` evaluating the latest surrogate (loads from artifacts if not already in memory).

- **GET /surrogate/status**
  - Input: none.
  - Response example:

    ```json
    {
      "has_model": true,
      "model_path": "backend/artifacts/surrogate_latest.pt",
      "dataset_size": 3200,
      "last_trained_epochs": 120,
      "last_trained_at": "2025-11-08T06:40:00.123456+00:00",
      "training_history": [{"epoch":0,"total_loss":0.12,...}, ...],
      "sample_prediction": {"x": [...], "u_hat": [...], "t": 0.6, "D": 0.12, "v": 0.65}
    }
    ```

  - Use this endpoint (or its WebSocket-friendly equivalent you can build) to drive live plots of training curves and surrogate predictions.

- **POST /optimize**
  - Input: `{ "max_iters": int, "population": int, "initial_D": optional float, "initial_V": optional float }`.
  - Response: `{ "status": "started" }`. Live updates stream via `/ws/updates`; `/status` and `/result` provide pull-based snapshots. Upon completion, the backend writes `backend/exports/optimization_latest.json` (plus `.xlsx` sheets) for visualization. If an in-memory surrogate is missing, the endpoint automatically loads `backend/artifacts/surrogate_latest.pt` (and regenerates the reference target profile) before starting.

- **GET /optimize/history**
  - Input: none.
  - Response: `{ "history": [{ "iteration": n, "cost": ..., "D": ..., "v": ..., "grad_D": ..., "grad_V": ..., "grad_norm": ..., "u_profile": [...] }, ...], "result": {...} }`.
  - Use this to reload the last run’s telemetry—including gradient traces—even after the optimizer stops.

- **POST /optimize/stop**
  - Input: none. Response: `{ "status": "stopping" }` or `{ "status": "idle" }`.

- **GET /status**
  - Response: `{ "running": bool, "iterations": [...], "result": {... or null} }`.

- **GET /result**
  - Response: `{ "best_params": {"D": float, "v": float}, "best_cost": float, "best_profile": [...] }` once optimization completes.

- **WebSocket /ws/updates**
  - Messages: `{ "type": "iteration", "payload": {"iteration": int, ...} }` and `{ "type": "complete", "payload": {...} }`.

## Frontend: local development

```bash
cd frontend
npm install        # already executed once, repeat if deps change
npm start          # serves at http://localhost:4200
```

The UI expects the backend at `http://localhost:8000`. Adjust `BackendService` if you deploy elsewhere.

### Angular dashboard walkthrough

The dashboard stitches together every API described above so you can inspect the full workflow without touching Swagger:

1. **Controls panel (top-left)** –  
   - *Simulate & Train* calls `/simulate-train`, regenerating surrogate data, training the network, and refreshing the training-loss chart.  
   - *Refresh Status* calls `/surrogate/status` to pull the latest metadata/artifacts, so you can verify whether a model is loaded (even after a backend restart).  
   - *Load Last Run* pulls `/optimize/history`, repopulating the cost chart & profile plot with the most recent optimization logs written to disk.  
   - The sliders & numeric inputs configure the optimizer’s Gaussian search (`initialD`, `initialV`, `maxIters`, `population`).  
   - Status badges beneath the buttons echo the surrogate metadata (dataset size, last trained timestamp, model path).

2. **Pressure profile chart (top row, center)** –  
   Overlays three curves: the red dashed **target profile** from the physics simulation, the blue **surrogate prediction** at the preview point (or live during training), and the teal dashed **optimizer best profile** that updates every iteration. This makes it obvious how the surrogate approximates the target and how the optimizer steers the surrogate output toward that target over time. The note under the chart indicates the \(t, D, v\) values used for the current preview.

3. **Cost convergence chart (top row, right)** –  
   Plots the objective \(J(D,v)\) versus iteration for the current/most recent optimization run. The “Reload history” button replays whatever is saved in `backend/exports/optimization_latest.*`.

4. **Training loss chart (middle row)** –  
   Plots `train_total_loss` and `val_total_loss` vs. epoch using the telemetry emitted by `/simulate-train` or `/surrogate/train`. This is powered by the `training_history` array (train/validation curves). If you train offline, you can load the same history via `/surrogate/status`.

5. **Gradient trend chart (middle row)** –  
   Uses the gradient telemetry (`grad_D`, `grad_V`, `grad_norm`) computed each optimizer iteration. It helps diagnose how sensitive the objective is to each parameter during stochastic search—flat gradients often signal convergence or surrogate saturation.

6. **Result panel & surrogate snapshot (bottom-left)** –  
   The `ResultPanelComponent` shows the live/optimized parameters: the top line displays the current iteration’s D, v, cost; once optimization finishes, it locks in the best parameters, cost, and profile. Adjacent to it, the “Surrogate Snapshot” card lists dataset size, epoch count, and whether a model is present—all derived from `/surrogate/status`.

7. **Activity log (bottom-right)** –  
   Streams textual events (start/stop, iteration summaries, API fallback messages) so you can trace what just happened.

8. **Dark-mode toggle (top-right corner)** –  
   The sun/moon switch applies a full-surface theme (page background, header, panes, metric subcards, and chart canvases). The state is stored in the dashboard component so the UI keeps your preference while you navigate.

#### UI controls & optimization status quick reference

- **Knobs (left column).**  
  - *Synthetic systems* – number of random PDE simulations used when retraining the surrogate.  
  - *D (diffusion) slider* – seeds the Gaussian-search mean for \(D\); use it to start the optimizer closer to your expected regime.  
  - *v (velocity) slider* – same idea for \(v\).  
  - *Iterations / Population* – control the number of optimizer iterations and the number of samples per iteration.

- **Buttons & actions (second column).**  
  - *Simulate & Train* → `/simulate-train`.  
  - *Refresh Status* → `/surrogate/status`.  
  - *Load Last Run* → `/optimize/history`.  
  - *Start Optimization* / *Stop* → `/optimize` / `/optimize/stop`.  
  Any errors surface in the red banner below the buttons.

- **Optimization Status panel (full-width row beneath the first two columns).**  
  - Shows whether a surrogate is currently loaded (`Surrogate ready`).  
  - Lists dataset size, last-trained timestamp, and model path from `/surrogate/status`.  
  - Embeds the live `ResultPanel` so you always see the current/optimal \(D, v, J\).  
  - Includes the “Surrogate Snapshot” card (epochs, model presence) and the consolidated Activity Log.

These three sections align with the physical layout: knobs on the far left, buttons/status next to them, a combined optimization-status strip underneath, and the plots/logs occupying the right half of the view.

**Where do you see the optimal inputs?**  
Inside the `ResultPanel` (bottom-left), the `D`, `v`, and `Cost` fields update every iteration. When the optimizer finishes you’ll see “Final Result” with the best \(D, v\) and \(J\) captured from `/result`. The full iteration history (including profiles) still lives under `/optimize/history` and is used to populate the convergence chart.

**Where’s the “sample prediction” mentioned earlier?**  
The pressure chart’s subtitle and line labeled “Surrogate u(x)” are sourced from `/surrogate/status.sample_prediction`. That endpoint evaluates the current surrogate at the canonical \(t, D, v\) and returns the `x`/`u_hat` arrays the chart renders when the optimizer isn’t running. Once optimization starts, those values are replaced by each iteration’s `u_profile`, so you can watch the surrogate prediction approach the target profile in real time.

## Data export & offline workflows

Use `backend/offline_tools.py` when you want to generate data files, train the surrogate, or run optimization outside the FastAPI server.

### Generate surrogate datasets (JSON or Excel)

```bash
cd backend
python3 offline_tools.py export-data --num-systems 15 --format json --output data/surrogate.json
python3 offline_tools.py export-data --num-systems 15 --format excel --output data/surrogate.xlsx
```

Each record stores `(x, t, D, v, u)` so you can inspect it in Excel or load it into notebooks. The `/simulate` endpoint returns the same structure in JSON, automatically saves it to `backend/data/surrogate_latest.json` (plus `surrogate_latest.xlsx`), and you can also capture it manually via `curl`:

```bash
curl -X POST http://localhost:8000/simulate -H 'Content-Type: application/json' -d '{"num_systems": 15}' > data/surrogate.json
```

> **Where is the dataset stored?** When using the CLI export commands you control the `--output` path. When using the UI or Swagger (`POST /simulate`), the backend automatically writes `backend/data/surrogate_latest.json` and `backend/data/surrogate_latest.xlsx`. Feel free to rename or copy these files for archival.

> Need to inspect the most recent dataset via Swagger? Call `GET /dataset` after `/simulate`; it streams back the cached `inputs`/`targets` that the backend keeps in memory for the UI.

> Want to retrain directly from Swagger? Call `POST /surrogate/train`. By default it will load `backend/data/surrogate_latest.json`, but you can also include a `dataset` field (`{"inputs": [...], "targets": [...]}`), a `dataset_path` (e.g., `"data/surrogate_latest.xlsx"`), plus hyperparameters. Every training call writes the weights to `backend/artifacts/surrogate_latest.pt` and its per-epoch losses to `backend/artifacts/surrogate_training_history.json`.

> **Where is the surrogate stored?** Both `/simulate` and `/surrogate/train` export the latest weights to `backend/artifacts/surrogate_latest.pt` and log per-epoch metrics to `backend/artifacts/surrogate_training_history.json`. The inference endpoint and optimizer automatically load from these artifacts if no in-memory model exists.

> **Where is the optimization log stored?** After every `/optimize` run finishes, the backend persists the iteration history and summary to `backend/exports/optimization_latest.json` (and `.xlsx`). Use these files for downstream dashboards without re-running the optimizer.

> Need a consolidated surrogate status feed? Poll `GET /surrogate/status`—it reports whether a model is ready, the dataset size that produced it, timestamps/epochs, the full train/validation loss history, and a sample prediction you can pipe directly into charts.

### PhysicsNeMo pipeline (branch `physics_nemo`)

This branch adds a first-class integration with **NVIDIA PhysicsNeMo** (the successor to Modulus). The adapter uses PhysicsNeMo’s `FullyConnected` PINN block to learn the 1-D diffusion–advection process directly from physics constraints, then exports artifacts that the rest of the stack (FastAPI + Angular) can reuse.

1. **Install the extra dependency**

   ```bash
   cd backend
   pip install -r requirements.txt  # pulls nvidia-physicsnemo==1.2.0
   ```

   > On GPUs you can keep `device="auto"` (default) and the adapter will switch to CUDA whenever it’s available. On CPU-only hosts, PhysicsNeMo will fall back automatically.

2. **Run the full pipeline (simulation + surrogate training + export)**

   ```bash
   curl -X POST http://localhost:8000/physics-nemo/run \
     -H "Content-Type: application/json" \
     -d '{
       "epochs": 400,
       "collocation_points": 8192,
       "diffusion_range": [0.01, 0.2],
       "velocity_range": [0.05, 1.2],
       "dataset_profiles": 8,
       "export_onnx": true
     }'
   ```

   Behind the scenes the adapter:
   - Samples PhysicsNeMo collocation points to “simulate” the PDE without relying on the legacy finite-difference solver.
   - Trains a PhysicsNeMo PINN with PDE, boundary, and initial-condition losses.
   - Generates a dataset (`backend/data/surrogate_nemo_latest.json`) by evaluating the trained PINN at the desired time slice, exposing it via `/physics-nemo/dataset`.
   - Exports PyTorch weights (`backend/artifacts/physics_nemo/physics_nemo_surrogate.pt`), TorchScript, and ONNX files for downstream deployment.

3. **Query pipeline metadata**

   - `GET /physics-nemo/status` → shows the exact configuration used, loss history, and artifact paths, plus whether the PhysicsNeMo runtime is currently importable.
   - `GET /physics-nemo/dataset` → returns the dataset created in step 2 (same schema as `/dataset`), so you can retrain the classic surrogate or visualize the PhysicsNeMo profiles inside the Angular dashboard.

   The metadata is also written to `backend/artifacts/physics_nemo/physics_nemo_meta.json`, which UI clients can consume via REST or directly from disk.

The existing FastAPI/Angular flows stay untouched: you can continue using `/simulate` + `/optimize`, or switch entirely to the PhysicsNeMo outputs if you prefer NVIDIA’s stack for simulation and export.

### Omniverse plotting (branch `omniverse_plot`)

Need a richer 3D visualization? The `omniverse_plot` branch layers on a USD exporter so you can inspect every profile inside **NVIDIA Omniverse USD Composer/Kit**:

1. Export the scene:

   ```bash
   curl -X POST http://localhost:8000/omniverse/export \
     -H "Content-Type: application/json" \
     -d '{
       "include_target": true,
       "include_surrogate": true,
       "include_optimization": true,
       "include_nemo_profiles": true,
       "z_scale": 1.0
     }'
   ```

   The backend writes `backend/exports/omniverse_pressure.usda` (override the path with `output_path`). Each curve gets a color-coded `UsdGeomPoints` prim (red = target, blue = surrogate, teal = optimizer, gold = PhysicsNeMo samples).

2. Open the USD in Omniverse:
   - Launch **USD Composer** (or any Kit-based viewer).
   - `File → Open...` and pick `backend/exports/omniverse_pressure.usda`.
   - Toggle prim visibility to compare datasets; all metadata (export timestamp, curve count, z-scale) is stored in the root Xform’s `customData`.

Because the exporter writes plain `.usda`, no Omniverse SDK binaries are required on the FastAPI host. You can even push the file to Nucleus for collaborative reviews once it’s generated.

### Train the surrogate offline or from the UI

- **UI path:** Click **Generate Surrogate**. This calls `/simulate`, writes the dataset in memory, and immediately trains the surrogate used by the optimizer buttons.
- **Offline path:**

  ```bash
  python3 offline_tools.py train --data data/surrogate.json --format json \
    --epochs 150 --batch-size 512 --model-out artifacts/surrogate.pt

  # Training from Excel instead
  python3 offline_tools.py train --data data/surrogate.xlsx --format excel --model-out artifacts/surrogate.pt
  ```

  Afterwards, you can load the trained weights manually inside `api.py` if you want the API/UI to reuse the offline model (`state["surrogate"].load_state_dict(torch.load('artifacts/surrogate.pt'))`), or call `/surrogate/train` with a dataset payload / persisted dataset to refresh the live surrogate without regenerating synthetic data. Pass `--history-out logs/training.json` (and, if desired, `--val-split 0.2`) to archive the per-epoch train/validation curves from the CLI helper.

  **API usage examples**

  - Retrain from the most recent persisted data (no body required):

    ```json
    POST /surrogate/train
    {}
    ```

  - Retrain from a file path (relative to `backend/` or absolute):

    ```json
    {
      "dataset_path": "data/surrogate_latest.json",
      "epochs": 150,
      "batch_size": 512,
      "learning_rate": 0.0005,
      "pde_weight": 0.2
    }
    ```

  - Retrain from an inline dataset payload (shape requirements: inputs `(N,4)` ordered `[x,t,D,v]`, targets `(N,1)` for `u`):

    ```json
    {
      "dataset": {
        "inputs": [[0.0, 0.0, 0.08, 0.5], [0.02, 0.0, 0.08, 0.5]],
        "targets": [[0.95], [0.93]]
      },
      "epochs": 50
    }
    ```

  - Run inference with the latest surrogate:

    ```json
    {
      "x": [0.0, 0.02, 0.04, 0.06],
      "t": 0.6,
      "D": 0.1,
      "v": 0.65
    }
    ```

    Response shape: `{ "x": [...], "u_hat": [...] }`.

> **Where is the surrogate stored?** The `--model-out` argument controls the location of the `.pt` file (e.g., `backend/artifacts/surrogate.pt`). Keep that file around to seed new API sessions or to archive trained models.

### Optimization + visualization data

- **UI path:** After training, adjust the sliders for the initial Gaussian mean, set iteration/population counts, and press **Start**. Use **Stop** to request early termination. The charts update live, and you can always query `/status` for JSON snapshots.
- **Offline path:**

  ```bash
  python3 offline_tools.py optimize --model artifacts/surrogate.pt \
    --history-out exports/optimization.json --history-format json

  # Excel export for analysts
  python3 offline_tools.py optimize --model artifacts/surrogate.pt \
    --history-out exports/optimization.xlsx --history-format excel
  ```

  JSON output contains per-iteration tuples (`iteration, D, v, cost, u_profile`). The Excel export writes an `iterations` sheet plus a `summary` sheet suitable for Power BI or Excel charts. For visualization elsewhere, load the JSON, plot `cost` versus `iteration`, or expand `u_profile` to animate spatial curves. The same data can be captured from the running backend by streaming `/ws/updates` and persisting the payloads to disk.

> **Where is the optimization log stored?** Just like the dataset, it lands wherever you point `--history-out` (for example `backend/exports/optimization.json`). Use that folder as a hand-off location for visualization teams.

## Typical workflow

1. Launch backend and frontend.
2. In the UI, click **Generate Surrogate** to create synthetic data and train the PINN-like network.
3. Adjust the D/v sliders (used as the optimizer's initial Gaussian mean), choose iteration/population sizes, then click **Start**.
4. Watch live pressure-profile and cost charts, while the activity log and result panel update via WebSocket.
5. Use **Stop** to request early termination, or fetch `/result` once the optimizer completes.

## Notes & extensibility

- The surrogate network is intentionally small; swap in a real PINN by editing `surrogate_model.py`.
- The optimizer loop is modular (see `optimizer.py`) and can be replaced with gradient-based or hybrid schemes.
- WebSocket broadcasting is centralized in `StreamBroker`, so you can plug other clients in easily.
- Optional enhancements (noise, CSV export, alternate objectives, WebGL visuals) can be layered on top without restructuring the current modules.
