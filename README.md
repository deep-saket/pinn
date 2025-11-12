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

- **Facility surrogate + optimizer** – A second-stage MLP (`facility_surrogate.py`) consumes the pipe outlet pressure, pipe outlet temperature, throughput, and separator settings \((P_\text{sep}, T_\text{sep}, \phi_\text{gas})\) to predict the facility observables `vapor_fraction`, `gas_flow_rate`, and `liquid_flow_rate`. The standalone facility optimizer minimizes

  $$
  J_\text{fac} =
    w_v \big(v_\text{pred} - v_\text{target}\big)^2 +
    w_g \big(\dot{m}_g - \dot{m}_{g,\text{target}}\big)^2 +
    w_\ell \big(\dot{m}_\ell - \dot{m}_{\ell,\text{target}}\big)^2 +
    \lambda_\text{reg} \|\theta_\text{sep}\|_2^2,
  $$

  by sampling separator pressure/temperature/gas-fraction triplets, evaluating the facility surrogate, and adapting the Gaussian mean/std exactly like the pipe-stage optimizer. Every run is persisted to `backend/exports/facility_optimization_latest.{json,xlsx}` with the per-iteration metrics and gradients.

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
| `/facility/simulate` | POST | Generates a synthetic facility/separator dataset (pipe outputs + separator setpoints) and caches it under `backend/data/facility_latest.json`. |
| `/facility/train` | POST | Trains the facility surrogate using the cached dataset. Body: `{"epochs": int, "batch_size": int, "learning_rate": float, "val_split": float}`. Stores weights under `backend/artifacts/facility_surrogate.pt`. |
| `/facility/status` | GET | Indicates whether the facility surrogate is trained, shows dataset size, last-trained timestamp, and loss history. |
| `/pipeline/run` | POST | Runs the current pipe → facility surrogate chain given `{ "D": float, "v": float, "t": float, "separator_pressure": float, "separator_temp": float, "gas_fraction": float }` and returns both stage outputs. |
| `/optimize/history` | GET | Returns the most recent optimization iteration history and summary (each record now includes `du_dx_profile` alongside `u_profile`). |
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
  - Input: `{ "max_iters": int, "population": int, "initial_D": optional float, "initial_V": optional float, "target_outlet_pressure": optional float, "target_mean_pressure": optional float, "target_gradient": optional float }`.
  - Response: `{ "status": "started" }`. The optional `target_*` fields come directly from the UI knobs and let you optimize toward desired **outputs** (outlet pressure, mean line pressure, and outlet gradient) while keeping the surrogate weights frozen. Live updates stream via `/ws/updates`; `/status` and `/result` provide pull-based snapshots. Upon completion, the backend writes `backend/exports/optimization_latest.json` (plus `.xlsx` sheets) for visualization. If an in-memory surrogate is missing, the endpoint automatically loads `backend/artifacts/surrogate_latest.pt` (and regenerates the reference target profile) before starting.

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

### Facility + sequential pipeline endpoints

- **POST /facility/simulate**
  - Input: `{ "num_samples": int, "seed": optional }`.
  - Generates a synthetic facility-stage dataset (pipe outlet pressure/temperature/throughput + separator settings → vapor/gas/liquid outputs). Saves the JSON payload to `backend/data/facility_latest.json` and caches it in-memory for `/facility/train`.

- **POST /facility/train**
  - Input mirrors the pipe surrogate (`epochs`, `batch_size`, `learning_rate`, `val_split`). Uses the cached dataset or reloads `backend/data/facility_latest.json` if needed. Persists weights to `backend/artifacts/facility_surrogate.pt` and the epoch-by-epoch losses to `backend/artifacts/facility_training_history.json`.

- **GET /facility/status**
  - Response: `{ "has_model": bool, "dataset_size": int | null, "last_trained_at": str | null, "training_history": [...] }`.
  - Lets the UI/CLI confirm whether the facility surrogate is ready, how large the latest dataset is, and when it was last trained.

- **POST /pipeline/run**
  - Input: pipe controls (`D`, `v`, `t`) plus facility knobs (`separator_pressure`, `separator_temp`, `gas_fraction`).
  - Runs the sequential evaluation (pipe surrogate → facility surrogate) and returns both stage outputs as well as the feature vector fed into the facility surrogate. The result is also cached under `state["pipeline_snapshot"]` so the UI can refresh without re-running the API call.

- **POST /facility/optimize**
  - Input: `{ "max_iters": 20, "population": 20, "initial_separator_pressure": 100, "initial_separator_temp": 45, "initial_gas_fraction": 0.5, "target_vapor_fraction": 0.65, "target_gas_flow_rate": 18, "target_liquid_flow_rate": 20, "pipe_D": optional, "pipe_v": optional, "pipe_t": optional }`.
  - Launches the facility-only Gaussian optimizer. It samples separator settings, evaluates the facility surrogate (conditioned on the supplied pipe context), and adapts the Gaussian mean/std exactly like the pipe optimizer. Returns `{ "history": [...], "result": {...}, "pipe_context": {...}, "timestamp": ... }` and persists summaries to `backend/exports/facility_optimization_latest.{json,xlsx}` for later visualization.

- **GET /facility/optimize/status**
  - Returns the latest `{ history, result, pipe_context, timestamp }` bundle so the UI retains facility cost/gradient curves even after restarting the backend.

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
   - The stage toggle (Pipe ⇄ Facility) swaps the entire knob stack, Actions pane, and chart context. **Pipe mode now exposes only output knobs**—target outlet pressure, target mean pressure, and target outlet gradient—which feed directly into `/optimize` as `target_*` fields. Facility mode exposes separator pressure/temperature/gas sliders (still inputs), the facility objective knobs (target vapor/gas/liquid), and facility optimizer iteration/population sliders.  
   - *Simulate & Train* calls `/simulate-train`, regenerating surrogate data, training the network, and refreshing the training-loss chart. In Facility mode, the equivalent **Simulate Facility** / **Train Facility Surrogate** buttons call `/facility/simulate` and `/facility/train`.  
   - *Refresh Status* calls `/surrogate/status`; in Facility mode the surrogate status card switches automatically to `/facility/status`.  
   - *Load Last Run* pulls `/optimize/history`, repopulating the cost chart & profile plot with the most recent pipe-stage optimization logs. Facility mode keeps the same button but displays `/facility/optimize/status` results in the charts.  
   - Status badges beneath the buttons echo the active stage’s metadata (dataset size, last trained timestamp, model path, and—when available—the facility optimizer summary).

2. **Pressure profile chart (top row, center)** –  
   Overlays three curves: the red dashed **target profile** from the physics simulation, the blue **surrogate prediction** at the preview point (or live during training), and the teal dashed **optimizer best profile** that updates every iteration. This makes it obvious how the surrogate approximates the target and how the optimizer steers the surrogate output toward that target over time. The note under the chart indicates the \(t, D, v\) values used for the current preview.

3. **Cost convergence chart (top row, right)** –  
   Plots the objective \(J(D,v)\) versus iteration for the current/most recent optimization run. The “Reload history” button replays whatever is saved in `backend/exports/optimization_latest.*`.

4. **Training loss chart (middle row)** –  
   Plots `train_total_loss` and `val_total_loss` vs. epoch using the telemetry emitted by `/simulate-train` or `/surrogate/train`. This is powered by the `training_history` array (train/validation curves). If you train offline, you can load the same history via `/surrogate/status`.

5. **Gradient trend chart (middle row)** –  
   Uses the gradient telemetry (`grad_D`, `grad_V`, `grad_norm`) computed each optimizer iteration. It helps diagnose how sensitive the objective is to each parameter during stochastic search—flat gradients often signal convergence or surrogate saturation.

6. **Spatial gradient chart (bottom row)** –  
   Shows the surrogate’s spatial derivative \( \partial u/\partial x \) alongside the target and optimizer best-so-far gradient. Sharp peaks line up with pressure drops, so you can confirm the PINN matches slope as well as absolute values. The chart pulls the new `du_dx_profile` field streamed during optimization, plus the live surrogate preview (`/surrogate/status`).

7. **Result panel & surrogate snapshot (bottom-left)** –  
   The `ResultPanelComponent` shows the live/optimized parameters: the top line displays the current iteration’s D, v, cost; once optimization finishes, it locks in the best parameters, cost, and profile. Adjacent to it, the “Surrogate Snapshot” card lists dataset size, epoch count, and whether a model is present—all derived from `/surrogate/status`.

8. **Activity log (bottom-right)** –  
   Streams textual events (start/stop, iteration summaries, API fallback messages) so you can trace what just happened.

9. **Dark-mode toggle (top-right corner)** –  
   The sun/moon switch applies a full-surface theme (page background, header, panes, metric subcards, and chart canvases). The state is stored in the dashboard component so the UI keeps your preference while you navigate.

10. **Facility optimizer view (toggle to “Facility”)** –  
    - The pressure chart switches to a facility KPI bar chart (vapor fraction, gas flow, liquid flow) driven either by `/pipeline/run` or the latest `/facility/optimize` run.  
    - The cost chart plots the facility optimizer’s \(J_\text{fac}\) vs. iteration, and the gradient chart shows the three partial derivatives `∂J/∂P`, `∂J/∂T`, and `∂J/∂Gas`.  
    - The Actions pane exposes **Simulate Facility**, **Train Facility Surrogate**, **Run Pipe ➜ Facility**, and **Optimize Facility** (which calls `/facility/optimize`).  
    - The surrogate status card morphs into a facility summary, echoing dataset size/model path plus the most recent best separator settings. The summary includes the timestamp and the pipe context that the optimizer conditioned on, so you can tell which upstream conditions produced the shown optimum.

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

> Facility datasets follow the same convention: `/facility/simulate` writes `backend/data/facility_latest.json`, `/facility/train` produces `backend/artifacts/facility_surrogate.pt`, and the per-epoch losses land in `backend/artifacts/facility_training_history.json`. These files power the facility status card and the facility-only optimizer.

> **Where is the surrogate stored?** Both `/simulate` and `/surrogate/train` export the latest weights to `backend/artifacts/surrogate_latest.pt` and log per-epoch metrics to `backend/artifacts/surrogate_training_history.json`. The inference endpoint and optimizer automatically load from these artifacts if no in-memory model exists.

> **Where is the optimization log stored?** After every `/optimize` run finishes, the backend persists the iteration history and summary to `backend/exports/optimization_latest.json` (and `.xlsx`). Use these files for downstream dashboards without re-running the optimizer.

> Each iteration row now includes the observed outlet pressure, mean pressure, and outlet gradient so you can confirm whether the optimizer is steering the surrogate toward the target knobs. The summary tab also records the best-achieved metrics alongside the best `(D, v)` pair.

> Facility-only optimizer runs are stored the same way under `backend/exports/facility_optimization_latest.json` (plus `.xlsx`). Each row includes separator settings, cost, predicted vapor/gas/liquid outputs, and gradients (`dJ/dP`, `dJ/dT`, `dJ/dGas`) so you can graph sensitivities over time.

> Need a consolidated surrogate status feed? Poll `GET /surrogate/status`—it reports whether a model is ready, the dataset size that produced it, timestamps/epochs, the full train/validation loss history, and a sample prediction you can pipe directly into charts.

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
