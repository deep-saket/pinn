# Sequential Surrogates Implementation Plan

## 1. Overview
We will extend the Geminus demo into a full three-stage pipeline (Pipe → Facility → Financial) with sequential surrogates, orchestration APIs, and an upgraded Angular dashboard. Work is organized into backend, frontend, documentation, and testing streams so we can iterate incrementally while keeping the system runnable.

## 2. Phased Workstreams

### Phase A – Backend Foundations
1. **Data schema & artifacts**
   - Define Pydantic/Typed dataclasses for each stage’s datasets (pipe, facility, financial) using the variables from *im3.jpeg*.
   - Establish artifact paths (`artifacts/pipe`, `artifacts/facility`, `artifacts/financial`) and persistent dataset locations under `data/pipeline/`.
2. **Surrogate modules**
   - Create `pipe_surrogate.py`, `facility_surrogate.py`, `financial_surrogate.py` with configurable `TrainingConfig`, `Surrogate` classes, and helper functions (train, evaluate, export).
   - Implement synthetic dataset generation for facility/financial stages (deterministic transforms + noise sourced from preceding stage outputs).
3. **Pipeline orchestrator**
   - Build a `pipeline.py` service that handles `simulate → train → optimize` across stages, caching intermediate tensors and enforcing dependency ordering.
   - Add sequential optimizer logic (pipe parameters as decision variables, facility/financial predicted outputs propagate forward, objective = financial profit).
4. **REST API expansion**
   - Add endpoints `/pipeline/simulate`, `/surrogates/train`, `/pipeline/optimize`, `/pipeline/status`, `/pipeline/history`.
   - Extend WebSocket payloads to include per-stage profiles & gradients for real-time charts.

### Phase B – Frontend & UX
1. **Service layer**
   - Update `BackendService` interfaces/types for the new endpoints and payloads (per-stage datasets, gradients, KPIs).
2. **Dashboard layout**
   - Redesign the dashboard per *im2.jpeg*: three blocks (Pipe, Facility, Financial) with workflow buttons (`Simulate All`, `Train All`, `Sequential Optimize`, `Refresh Status`).
3. **Visualization**
   - Stage 1: reuse pressure + ∂u/∂x chart.
   - Stage 2: add separator KPIs (e.g., vapor fraction, flow rates) and gradient chart (sensitivities wrt key facility variables).
   - Stage 3: add financial metrics (profit vs iteration, cumulative revenue/cost, risk band).
   - Activity log expanded to tag entries by stage.
4. **Dark-mode polish**
   - Ensure new panels/charts inherit the existing dark-mode toggle styling.

### Phase C – Documentation & Ops
1. **README updates**
   - Describe the tri-stage architecture, data flow diagram, API usage, and dataset/artifact locations.
   - Summarize variable tables (pipe/facility/financial) and how they map to surrogate inputs/outputs.
2. **Usage guides**
   - Document CLI workflows (e.g., `offline_tools.py pipeline-simulate`, `pipeline-train`, `pipeline-optimize`).
   - Add troubleshooting notes for sequential training/optimization.

### Phase D – Testing & Validation
1. **Unit/integration tests**
   - Backend tests for dataset generation, surrogate training, and sequential optimizer correctness (mock targets).
   - End-to-end smoke test hitting `/pipeline/simulate` → `/surrogates/train` → `/pipeline/optimize`.
2. **Frontend validation**
   - Run `npm run build` (track SCSS budget warnings) and add Cypress or Playwright smoke (optional stretch).
3. **Performance sanity**
   - Ensure sequential run times remain manageable (limit dataset sizes, expose config knobs).

## 3. Milestones & Deliverables
| Milestone | Deliverables |
| --- | --- |
| A1 | Three surrogate modules with dataset helpers + pipeline orchestrator API stubs |
| A2 | `/pipeline/*` endpoints live, WebSocket streaming per-stage data, sequential optimizer working |
| B1 | Updated Angular service + new layout skeleton |
| B2 | All stage-specific charts/KPIs wired, dark-mode verified |
| C1 | README + diagrams + CLI docs updated |
| D1 | Tests + build verification scripts committed |

## 4. Risks & Mitigations
- **Large scope** → Tackle in phases, keep branch deployable after each milestone.
- **Data explosion** → Limit synthetic datasets (configurable sample counts) and reuse cached data when possible.
- **UI complexity** → Add feature flags / placeholder charts early to keep dashboard functional while wiring data.
- **Optimization stability** → Start with deterministic sequential evaluation; add noise/uncertainty only after baseline works.

## 5. Next Actions
1. Implement Phase A.1–A.2 (surrogate modules + pipeline endpoints) on `sequential_surrogates` branch.
2. Commit & push after backend APIs are stable, then move to frontend (Phase B).
