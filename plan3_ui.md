# Facility UI Integration Plan

## Goals
1. Surface the new facility surrogate endpoints (`/facility/simulate`, `/facility/train`, `/facility/status`, `/pipeline/run`) in the Angular app.
2. Present a clear sequential workflow (Pipe → Facility) with controls, buttons, and metrics per stage.
3. Reuse existing styling/dark-mode behaviors while keeping the layout close to the mock (im2.jpeg).

## Tasks
### 1. Service Layer
- Extend `BackendService`:
  - Add TypeScript interfaces for `FacilityStatusResponse`, `PipelineRunResponse`.
  - Implement methods: `facilitySimulate`, `facilityTrain`, `fetchFacilityStatus`, `runPipeline`.
  - Update `IterationPayload` if pipeline data is later streamed.

### 2. Dashboard Layout Enhancements
- Introduce a “Facility Controls” panel:
  - Separator pressure/temp sliders.
  - Gas fraction slider.
  - Buttons: “Simulate Facility Data”, “Train Facility Surrogate”, “Run Pipe→Facility”.
- Add status badges mirroring the surrogate section (dataset size, last trained).

### 3. Visualizations
- Add a facility metrics card/chart (e.g., vapor fraction, gas/liquid flow). For v1, a bar or numeric panel is sufficient.
- Extend the existing spatial gradient chart to mention facility context (optional).
- Update the activity log to tag messages (`[Pipe]`, `[Facility]`).

### 4. Workflow Buttons
- Provide top-level buttons to run the sequential pipeline after pipe training.
- Disable buttons appropriately when prerequisites (pipe/facility surrogates) are missing.

### 5. Dark Mode / Styling
- Ensure new panels inherit `.panel` styles and dark-mode colors.
- Keep SCSS additions tight to avoid inflating the bundle (re-use existing classes where possible).

### 6. Documentation Touch-up
- Note new UI controls and pipeline flow in README once UI changes are merged.

## Implementation Order
1. Service updates.
2. Dashboard template & component logic (state vars, methods).
3. Styling tweaks.
4. Quick manual test (npm start → click facility buttons).
