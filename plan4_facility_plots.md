# Facility Plot Integration Plan

## Objective
Extend the Angular dashboard so the “Pipe / Facility” stage toggle affects both the knobs and the visualization area:

- **Pipe view** (existing): pressure profile, ∂u/∂x, cost convergence, gradient trend.
- **Facility view**: new charts showing separator outputs (vapor fraction, gas/liquid flow rates) and their gradients/sensitivities.

## Work Breakdown

### 1. Backend data exposure
1.1 Update `/pipeline/run` response to carry per-iteration facility history (vapor fraction, gas/liquid flow, separator setpoints).  
1.2 When `/optimize` runs, store facility predictions alongside pipe profiles so history endpoints can replay them.  
1.3 Expose a short `/facility/history` endpoint (or reuse `/optimize/history`) that returns facility metrics per iteration for the frontend.

### 2. Frontend service updates
2.1 Extend `BackendService.IterationPayload` with optional `facility_metrics` ({ vapor_fraction, gas_flow, liquid_flow }).  
2.2 Add helper methods to request facility-specific history if a separate endpoint is created.  
2.3 Ensure `pipelinePreview` also captures facility arrays for static plotting.

### 3. Chart logic
3.1 Pressure chart: when `stageView === 'pipe'`, show existing datasets; when `facility`, swap to a new line/bar chart (e.g., vapor vs. iteration).  
3.2 Spatial gradient chart: in facility mode show sensitivity curves (e.g., derivative of vapor fraction w.r.t. separator pressure).  
3.3 Cost chart: in facility mode, display facility objectives (e.g., throughput, efficiency) instead of pipe cost.  
3.4 Gradient trend chart: in facility mode, show gradients for separator pressure/temp instead of ∂J/∂D, ∂J/∂v.

### 4. UI/UX
4.1 When the stage toggle flips, animate chart transitions (or simply refresh datasets) without rebuilding canvases.  
4.2 Add legends/labels so users know which stage is currently plotted.  
4.3 Keep the Components panel overlay in sync (pipe radio updates pipe charts, facility radio updates facility charts).  
4.4 Add tooltips/empty states (“Run facility workflow to populate charts”) when facility data is missing.

### 5. Testing
5.1 Verify `/pipeline/run` + `/optimize` produce facility telemetry.  
5.2 `npm run build` + manual UI check for both stages.  
5.3 Optional: add Cypress smoke to flip stages and assert chart labels update.

## Next Actions
1. Implement backend telemetry changes (Steps 1 + 2.1).  
2. Update Angular service and chart code to react to `stageView`.  
3. Polish UX (legends/tooltips).  
4. Run builds/tests.
