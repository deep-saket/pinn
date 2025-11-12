# Next Steps & To‑Dos

## Workflow Coverage

| Workflow / Tool | Status | Notes |
| --- | --- | --- |
| **Workflow 1:** optimize each system individually | ✅ | Pipe, facility, and financial stages each have their own targets/optimizers with frozen surrogates. |
| **Workflow 2:** optimize sequentially | ✅ | `/pipeline/run` plus the cascading surrogates let users propagate pipe → facility → financial and observe effects. |
| **Workflow 3:** optimize whole system together | ⛔ | Need a multi-stage objective & optimizer that perturbs all controls simultaneously. |
| **Workflow 4:** robust optimization | ⛔ | No uncertainty ensembles yet; objectives are deterministic. |
| **Workflow 5:** robust optimization with surrogate | ⛔ | Same gap as Workflow 4 but with surrogate speedups. |
| **Tool 1:** what-if analysis | ⚠️ Partial | `/pipeline/run` + charts allow manual what-ifs, but there’s no dedicated panel or uncertainty reporting. |
| **Tool 2:** create surrogates of individual systems | ✅ | Pipe, facility, and financial simulate/train controls are in the UI and API. |
| **Tool 3:** create surrogate of aggregated system | ⛔ | Need dataset generation + trainer for an end-to-end surrogate. |

## Workflow Coverage Gaps / Next Steps
1. **Workflow 3 – Global Deterministic Optimization**
   - Implement a single optimizer that perturbs pipe, facility, and financial controls simultaneously to hit a unified objective (e.g., profit vs. risk and target pressure profile).
   - Expose `/pipeline/optimize` endpoint + UI button, stream per-stage gradients.
2. **Workflows 4 & 5 – Robust Optimization**
   - Add uncertainty models (e.g., random perturbations on D, separator settings, market price) and run ensemble evaluations.
   - Provide reporting that summarizes min/mean/max or percentile bands for each stage.
   - Optional: support surrogate-driven robust mode (Workf.5) with faster sampling.
3. **Tool 1 – What‑If Analysis**
   - Build a dedicated UI panel/endpoint for one-off scenarios: user sets controls, system reports outputs plus uncertainty bands without running an optimizer.
4. **Tool 3 – Aggregated Surrogate**
   - Train a single surrogate that maps pipe+facility controls directly to financial KPIs for fast what-if/optimization.
   - Requires dataset generation from the sequential pipeline and a new offline trainer.

## Plotting / Visualization Ideas
- **Pipe Stage**
  - Pressure profile (already implemented – target vs surrogate vs optimizer).
  - Spatial gradient \(\partial u/\partial x\).
  - Iteration cost \(J(D,v)\) vs iteration & gradient magnitudes (current charts).
  - Histogram / violin of residuals once robust mode is added.
- **Facility Stage**
  - Bar chart for vapor fraction / gas / liquid (current).
  - Line plot of facility cost vs iteration & gradients `∂J/∂P`, `∂J/∂T`, `∂J/∂Gas` (current).
  - Sankey/stacked bars showing throughput split (nice-to-have).
  - Uncertainty whiskers (min/max vapor fraction) for robust runs.
- **Financial Stage**
  - Bar chart for net profit & risk (new).
  - Financial cost vs iteration; gradient traces for price/hedge/opex (new).
  - Profit vs risk Pareto scatter if we run ensembles.
  - Time-series of profit/risk when sweeping price inputs.
- **Cross-Stage / Global**
  - Pipeline view that shows how pipe outputs feed facility and financial metrics (line + bar combo).
  - Correlation matrix (heatmap) between controls and KPIs once enough samples exist.
  - What-if comparison chart (overlay two scenarios).

## Implementation Notes
- Extend backend with `/pipeline/optimize` for Workflow 3; share Gaussian optimizer but feed multi-stage cost.
- Add `uncertainty_config` to APIs for robust runs; log ensembles in exports.
- Introduce `what_if` endpoints + UI table/plot.
- Consider storing aggregated training datasets for Tool 3, then implement `aggregate_surrogate.py` and its trainer/optimizer.
