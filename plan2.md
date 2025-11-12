# Plan 2: Sequential Surrogate Backend Build

## Scope
Incrementally wire the existing pipe surrogate and the new facility surrogate into a sequential pipeline, expose orchestration endpoints, and lay groundwork for the future financial stage.

## Tasks
1. **Pipeline runner module**
   - Create `backend/pipeline.py` with dataclasses for pipe/facility inputs and outputs.
   - Implement helper functions to evaluate the pipe surrogate, derive facility feature vectors, and call the facility surrogate.
   - Provide a single `run_pipeline` function returning both stage outputs (pressure profile + facility KPIs).

2. **API integration**
   - Define `PipelineRunRequest`/`PipelineRunResponse` Pydantic models.
   - Add `/pipeline/run` endpoint that:
     * Ensures both surrogates are trained/loaded.
     * Evaluates the pipeline runner with the provided controls (D, v, separator setpoints, gas fraction).
     * Returns pipe metrics (full profile, outlet pressure, gradients placeholder) plus facility predictions (vapor fraction, gas/liquid flows).

3. **State bookkeeping**
   - Track last pipeline run in `state["pipeline_snapshot"]` for reuse by later endpoints (optimization/history).
   - If facility data is missing, auto-generate a default dataset via `generate_facility_dataset`.

4. **Testing hooks**
   - Add a minimal unit helper (in `pipeline.py` or `tests/`) to validate the runner with mock surrogates.
   - Update README to describe `/facility/*` and `/pipeline/run`.

5. **Future steps (not in this commit)**
   - Extend optimizer to iterate over facility controls.
   - Wire Angular components to call the new endpoints and display facility KPIs.
