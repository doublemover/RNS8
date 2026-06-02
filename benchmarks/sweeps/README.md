# Benchmark Sweeps

Reserved for reviewed benchmark sweep definitions and scripts.

Generated captures, raw outputs, and exploratory sweep results belong under
`temp/` unless they have been reviewed and summarized into durable docs.

Reviewed captures must carry explicit `rns8-bench` JSON `"schema_version": 4`
metadata and the structured timing fields documented in
`docs/performance-model.md`. Sweep summaries compare `timing_summary_us` phases
from current schema v4 captures. Older or versionless captures are historical
evidence and are not accepted by current sweep tooling.

GPU HIP event timing is intentionally nullable for non-HIP captures and any
capture with incomplete backend event data. A sweep must only compare
`gpu_event_timing_summary_us` when `timing_metadata.gpu_event_timing` is true for
every compared capture and the event timing source/scope metadata matches.
Missing event timings are an unsupported measurement, not zero time and not a
host-timing proxy.
