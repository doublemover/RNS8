# Benchmark Sweeps

Reserved for reviewed benchmark sweep definitions and scripts.

Generated captures, raw outputs, and exploratory sweep results belong under
`temp/` unless they have been reviewed and summarized into durable docs.

Reviewed captures should preserve the `rns8-bench` JSON schema version and the
structured timing fields documented in `docs/performance-model.md`. Sweep
summaries should compare `timing_summary_us` phases when present and may fall
back to legacy top-level averages for older captures.

GPU HIP event timing is intentionally nullable for non-HIP captures and any
capture with incomplete backend event data. A sweep must only compare
`gpu_event_timing_summary_us` when `timing_metadata.gpu_event_timing` is true for
every compared capture and the event timing source/scope metadata matches.
Missing event timings are an unsupported measurement, not zero time and not a
host-timing proxy.
