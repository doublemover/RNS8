# Benchmark Sweeps

Reserved for reviewed benchmark sweep definitions and scripts.

Generated captures, raw outputs, and exploratory sweep results belong under
`temp/` unless they have been reviewed and summarized into durable docs.

Reviewed captures should preserve the `rns8-bench` JSON schema version and the
structured timing fields documented in `docs/performance-model.md`. Sweep
summaries should compare `timing_summary_us` phases when present and may fall
back to legacy top-level averages for older captures.
