"""Validate censored event observations without admitting them as GPU timings."""

_PHASES = frozenset({
    "create_or_record_start", "record_stop", "stop_synchronize", "elapsed_time",
    "invalid_elapsed_time", "invalid_event_pair", "invalid_host_sample",
    "synchronized_host_fallback",
})


def timing_failure_metadata_errors(capture: dict) -> list[str]:
    metadata = capture.get("timing_metadata")
    if not isinstance(metadata, dict):
        return []
    keys = {"gpu_event_failure_diagnostics", "gpu_event_failure_observer_complete"}
    if not keys.intersection(metadata):
        return []  # Earlier v4 captures do not carry this optional extension.
    failures = metadata.get("gpu_event_failure_diagnostics")
    complete = metadata.get("gpu_event_failure_observer_complete")
    error = "invalid censored GPU event diagnostics"
    if not isinstance(complete, bool) or not isinstance(failures, list) or len(failures) > 1024:
        return [error]
    seen = set()
    required_reasons = set()
    for failure in failures:
        if not isinstance(failure, dict) or set(failure) != {"label", "phase", "backend_status"}:
            return [error]
        label, phase, status = (failure.get(k) for k in ("label", "phase", "backend_status"))
        if (not isinstance(label, str) or not label.isascii() or not 1 <= len(label) <= 127 or
                any(ord(c) < 32 for c in label) or not isinstance(phase, str) or phase not in _PHASES or
                type(status) is not int or not -1 <= status <= 2147483647 or
                (status == -1 and phase != "synchronized_host_fallback")):
            return [error]
        identity = label, phase, status
        if identity in seen:
            return [error]
        seen.add(identity)
        required_reasons.add(f"HIP timing unavailable: {label} {phase} status={status}")
    if not failures and complete:
        return []
    if not complete:
        required_reasons.add("HIP timing failure observer incomplete")
    reasons = metadata.get("gpu_event_timing_unavailable_reasons")
    if (metadata.get("gpu_event_timing") is not False or
            metadata.get("gpu_event_timing_status") != "unavailable_missing_expected_events" or
            metadata.get("gpu_event_timing_reason") != "backend_event_capture_incomplete" or
            any(metadata.get(k) is not None for k in (
                "gpu_event_timing_source", "gpu_event_timing_source_scope", "gpu_event_phase_order")) or
            capture.get("gpu_event_timings_us") is not None or
            capture.get("gpu_event_timing_summary_us") is not None or
            not isinstance(reasons, list) or not all(isinstance(x, str) and x for x in reasons) or
            not required_reasons.issubset(reasons)):
        return [error]
    return []


def is_censored_gpu_event_capture(capture: dict) -> bool:
    metadata = capture.get("timing_metadata")
    return (isinstance(metadata, dict) and
            isinstance(metadata.get("gpu_event_failure_diagnostics"), list) and
            (bool(metadata["gpu_event_failure_diagnostics"]) or
             metadata.get("gpu_event_failure_observer_complete") is False) and
            not timing_failure_metadata_errors(capture))
