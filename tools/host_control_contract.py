"""Typed calling-thread host diagnostics; inclusive times never imply latency."""

PHASE_TIME_CONTRACT = "calling_thread_steady_ns_inclusive_scopes_nonadditive_v1"
HOST_CONTROL_FIELDS = {
    "device_gets",
    "device_sets",
    "stream_device_queries",
    "stream_capture_queries",
    "event_creates",
    "event_destroys",
    "event_records",
    "event_queries",
    "event_waits",
    "submission_proofs",
    "proof_builds",
    "proof_reuses",
    "graph_launches",
    "range_proof_calls",
    "range_proof_host_ns",
    "proof_identity_host_ns",
    "proof_metadata_host_ns",
    "graph_preflight_host_ns",
}


def validate_host_counts(value):
    if (
        not isinstance(value, dict)
        or set(value) != HOST_CONTROL_FIELDS
        or any(type(item) is not int or not 0 <= item < 2**64 for item in value.values())
    ):
        raise ValueError("complete unsigned64 calling-thread host-control fields required")
    return value
