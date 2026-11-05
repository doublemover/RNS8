"""Shared producer/reader version and explicit review policy domains."""

REVIEW_SCHEMA_VERSION = 3
REVIEW_MODES = frozenset(('smoke', 'release'))
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
FINAL_COMPARISON_STATUSES = frozenset(('checksum_recorded_reference_required', 'reference_required',
                                      'exact_cpu_reference_compared', 'passed'))
COMPLETE_COMPARISON_STATUSES = frozenset(('exact_cpu_reference_compared', 'passed'))


def release_capture_satisfied(capture: dict) -> bool:
    return (type(capture.get('warmups')) is int and type(capture.get('repeats')) is int
            and capture['warmups'] >= RELEASE_MIN_WARMUPS and capture['repeats'] >= RELEASE_MIN_REPEATS)
