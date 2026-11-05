"""Keep distinct benchmark callback boundaries out of matched evidence cohorts."""

import math

CURRENT = "typed_observation_no_phase_clocks_in_sustained_v2"
HISTORICAL = "historical_json_in_interval"
KNOWN = frozenset(
    (
        CURRENT,
        "typed_observation_json_after_interval_v1",
        "typed_observations_json_after_intervals_v1",
    )
)


def observation_boundary(capture):
    if "hot_loop_serialization" not in capture:
        return HISTORICAL
    value = capture["hot_loop_serialization"]
    if type(value) is not str or value not in KNOWN:
        raise ValueError("unknown completed-call observation boundary")
    return value


def validate_completed_call_samples(capture, repeats, *, resident_output=False):
    def positive(value, *, zero=False):
        return (
            type(value) in (int, float)
            and math.isfinite(value)
            and (value >= 0 if zero else value > 0)
        )

    samples = capture.get("samples")
    if (
        not positive(capture.get("setup_us"))
        or not isinstance(samples, list)
        or len(samples) != repeats
    ):
        raise ValueError("completed-call setup/sample count")
    for sample in (capture.get("first_call"), *samples):
        if not isinstance(sample, dict):
            raise ValueError("completed-call phase observation")
        phases = [
            sample.get(key) for key in ("prepare_us", "execute_us", "export_us", "cleanup_us")
        ]
        if (
            not all(positive(value, zero=True) for value in phases)
            or not positive(sample.get("total_us"))
            or not math.isclose(sum(phases), sample["total_us"], rel_tol=1e-7, abs_tol=1e-6)
        ):
            raise ValueError("completed-call phase/total accounting")
        if resident_output and sample["export_us"] != 0:
            raise ValueError("resident route must not hide a timed host export")
    bursts = capture.get("sustained")
    if not isinstance(bursts, list) or len(bursts) != 3:
        raise ValueError("completed-call sustained observations")
    for burst in bursts:
        if not isinstance(burst, dict):
            raise ValueError("completed-call sustained observation")
        calls, duration, per_call = (burst.get(key) for key in ("calls", "total_us", "per_call_us"))
        if (
            type(calls) is not int
            or not 16 <= calls <= 4096
            or not positive(duration)
            or not positive(per_call)
            or (duration < 250000 and calls != 4096)
            or burst.get("minimum_duration_reached") is not (duration >= 250000)
            or not math.isclose(per_call, duration / calls, rel_tol=1e-7)
        ):
            raise ValueError("completed-call sustained work/duration")
