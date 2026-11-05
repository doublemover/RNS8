"""Keep distinct benchmark callback boundaries out of matched evidence cohorts."""

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
