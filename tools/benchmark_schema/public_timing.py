"""Bounded whole-call observations are not extra per-call latency samples."""
import math
from measurement_contract import host_instrumentation


def validate_sustained(capture):
    record = capture.get("sustained_production")
    if record is None: return []
    try:
        requested = record["requested_batches"]
        low, high = record["min_calls"], record["max_calls"]
        target = record["min_duration_us"]
        if any(type(x) is not int for x in (requested,low,high)) or not 0 <= requested <= 64 or not 1 <= low <= high <= 65536:
            raise ValueError("invalid interval count bounds")
        if type(target) not in (int,float) or not math.isfinite(target) or not 0 <= target <= 60000000:
            raise ValueError("invalid duration bound")
        observations = record["observations"]
        if not isinstance(observations,list) or len(observations)!=requested: raise ValueError("missing bounded observations")
        if requested and host_instrumentation(capture) != "hooks_disabled": raise ValueError("sustained production must disable hooks")
        if record["boundary"] != "same_completed_output_as_latency_lane" or record["instrumentation"] != "hooks_disabled":
            raise ValueError("sustained output/instrumentation contract")
        tasks = max((capture.get("host_api_batch") or {}).get("tasks_per_measured_repeat",1),
                    (capture.get("grouped_dispatch") or {}).get("task_count",1))
        for observation in observations:
            calls,duration=observation["calls"],observation["duration_us"]
            if type(calls) is not int or not 1<=calls<=high or observation["tasks"]!=calls*tasks: raise ValueError("actual completed call count")
            if type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0: raise ValueError("duration")
            if not math.isclose(observation["per_call_us"],duration/calls,rel_tol=1e-5,abs_tol=1e-5): raise ValueError("quotient")
            reached=duration>=target and calls>=low
            if observation["termination"] != ("duration_and_minimum_calls" if reached else "hard_call_cap"):
                raise ValueError("termination reason")
            if not reached and calls!=high: raise ValueError("early truncated interval")
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        return [f"invalid sustained production receipt: {error}"]
    return []
