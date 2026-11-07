"""Describe one explicit counter group without merging instances or replay passes."""

from __future__ import annotations

import re


def counter_names(values) -> tuple[str, ...]:
    if (
        not isinstance(values, (list, tuple))
        or len(values) > 8
        or any(
            type(name) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None
            for name in values
        )
        or len(set(values)) != len(values)
    ):
        raise ValueError("request at most eight unique explicit counter names")
    return tuple(values)


def summarize(projection: dict, selected: list[dict], requested) -> dict:
    names = counter_names(requested)
    if not names:
        return dict(state="not_requested", production_latency=False)
    # Native identities are already checked by the immutable database decoder.
    # Group once; neither counter-instance values nor separate passes are summed.
    index = {}
    for metric in projection["metrics"]:
        key = tuple(metric[field] for field in ("agent_id", "queue_id", "dispatch_id"))
        if metric["raw_name"] in names:
            index.setdefault(key, {}).setdefault(metric["raw_name"], []).append(metric)
    dispatches = []
    for dispatch in selected:
        key = tuple(dispatch[field] for field in ("agent_id", "queue_id", "dispatch_id"))
        observed = index.get(key, {})
        missing = [name for name in names if name not in observed]
        incomplete = [
            name
            for name in names
            if any(row["missingness"] is not None for row in observed.get(name, ()))
        ]
        dispatches.append(
            dict(
                **dispatch,
                observations=observed,
                missing_counters=missing,
                incomplete_counters=incomplete,
            )
        )
    complete = (
        bool(dispatches)
        and not projection["unresolved_metrics"]
        and all(
            not row["missing_counters"] and not row["incomplete_counters"] for row in dispatches
        )
    )
    return dict(
        schema="rns8_native_counter_pass_v1",
        state="observed" if complete else "partial",
        requested_counters=list(names),
        application_executions=1,
        counter_groups=1,
        kernel_replay=False,
        dispatches=dispatches,
        unresolved_metrics=projection["unresolved_metrics"],
        loss_status=projection["loss_status"],
        complete_selected_measurements=complete,
        complete_run_coverage=False,
        aggregation="individual_native_observations_no_cross_instance_or_replay_aggregation",
        production_latency=False,
        performance_promotion_allowed=False,
    )
