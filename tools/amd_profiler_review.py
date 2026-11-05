"""Falsifiable review notes from bound dispatch observations, never promotion."""

from __future__ import annotations


def selected_interval_summary(observations: list[dict]) -> dict:
    # One receipt is one process/clock domain. Counter replay receipts remain
    # separate; their durations cannot be pooled into production throughput.
    intervals = sorted((row["start_ns"], row["end_ns"]) for row in observations)
    service = sum(end - start for start, end in intervals)
    union = 0
    start = end = None
    for first, last in intervals:
        if start is None:
            start, end = first, last
        elif first <= end:
            end = max(end, last)
        else:
            union += end - start
            start, end = first, last
    if start is not None:
        union += end - start
    # The final interval by start is not necessarily the one with maximum end.
    span = max(last for _, last in intervals) - intervals[0][0] if intervals else 0
    return dict(
        selected_dispatch_count=len(intervals),
        service_ns=service,
        interval_union_ns=union,
        span_ns=span,
        uncovered_span_ns=span - union,
        overlap_service_ns=service - union,
        scope="selected_dispatches_in_one_profiled_process_not_complete_workload",
        production_latency=False,
    )


def review(receipts: list[dict]) -> dict:
    passes, next_steps = [], []
    for receipt in receipts:
        intervals = selected_interval_summary(receipt["observations"])
        unknown = [item for item in receipt.get("native_metrics", []) if item.get("missingness")]
        hypotheses = []
        if intervals["uncovered_span_ns"]:
            hypotheses.append(
                dict(
                    hypothesis="selected_dispatch_set_contains_uncovered_intervals",
                    evidence=dict(uncovered_span_ns=intervals["uncovered_span_ns"]),
                    contradiction="unselected kernels or copies may occupy these intervals",
                    next_experiment="inspect a complete matched API/kernel/copy timeline before attributing gaps to host overhead",
                )
            )
        if unknown:
            next_steps.append(
                "resolve native metric units/value availability before deriving traffic, utilization or a roofline"
            )
        if receipt.get("loss_status", "").startswith("unknown") or not receipt.get(
            "complete_run_coverage"
        ):
            next_steps.append(
                "establish capture loss and full requested dispatch/pass coverage before whole-workload conclusions"
            )
        passes.append(
            dict(
                receipt_sha256=receipt["receipt_sha256"],
                intervals=intervals,
                hypotheses=hypotheses,
                native_metric_count=len(receipt.get("native_metrics", [])),
                unresolved_metric_count=len(receipt.get("unresolved_metrics", [])),
                metrics_missing_definition_or_value=len(unknown),
                loss_status=receipt.get("loss_status"),
            )
        )
    if not passes:
        next_steps.append(
            "collect one supported, bound timeline for this exact artifact and work contract"
        )
    return dict(
        schema="rns8_amd_profiler_hypothesis_review_v1",
        passes=passes,
        conclusion="inconclusive_for_whole_workload_bottleneck",
        next_experiments=sorted(set(next_steps)),
        automatic_optimization=False,
        performance_promotion_allowed=False,
    )
