"""Falsifiable review notes from bound dispatch observations, never promotion."""

from __future__ import annotations
from interval_observations import interval_summary


def selected_interval_summary(observations: list[dict]) -> dict:
    # One receipt is one process/clock domain. Counter replay receipts remain
    # separate; their durations cannot be pooled into production throughput.
    summary = interval_summary([(row["start_ns"], row["end_ns"]) for row in observations])
    return dict(
        selected_dispatch_count=summary["count"],
        **{key: value for key, value in summary.items() if key != "count"},
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
