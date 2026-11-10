"""Qualify a declared two-source experiment without granting cache authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paired_experiment import ARMS, load_experiment
from report_comparison import ComparisonPolicy, ReportCohorts


def build_report(plan: Path, receipt: Path) -> dict:
    experiment, captures, bindings, owner = load_experiment(plan, receipt)
    cohorts = ReportCohorts(captures, bindings)
    policy = ComparisonPolicy(
        "explicit_paired_source_completed_output",
        require_cpu_anchor=False,
        amortize_prepack_setup=True,
        paired_experiment=experiment,
    )
    arms = {
        arm: [view for view in cohorts.observations if experiment.arm(view) == arm] for arm in ARMS
    }
    comparison = None
    blockers = []
    if any(len(arms[arm]) != 1 for arm in ARMS):
        blockers.append("exactly_one_qualified_implementation_cohort_per_declared_arm_required")
    else:
        comparison = cohorts.compare(arms["candidate"][0], arms["baseline"][0], policy)
        blockers.extend(comparison["blockers"])
    report = dict(
        schema="rns8_paired_source_experiment_report_v1",
        plan_identity_sha256=experiment.plan_identity,
        collection_receipt_sha256=experiment.receipt_sha256,
        allowed_delta=experiment.allowed_delta,
        arms={arm: [cohorts.metadata(view, policy) for view in arms[arm]] for arm in ARMS},
        comparison=comparison,
        blockers=sorted(set(blockers)),
        qualified=comparison is not None and comparison["qualified"] and not blockers,
        beneficial=comparison is not None
        and comparison["qualified"]
        and comparison["speedup"] > comparison["required_speedup_margin"]
        and not blockers,
        promotion_eligible=False,
        evidence_scope="declared_source_delta_same_work_same_environment_not_cache_authority",
    )
    owner.finish()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.plan, args.collection)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
