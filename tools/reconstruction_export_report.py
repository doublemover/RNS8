#!/usr/bin/env python3
"""Classify GPU CRT/export and reconstruction variant A/B evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from review_reference import ReviewIndex, load_review_index

from report_capture_inputs import expand_report_inputs, load_report_capture
from report_comparison import ComparisonPolicy, ReportCohorts
from observation_cohorts import phase_median, evidence
from result_compare import EXPORT_SELECTOR_REVIEW_DIFF_KEYS
from diagnostic_companion import load_bindings
from timing_eligibility import phase_blockers


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "temp" / "reconstruction-export-reports"
REQUIRED_VARIANT_CLASSES = {
    "compact_d2h_host_scatter",
    "status_elided_exact_proof",
    "prefix20_fixed_export",
    "tree_crt_reconstruction",
}
BASELINE_EXPORT_VARIANTS = {"default", "exact-wide-fixed-limb-export"}
POLICY = ComparisonPolicy(
    "reconstruction_export_completed_call",
    tuple(sorted(EXPORT_SELECTOR_REVIEW_DIFF_KEYS)),
    require_cpu_anchor=False,
    amortize_prepack_setup=True,
)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _load_captures(inputs: list[Path]) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    captures: list[tuple[Path, dict[str, Any]]] = []
    skipped: list[str] = []
    for path, from_directory in expand_report_inputs(inputs):
        capture = load_report_capture(path, from_directory=from_directory)
        if capture is None:
            continue
        captures.append((path, capture))
    return captures, skipped


def _target_id(capture: dict[str, Any], export: dict[str, Any]) -> str | None:
    for source in (export, capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict):
            for key in ("target_id", "target_arch", "gcn_arch"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _gpu_events_complete(capture: dict[str, Any]) -> bool:
    if capture.get("backend_selected") in {"cpu", "cpu-reference"}:
        return True
    events = capture.get("gpu_events")
    if isinstance(events, dict) and events.get("requested") and events.get("complete") is False:
        return False
    timings = capture.get("gpu_event_timings_us")
    return isinstance(timings, dict) and bool(timings)


def _variant_class(export: dict[str, Any], reconstruction: dict[str, Any]) -> str:
    export_name = export.get("name")
    reconstruction_name = reconstruction.get("name")
    if export_name == "compact-d2h-export-candidate":
        return "compact_d2h_host_scatter"
    if export_name == "status-elided-exact-proof-export-candidate":
        return "status_elided_exact_proof"
    if export_name == "prefix20-fixed-export-candidate":
        return "prefix20_fixed_export"
    if export_name == "tree-crt-export-candidate" or reconstruction_name == "tree_crt_candidate":
        return "tree_crt_reconstruction"
    if export_name in BASELINE_EXPORT_VARIANTS and reconstruction_name in {None, "default_garner"}:
        return "baseline_default_garner"
    return "other_export_reconstruction_variant"


def _row_for_capture(
    path: Path, capture: dict[str, Any], review_index: ReviewIndex, cohorts: ReportCohorts
) -> dict[str, Any]:
    export = (
        capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    )
    reconstruction = (
        capture.get("reconstruction_variant")
        if isinstance(capture.get("reconstruction_variant"), dict)
        else {}
    )
    reviews = [
        review_index.lookup(Path(raw["_path"]), raw) for raw in cohorts.raw_observations(capture)
    ]
    numerical = cohorts.numerical(capture, require_cpu_anchor=False).to_json()
    reference_blockers, missing, duplicates = [], [], []
    release_ready = bool(reviews)
    for review in reviews:
        if not isinstance(review, dict):
            reference_blockers.append("review_reference_missing")
            release_ready = False
            continue
        reference_blockers.extend(review.get("review_reader_blockers") or [])
        missing.extend(review.get("missing_required_baselines") or [])
        duplicates.extend(review.get("duplicate_backends") or [])
        release_ready = release_ready and (
            review.get("review_mode") == "release"
            and review.get("release_review_satisfied") is True
            and isinstance(review.get("numerical_reference"), dict)
            and review["numerical_reference"].get("passed") is True
        )
    admission = cohorts.admission(capture, POLICY)
    diagnostics = []
    if not cohorts.production.blockers(capture):
        try:
            diagnostics = [row["capture"] for row in cohorts.diagnostic_observations(capture)]
        except (OSError, ValueError):
            admission.append("matched_diagnostic_capture_unavailable_or_changed")
    export_times = [phase_median(raw, "crt_export") for raw in diagnostics]
    export_qualified = bool(export_times) and all(
        value is not None and value > 0 and not phase_blockers(raw, "crt_export")
        for raw, value in zip(diagnostics, export_times)
    )
    release_ready = (
        release_ready
        and not (missing or duplicates or reference_blockers or admission)
        and numerical["passed"]
    )
    return {
        "capture_path": _relative(path),
        "numerical_reference": numerical,
        "reference_blockers": reference_blockers,
        "admission_blockers": sorted(set(admission)),
        "measurement_cohort": evidence(capture),
        "production_observations": cohorts.production.receipts(capture),
        "semantics": capture.get("semantics"),
        "backend": capture.get("backend_selected"),
        "target_id": _target_id(capture, export),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "selected_prefix": capture.get("selected_prefix"),
        "requested_max_prefix": capture.get("requested_max_prefix"),
        "output_ld_padding": capture.get("output_ld_padding"),
        "export_variant": export.get("name"),
        "reconstruction_variant": reconstruction.get("name"),
        "variant_class": _variant_class(export, reconstruction),
        "limb_count": export.get("limb_count"),
        "signedness": export.get("signedness"),
        "output_layout": export.get("output_layout"),
        "status_policy": export.get("selector_status_policy"),
        "d2h_policy": export.get("d2h_policy"),
        "final_output_mode": export.get("final_output_mode"),
        "selected_kernel": export.get("selected_kernel") or capture.get("selected_kernel"),
        "selector_key": export.get("selector_key"),
        "promotion_eligible": export.get("promotion_eligible") is True
        and reconstruction.get("promotion_eligible") is not False,
        "promotion_blocker": export.get("promotion_blocker")
        or reconstruction.get("promotion_blocker"),
        "median_end_to_end_us": POLICY.measurement(capture)["median_us"],
        "median_export_us": statistics.median(export_times) if export_qualified else None,
        "export_timing_scope": "matched_separate_diagnostic_process_medians",
        "gpu_events_complete": bool(diagnostics)
        and all(_gpu_events_complete(raw) for raw in diagnostics),
        "release_review_ready": release_ready,
        "review_report": sorted(
            {
                review["review_report"]
                for review in reviews
                if isinstance(review, dict) and review.get("review_report")
            }
        ),
        "review_missing_required_baselines": missing if isinstance(missing, list) else [],
        "review_duplicate_backends": duplicates if isinstance(duplicates, list) else [],
    }


def _comparison_key(row: dict[str, Any]) -> tuple[Any, ...]:
    shape = row["shape"]
    return (
        row["semantics"],
        row["backend"],
        row["target_id"],
        shape["m"],
        shape["n"],
        shape["k"],
        row["selected_prefix"],
        row["requested_max_prefix"],
        row["output_ld_padding"],
        row["limb_count"],
        row["signedness"],
        row["output_layout"],
        row["final_output_mode"],
    )


def _ratio(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or candidate <= 0:
        return None
    return baseline / candidate


def _candidate_blockers(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    blockers: list[str] = []
    blockers.extend("candidate:" + item for item in candidate.get("reference_blockers", []))
    if not candidate["release_review_ready"]:
        blockers.append("candidate_release_review_not_ready")
    if baseline is None:
        return blockers + ["missing_default_baseline"]
    blockers.extend("baseline:" + item for item in baseline.get("reference_blockers", []))
    if not baseline["release_review_ready"]:
        blockers.append("baseline_release_review_not_ready")
    if not candidate["gpu_events_complete"]:
        blockers.append("candidate_gpu_events_incomplete")
    if not baseline["gpu_events_complete"]:
        blockers.append("baseline_gpu_events_incomplete")
    if candidate["median_end_to_end_us"] is None or baseline["median_end_to_end_us"] is None:
        blockers.append("missing_end_to_end_median")
    if candidate["median_export_us"] is None or baseline["median_export_us"] is None:
        blockers.append("missing_export_median")
    return blockers


def _comparison_for(
    candidate: dict[str, Any], baseline: dict[str, Any] | None, comparison: dict
) -> dict[str, Any]:
    end_to_end_speedup = comparison["speedup"]
    export_speedup = _ratio(
        baseline.get("median_export_us") if baseline else None,
        candidate.get("median_export_us"),
    )
    blockers = _candidate_blockers(candidate, baseline)
    blockers.extend(comparison["blockers"])
    blockers.extend(candidate["admission_blockers"])
    if baseline is not None:
        blockers.extend("baseline:" + value for value in baseline["admission_blockers"])
    margin = comparison["required_speedup_margin"]
    if blockers:
        disposition = "blocked"
        end_to_end_speedup = export_speedup = None
    elif margin is not None and 1 / margin <= end_to_end_speedup <= margin:
        disposition = "keep experimental"
        blockers.append("speedup_inside_variance_margin")
    elif (
        end_to_end_speedup is not None
        and margin is not None
        and end_to_end_speedup > margin
        and candidate["promotion_eligible"]
    ):
        disposition = "promote locally"
    elif end_to_end_speedup is not None and margin is not None and end_to_end_speedup > margin:
        disposition = "keep experimental"
    else:
        disposition = "drop/deprioritize"
    return {
        "variant_class": candidate["variant_class"],
        "candidate": candidate,
        "baseline": baseline,
        "speedup_end_to_end": end_to_end_speedup,
        "speedup_export": export_speedup,
        "disposition": disposition,
        "blockers": sorted(set(blockers)),
        "comparison": comparison,
        "required_speedup_margin": margin,
        "export_timing_scope": "matched_separate_diagnostic_process_medians",
    }


def build_report(
    inputs: list[Path], diagnostic_binding_paths: list[Path] | None = None
) -> dict[str, Any]:
    captures, skipped = _load_captures(inputs)
    review_index = load_review_index(inputs, captures)
    cohorts = ReportCohorts(
        [capture for _, capture in captures], load_bindings(diagnostic_binding_paths or [])
    )
    observations = [
        capture
        for capture in cohorts.observations
        if isinstance(capture.get("export_variant"), dict)
        or isinstance(capture.get("reconstruction_variant"), dict)
    ]
    rows = [
        _row_for_capture(Path(capture["_path"]), capture, review_index, cohorts)
        for capture in observations
    ]
    capture_by_row = {id(row): capture for row, capture in zip(rows, observations)}
    row_by_capture = {id(capture): row for row, capture in zip(rows, observations)}
    baseline_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["variant_class"] == "baseline_default_garner":
            baseline_by_key[_comparison_key(row)].append(row)

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row["variant_class"] in REQUIRED_VARIANT_CLASSES:
            baselines = baseline_by_key.get(_comparison_key(row), [])
            candidate = capture_by_row[id(row)]
            baseline_capture = cohorts.best(
                [capture_by_row[id(value)] for value in baselines if value["release_review_ready"]],
                candidate,
                POLICY,
            )
            baseline = (
                row_by_capture[id(baseline_capture)] if baseline_capture is not None else None
            )
            comparisons.append(
                _comparison_for(row, baseline, cohorts.compare(candidate, baseline_capture, POLICY))
            )

    class_summary: dict[str, dict[str, Any]] = {}
    for variant_class in sorted(REQUIRED_VARIANT_CLASSES):
        class_comparisons = [item for item in comparisons if item["variant_class"] == variant_class]
        classified = [item for item in class_comparisons if item["disposition"] != "blocked"]
        class_summary[variant_class] = {
            "comparison_count": len(class_comparisons),
            "classified_count": len(classified),
            "ready": bool(classified),
            "dispositions": sorted({item["disposition"] for item in class_comparisons}),
            "blockers": sorted(
                {blocker for item in class_comparisons for blocker in item["blockers"]}
            ),
        }

    return {
        "schema": "rns8_reconstruction_export_report_v1",
        "policy": "same_target_setup_inclusive_gpu_crt_export_variant_classification",
        "required_variant_classes": sorted(REQUIRED_VARIANT_CLASSES),
        "capture_count": len(captures),
        "implementation_cohort_count": len(rows),
        "skipped_json_count": len(skipped),
        "skipped_json": skipped[:20],
        "comparison_count": len(comparisons),
        "rank33_classification_complete": all(item["ready"] for item in class_summary.values()),
        "variant_class_summary": class_summary,
        "comparisons": sorted(
            comparisons,
            key=lambda item: (
                item["variant_class"],
                str(item["candidate"].get("semantics")),
                str(item["candidate"].get("backend")),
                item["candidate"].get("shape", {}).get("m") or 0,
            ),
        ),
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "reconstruction-export-report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "reconstruction-export-report.md"
    lines = [
        "# Reconstruction Export Report",
        "",
        f"- Classification complete: `{report['rank33_classification_complete']}`.",
        f"- Captures: `{report['capture_count']}`.",
        f"- Comparisons: `{report['comparison_count']}`.",
        "",
        "| Variant Class | Ready | Comparisons | Dispositions | Blockers |",
        "|---|---:|---:|---|---|",
    ]
    for name, summary in report["variant_class_summary"].items():
        lines.append(
            "| `{name}` | `{ready}` | {count} | `{dispositions}` | `{blockers}` |".format(
                name=name,
                ready=summary["ready"],
                count=summary["comparison_count"],
                dispositions=", ".join(summary["dispositions"]),
                blockers=", ".join(summary["blockers"]),
            )
        )
    lines.extend(
        [
            "",
            "| Variant | Semantics | Shape | Backend | Baseline us | Candidate us | E2E Speedup | Export Speedup | Disposition |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["comparisons"]:
        candidate = item["candidate"]
        baseline = item.get("baseline") or {}
        shape = candidate.get("shape") or {}
        lines.append(
            "| `{variant}` | `{semantics}` | `{shape}` | `{backend}` | {baseline_us} | {candidate_us} | {e2e} | {export} | `{disposition}` |".format(
                variant=item["variant_class"],
                semantics=candidate.get("semantics"),
                shape=f"{shape.get('m')}x{shape.get('n')}x{shape.get('k')}",
                backend=candidate.get("backend"),
                baseline_us=baseline.get("median_end_to_end_us", ""),
                candidate_us=candidate.get("median_end_to_end_us", ""),
                e2e=f"{item['speedup_end_to_end']:.3f}"
                if item.get("speedup_end_to_end") is not None
                else "",
                export=f"{item['speedup_export']:.3f}"
                if item.get("speedup_export") is not None
                else "",
                disposition=item["disposition"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="capture files or directories containing captures"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--diagnostic-binding", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = build_report(args.inputs, args.diagnostic_binding)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        outputs = write_outputs(report, args.out_dir)
        for label, path in outputs.items():
            print(f"{label}: {path}")
    if args.require_complete and not report["rank33_classification_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
