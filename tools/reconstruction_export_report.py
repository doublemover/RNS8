#!/usr/bin/env python3
"""Classify GPU CRT/export and reconstruction variant A/B evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from report_capture_inputs import expand_report_inputs, load_report_capture


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "temp" / "reconstruction-export-reports"
REQUIRED_VARIANT_CLASSES = {
    "compact_d2h_host_scatter",
    "status_elided_exact_proof",
    "prefix20_fixed_export",
    "tree_crt_reconstruction",
}
BASELINE_EXPORT_VARIANTS = {"default", "exact-wide-fixed-limb-export"}


def _normalize_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate.absolute()
    text = str(resolved).replace("\\", "/")
    return text.lower() if sys.platform.startswith("win") else text


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _json_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.json")))
        elif item.exists():
            paths.append(item)
    return paths


def _load_captures(inputs: list[Path]) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    captures: list[tuple[Path, dict[str, Any]]] = []
    skipped: list[str] = []
    for path, from_directory in expand_report_inputs(inputs):
        capture = load_report_capture(path, from_directory=from_directory)
        if capture is None:
            continue
        if isinstance(capture.get("export_variant"), dict) or isinstance(capture.get("reconstruction_variant"), dict):
            captures.append((path, capture))
    return captures, skipped


def _review_capture_keys(capture_path: str, report_path: Path) -> set[str]:
    raw = Path(capture_path.replace("\\", "/"))
    paths = [raw]
    if not raw.is_absolute():
        paths.append(REPO_ROOT / raw)
        paths.append(report_path.parent / raw)
    return {_normalize_path(path) for path in paths}


def _load_review_index(inputs: list[Path]) -> dict[str, dict[str, Any]]:
    reports = [
        path
        for path in _json_inputs(inputs)
        if path.name == "review_report.json"
    ]
    index: dict[str, dict[str, Any]] = {}
    for report_path in reports:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        for group_index, group in enumerate(report.get("groups", [])):
            if not isinstance(group, dict):
                continue
            group_entry = {
                "review_report": _relative(report_path),
                "review_report_group_index": group_index,
                "review_mode": group.get("review_mode") or report.get("review_mode"),
                "release_review_satisfied": group.get("release_review_satisfied"),
                "missing_required_baselines": group.get("missing_required_baselines") or [],
                "duplicate_backends": group.get("duplicate_backends") or [],
                "contract_key": group.get("contract_key"),
            }
            for candidate in group.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                capture_path = candidate.get("capture")
                if not isinstance(capture_path, str) or not capture_path:
                    continue
                entry = {
                    **group_entry,
                    "candidate_promotable": candidate.get("promotable") is True,
                    "candidate_promotion_reason": candidate.get("promotion_reason"),
                    "candidate_promotion_blockers": candidate.get("promotion_blockers") or [],
                    "candidate_cache_write_status": candidate.get("cache_write_status"),
                }
                for key in _review_capture_keys(capture_path, report_path):
                    index[key] = entry
    return index


def _timing_median(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("timing_summary_us")
    item = summary.get(phase) if isinstance(summary, dict) else None
    if isinstance(item, dict) and isinstance(item.get("median"), (int, float)):
        return float(item["median"])
    return None


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


def _row_for_capture(path: Path, capture: dict[str, Any], review_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    export = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    reconstruction = (
        capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
    )
    review = review_index.get(_normalize_path(path))
    missing = review.get("missing_required_baselines") if isinstance(review, dict) else []
    duplicates = review.get("duplicate_backends") if isinstance(review, dict) else []
    release_ready = (
        isinstance(review, dict)
        and review.get("review_mode") == "release"
        and review.get("release_review_satisfied") is True
        and not missing
        and not duplicates
    )
    return {
        "capture_path": _relative(path),
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
        "promotion_eligible": export.get("promotion_eligible") is True and reconstruction.get("promotion_eligible") is not False,
        "promotion_blocker": export.get("promotion_blocker") or reconstruction.get("promotion_blocker"),
        "median_end_to_end_us": _timing_median(capture, "end_to_end"),
        "median_export_us": _timing_median(capture, "crt_export"),
        "gpu_events_complete": _gpu_events_complete(capture),
        "release_review_ready": release_ready,
        "review_report": review.get("review_report") if isinstance(review, dict) else None,
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
    if baseline is None:
        return ["missing_default_baseline"]
    if not candidate["release_review_ready"]:
        blockers.append("candidate_release_review_not_ready")
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


def _comparison_for(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    end_to_end_speedup = _ratio(
        baseline.get("median_end_to_end_us") if baseline else None,
        candidate.get("median_end_to_end_us"),
    )
    export_speedup = _ratio(
        baseline.get("median_export_us") if baseline else None,
        candidate.get("median_export_us"),
    )
    blockers = _candidate_blockers(candidate, baseline)
    if blockers:
        disposition = "blocked"
    elif end_to_end_speedup is not None and end_to_end_speedup > 1.01 and candidate["promotion_eligible"]:
        disposition = "promote locally"
    elif end_to_end_speedup is not None and end_to_end_speedup > 1.01:
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
        "blockers": blockers,
    }


def build_report(inputs: list[Path]) -> dict[str, Any]:
    captures, skipped = _load_captures(inputs)
    review_index = _load_review_index(inputs)
    rows = [_row_for_capture(path, capture, review_index) for path, capture in captures]
    baseline_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["variant_class"] == "baseline_default_garner":
            baseline_by_key[_comparison_key(row)].append(row)

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row["variant_class"] in REQUIRED_VARIANT_CLASSES:
            baselines = baseline_by_key.get(_comparison_key(row), [])
            baseline = min(
                baselines,
                key=lambda item: item.get("median_end_to_end_us") or float("inf"),
            ) if baselines else None
            comparisons.append(_comparison_for(row, baseline))

    class_summary: dict[str, dict[str, Any]] = {}
    for variant_class in sorted(REQUIRED_VARIANT_CLASSES):
        class_comparisons = [item for item in comparisons if item["variant_class"] == variant_class]
        classified = [item for item in class_comparisons if item["disposition"] != "blocked"]
        class_summary[variant_class] = {
            "comparison_count": len(class_comparisons),
            "classified_count": len(classified),
            "ready": bool(classified),
            "dispositions": sorted({item["disposition"] for item in class_comparisons}),
            "blockers": sorted({blocker for item in class_comparisons for blocker in item["blockers"]}),
        }

    return {
        "schema": "rns8_reconstruction_export_report_v1",
        "policy": "same_target_setup_inclusive_gpu_crt_export_variant_classification",
        "required_variant_classes": sorted(REQUIRED_VARIANT_CLASSES),
        "capture_count": len(rows),
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
                e2e=f"{item['speedup_end_to_end']:.3f}" if item.get("speedup_end_to_end") is not None else "",
                export=f"{item['speedup_export']:.3f}" if item.get("speedup_export") is not None else "",
                disposition=item["disposition"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="capture files or directories containing captures")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = build_report(args.inputs)
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
