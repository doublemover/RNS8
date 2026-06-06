#!/usr/bin/env python3
"""Build a focused bounded-i64 1024 hipBLASLt disposition report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture
from benchmark_schema.core_shared import _percentile
from promotion_ledger import load_variance_entries, path_key
import target_validation_report


DEFAULT_OUT_DIR = Path("temp") / "bounded-i64-1024-review"
DEFAULT_MARGIN = 1.02
POLICY = "setup_inclusive_bounded_i64_1024_same_target_disposition_only"


def _median(values: Any) -> float | None:
    if isinstance(values, list) and values:
        numbers = [float(value) for value in values if isinstance(value, (int, float))]
        return _percentile(numbers, 0.50) if numbers else None
    return None


def end_to_end_median(capture: dict[str, Any]) -> float | None:
    summary = capture.get("timing_summary_us")
    if isinstance(summary, dict):
        item = summary.get("end_to_end")
        if isinstance(item, dict) and isinstance(item.get("median"), (int, float)):
            return float(item["median"])
    raw = capture.get("raw_timings_us")
    if isinstance(raw, dict):
        return _median(raw.get("end_to_end"))
    value = capture.get("avg_end_to_end_us")
    return float(value) if isinstance(value, (int, float)) else None


def pack_mode(capture: dict[str, Any]) -> str:
    timing = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    for key in ("prepack_reuse_strategy", "pack_mode"):
        value = timing.get(key) or capture.get(key)
        if isinstance(value, str) and value:
            return value
    if capture.get("reuse_packed_inputs") is True:
        return "prepacked_reuse"
    if capture.get("reuse_packed_a") is True:
        return "prepacked_reuse_a"
    if capture.get("reuse_packed_b") is True:
        return "prepacked_reuse_b"
    return "per_repeat_repack"


def has_required_events(capture: dict[str, Any]) -> bool:
    backend = capture.get("backend_selected")
    if backend == "cpu-reference" or backend == "cpu":
        return True
    timing = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    events = capture.get("gpu_event_timings_us")
    return timing.get("gpu_event_timing_status") == "available" and isinstance(events, dict) and bool(events)


def exact_correct(capture: dict[str, Any]) -> bool:
    backend = capture.get("backend_selected")
    if backend == "cpu-reference" or backend == "cpu":
        return True
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    return metadata.get("exact_differential_validated") is True


def capture_row(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    if capture.get("semantics") != "bounded_i64" or any(capture.get(key) != 1024 for key in ("m", "n", "k")):
        raise RuntimeError(f"{path}: expected bounded_i64 1024x1024x1024 capture")
    target = target_validation_report.capture_target(path)
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    return {
        "path": str(path),
        "path_key": path_key(path),
        "backend": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "pack_mode": pack_mode(capture),
        "autotune_key": metadata.get("autotune_key"),
        "target_validation_group": target.get("target_validation_group"),
        "target_id": target.get("target_id"),
        "target_class": target.get("target_class"),
        "host_os": target.get("host_os"),
        "median_end_to_end_us": end_to_end_median(capture),
        "required_events": has_required_events(capture),
        "exact_correct": exact_correct(capture),
        "capture": capture,
    }


def load_target_groups(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(group.get("target_validation_group")): group
        for group in data.get("groups", [])
        if isinstance(group, dict) and group.get("target_validation_group")
    }


def load_ledger_entries(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry.get("autotune_key")): entry
        for entry in data.get("entries", [])
        if isinstance(entry, dict) and entry.get("autotune_key")
    }


def load_counter_reports(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    reports = data.get("reports") if isinstance(data.get("reports"), list) else [data]
    rows: dict[str, dict[str, Any]] = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        capture = report.get("capture")
        if isinstance(capture, dict) and isinstance(capture.get("path"), str):
            rows[path_key(capture["path"])] = report
    return rows


def row_gate_state(
    row: dict[str, Any],
    *,
    target_groups: dict[str, dict[str, Any]],
    ledger_entries: dict[str, dict[str, Any]],
    variance_entries: dict[str, dict[str, Any]],
    counter_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_group = target_groups.get(str(row.get("target_validation_group")))
    target_eligibility = target_group.get("cache_eligibility") if isinstance(target_group, dict) else {}
    ledger = ledger_entries.get(str(row.get("autotune_key")))
    variance = variance_entries.get(str(row.get("path_key")))
    counter = counter_reports.get(str(row.get("path_key")))
    resource_summary = counter.get("resource_summary") if isinstance(counter, dict) else None
    return {
        "target_validation_available": target_group is not None,
        "target_cache_eligible": isinstance(target_eligibility, dict) and target_eligibility.get("eligible") is True,
        "ledger_available": ledger is not None,
        "ledger_unblocked": isinstance(ledger, dict) and not ledger.get("promotion_blockers"),
        "ledger_installed_cache_entry": isinstance(ledger, dict) and ledger.get("installed_cache_entry") is True,
        "variance_available": variance is not None,
        "variance_ready": isinstance(variance, dict) and variance.get("promotion_ready") is True,
        "variance_required_speedup_margin": variance.get("required_speedup_margin") if isinstance(variance, dict) else None,
        "resource_explanation_available": isinstance(resource_summary, dict) and bool(resource_summary),
        "resource_summary": resource_summary,
    }


def decide_disposition(row: dict[str, Any], direct: dict[str, Any] | None, gate: dict[str, Any], margin: float) -> dict[str, Any]:
    blockers: list[str] = []
    if direct is None:
        blockers.append("missing_direct_hip_baseline")
    if row.get("median_end_to_end_us") is None:
        blockers.append("missing_candidate_timing")
    if row.get("exact_correct") is not True:
        blockers.append("missing_exact_correctness")
    if row.get("required_events") is not True:
        blockers.append("missing_required_gpu_events")
    if gate.get("target_validation_available") is not True:
        blockers.append("missing_target_validation")
    if gate.get("target_cache_eligible") is not True:
        blockers.append("target_not_cache_eligible")
    if gate.get("variance_available") is not True:
        blockers.append("missing_variance_gate")
    if gate.get("variance_ready") is not True:
        blockers.append("variance_not_ready")
    if gate.get("resource_explanation_available") is not True:
        blockers.append("missing_resource_explanation")
    if gate.get("ledger_available") is not True:
        blockers.append("missing_promotion_ledger")
    if gate.get("ledger_unblocked") is not True:
        blockers.append("promotion_ledger_blocked")

    speedup = None
    if direct is not None and direct.get("median_end_to_end_us") and row.get("median_end_to_end_us"):
        speedup = float(direct["median_end_to_end_us"]) / float(row["median_end_to_end_us"])
        if speedup < margin:
            blockers.append("does_not_clear_speedup_margin")

    promotable = not blockers
    if promotable and gate.get("ledger_installed_cache_entry") is True:
        disposition = "keep cache"
    elif promotable:
        disposition = "replace cache"
    elif direct is not None and speedup is not None and speedup < 1.0 and row.get("exact_correct") and row.get("required_events"):
        disposition = "drop/deprioritize"
    else:
        disposition = "keep experimental"
    return {
        "disposition": disposition,
        "speedup_vs_direct_hip": speedup,
        "required_speedup_margin": margin,
        "blockers": sorted(set(blockers)),
    }


def build_report(
    capture_paths: list[Path],
    *,
    target_validation: Path | None = None,
    counter_report: Path | None = None,
    promotion_ledger: Path | None = None,
    variance_reports: list[Path] | None = None,
    default_margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    rows = [capture_row(path) for path in capture_paths]
    target_groups = load_target_groups(target_validation)
    ledger_entries = load_ledger_entries(promotion_ledger)
    variance_entries = load_variance_entries(variance_reports or [])
    counter_reports = load_counter_reports(counter_report)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("target_validation_group")), []).append(row)

    groups = []
    for group_key, group_rows in sorted(by_group.items()):
        direct_candidates = [
            row for row in group_rows if row.get("backend") == "hip-direct" and row.get("pack_mode") == "per_repeat_repack"
        ]
        direct = min(
            (row for row in direct_candidates if row.get("median_end_to_end_us") is not None),
            key=lambda item: float(item["median_end_to_end_us"]),
            default=None,
        )
        hipblaslt_rows = [row for row in group_rows if row.get("backend") == "hipblaslt"]
        candidates = []
        for row in hipblaslt_rows:
            gate = row_gate_state(
                row,
                target_groups=target_groups,
                ledger_entries=ledger_entries,
                variance_entries=variance_entries,
                counter_reports=counter_reports,
            )
            margin = default_margin
            required = gate.get("variance_required_speedup_margin")
            if isinstance(required, (int, float)):
                margin = max(margin, float(required))
            decision = decide_disposition(row, direct, gate, margin)
            candidates.append({k: v for k, v in row.items() if k != "capture"} | {"gates": gate, **decision})
        if not candidates:
            candidates.append(
                {
                    "backend": "hipblaslt",
                    "disposition": "unsupported accelerator",
                    "blockers": ["no_hipblaslt_capture_for_target_group"],
                }
            )
        groups.append(
            {
                "target_validation_group": group_key,
                "direct_hip_baseline": {k: v for k, v in direct.items() if k != "capture"} if direct else None,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )
    return {
        "schema_version": 1,
        "policy": POLICY,
        "default_margin": default_margin,
        "capture_count": len(rows),
        "group_count": len(groups),
        "groups": groups,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bounded-i64-1024-review.json"
    md_path = out_dir / "bounded-i64-1024-review.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RNS8 Bounded-i64 1024 hipBLASLt Review",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Default margin: `{report['default_margin']}`",
        "",
        "| group | pack mode | median us | speedup vs Direct-HIP | disposition | blockers |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for group in report["groups"]:
        for candidate in group["candidates"]:
            speedup = candidate.get("speedup_vs_direct_hip")
            lines.append(
                "| `{group}` | `{pack}` | {median} | {speedup} | `{disposition}` | `{blockers}` |".format(
                    group=group["target_validation_group"],
                    pack=candidate.get("pack_mode"),
                    median=candidate.get("median_end_to_end_us"),
                    speedup=f"{float(speedup):.4g}" if isinstance(speedup, (int, float)) else "",
                    disposition=candidate.get("disposition"),
                    blockers=", ".join(candidate.get("blockers") or []),
                )
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--target-validation", type=Path)
    parser.add_argument("--counter-report", type=Path)
    parser.add_argument("--promotion-ledger", type=Path)
    parser.add_argument("--variance-report", type=Path, action="append", default=[])
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.captures,
        target_validation=args.target_validation,
        counter_report=args.counter_report,
        promotion_ledger=args.promotion_ledger,
        variance_reports=args.variance_report,
        default_margin=args.margin,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
