#!/usr/bin/env python3
"""Classify vector/native producer to Direct-HIP RNS consumer chain captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from report_capture_inputs import load_report_captures


PROMOTION_SPEEDUP_THRESHOLD = 1.02
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9


def timing_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("timing_metadata")
    return value if isinstance(value, dict) else {}


def scenario_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("scenario_metadata")
    return value if isinstance(value, dict) else {}


def scenario_inner_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = scenario_metadata(capture).get("metadata")
    return value if isinstance(value, dict) else {}


def control_mode(capture: dict[str, Any]) -> str:
    metadata = timing_metadata(capture)
    value = metadata.get("vector_to_rns_chain_control_mode")
    if isinstance(value, str) and value:
        return value
    scenario_value = scenario_inner_metadata(capture).get("chain_control_mode")
    return str(scenario_value or "")


def target_id(capture: dict[str, Any]) -> str:
    target = capture.get("target_variant")
    if isinstance(target, dict) and target.get("target_id"):
        return str(target["target_id"])
    return str(capture.get("target_id") or "")


def timing_median(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    item = summary.get(phase)
    if not isinstance(item, dict):
        return None
    value = item.get("median")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def event_median(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("gpu_event_timing_summary_us")
    if not isinstance(summary, dict):
        return None
    item = summary.get(phase)
    if not isinstance(item, dict):
        return None
    value = item.get("median")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def chain_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    return (
        target_id(capture),
        capture.get("semantics"),
        capture.get("m"),
        capture.get("n"),
        capture.get("k"),
        capture.get("pack_mode"),
        capture.get("backend_selected"),
        capture.get("backend_requested"),
    )


def release_reviewed(capture: dict[str, Any]) -> bool:
    return int(capture.get("warmups") or 0) >= RELEASE_MIN_WARMUPS and int(capture.get("repeats") or 0) >= RELEASE_MIN_REPEATS


def event_complete(capture: dict[str, Any], required: list[str]) -> bool:
    metadata = timing_metadata(capture)
    if metadata.get("gpu_event_timing") is not True:
        return False
    phases = metadata.get("gpu_event_phase_order")
    timings = capture.get("gpu_event_timings_us")
    if not isinstance(phases, list) or not isinstance(timings, dict):
        return False
    return all(phase in phases and phase in timings for phase in required)


def vector_chain_captures(captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        capture
        for capture in captures
        if capture.get("benchmark_execution_mode")
        in {
            "vector_native_to_direct_rns_chain",
            "vector_native_host_export_repack_direct_rns_chain",
        }
    ]


def classify_pair(candidate: dict[str, Any] | None, control: dict[str, Any] | None) -> dict[str, Any]:
    blockers: list[str] = []
    if candidate is None:
        blockers.append("missing_fused_device_candidate")
    if control is None:
        blockers.append("missing_host_export_repack_control")
    if candidate is None or control is None:
        return {
            "disposition": "keep experimental",
            "blockers": blockers,
        }

    candidate_median = timing_median(candidate, "end_to_end")
    control_median = timing_median(control, "end_to_end")
    if candidate_median is None:
        blockers.append("missing_candidate_end_to_end_median")
    if control_median is None:
        blockers.append("missing_control_end_to_end_median")
    if candidate.get("checksum") != control.get("checksum"):
        blockers.append("final_checksum_mismatch")
    if not release_reviewed(candidate):
        blockers.append("candidate_not_release_reviewed")
    if not release_reviewed(control):
        blockers.append("control_not_release_reviewed")
    if not event_complete(candidate, ["vector_alu_status_memset", "rns_gemm_kernel_group", "crt_export"]):
        blockers.append("candidate_missing_required_events")
    if not event_complete(control, ["vector_alu_output_d2h", "vector_to_rns_host_repack_a", "rns_gemm_kernel_group"]):
        blockers.append("control_missing_required_events")

    speedup = None
    if candidate_median and control_median and candidate_median > 0:
        speedup = control_median / candidate_median

    if blockers:
        disposition = "keep experimental"
    elif speedup is not None and speedup >= PROMOTION_SPEEDUP_THRESHOLD:
        disposition = "local promote"
    elif speedup is not None and speedup < 1.0:
        disposition = "drop/deprioritize"
    else:
        disposition = "keep experimental"

    return {
        "disposition": disposition,
        "blockers": blockers,
        "candidate_end_to_end_median_us": candidate_median,
        "control_end_to_end_median_us": control_median,
        "speedup_vs_host_repack_control": speedup,
        "checksum_match": candidate.get("checksum") == control.get("checksum"),
        "candidate_pack_median_us": timing_median(candidate, "pack"),
        "candidate_chain_median_us": timing_median(candidate, "rns_gemm"),
        "candidate_export_median_us": timing_median(candidate, "crt_export"),
        "control_pack_median_us": timing_median(control, "pack"),
        "control_chain_median_us": timing_median(control, "rns_gemm"),
        "control_export_median_us": timing_median(control, "crt_export"),
        "candidate_conversion_event_us": event_median(
            candidate,
            "native_i64_to_rns_kernel"
            if candidate.get("semantics") == "bounded_i64"
            else "native_u64_to_rns_kernel",
        ),
        "control_vector_output_d2h_us": event_median(control, "vector_alu_output_d2h"),
        "control_host_repack_us": event_median(control, "vector_to_rns_host_repack_a"),
    }


def build_report(captures: list[dict[str, Any]]) -> dict[str, Any]:
    selected = vector_chain_captures(captures)
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for capture in selected:
        groups.setdefault(chain_key(capture), {})[control_mode(capture)] = capture

    rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        candidate = group.get("fused_device_native_to_rns")
        control = group.get("host_export_repack_control")
        classified = classify_pair(candidate, control)
        representative = candidate or control or {}
        rows.append(
            {
                "target_id": key[0],
                "semantics": key[1],
                "shape": {"m": key[2], "n": key[3], "k": key[4]},
                "pack_mode": key[5],
                "backend_selected": key[6],
                "backend_requested": key[7],
                "candidate_capture": candidate.get("_path") if candidate else None,
                "control_capture": control.get("_path") if control else None,
                "scenario_name": scenario_metadata(representative).get("name"),
                **classified,
            }
        )

    disposition_counts: dict[str, int] = {}
    for row in rows:
        disposition = str(row["disposition"])
        disposition_counts[disposition] = disposition_counts.get(disposition, 0) + 1

    return {
        "schema": "rns8_vector_to_rns_chain_report_v1",
        "capture_count": len(selected),
        "group_count": len(rows),
        "disposition_counts": disposition_counts,
        "rows": rows,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vector-to-rns-chain-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Vector/Native-To-RNS Chain Report",
        "",
        f"- Captures: {report['capture_count']}",
        f"- Groups: {report['group_count']}",
        f"- Dispositions: {json.dumps(report['disposition_counts'], sort_keys=True)}",
        "",
        "| Semantics | Shape | Pack Mode | Candidate us | Control us | Speedup | Disposition | Blockers |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        shape = row["shape"]
        speedup = row.get("speedup_vs_host_repack_control")
        lines.append(
            "| {semantics} | {shape} | {pack} | {candidate} | {control} | {speedup} | {disp} | {blockers} |".format(
                semantics=row["semantics"],
                shape=f"{shape['m']}x{shape['n']}x{shape['k']}",
                pack=row["pack_mode"],
                candidate=f"{row.get('candidate_end_to_end_median_us'):.3f}"
                if isinstance(row.get("candidate_end_to_end_median_us"), float)
                else "",
                control=f"{row.get('control_end_to_end_median_us'):.3f}"
                if isinstance(row.get("control_end_to_end_median_us"), float)
                else "",
                speedup=f"{speedup:.3f}x" if isinstance(speedup, float) else "",
                disp=row["disposition"],
                blockers=", ".join(row.get("blockers") or []),
            )
        )
    (out_dir / "vector-to-rns-chain-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("temp/vector-to-rns-chain-report"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    captures = load_report_captures(args.captures)
    report = build_report(captures)
    write_outputs(report, args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(args.out_dir / "vector-to-rns-chain-report.md")


if __name__ == "__main__":
    main()
