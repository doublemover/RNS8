#!/usr/bin/env python3
"""Classify end-to-end layout-search benchmark candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


PROMOTION_SPEEDUP_THRESHOLD = 1.02


def expand_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.json")))
        else:
            expanded.append(path)
    return expanded


def load_validated_capture(path: Path) -> dict[str, Any]:
    try:
        data = load_capture(path)
        validate_capture(data, path)
    except BenchmarkSchemaError as exc:
        raise SystemExit(str(exc)) from exc
    data["_path"] = str(path)
    return data


def scenario_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("scenario_metadata")
    return value if isinstance(value, dict) else {}


def layout_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    scenario = scenario_metadata(capture)
    value = scenario.get("metadata")
    return value if isinstance(value, dict) else {}


def backend_id(capture: dict[str, Any]) -> str:
    return str(capture.get("backend_selected") or capture.get("backend_requested") or "")


def target_id(capture: dict[str, Any]) -> str:
    target = capture.get("target_variant")
    if isinstance(target, dict) and target.get("target_id"):
        return str(target["target_id"])
    return str(capture.get("target_id") or "")


def shape(capture: dict[str, Any]) -> dict[str, Any]:
    return {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")}


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


def phase_attribution(capture: dict[str, Any]) -> dict[str, float | None]:
    return {
        "pack_us": timing_median(capture, "pack"),
        "gemm_us": timing_median(capture, "rns_gemm"),
        "export_us": timing_median(capture, "crt_export"),
        "conversion_setup_us": timing_median(capture, "native_to_rns_bridge")
        or event_median(capture, "native_to_rns_convert")
        or event_median(capture, "vector_to_rns_convert"),
        "d2h_us": event_median(capture, "crt_export_d2h")
        or event_median(capture, "exact_wide_export_d2h")
        or event_median(capture, "finite_export_d2h")
        or event_median(capture, "wrap64_export_d2h")
        or event_median(capture, "vector_alu_output_d2h"),
        "end_to_end_us": timing_median(capture, "end_to_end"),
    }


def gpu_events_available(capture: dict[str, Any]) -> bool | None:
    backend = backend_id(capture)
    if backend in {"cpu", "cpu-reference", "wrap64-byte-limb"}:
        return None
    timing = capture.get("timing_metadata")
    if not isinstance(timing, dict):
        return False
    return (
        timing.get("gpu_event_timing") is True
        and timing.get("gpu_event_timing_status") == "available"
        and isinstance(timing.get("gpu_event_phase_order"), list)
    )


def release_reviewed(capture: dict[str, Any]) -> bool:
    return int(capture.get("warmups", 0) or 0) >= 3 and int(capture.get("repeats", 0) or 0) >= 9


def output_policy(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("output_policy")
    return value if isinstance(value, dict) else {}


def timing_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("timing_metadata")
    return value if isinstance(value, dict) else {}


def contract_key(capture: dict[str, Any], baseline_name: str | None = None) -> tuple[Any, ...]:
    scenario = scenario_metadata(capture)
    return (
        baseline_name or scenario.get("name"),
        backend_id(capture),
        target_id(capture),
        scenario.get("semantics") or capture.get("semantics"),
        capture.get("m"),
        capture.get("n"),
        capture.get("k"),
        scenario.get("modulus"),
        scenario.get("exact_wide_limb_count"),
        scenario.get("output_domain"),
    )


def actual_layout_deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    deltas: list[str] = []
    cand_timing = timing_metadata(candidate)
    base_timing = timing_metadata(baseline)
    cand_output = output_policy(candidate)
    base_output = output_policy(baseline)
    cand_scenario = scenario_metadata(candidate)
    base_scenario = scenario_metadata(baseline)
    comparisons = [
        ("pack_layout_changed", cand_timing.get("pack_layout"), base_timing.get("pack_layout")),
        ("residue_group_layout_changed", cand_timing.get("residue_group_layout"), base_timing.get("residue_group_layout")),
        ("output_destination_changed", cand_output.get("destination_layout"), base_output.get("destination_layout")),
        ("output_ld_changed", cand_output.get("logical_ld"), base_output.get("logical_ld")),
        ("output_padding_changed", cand_output.get("ld_padding"), base_output.get("ld_padding")),
        ("next_op_changed", cand_scenario.get("next_op_hint"), base_scenario.get("next_op_hint")),
        ("export_variant_changed", cand_scenario.get("export_variant"), base_scenario.get("export_variant")),
        (
            "reconstruction_variant_changed",
            cand_scenario.get("reconstruction_variant"),
            base_scenario.get("reconstruction_variant"),
        ),
    ]
    for name, candidate_value, baseline_value in comparisons:
        if candidate_value != baseline_value:
            deltas.append(name)
    if int(cand_scenario.get("residue_chain_length") or 1) != int(base_scenario.get("residue_chain_length") or 1):
        deltas.append("residue_chain_layout_changed")
    if cand_scenario.get("residue_chain_final_export") != base_scenario.get("residue_chain_final_export"):
        deltas.append("residue_chain_final_export_policy_changed")
    if cand_scenario.get("residue_chain_independent_final_export") != base_scenario.get(
        "residue_chain_independent_final_export"
    ):
        deltas.append("residue_chain_independent_export_policy_changed")
    if cand_scenario.get("residue_channel_fusion") is True:
        deltas.append("residue_channel_fusion_enabled")
    if cand_scenario.get("native_to_rns_bridge") is True:
        deltas.append("native_to_rns_bridge_enabled")
    if cand_scenario.get("vector_to_rns_chain") is True:
        deltas.append("vector_to_rns_chain_enabled")
    return list(dict.fromkeys(deltas))


def candidate_rows(captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    for capture in captures:
        metadata = layout_metadata(capture)
        if metadata.get("layout_variant_role") == "baseline":
            baselines[contract_key(capture)] = capture

    rows: list[dict[str, Any]] = []
    for capture in sorted(captures, key=lambda item: str(item.get("_path", ""))):
        metadata = layout_metadata(capture)
        if metadata.get("layout_variant_role") != "candidate":
            continue
        baseline_name = str(metadata.get("layout_baseline_name") or "")
        baseline = baselines.get(contract_key(capture, baseline_name=baseline_name))
        blockers: list[str] = []
        baseline_median = None
        speedup = None
        deltas: list[str] = []
        if baseline is None:
            blockers.append("missing_default_layout_baseline")
        else:
            deltas = actual_layout_deltas(capture, baseline)
            if not deltas:
                blockers.append("metadata_only_layout_candidate")
            if scenario_metadata(capture).get("output_domain") != scenario_metadata(baseline).get("output_domain"):
                blockers.append("output_contract_mismatch")
            if int(scenario_metadata(capture).get("residue_chain_length") or 1) != int(
                scenario_metadata(baseline).get("residue_chain_length") or 1
            ):
                blockers.append("residue_chain_length_mismatch")
            baseline_median = timing_median(baseline, "end_to_end")
        candidate_median = timing_median(capture, "end_to_end")
        if candidate_median is None:
            blockers.append("missing_candidate_end_to_end_timing")
        if baseline is not None and baseline_median is None:
            blockers.append("missing_baseline_end_to_end_timing")
        if candidate_median is not None and baseline_median and baseline_median > 0:
            speedup = baseline_median / candidate_median
        if not release_reviewed(capture):
            blockers.append("candidate_not_release_reviewed")
        if baseline is not None and not release_reviewed(baseline):
            blockers.append("baseline_not_release_reviewed")
        if gpu_events_available(capture) is False:
            blockers.append("missing_candidate_gpu_events")
        if baseline is not None and gpu_events_available(baseline) is False:
            blockers.append("missing_baseline_gpu_events")
        attribution = phase_attribution(capture)
        for phase in ("pack_us", "gemm_us", "export_us", "end_to_end_us"):
            if attribution[phase] is None:
                blockers.append(f"missing_{phase}")
        if attribution["d2h_us"] is None:
            blockers.append("missing_d2h_attribution")
        decision = "keep experimental"
        if speedup is not None and baseline is not None:
            hard_blockers = [
                blocker
                for blocker in blockers
                if blocker
                not in {
                    "missing_d2h_attribution",
                }
            ]
            if speedup < 1.0 and not hard_blockers:
                decision = "drop/deprioritize"
            elif speedup >= PROMOTION_SPEEDUP_THRESHOLD and not blockers:
                decision = "promote locally"
            elif speedup < 1.0 and baseline is not None and "missing_default_layout_baseline" not in blockers:
                decision = "drop/deprioritize"
        rows.append(
            {
                "capture": capture.get("_path"),
                "baseline_capture": baseline.get("_path") if baseline else None,
                "layout_variant_name": metadata.get("layout_variant_name") or scenario_metadata(capture).get("name"),
                "layout_role": metadata.get("layout_role"),
                "baseline_name": baseline_name or None,
                "backend": backend_id(capture),
                "target_id": target_id(capture),
                "semantics": scenario_metadata(capture).get("semantics") or capture.get("semantics"),
                "shape": shape(capture),
                "modulus": scenario_metadata(capture).get("modulus"),
                "exact_wide_limb_count": scenario_metadata(capture).get("exact_wide_limb_count"),
                "output_domain": scenario_metadata(capture).get("output_domain"),
                "pack_layout": timing_metadata(capture).get("pack_layout"),
                "residue_group_layout": timing_metadata(capture).get("residue_group_layout"),
                "output_policy": output_policy(capture),
                "layout_deltas": deltas,
                "actual_layout_variant": bool(deltas),
                "candidate_phase_medians_us": attribution,
                "baseline_end_to_end_us": baseline_median,
                "candidate_end_to_end_us": candidate_median,
                "speedup_vs_default_layout": speedup,
                "candidate_gpu_events": gpu_events_available(capture),
                "release_reviewed": release_reviewed(capture),
                "promotion_blockers": list(dict.fromkeys(blockers)),
                "decision": decision,
                "promotion_eligible": decision == "promote locally",
            }
        )
    return rows


def build_report(captures: list[dict[str, Any]]) -> dict[str, Any]:
    rows = candidate_rows(captures)
    decisions: dict[str, int] = {}
    for row in rows:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    return {
        "schema": "rns8_layout_search_report_v1",
        "capture_count": len(captures),
        "baseline_count": sum(1 for capture in captures if layout_metadata(capture).get("layout_variant_role") == "baseline"),
        "candidate_count": len(rows),
        "decision_counts": decisions,
        "rows": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Layout Search Report",
        "",
        f"- Captures: {report['capture_count']}",
        f"- Baselines: {report['baseline_count']}",
        f"- Candidates: {report['candidate_count']}",
        f"- Decisions: {json.dumps(report['decision_counts'], sort_keys=True)}",
        "",
        "| Variant | Semantic | Shape | Decision | Speedup vs default | Blockers |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in report["rows"]:
        shape_text = f"{row['shape']['m']}x{row['shape']['n']}x{row['shape']['k']}"
        speedup = row["speedup_vs_default_layout"]
        speedup_text = "" if speedup is None else f"{speedup:.3f}x"
        blockers = ", ".join(row["promotion_blockers"])
        lines.append(
            f"| {row['layout_variant_name']} | {row['semantics']} | {shape_text} | "
            f"{row['decision']} | {speedup_text} | {blockers} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--json", action="store_true", dest="print_json")
    args = parser.parse_args()
    captures = [load_validated_capture(path) for path in expand_inputs(args.captures)]
    report = build_report(captures)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "layout-search-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_markdown(report, args.out_dir / "layout-search-report.md")
    if args.print_json or not args.out_dir:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
