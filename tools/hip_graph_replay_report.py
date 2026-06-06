#!/usr/bin/env python3
"""Summarize HIP Graph replay evidence for queue rank 30."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture
from benchmark_sweep_lib.capture_metadata import capture_execution_mode


DEFAULT_OUT_DIR = Path("temp") / "hip-graph-replay-reports"
NON_CAPTURE_JSON_NAMES = {
    "hip-graph-replay-report.json",
    "review_report.json",
    "scenario_manifest.json",
    "validation-summary.json",
}


def timing_summary_value(capture: dict[str, Any] | None, phase: str, statistic: str) -> float | None:
    if capture is None:
        return None
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    phase_summary = summary.get(phase)
    if not isinstance(phase_summary, dict):
        return None
    value = phase_summary.get(statistic)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def numeric_value(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def prepack_setup_us(capture: dict[str, Any]) -> float | None:
    value = numeric_value(capture, "avg_prepack_setup_us")
    if value is None:
        value = numeric_value(capture, "prepack_setup_us")
    return value


def graph_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    graph = capture.get("hip_graph_replay")
    return graph if isinstance(graph, dict) else {}


def graph_requested(capture: dict[str, Any]) -> bool:
    graph = graph_metadata(capture)
    return (
        graph.get("requested") is True
        or graph.get("used") is True
        or capture_execution_mode(capture) == "hip_graph_replay_resident_rns_chain"
    )


def scenario_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("scenario_metadata")
    return value if isinstance(value, dict) else {}


def scenario_inner_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = scenario_metadata(capture).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def exact_limb_count(capture: dict[str, Any]) -> Any:
    exact_contract = capture.get("exact_output_contract")
    if isinstance(exact_contract, dict):
        limb_count = exact_contract.get("limb_count")
        if limb_count is not None:
            return limb_count
    return capture.get("exact_wide_limb_count")


def output_domain(capture: dict[str, Any]) -> Any:
    exact_contract = capture.get("exact_output_contract")
    if isinstance(exact_contract, dict):
        domain = exact_contract.get("output_domain_after_measured_repeats")
        if domain is not None:
            return domain
    return capture.get("residue_output_mode")


def requested_next_op(capture: dict[str, Any]) -> Any:
    next_op = capture.get("requested_next_op")
    if isinstance(next_op, dict):
        return next_op.get("resolved") or next_op.get("requested")
    return None


def comparison_key(capture: dict[str, Any]) -> str:
    parts = [
        ("semantics", capture.get("semantics")),
        ("m", capture.get("m")),
        ("n", capture.get("n")),
        ("k", capture.get("k")),
        ("tile_m", capture.get("tile_m")),
        ("tile_n", capture.get("tile_n")),
        ("bound_kind", capture.get("bound_kind")),
        ("bound_mode", capture.get("bound_mode")),
        ("bound_source", capture.get("bound_source", "static_profile")),
        ("input_distribution", capture.get("input_distribution")),
        ("selected_prefix", capture.get("selected_prefix")),
        ("requested_max_prefix", capture.get("requested_max_prefix")),
        ("prefix_policy", capture.get("contract_prefix_policy")),
        ("exact_limb_count", exact_limb_count(capture)),
        ("residue_chain_length", capture.get("residue_chain_length", 1)),
        ("output_domain", output_domain(capture)),
        ("next_op", requested_next_op(capture)),
        ("seed", capture.get("seed")),
    ]
    return ";".join(f"{key}={value}" for key, value in parts)


def setup_inclusive_per_repeat(capture: dict[str, Any], *, graph: bool) -> float | None:
    median = timing_summary_value(capture, "end_to_end", "median")
    repeats = capture.get("repeats")
    setup = prepack_setup_us(capture)
    if median is None or not isinstance(repeats, int) or repeats <= 0 or setup is None:
        return None
    if graph:
        graph_info = graph_metadata(capture)
        capture_us = numeric_value(graph_info, "capture_us")
        instantiate_us = numeric_value(graph_info, "instantiate_us")
        if capture_us is None or instantiate_us is None:
            return None
        setup += capture_us + instantiate_us
    return median + setup / float(repeats)


def release_capture_satisfied(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
        and isinstance(capture.get("repeats"), int)
        and capture["warmups"] >= 3
        and capture["repeats"] >= 9
    )


def baseline_events_available(capture: dict[str, Any]) -> bool:
    metadata = capture.get("timing_metadata")
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("gpu_event_timing") is True
        and metadata.get("gpu_event_timing_status") == "available"
        and isinstance(metadata.get("gpu_event_phase_order"), list)
        and bool(metadata.get("gpu_event_timing_source"))
    )


def graph_wall_clock_policy_valid(capture: dict[str, Any]) -> bool:
    metadata = capture.get("timing_metadata")
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("gpu_event_timing") is False
        and metadata.get("gpu_event_timing_reason") == "hip_graph_replay_wall_clock_only"
        and metadata.get("gpu_event_timing_status") == "not_requested_graph_replay"
    )


def graph_status_available(capture: dict[str, Any]) -> bool:
    graph = graph_metadata(capture)
    return graph.get("status") == "available" and graph.get("capture_status") == "replayed"


def checksum_value(capture: dict[str, Any] | None) -> Any:
    if capture is None:
        return None
    if capture.get("checksum_u64") is not None:
        return capture.get("checksum_u64")
    return capture.get("checksum")


def shape(capture: dict[str, Any]) -> dict[str, Any]:
    return {
        "m": capture.get("m"),
        "n": capture.get("n"),
        "k": capture.get("k"),
        "residue_chain_length": capture.get("residue_chain_length", 1),
    }


def compare_graph_capture(graph: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    graph_setup = setup_inclusive_per_repeat(graph, graph=True)
    baseline_setup = setup_inclusive_per_repeat(baseline, graph=False) if baseline is not None else None
    graph_median = timing_summary_value(graph, "end_to_end", "median")
    baseline_median = timing_summary_value(baseline, "end_to_end", "median") if baseline is not None else None
    speedup = baseline_setup / graph_setup if baseline_setup not in (None, 0.0) and graph_setup else None
    steady_speedup = baseline_median / graph_median if baseline_median not in (None, 0.0) and graph_median else None

    blockers: list[str] = []
    if baseline is None:
        blockers.append("missing_same_contract_non_graph_baseline")
    else:
        if not release_capture_satisfied(baseline):
            blockers.append("baseline_not_release_review")
        if not baseline_events_available(baseline):
            blockers.append("baseline_missing_required_gpu_events")
        if (
            checksum_value(graph) is not None
            and checksum_value(baseline) is not None
            and checksum_value(graph) != checksum_value(baseline)
        ):
            blockers.append("checksum_mismatch")
    if not release_capture_satisfied(graph):
        blockers.append("graph_not_release_review")
    if not graph_status_available(graph):
        blockers.append("graph_replay_not_available")
    if not graph_wall_clock_policy_valid(graph):
        blockers.append("graph_timing_policy_invalid")
    if graph_setup is None or baseline_setup is None:
        blockers.append("missing_setup_inclusive_timing")

    if blockers:
        decision = "keep_experimental"
    elif speedup is not None and speedup > 1.0:
        decision = "candidate_workload_win"
    else:
        decision = "deprioritize"

    graph_info = graph_metadata(graph)
    return {
        "comparison_key": comparison_key(graph),
        "scenario_name": scenario_metadata(graph).get("name"),
        "scenario_promotion_eligibility": scenario_metadata(graph).get("promotion_eligibility"),
        "graph_role": scenario_inner_metadata(graph).get("graph_role"),
        "semantics": graph.get("semantics"),
        "shape": shape(graph),
        "backend": graph.get("backend_selected"),
        "graph_capture": graph.get("_path"),
        "baseline_capture": baseline.get("_path") if baseline else None,
        "graph_capture_us": numeric_value(graph_info, "capture_us"),
        "graph_instantiate_us": numeric_value(graph_info, "instantiate_us"),
        "graph_prepack_setup_us": prepack_setup_us(graph),
        "baseline_prepack_setup_us": prepack_setup_us(baseline) if baseline else None,
        "graph_median_end_to_end_us": graph_median,
        "baseline_median_end_to_end_us": baseline_median,
        "graph_setup_inclusive_per_repeat_us": graph_setup,
        "baseline_setup_inclusive_per_repeat_us": baseline_setup,
        "steady_state_speedup": steady_speedup,
        "setup_inclusive_speedup": speedup,
        "release_review_satisfied": not any(
            blocker in blockers for blocker in {"baseline_not_release_review", "graph_not_release_review"}
        ),
        "baseline_gpu_events_available": baseline_events_available(baseline) if baseline else False,
        "graph_wall_clock_policy_valid": graph_wall_clock_policy_valid(graph),
        "checksum_match": bool(
            baseline
            and checksum_value(graph) is not None
            and checksum_value(baseline) is not None
            and checksum_value(graph) == checksum_value(baseline)
        ),
        "decision": decision,
        "blockers": blockers,
        "promotion_eligible": False,
    }


def build_hip_graph_replay_report_from_captures(captures: list[dict[str, Any]]) -> dict[str, Any]:
    for capture in captures:
        capture.setdefault("_path", "")
    baselines: dict[str, dict[str, Any]] = {}
    graphs: list[dict[str, Any]] = []
    for capture in captures:
        if graph_requested(capture):
            graphs.append(capture)
        elif scenario_inner_metadata(capture).get("graph_role") == "same_contract_non_graph_baseline" or (
            scenario_metadata(capture).get("family") == "hip-graph-replay"
        ):
            baselines[comparison_key(capture)] = capture

    comparisons = [compare_graph_capture(graph, baselines.get(comparison_key(graph))) for graph in graphs]
    decision_counts = Counter(str(item["decision"]) for item in comparisons)
    summary = {
        "capture_count": len(captures),
        "graph_candidate_count": len(graphs),
        "comparison_count": len(comparisons),
        "candidate_workload_wins": decision_counts.get("candidate_workload_win", 0),
        "deprioritized": decision_counts.get("deprioritize", 0),
        "experimental": decision_counts.get("keep_experimental", 0),
        "missing_baselines": sum(
            1 for item in comparisons if "missing_same_contract_non_graph_baseline" in item["blockers"]
        ),
        "release_review_satisfied": sum(1 for item in comparisons if item["release_review_satisfied"]),
        "baseline_gpu_events_available": sum(1 for item in comparisons if item["baseline_gpu_events_available"]),
        "graph_wall_clock_policy_valid": sum(1 for item in comparisons if item["graph_wall_clock_policy_valid"]),
        "checksum_matches": sum(1 for item in comparisons if item["checksum_match"]),
        "decisions": dict(sorted(decision_counts.items())),
    }
    return {
        "schema_version": 1,
        "policy": "hip_graph_replay_setup_inclusive_evidence_only_not_autotune_promotion",
        "summary": summary,
        "comparisons": comparisons,
    }


def expand_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                item
                for item in sorted(path.rglob("*.json"))
                if item.name not in NON_CAPTURE_JSON_NAMES and not item.name.endswith("-report.json")
            )
        else:
            expanded.append(path)
    return expanded


def load_validated_capture(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    capture["_path"] = str(path)
    return capture


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = [load_validated_capture(path) for path in expand_inputs(paths)]
    return build_hip_graph_replay_report_from_captures(captures)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "none"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# HIP Graph Replay Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| semantics | shape | graph setup-inclusive us | baseline setup-inclusive us | speedup | decision | blockers |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for item in report["comparisons"]:
        item_shape = item.get("shape", {})
        blockers = ",".join(item.get("blockers") or []) or "none"
        lines.append(
            "| {semantics} | {m}x{n}x{k} chain{chain} | {graph_us} | {baseline_us} | {speedup} | {decision} | {blockers} |".format(
                semantics=item.get("semantics"),
                m=item_shape.get("m"),
                n=item_shape.get("n"),
                k=item_shape.get("k"),
                chain=item_shape.get("residue_chain_length"),
                graph_us=fmt(item.get("graph_setup_inclusive_per_repeat_us")),
                baseline_us=fmt(item.get("baseline_setup_inclusive_per_repeat_us")),
                speedup=fmt(item.get("setup_inclusive_speedup")),
                decision=item.get("decision"),
                blockers=blockers,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "hip-graph-replay-report.json"
    md_path = out_dir / "hip-graph-replay-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": str(json_path), "markdown": str(md_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="*", help="schema-v4 benchmark JSON captures or directories")
    parser.add_argument("--capture", type=Path, action="append", help="additional capture file or directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true", help="print report JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [*(args.capture or []), *args.captures]
    if not paths:
        raise SystemExit("hip_graph_replay_report requires at least one capture path")
    report = build_report(paths)
    outputs = write_outputs(report, args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in outputs.items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
