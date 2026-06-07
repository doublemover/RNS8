#!/usr/bin/env python3
"""Summarize Direct-HIP reuse expansion evidence for queue rank 20."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import reuse_contract_report


DEFAULT_OUT_DIR = Path("temp") / "direct-hip-reuse-expansion-reports"


def _capture_path(capture: dict[str, Any]) -> str:
    return str(capture.get("_path") or "")


def _scenario_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    scenario = capture.get("scenario_metadata")
    return scenario if isinstance(scenario, dict) else {}


def _scenario_inner_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = _scenario_metadata(capture).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _reuse_profile(capture: dict[str, Any]) -> str:
    metadata = _scenario_inner_metadata(capture)
    value = metadata.get("reuse_profile")
    if isinstance(value, str) and value:
        return value
    semantics = str(capture.get("semantics"))
    pack_mode = reuse_contract_report.pack_mode(capture)
    return f"{semantics}_{pack_mode}"


def _rank20_scope(capture: dict[str, Any]) -> bool:
    semantics = str(capture.get("semantics"))
    if semantics not in {"bounded_i64", "bounded_u64"}:
        return True
    return capture.get("input_distribution") not in {"signed_uniform_-16_16", "unsigned_uniform_0_16"}


def _comparison_profile(
    item: dict[str, Any],
    captures_by_path: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    capture = captures_by_path.get(str(item.get("reuse_capture"))) or {}
    scenario = _scenario_metadata(capture)
    profile = _reuse_profile(capture)
    return {
        "profile": profile,
        "rank20_scope": _rank20_scope(capture),
        "scenario_family": scenario.get("family"),
        "scenario_name": scenario.get("name"),
        "scenario_promotion_eligibility": scenario.get("promotion_eligibility"),
        "scenario_reuse_role": _scenario_inner_metadata(capture).get("reuse_contract_role"),
    }


def build_direct_hip_report_from_captures(captures: list[dict[str, Any]]) -> dict[str, Any]:
    base = reuse_contract_report.compare_reuse_contracts(captures)
    captures_by_path = {_capture_path(capture): capture for capture in captures if _capture_path(capture)}
    comparisons: list[dict[str, Any]] = []
    by_profile: dict[str, Counter[str]] = defaultdict(Counter)
    rank20_scope_count = 0
    for item in base["comparisons"]:
        if item.get("backend") != "hip-direct":
            continue
        profile_metadata = _comparison_profile(item, captures_by_path)
        enriched = {**item, **profile_metadata}
        comparisons.append(enriched)
        by_profile[str(profile_metadata["profile"])][str(item.get("decision"))] += 1
        if profile_metadata["rank20_scope"]:
            rank20_scope_count += 1

    summary = {
        "direct_hip_reuse_comparisons": len(comparisons),
        "rank20_scope_comparisons": rank20_scope_count,
        "candidate_workload_wins": sum(
            1 for item in comparisons if item.get("decision") == "candidate_workload_win"
        ),
        "explicit_workload_selector_ready": sum(
            1
            for item in comparisons
            if item.get("selector_eligibility", {}).get("explicit_workload_selector_eligible") is True
        ),
        "same_workload_family_ready": sum(
            1 for item in comparisons if item.get("same_workload_family", {}).get("available") is True
        ),
        "missing_baselines": sum(1 for item in comparisons if item.get("decision") == "missing_baseline"),
        "experimental": sum(1 for item in comparisons if item.get("decision") == "keep_experimental"),
        "deprioritized": sum(1 for item in comparisons if item.get("decision") == "deprioritize"),
        "profiles": {
            profile: dict(sorted(counter.items()))
            for profile, counter in sorted(by_profile.items())
        },
    }
    return {
        "schema_version": 1,
        "policy": "direct_hip_reuse_expansion_setup_inclusive_evidence_only",
        "source_report_policy": base.get("policy", reuse_contract_report.REUSE_SELECTOR_POLICY),
        "summary": summary,
        "comparisons": comparisons,
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = [reuse_contract_report.load_validated_capture(path) for path in reuse_contract_report.expand_inputs(paths)]
    report = build_direct_hip_report_from_captures(captures)
    return {"capture_count": len(captures), **report}


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "none"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Direct-HIP Reuse Expansion Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        if key == "profiles":
            continue
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Profiles", "", "| Profile | Decisions |", "|---|---|"])
    for profile, decisions in report["summary"]["profiles"].items():
        text = ", ".join(f"{key}={value}" for key, value in decisions.items())
        lines.append(f"| {profile} | {text} |")
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| profile | backend | semantics | shape | mode | repeats | setup us | same-backend speedup | best non-reuse speedup | decision | blockers |",
            "|---|---|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in report["comparisons"]:
        shape = item.get("shape", {})
        phase = item.get("phases", {}).get("end_to_end", {})
        blockers = ",".join(item.get("blockers") or []) or "none"
        lines.append(
            "| {profile} | {backend} | {semantics} | {m}x{n}x{k} | {mode} | {repeats} | {setup} | {same_speedup} | {best_speedup} | {decision} | {blockers} |".format(
                profile=item.get("profile"),
                backend=item.get("backend"),
                semantics=item.get("semantics"),
                m=shape.get("m"),
                n=shape.get("n"),
                k=shape.get("k"),
                mode=item.get("pack_mode"),
                repeats=item.get("repeats"),
                setup=fmt(item.get("prepack_setup_us")),
                same_speedup=fmt(phase.get("setup_inclusive_speedup")),
                best_speedup=fmt(item.get("speedup_vs_best_nonreuse_setup_inclusive")),
                decision=item.get("decision"),
                blockers=blockers,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "direct-hip-reuse-expansion-report.json"
    md_path = out_dir / "direct-hip-reuse-expansion-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": str(json_path), "markdown": str(md_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="*", help="schema-v4 benchmark JSON captures or directories")
    parser.add_argument("--capture", type=Path, action="append", help="additional capture file or directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="write JSON and Markdown reports here")
    parser.add_argument("--json", action="store_true", help="print report JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [*(args.capture or []), *args.captures]
    if not paths:
        raise SystemExit("direct_hip_reuse_expansion_report requires at least one capture path")
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
