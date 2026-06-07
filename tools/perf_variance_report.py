#!/usr/bin/env python3
"""Report repeatability gates for reviewed RNS8 performance captures."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture
from benchmark_sweep_lib.capture_metadata import (
    backend_id,
    capture_contract_key,
    capture_execution_mode,
    capture_pack_mode,
    median_phase,
    selected_kernel,
)


DEFAULT_OUT_DIR = Path("temp") / "perf-variance-reports"
DEFAULT_MIN_REPEATS = 9
DEFAULT_MIN_RUNS = 2
DEFAULT_MAX_WITHIN_CAPTURE_REL_SPREAD = 0.15
DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD = 0.10
DEFAULT_MIN_NOISE_SPEEDUP_MARGIN = 0.03


def path_key(path: str | Path) -> str:
    return str(Path(path).resolve())


def _percentile_summary(capture: dict[str, Any], phase: str, key: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    item = summary.get(phase)
    if not isinstance(item, dict):
        return None
    value = item.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _export_kernel(capture: dict[str, Any]) -> str:
    variant = capture.get("export_variant")
    if isinstance(variant, dict) and isinstance(variant.get("selected_kernel"), str):
        return str(variant["selected_kernel"])
    return ""


def variance_group_key(capture: dict[str, Any]) -> str:
    return ";".join(
        [
            f"contract=({capture_contract_key(capture)})",
            f"backend={backend_id(capture)}",
            f"kernel={selected_kernel(capture)}",
            f"export_kernel={_export_kernel(capture)}",
            f"execution_mode={capture_execution_mode(capture)}",
            f"pack_mode={capture_pack_mode(capture)}",
        ]
    )


def _relative_spread(values: list[float]) -> float | None:
    positive = [float(value) for value in values if value > 0.0]
    if not positive:
        return None
    center = statistics.median(positive)
    if center <= 0.0:
        return None
    return (max(positive) - min(positive)) / center


def _within_capture_rel_spread(capture: dict[str, Any]) -> float | None:
    median = median_phase(capture, "end_to_end")
    if median is None or median <= 0.0:
        return None
    p95 = _percentile_summary(capture, "end_to_end", "p95")
    if p95 is None:
        raw = capture.get("raw_timings_us")
        values = raw.get("end_to_end") if isinstance(raw, dict) else None
        if isinstance(values, list):
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            p95 = max(numeric) if numeric else None
    if p95 is None:
        return None
    return max(0.0, (float(p95) - float(median)) / float(median))


def _timing_sample_count(capture: dict[str, Any]) -> int:
    raw = capture.get("raw_timings_us")
    values = raw.get("end_to_end") if isinstance(raw, dict) else None
    if isinstance(values, list):
        return sum(1 for value in values if isinstance(value, (int, float)))
    repeats = capture.get("repeats")
    return int(repeats) if isinstance(repeats, int) and not isinstance(repeats, bool) else 0


def _capture_entry(
    *,
    capture: dict[str, Any],
    path: Path,
    group_count: int,
    group_rel_spread: float | None,
    min_repeats: int,
    min_runs: int,
    require_multi_run: bool,
    max_within_capture_rel_spread: float,
    max_run_to_run_rel_spread: float,
    min_noise_speedup_margin: float,
) -> dict[str, Any]:
    median_us = median_phase(capture, "end_to_end")
    p95_us = _percentile_summary(capture, "end_to_end", "p95")
    within_rel_spread = _within_capture_rel_spread(capture)
    sample_count = _timing_sample_count(capture)
    blockers: list[str] = []
    if median_us is None or median_us <= 0.0:
        blockers.append("missing_end_to_end_median")
    if sample_count < min_repeats:
        blockers.append("insufficient_repeat_samples")
    if within_rel_spread is None:
        blockers.append("missing_within_capture_spread")
    elif within_rel_spread > max_within_capture_rel_spread:
        blockers.append("high_within_capture_variance")
    if require_multi_run and group_count < min_runs:
        blockers.append("multi_run_variance_missing")
    if group_count >= min_runs and group_rel_spread is not None and group_rel_spread > max_run_to_run_rel_spread:
        blockers.append("high_run_to_run_variance")

    observed_noise = max(
        value
        for value in [
            within_rel_spread if within_rel_spread is not None else 0.0,
            group_rel_spread if group_rel_spread is not None else 0.0,
            min_noise_speedup_margin,
        ]
    )
    required_speedup_margin = 1.0 + observed_noise
    return {
        "capture": str(path),
        "capture_key": path_key(path),
        "group_key": variance_group_key(capture),
        "backend": backend_id(capture),
        "selected_kernel": selected_kernel(capture),
        "export_selected_kernel": _export_kernel(capture) or None,
        "median_end_to_end_us": median_us,
        "p95_end_to_end_us": p95_us,
        "repeat_samples": sample_count,
        "group_capture_count": group_count,
        "within_capture_rel_spread": within_rel_spread,
        "run_to_run_rel_spread": group_rel_spread,
        "observed_max_relative_noise": observed_noise,
        "required_speedup_margin": required_speedup_margin,
        "promotion_ready": not blockers,
        "blockers": blockers,
    }


def build_report(
    captures: list[Path],
    *,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    min_runs: int = DEFAULT_MIN_RUNS,
    require_multi_run: bool = False,
    max_within_capture_rel_spread: float = DEFAULT_MAX_WITHIN_CAPTURE_REL_SPREAD,
    max_run_to_run_rel_spread: float = DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD,
    min_noise_speedup_margin: float = DEFAULT_MIN_NOISE_SPEEDUP_MARGIN,
) -> dict[str, Any]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in captures:
        capture = load_capture(path)
        validate_capture(capture, path)
        loaded.append((path, capture))

    by_group: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, capture in loaded:
        by_group.setdefault(variance_group_key(capture), []).append((path, capture))

    group_stats: dict[str, dict[str, Any]] = {}
    for key, items in by_group.items():
        medians = [
            value
            for _, capture in items
            if (value := median_phase(capture, "end_to_end")) is not None and value > 0.0
        ]
        group_stats[key] = {
            "capture_count": len(items),
            "median_end_to_end_us": statistics.median(medians) if medians else None,
            "run_to_run_rel_spread": _relative_spread(medians),
        }

    entries = [
        _capture_entry(
            capture=capture,
            path=path,
            group_count=int(group_stats[variance_group_key(capture)]["capture_count"]),
            group_rel_spread=group_stats[variance_group_key(capture)]["run_to_run_rel_spread"],
            min_repeats=min_repeats,
            min_runs=min_runs,
            require_multi_run=require_multi_run,
            max_within_capture_rel_spread=max_within_capture_rel_spread,
            max_run_to_run_rel_spread=max_run_to_run_rel_spread,
            min_noise_speedup_margin=min_noise_speedup_margin,
        )
        for path, capture in loaded
    ]
    return {
        "schema": "rns8_perf_variance_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "reviewed_performance_wins_must_clear_observed_repeatability_noise",
        "thresholds": {
            "min_repeats": min_repeats,
            "min_runs": min_runs,
            "require_multi_run": require_multi_run,
            "max_within_capture_rel_spread": max_within_capture_rel_spread,
            "max_run_to_run_rel_spread": max_run_to_run_rel_spread,
            "min_noise_speedup_margin": min_noise_speedup_margin,
        },
        "group_count": len(group_stats),
        "capture_count": len(entries),
        "blocked_count": sum(1 for entry in entries if entry["blockers"]),
        "groups": [
            {"group_key": key, **stats}
            for key, stats in sorted(group_stats.items(), key=lambda item: item[0])
        ],
        "entries": sorted(entries, key=lambda entry: str(entry["capture"])),
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "perf-variance-report.json"
    md_path = out_dir / "perf-variance-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RNS8 Performance Variance Report",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Captures: `{report['capture_count']}`",
        f"- Blocked captures: `{report['blocked_count']}`",
        "",
        "| capture | backend | median us | p95 us | required speedup | blockers |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for entry in report["entries"]:
        lines.append(
            "| `{capture}` | `{backend}` | `{median}` | `{p95}` | `{margin:.4f}` | `{blockers}` |".format(
                capture=entry["capture"],
                backend=entry["backend"],
                median=entry["median_end_to_end_us"],
                p95=entry["p95_end_to_end_us"],
                margin=entry["required_speedup_margin"],
                blockers=", ".join(entry["blockers"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-repeats", type=int, default=DEFAULT_MIN_REPEATS)
    parser.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS)
    parser.add_argument("--require-multi-run", action="store_true")
    parser.add_argument("--max-within-capture-rel-spread", type=float, default=DEFAULT_MAX_WITHIN_CAPTURE_REL_SPREAD)
    parser.add_argument("--max-run-to-run-rel-spread", type=float, default=DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD)
    parser.add_argument("--min-noise-speedup-margin", type=float, default=DEFAULT_MIN_NOISE_SPEEDUP_MARGIN)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.captures,
        min_repeats=args.min_repeats,
        min_runs=args.min_runs,
        require_multi_run=args.require_multi_run,
        max_within_capture_rel_spread=args.max_within_capture_rel_spread,
        max_run_to_run_rel_spread=args.max_run_to_run_rel_spread,
        min_noise_speedup_margin=args.min_noise_speedup_margin,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    return 0 if report["blocked_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
