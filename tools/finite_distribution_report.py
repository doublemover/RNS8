#!/usr/bin/env python3
"""Summarize finite-u8 input-distribution sensitivity captures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture
from benchmark_sweep_lib.capture_metadata import backend_id, median_phase, selected_kernel
from benchmark_sweep_lib.config import RELEASE_MIN_REPEATS, RELEASE_MIN_WARMUPS


FINITE_SEMANTICS = {"finite_ring_u8", "finite_field_u8"}
SIDECAR_NAMES = {
    "review_report.json",
    "scenario_manifest.json",
    "command-plan.json",
    "finite-distribution-report.json",
}
FULL_UNIFORM_DISTRIBUTIONS = {
    "u8_full_uniform_0_modulus_minus_1",
    "u8_uniform_0_modulus_minus_1",
}


def _distribution(value: Any) -> str:
    text = str(value) if isinstance(value, str) and value else "unknown"
    return "u8_full_uniform_0_modulus_minus_1" if text in FULL_UNIFORM_DISTRIBUTIONS else text


def _distribution_role(distribution: str) -> str:
    return {
        "u8_binary_0_1": "binary",
        "u8_sparse_90pct_zero_uniform_nonzero": "sparse",
        "u8_low_hamming_powers_of_two_mod_q": "low_hamming",
        "u8_small_centered_minus2_2_mod_q": "small_centered",
        "u8_full_uniform_0_modulus_minus_1": "full_uniform",
    }.get(distribution, "unknown")


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    factor = 3
    while factor * factor <= value:
        if value % factor == 0:
            return False
        factor += 2
    return True


def _modulus_role(semantics: str, modulus: int) -> str:
    if semantics == "finite_field_u8":
        return "hot_prime" if modulus == 251 else "generic_prime"
    if modulus == 251:
        return "hot_prime"
    if modulus == 255:
        return "hot_composite"
    if modulus == 256:
        return "hot_power_of_two"
    if _is_prime(modulus):
        return "generic_prime"
    return "generic_composite"


def _release_ready(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
        and capture["warmups"] >= RELEASE_MIN_WARMUPS
        and isinstance(capture.get("repeats"), int)
        and capture["repeats"] >= RELEASE_MIN_REPEATS
    )


def _gpu_events_available(capture: dict[str, Any]) -> bool:
    timing = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    return (
        timing.get("gpu_event_timing") is True
        and timing.get("gpu_event_timing_status") == "available"
        and isinstance(timing.get("gpu_event_phase_order"), list)
        and bool(timing.get("gpu_event_phase_order"))
    )


def _target_id(capture: dict[str, Any]) -> str | None:
    for source in (capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict):
            for key in ("target_id", "target_arch", "gcn_arch"):
                value = source.get(key)
                if isinstance(value, str) and value and value not in {"none", "unknown"}:
                    return value
    return None


def _iter_paths(inputs: list[Path]) -> list[tuple[Path, bool]]:
    paths: list[tuple[Path, bool]] = []
    for item in inputs:
        if item.is_dir():
            for path in sorted(item.rglob("*.json")):
                if path.name in SIDECAR_NAMES:
                    continue
                paths.append((path, True))
        else:
            paths.append((item, False))
    return paths


def _load_capture(path: Path, *, from_directory: bool) -> dict[str, Any] | None:
    try:
        capture = load_capture(path)
        validate_capture(capture, path)
    except (BenchmarkSchemaError, OSError, JSONDecodeError):
        if from_directory:
            return None
        raise
    if capture.get("semantics") not in FINITE_SEMANTICS:
        return None
    capture["_path"] = str(path)
    return capture


def _row(capture: dict[str, Any]) -> dict[str, Any]:
    backend = backend_id(capture)
    distribution = _distribution(capture.get("input_distribution"))
    return {
        "backend": backend,
        "capture_path": capture.get("_path"),
        "distribution": distribution,
        "distribution_role": _distribution_role(distribution),
        "selected_kernel": selected_kernel(capture),
        "target_id": "cpu" if backend == "cpu-reference" else _target_id(capture),
        "release_ready": _release_ready(capture),
        "gpu_events_required": backend != "cpu-reference",
        "gpu_events_available": True if backend == "cpu-reference" else _gpu_events_available(capture),
        "median_end_to_end_us": median_phase(capture, "end_to_end"),
        "median_pack_us": median_phase(capture, "pack"),
        "median_gemm_us": median_phase(capture, "rns_gemm"),
        "median_export_us": median_phase(capture, "crt_export"),
    }


def _group_key(capture: dict[str, Any]) -> tuple[str, int, int, int, int, str]:
    distribution = _distribution(capture.get("input_distribution"))
    return (
        str(capture.get("semantics")),
        int(capture.get("finite_modulus") or 0),
        int(capture.get("m") or 0),
        int(capture.get("n") or 0),
        int(capture.get("k") or 0),
        distribution,
    )


def _baseline_key(key: tuple[str, int, int, int, int, str]) -> tuple[str, int, int, int, int, str]:
    semantics, modulus, m, n, k, _distribution_name = key
    return (semantics, modulus, m, n, k, "u8_full_uniform_0_modulus_minus_1")


def _winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    timed = [row for row in rows if isinstance(row.get("median_end_to_end_us"), (int, float))]
    if not timed:
        return None
    best = min(timed, key=lambda row: float(row["median_end_to_end_us"]))
    return {
        "backend": best.get("backend"),
        "selected_kernel": best.get("selected_kernel"),
        "median_end_to_end_us": best.get("median_end_to_end_us"),
    }


def _backend_row(rows: list[dict[str, Any]], backend: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("backend") == backend:
            return row
    return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)) or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _blockers(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    backends = {row.get("backend") for row in rows}
    if "cpu-reference" not in backends:
        blockers.append("missing_cpu_reference")
    if "hip-direct" not in backends:
        blockers.append("missing_direct_hip")
    if not baseline_rows:
        blockers.append("missing_full_uniform_baseline")
    if any(row.get("release_ready") is not True for row in rows):
        blockers.append("not_release_reviewed")
    if any(row.get("gpu_events_required") is True and row.get("gpu_events_available") is not True for row in rows):
        blockers.append("missing_required_gpu_events")
    return blockers


def _classification(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    distribution: str,
) -> dict[str, Any]:
    winner = _winner(rows)
    baseline_winner = _winner(baseline_rows)
    direct = _backend_row(rows, "hip-direct")
    baseline_direct = _backend_row(baseline_rows, "hip-direct")
    cpu = _backend_row(rows, "cpu-reference")
    baseline_cpu = _backend_row(baseline_rows, "cpu-reference")
    blockers = _blockers(rows, baseline_rows)
    backend_changed = bool(winner and baseline_winner and winner.get("backend") != baseline_winner.get("backend"))
    pack_ratio = _ratio(
        direct.get("median_pack_us") if direct else None,
        baseline_direct.get("median_pack_us") if baseline_direct else None,
    )
    gemm_ratio = _ratio(
        direct.get("median_gemm_us") if direct else None,
        baseline_direct.get("median_gemm_us") if baseline_direct else None,
    )
    export_ratio = _ratio(
        direct.get("median_export_us") if direct else None,
        baseline_direct.get("median_export_us") if baseline_direct else None,
    )
    cpu_ratio = _ratio(
        cpu.get("median_end_to_end_us") if cpu else None,
        baseline_cpu.get("median_end_to_end_us") if baseline_cpu else None,
    )
    if distribution == "u8_full_uniform_0_modulus_minus_1":
        disposition = "baseline_reference"
    elif blockers:
        disposition = "keep experimental"
    elif backend_changed or any(
        isinstance(value, float) and (value <= 0.9 or value >= 1.1)
        for value in (pack_ratio, gemm_ratio, export_ratio, cpu_ratio)
    ):
        disposition = "keep experimental"
    else:
        disposition = "drop/deprioritize"
    return {
        "winner": winner,
        "full_uniform_winner": baseline_winner,
        "backend_winner_changed": backend_changed,
        "cpu_cutoff_shift_ratio_vs_full_uniform": cpu_ratio,
        "direct_hip_pack_ratio_vs_full_uniform": pack_ratio,
        "direct_hip_gemm_ratio_vs_full_uniform": gemm_ratio,
        "direct_hip_export_ratio_vs_full_uniform": export_ratio,
        "blockers": blockers,
        "disposition": disposition,
    }


def build_report(inputs: list[Path]) -> dict[str, Any]:
    loaded = [
        capture
        for path, from_directory in _iter_paths(inputs)
        if (capture := _load_capture(path, from_directory=from_directory)) is not None
    ]
    grouped: dict[tuple[str, int, int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for capture in loaded:
        grouped[_group_key(capture)].append(_row(capture))

    groups: list[dict[str, Any]] = []
    disposition_counts: dict[str, int] = defaultdict(int)
    distribution_counts: dict[str, int] = defaultdict(int)
    for key in sorted(grouped):
        semantics, modulus, m, n, k, distribution = key
        rows = sorted(grouped[key], key=lambda row: (str(row.get("backend")), str(row.get("selected_kernel"))))
        baseline_rows = grouped.get(_baseline_key(key), [])
        classification = _classification(rows, baseline_rows, distribution)
        disposition_counts[classification["disposition"]] += 1
        distribution_counts[_distribution_role(distribution)] += 1
        groups.append(
            {
                "semantics": semantics,
                "finite_modulus": modulus,
                "modulus_role": _modulus_role(semantics, modulus),
                "shape": {"m": m, "n": n, "k": k},
                "distribution": distribution,
                "distribution_role": _distribution_role(distribution),
                "rows": rows,
                **classification,
            }
        )

    return {
        "schema": "rns8_finite_distribution_report_v1",
        "capture_count": len(loaded),
        "group_count": len(groups),
        "distribution_counts": dict(sorted(distribution_counts.items())),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "groups": groups,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finite-distribution-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Finite Distribution Report",
        "",
        f"- captures: `{report['capture_count']}`",
        f"- groups: `{report['group_count']}`",
        f"- dispositions: `{report['disposition_counts']}`",
        "",
        "| Semantics | Modulus | Shape | Distribution | Winner | Full Uniform Winner | CPU Ratio | Direct Pack Ratio | Direct GEMM Ratio | Direct Export Ratio | Disposition |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for group in report["groups"]:
        winner = group.get("winner") or {}
        baseline = group.get("full_uniform_winner") or {}
        shape = group["shape"]
        lines.append(
            "| {semantics} | {modulus} | {shape} | {distribution} | {winner} | {baseline} | {cpu} | {pack} | {gemm} | {export} | {disposition} |".format(
                semantics=group["semantics"],
                modulus=group["finite_modulus"],
                shape=f"{shape['m']}x{shape['n']}x{shape['k']}",
                distribution=group["distribution_role"],
                winner=winner.get("backend", "n/a"),
                baseline=baseline.get("backend", "n/a"),
                cpu=_format_ratio(group.get("cpu_cutoff_shift_ratio_vs_full_uniform")),
                pack=_format_ratio(group.get("direct_hip_pack_ratio_vs_full_uniform")),
                gemm=_format_ratio(group.get("direct_hip_gemm_ratio_vs_full_uniform")),
                export=_format_ratio(group.get("direct_hip_export_ratio_vs_full_uniform")),
                disposition=group["disposition"],
            )
        )
    (out_dir / "finite-distribution-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_ratio(value: Any) -> str:
    return f"{float(value):.3f}x" if isinstance(value, (int, float)) else "n/a"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("temp") / "finite-distribution-report")
    parser.add_argument("--json", action="store_true", help="also print the report JSON to stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.captures)
    write_outputs(report, args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
