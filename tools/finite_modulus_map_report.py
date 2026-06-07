#!/usr/bin/env python3
"""Summarize finite-u8 modulus-family coverage and backend reducer choices."""

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


DEFAULT_OUT_DIR = Path("temp") / "finite-modulus-map-report"
FINITE_SEMANTICS = {"finite_ring_u8", "finite_field_u8"}
DEFAULT_SHAPES = [128, 512, 1024, 2048]
DEFAULT_RING_MODULI = [127, 241, 243, 251, 253, 255, 256]
DEFAULT_FIELD_MODULI = [127, 241, 251]
DEFAULT_BACKENDS = ["cpu-reference", "hip-direct", "hipblaslt", "ck", "rocwmma"]
SIDECAR_NAMES = {
    "review_report.json",
    "scenario_manifest.json",
    "command-plan.json",
    "validation-summary.json",
    "finite-modulus-map-report.json",
}


def _parse_csv_ints(value: str) -> list[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part, 10))
    return items


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


def _modulus_class(modulus: int) -> str:
    if modulus > 0 and modulus & (modulus - 1) == 0:
        return "power_of_two"
    if _is_prime(modulus):
        return "prime"
    return "composite"


def _modulus_role(semantics: str, modulus: int) -> str:
    cls = _modulus_class(modulus)
    if semantics == "finite_field_u8":
        return "hot_prime" if modulus == 251 else "generic_non_hot_prime"
    if modulus in {251, 255, 256}:
        return f"hot_{cls}"
    return f"generic_non_hot_{cls}"


def _target_id(capture: dict[str, Any]) -> str | None:
    for source in (capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict):
            for key in ("target_id", "target_arch", "gcn_arch"):
                value = source.get(key)
                if isinstance(value, str) and value and value not in {"none", "unknown"}:
                    return value
    return None


def _gpu_events_available(capture: dict[str, Any]) -> bool:
    timing = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    return (
        timing.get("gpu_event_timing") is True
        and timing.get("gpu_event_timing_status") == "available"
        and bool(timing.get("gpu_event_timing_source"))
        and isinstance(timing.get("gpu_event_phase_order"), list)
    )


def _release_ready(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
        and capture["warmups"] >= RELEASE_MIN_WARMUPS
        and isinstance(capture.get("repeats"), int)
        and capture["repeats"] >= RELEASE_MIN_REPEATS
    )


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


def _load_finite_capture(path: Path, *, from_directory: bool) -> dict[str, Any] | None:
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
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    backend = backend_id(capture)
    end_to_end = median_phase(capture, "end_to_end")
    return {
        "backend": backend,
        "capture_path": capture.get("_path"),
        "selected_kernel": selected_kernel(capture),
        "epilogue_mode": metadata.get("epilogue_mode"),
        "accelerator_backend": metadata.get("accelerator_backend") is True,
        "target_id": _target_id(capture) if backend != "cpu-reference" else "cpu",
        "release_ready": _release_ready(capture),
        "gpu_events_required": backend not in {"cpu-reference"},
        "gpu_events_available": True if backend == "cpu-reference" else _gpu_events_available(capture),
        "median_end_to_end_us": end_to_end,
        "median_pack_us": median_phase(capture, "pack"),
        "median_gemm_us": median_phase(capture, "rns_gemm"),
        "median_export_us": median_phase(capture, "crt_export"),
    }


def _expected_group_keys(
    shapes: list[int],
    ring_moduli: list[int],
    field_moduli: list[int],
) -> list[tuple[str, int, int, int, int]]:
    keys = []
    for shape in shapes:
        for modulus in ring_moduli:
            keys.append(("finite_ring_u8", modulus, shape, shape, shape))
        for modulus in field_moduli:
            keys.append(("finite_field_u8", modulus, shape, shape, shape))
    return keys


def _group_key(capture: dict[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(capture.get("semantics")),
        int(capture.get("finite_modulus") or 0),
        int(capture.get("m") or 0),
        int(capture.get("n") or 0),
        int(capture.get("k") or 0),
    )


def _winner(rows: list[dict[str, Any]], direct_median: float | None, cpu_median: float | None) -> dict[str, Any] | None:
    timed = [row for row in rows if isinstance(row.get("median_end_to_end_us"), (int, float))]
    if not timed:
        return None
    best = min(timed, key=lambda row: float(row["median_end_to_end_us"]))
    median = float(best["median_end_to_end_us"])
    return {
        "backend": best.get("backend"),
        "selected_kernel": best.get("selected_kernel"),
        "epilogue_mode": best.get("epilogue_mode"),
        "median_end_to_end_us": median,
        "speedup_vs_direct_hip": (direct_median / median) if direct_median and median > 0 else None,
        "speedup_vs_cpu": (cpu_median / median) if cpu_median and median > 0 else None,
    }


def build_report(
    captures: list[Path],
    *,
    expected_shapes: list[int] | None = None,
    expected_ring_moduli: list[int] | None = None,
    expected_field_moduli: list[int] | None = None,
    expected_backends: list[str] | None = None,
) -> dict[str, Any]:
    expected_shapes = DEFAULT_SHAPES if expected_shapes is None else expected_shapes
    expected_ring_moduli = DEFAULT_RING_MODULI if expected_ring_moduli is None else expected_ring_moduli
    expected_field_moduli = DEFAULT_FIELD_MODULI if expected_field_moduli is None else expected_field_moduli
    expected_backends = DEFAULT_BACKENDS if expected_backends is None else expected_backends

    loaded = [
        capture
        for path, from_directory in _iter_paths(captures)
        if (capture := _load_finite_capture(path, from_directory=from_directory)) is not None
    ]
    grouped: dict[tuple[str, int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for capture in loaded:
        grouped[_group_key(capture)].append(_row(capture))

    expected_keys = _expected_group_keys(expected_shapes, expected_ring_moduli, expected_field_moduli)
    missing_expected_groups = []
    groups = []
    for key in expected_keys:
        semantics, modulus, m, n, k = key
        rows = sorted(grouped.get(key, []), key=lambda row: (row["backend"], row["selected_kernel"] or ""))
        by_backend = {str(row["backend"]): row for row in rows}
        missing_backends = [backend for backend in expected_backends if backend not in by_backend]
        missing_gpu_events = [
            backend
            for backend, row in by_backend.items()
            if row.get("gpu_events_required") and not row.get("gpu_events_available")
        ]
        release_not_ready = [backend for backend, row in by_backend.items() if not row.get("release_ready")]
        cpu_median = by_backend.get("cpu-reference", {}).get("median_end_to_end_us")
        direct_median = by_backend.get("hip-direct", {}).get("median_end_to_end_us")
        if not rows:
            missing_expected_groups.append(
                {
                    "semantics": semantics,
                    "finite_modulus": modulus,
                    "shape": {"m": m, "n": n, "k": k},
                }
            )
        groups.append(
            {
                "key": {
                    "semantics": semantics,
                    "finite_modulus": modulus,
                    "modulus_class": _modulus_class(modulus),
                    "modulus_role": _modulus_role(semantics, modulus),
                    "shape": {"m": m, "n": n, "k": k},
                },
                "present_backends": sorted(by_backend),
                "missing_backends": missing_backends,
                "missing_required_baselines": [
                    backend for backend in ("cpu-reference", "hip-direct") if backend not in by_backend
                ],
                "missing_gpu_events": sorted(missing_gpu_events),
                "release_not_ready": sorted(release_not_ready),
                "map_group_ready": bool(rows) and not missing_backends and not missing_gpu_events and not release_not_ready,
                "winner": _winner(rows, direct_median, cpu_median),
                "rows": rows,
                "promotion_eligible": False,
                "promotion_blockers": ["non_promoting_modulus_map"],
            }
        )

    ready_groups = sum(1 for group in groups if group["map_group_ready"])
    return {
        "schema": "rns8_finite_modulus_map_report_v1",
        "policy": "finite_modulus_family_map_non_promoting_exact_modulus_shape_target",
        "capture_count": len(loaded),
        "expected": {
            "shapes": expected_shapes,
            "ring_moduli": expected_ring_moduli,
            "field_moduli": expected_field_moduli,
            "backends": expected_backends,
        },
        "summary": {
            "group_count": len(groups),
            "ready_groups": ready_groups,
            "missing_expected_groups": len(missing_expected_groups),
            "groups_with_missing_backends": sum(1 for group in groups if group["missing_backends"]),
            "groups_with_missing_gpu_events": sum(1 for group in groups if group["missing_gpu_events"]),
            "groups_not_release_ready": sum(1 for group in groups if group["release_not_ready"]),
            "map_complete": ready_groups == len(groups),
            "promotion_eligible": False,
        },
        "missing_expected_groups": missing_expected_groups,
        "groups": groups,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "finite-modulus-map-report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "finite-modulus-map-report.md"
    lines = [
        "# Finite-u8 Modulus Map Report",
        "",
        f"Policy: `{report['policy']}`",
        "",
        "| Semantic | Modulus | Class | Shape | Ready | Winner | Median us | Blockers |",
        "|---|---:|---|---:|---|---|---:|---|",
    ]
    for group in report["groups"]:
        key = group["key"]
        shape = key["shape"]
        winner = group.get("winner") or {}
        blockers = []
        if group["missing_backends"]:
            blockers.append("missing backends: " + ",".join(group["missing_backends"]))
        if group["missing_gpu_events"]:
            blockers.append("missing events: " + ",".join(group["missing_gpu_events"]))
        if group["release_not_ready"]:
            blockers.append("not release: " + ",".join(group["release_not_ready"]))
        lines.append(
            "| {semantics} | {modulus} | {cls} | {shape} | {ready} | {winner} | {median} | {blockers} |".format(
                semantics=key["semantics"],
                modulus=key["finite_modulus"],
                cls=key["modulus_class"],
                shape=f"{shape['m']}",
                ready="yes" if group["map_group_ready"] else "no",
                winner=winner.get("backend", ""),
                median=winner.get("median_end_to_end_us", ""),
                blockers="<br>".join(blockers),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--expected-shapes", default=",".join(str(item) for item in DEFAULT_SHAPES))
    parser.add_argument("--expected-ring-moduli", default=",".join(str(item) for item in DEFAULT_RING_MODULI))
    parser.add_argument("--expected-field-moduli", default=",".join(str(item) for item in DEFAULT_FIELD_MODULI))
    parser.add_argument("--expected-backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--require-complete-map", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.captures,
        expected_shapes=_parse_csv_ints(args.expected_shapes),
        expected_ring_moduli=_parse_csv_ints(args.expected_ring_moduli),
        expected_field_moduli=_parse_csv_ints(args.expected_field_moduli),
        expected_backends=[item.strip() for item in args.expected_backends.split(",") if item.strip()],
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    return 0 if (not args.require_complete_map or report["summary"]["map_complete"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
