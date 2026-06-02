#!/usr/bin/env python3
"""Run and review reproducible rns8-bench capture sweeps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


BOUNDED_BACKENDS = ["cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma"]
WRAP64_BACKENDS = ["wrap64-byte-limb", "hip-direct"]
PHASES = ["pack", "rns_gemm", "crt_export", "end_to_end"]


def capture_contract_key(capture: dict[str, Any]) -> str:
    tile_bounds = capture.get("tile_bounds_u64")
    tile_hash = tile_bounds.get("hash_u64") if isinstance(tile_bounds, dict) else None
    parts = [
        f"semantics={capture.get('semantics')}",
        f"bound_kind={capture.get('bound_kind')}",
        f"bound_mode={capture.get('bound_mode')}",
        f"bound={capture.get('bound')}",
        f"m={capture.get('m')}",
        f"n={capture.get('n')}",
        f"k={capture.get('k')}",
        f"prefix={capture.get('prefix')}",
        f"layout={capture.get('layout')}",
        f"tile_m={capture.get('tile_m')}",
        f"tile_n={capture.get('tile_n')}",
        f"k_block={capture.get('k_block_size')}",
        f"seed={capture.get('seed')}",
        f"input_distribution={capture.get('input_distribution')}",
        f"tile_hash={tile_hash}",
    ]
    return ";".join(str(part) for part in parts)


def median_phase(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    item = summary.get(phase)
    if not isinstance(item, dict):
        return None
    value = item.get("median")
    return float(value) if isinstance(value, (int, float)) else None


def backend_id(capture: dict[str, Any]) -> str:
    return str(capture.get("backend_selected"))


def selected_kernel(capture: dict[str, Any]) -> str:
    kernel = capture.get("selected_kernel")
    return str(kernel) if kernel is not None else ""


def parse_backend_bench(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--bench-for must use BACKEND=PATH, got {value!r}")
        backend, path = value.split("=", 1)
        if not backend or not path:
            raise SystemExit(f"--bench-for must use BACKEND=PATH, got {value!r}")
        result[backend] = Path(path)
    return result


def autotune_cache_path() -> Path:
    override = os.environ.get("RNS8_AUTOTUNE_CACHE_PATH")
    if override:
        return Path(override)
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("USERPROFILE")
        if root:
            suffix = ["rns8-gemm", "autotune.json"]
            if root == os.environ.get("USERPROFILE"):
                suffix = ["AppData", "Local", *suffix]
            return Path(root).joinpath(*suffix)
    root = os.environ.get("XDG_CACHE_HOME") or os.environ.get("HOME")
    if root:
        return Path(root) / ".cache" / "rns8-gemm" / "autotune.json"
    return Path("rns8-gemm") / "autotune.json"


def run_command(command: list[str], output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        failure = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return False
    output.write_text(completed.stdout, encoding="utf-8")
    return True


def validate_paths(paths: list[Path]) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = load_capture(path)
            validate_capture(data, path)
        except BenchmarkSchemaError as exc:
            raise SystemExit(str(exc)) from exc
        data["_path"] = str(path)
        captures.append(data)
    return captures


def review_captures(captures: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        grouped[capture_contract_key(capture)].append(capture)

    groups = []
    promotable_entries = []
    for key, items in sorted(grouped.items()):
        by_backend = {backend_id(item): item for item in items}
        semantics = items[0].get("semantics")
        required = (
            ["cpu-reference", "hip-direct", "hip-vector-alu-int64"]
            if semantics in {"bounded_i64", "bounded_u64"}
            else ["wrap64-byte-limb", "hip-direct"]
        )
        missing = [backend for backend in required if backend not in by_backend]
        phase_medians = {
            f"{backend_id(item)}/{selected_kernel(item)}": {
                phase: median_phase(item, phase)
                for phase in PHASES
            }
            for item in items
        }
        gpu_targets = {
            backend: capture.get("device", {}).get("gcn_arch")
            for backend, capture in by_backend.items()
            if backend not in {"cpu-reference", "wrap64-byte-limb"}
        }
        gpu_target_values = {value for value in gpu_targets.values() if value}
        gpu_target_compatible = len(gpu_target_values) <= 1
        candidates = []
        for item in items:
            backend = backend_id(item)
            metadata = item.get("backend_metadata") if isinstance(item.get("backend_metadata"), dict) else {}
            accelerator = metadata.get("accelerator_backend") is True
            end_to_end = median_phase(item, "end_to_end")
            direct = median_phase(by_backend["hip-direct"], "end_to_end") if "hip-direct" in by_backend else None
            vector = (
                median_phase(by_backend["hip-vector-alu-int64"], "end_to_end")
                if "hip-vector-alu-int64" in by_backend
                else None
            )
            promotable = (
                not missing
                and gpu_target_compatible
                and accelerator
                and end_to_end is not None
                and direct is not None
                and end_to_end < direct
                and (vector is None or end_to_end < vector)
            )
            candidate = {
                "backend": backend,
                "selected_kernel": selected_kernel(item),
                "capture": item.get("_path"),
                "accelerator_backend": accelerator,
                "median_end_to_end_us": end_to_end,
                "speedup_vs_direct_hip": (direct / end_to_end) if direct and end_to_end else None,
                "speedup_vs_vector_alu": (vector / end_to_end) if vector and end_to_end else None,
                "promotable": promotable,
                "promotion_reason": "beats_required_same_contract_gpu_baselines" if promotable else "missing_baseline_or_not_faster",
            }
            candidates.append(candidate)
            if promotable:
                promotable_entries.append(
                    {
                        "source_capture": item.get("_path"),
                        "autotune_key": metadata.get("autotune_key"),
                        "selected_backend": backend,
                        "selected_kernel": selected_kernel(item),
                        "median_end_to_end_us": end_to_end,
                    }
                )
        groups.append(
            {
                "contract_key": key,
                "semantics": semantics,
                "capture_count": len(items),
                "missing_required_baselines": missing,
                "gpu_targets": gpu_targets,
                "gpu_target_compatible": gpu_target_compatible,
                "phase_medians_us": phase_medians,
                "candidates": candidates,
            }
        )

    return {
        "schema_version": 1,
        "reviewed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "group_count": len(groups),
        "groups": groups,
        "promotable_autotune_entries": promotable_entries,
    }


def cache_entry_from_capture(capture: dict[str, Any], validation_status: str) -> dict[str, Any]:
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    medians = capture.get("timing_summary_us") if isinstance(capture.get("timing_summary_us"), dict) else {}
    schedule = capture.get("schedule_metadata") if isinstance(capture.get("schedule_metadata"), dict) else {}
    tile_bounds = capture.get("tile_bounds_u64") if isinstance(capture.get("tile_bounds_u64"), dict) else {}
    prefix_schedule_hash = (
        f"tile_rows={schedule.get('tile_rows')};tile_cols={schedule.get('tile_cols')};"
        f"groups={schedule.get('prefix_group_count')};"
        f"adaptive_prefix={int(bool(schedule.get('adaptive_prefix_active')))};"
        f"adaptive_skip={int(bool(schedule.get('adaptive_skip_active')))};"
        f"tile_bound_hash={tile_bounds.get('hash_u64', 0)}"
    )
    hip_toolchain = capture.get("hip_toolchain") if isinstance(capture.get("hip_toolchain"), dict) else {}
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    version = metadata.get("accelerator_version") or hip_toolchain.get("hip_sdk_or_rocm_version") or "unknown"

    def median(phase: str) -> float:
        item = medians.get(phase) if isinstance(medians, dict) else None
        if isinstance(item, dict) and isinstance(item.get("median"), (int, float)):
            return float(item["median"])
        return 0.0

    return {
        "key": metadata.get("autotune_key"),
        "selected_backend": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "target_id": device.get("gcn_arch") or "cpu",
        "hip_sdk_or_library_version": version,
        "semantic_contract": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "layout": capture.get("layout"),
        "prefix_schedule_hash": prefix_schedule_hash,
        "k_block_size": capture.get("k_block_size"),
        "tile_m": capture.get("tile_m"),
        "tile_n": capture.get("tile_n"),
        "epilogue": capture.get("epilogue_type"),
        "kernel_family": capture.get("selected_kernel"),
        "workspace_bytes": metadata.get("workspace_required_bytes", 0),
        "measured_medians_us": {
            "pack": median("pack"),
            "rns_gemm": median("rns_gemm"),
            "crt_export": median("crt_export"),
            "end_to_end": median("end_to_end"),
        },
        "performance_validated": True,
        "validation_status": validation_status,
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_promoted_cache_entries(report: dict[str, Any], captures: list[dict[str, Any]], path: Path) -> int:
    promotable = report.get("promotable_autotune_entries")
    if not isinstance(promotable, list) or not promotable:
        return 0
    by_path = {str(capture.get("_path")): capture for capture in captures}
    entries = []
    for item in promotable:
        if not isinstance(item, dict):
            continue
        capture = by_path.get(str(item.get("source_capture")))
        if not capture:
            continue
        entry = cache_entry_from_capture(capture, "reviewed_same_contract_fastest_windows_gfx1100")
        if entry.get("key"):
            entries.append(entry)
    if not entries:
        return 0

    existing: dict[str, Any] = {"schema_version": 1, "entries": []}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {"schema_version": 1, "entries": []}
    existing_entries = existing.get("entries")
    if not isinstance(existing_entries, list):
        existing_entries = []
    by_key = {
        entry.get("key"): entry
        for entry in existing_entries
        if isinstance(entry, dict) and entry.get("key")
    }
    for entry in entries:
        by_key[entry["key"]] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "entries": list(by_key.values())}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(entries)


def sweep_commands(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    backend_benches = parse_backend_bench(args.bench_for)
    commands: list[tuple[str, list[str], Path]] = []
    bounded_shapes = args.shapes or ["64", "128"]
    semantics_list = ["bounded-i64", "bounded-u64"]
    backends = args.backends or BOUNDED_BACKENDS
    for semantics in semantics_list:
        for shape in bounded_shapes:
            for backend in backends:
                bench = backend_benches.get(backend, args.bench)
                if bench is None:
                    raise SystemExit(f"no benchmark executable configured for backend {backend}")
                name = f"{semantics}-{shape}-{backend}.json".replace("/", "_")
                command = [
                    str(bench),
                    "--backend",
                    backend,
                    "--semantics",
                    semantics,
                    "--m",
                    shape,
                    "--n",
                    shape,
                    "--k",
                    shape,
                    "--warmups",
                    str(args.warmups),
                    "--repeats",
                    str(args.repeats),
                    "--seed",
                    str(args.seed),
                ]
                commands.append((name, command, args.out_root / name))
    if args.include_wrap64:
        for backend in WRAP64_BACKENDS:
            bench = backend_benches.get(backend, args.bench)
            if bench is None:
                raise SystemExit(f"no benchmark executable configured for backend {backend}")
            name = f"wrap-u64-64-{backend}.json"
            command = [
                str(bench),
                "--backend",
                backend,
                "--semantics",
                "wrap-u64",
                "--m",
                "64",
                "--n",
                "64",
                "--k",
                "64",
                "--warmups",
                str(args.warmups),
                "--repeats",
                str(args.repeats),
                "--seed",
                str(args.seed),
            ]
            commands.append((name, command, args.out_root / name))
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, help="path to rns8-bench executable")
    parser.add_argument(
        "--bench-for",
        action="append",
        default=[],
        metavar="BACKEND=PATH",
        help="override rns8-bench executable for one backend, useful for opt-in accelerator build dirs",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("temp") / "benchmark-sweeps" / "windows-gfx1100",
        help="ignored output directory for raw captures and review report",
    )
    parser.add_argument("--capture", type=Path, action="append", default=[], help="existing capture to review")
    parser.add_argument("--review-only", action="store_true", help="only review --capture files")
    parser.add_argument("--backend", dest="backends", action="append", help="bounded backend to sweep; repeatable")
    parser.add_argument("--shape", dest="shapes", action="append", help="square bounded shape to sweep; repeatable")
    parser.add_argument("--include-wrap64", action="store_true", help="include wrap64 CPU/direct-HIP captures")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--write-autotune-cache",
        action="store_true",
        help="write performance_validated cache entries only for promotable reviewed captures",
    )
    parser.add_argument("--autotune-cache", type=Path, help="override autotune cache output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    capture_paths = list(args.capture)
    if not args.review_only:
        if args.bench is None:
            raise SystemExit("--bench is required unless --review-only is used")
        for _name, command, output in sweep_commands(args):
            if run_command(command, output):
                capture_paths.append(output)

    captures = validate_paths(capture_paths)
    report = review_captures(captures)
    report_path = args.out_root / "review_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    promoted = 0
    cache_path = args.autotune_cache or autotune_cache_path()
    if args.write_autotune_cache:
        promoted = write_promoted_cache_entries(report, captures, cache_path)
    print(
        json.dumps(
            {
                "review_report": str(report_path),
                "captures": len(captures),
                "promoted_cache_entries": promoted,
                "autotune_cache": str(cache_path) if args.write_autotune_cache else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
