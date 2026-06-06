#!/usr/bin/env python3
"""Run the Windows gfx1100 pending-performance validation control pass."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture
import result_compare


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "temp" / "gfx1100-pending-validation-20260606"

BENCH_FOR = {
    "hip-direct": REPO_ROOT / "build" / "windows-msvc-hip-release" / "rns8-bench.exe",
    "hipblaslt": REPO_ROOT / "build" / "windows-msvc-hipblaslt-release" / "rns8-bench.exe",
    "ck": REPO_ROOT / "build" / "windows-msvc-ck-release" / "rns8-bench.exe",
    "rocwmma": REPO_ROOT / "build" / "windows-msvc-rocwmma-release" / "rns8-bench.exe",
}

BUILD_PRESETS = [
    "windows-release",
    "windows-hipblaslt-release",
    "windows-ck-release",
    "windows-rocwmma-release",
]

SCENARIOS = [
    "exact-wide-export",
    "export-bound-limb-variants",
    "resident-lifetime-arena",
    "k-block-tile-variants",
]


@dataclass(frozen=True)
class PlannedCommand:
    name: str
    command: list[str]
    log_path: Path


def _path_text(path: Path) -> str:
    return str(path)


def sweep_command(
    scenario: str,
    out_dir: Path,
    *,
    warmups: int,
    repeats: int,
    seed: int,
    max_new_captures: int | None,
    capture_timeout_seconds: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "tools/benchmark_sweep.py",
        "--scenario",
        scenario,
        "--review-mode",
        "release",
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--skip-existing",
        "--out-root",
        _path_text(out_dir / "captures" / scenario),
    ]
    for backend, bench in BENCH_FOR.items():
        command.extend(["--bench-for", f"{backend}={bench}"])
    command.extend(["--bench", _path_text(BENCH_FOR["hip-direct"])])
    if max_new_captures is not None:
        command.extend(["--max-new-captures", str(max_new_captures)])
    if capture_timeout_seconds is not None:
        command.extend(["--capture-timeout-seconds", str(capture_timeout_seconds)])
    return command


def build_command(preset: str) -> list[str]:
    return [sys.executable, "tools/windows_dev.py", "cmake", "--build", "--preset", preset]


def command_plan(args: argparse.Namespace) -> list[PlannedCommand]:
    commands: list[PlannedCommand] = []
    if not args.skip_build:
        for preset in BUILD_PRESETS:
            commands.append(
                PlannedCommand(
                    name=f"build_{preset}",
                    command=build_command(preset),
                    log_path=args.out_dir / "logs" / f"build-{preset}.log",
                )
            )
    for scenario in SCENARIOS:
        commands.append(
            PlannedCommand(
                name=f"sweep_{scenario}",
                command=sweep_command(
                    scenario,
                    args.out_dir,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    seed=args.seed,
                    max_new_captures=args.max_new_captures,
                    capture_timeout_seconds=args.capture_timeout_seconds,
                ),
                log_path=args.out_dir / "logs" / f"sweep-{scenario}.log",
            )
        )
    return commands


def write_command_plan(commands: list[PlannedCommand], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "rns8_gfx1100_pending_validation_plan_v1",
        "target": "gfx1100",
        "policy": "windows_rdna3_local_evidence_only_no_cache_or_readme_claim_update",
        "commands": [
            {"name": item.name, "command": item.command, "log_path": str(item.log_path)}
            for item in commands
        ],
    }
    json_path = out_dir / "command-plan.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "command-plan.md"
    lines = [
        "# gfx1100 Pending Validation Command Plan",
        "",
        "| Step | Command | Log |",
        "|---|---|---|",
    ]
    for item in commands:
        lines.append(f"| `{item.name}` | `{' '.join(item.command)}` | `{item.log_path}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def run_command(item: PlannedCommand) -> dict[str, Any]:
    item.log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        item.command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    item.log_path.write_text(result.stdout, encoding="utf-8")
    return {
        "name": item.name,
        "command": item.command,
        "returncode": result.returncode,
        "log_path": str(item.log_path),
        "passed": result.returncode == 0,
    }


def discover_captures(out_dir: Path) -> list[Path]:
    captures: list[Path] = []
    capture_root = out_dir / "captures"
    if not capture_root.exists():
        return captures
    for path in sorted(capture_root.glob("*/scenarios/**/*.json")):
        captures.append(path)
    return captures


def _valid_capture(path: Path) -> dict[str, Any] | None:
    try:
        capture = load_capture(path)
        validate_capture(capture, path)
        return capture
    except BenchmarkSchemaError:
        return None


def _is_reference_backend(capture: dict[str, Any]) -> bool:
    return capture.get("backend_selected") in {"cpu-reference", "cpu", "wrap64-byte-limb"}


def _capture_kind(capture: dict[str, Any]) -> str:
    if isinstance(capture.get("export_variant"), dict):
        return "export_selector"
    if isinstance(capture.get("workspace_arena"), dict):
        return "workspace_arena"
    if isinstance(capture.get("tile_shape_variant"), dict) or isinstance(capture.get("k_block_policy"), str):
        return "tile_k_block"
    return "baseline"


def _median_us(capture: dict[str, Any], phase: str = "end_to_end") -> float | None:
    timing = capture.get("timing_summary_us")
    if not isinstance(timing, dict):
        return None
    value = timing.get(phase)
    if isinstance(value, dict) and isinstance(value.get("median"), (int, float)):
        return float(value["median"])
    return None


def _gpu_events_complete(capture: dict[str, Any]) -> bool:
    events = capture.get("gpu_events")
    if isinstance(events, dict):
        if events.get("requested") and events.get("complete") is False:
            return False
    timings = capture.get("gpu_event_timings_us")
    return isinstance(timings, dict) and bool(timings)


def _decision_for_capture(path: Path) -> dict[str, Any]:
    try:
        capture = load_capture(path)
        validate_capture(capture, path)
        schema_status = "pass"
        schema_error = None
    except BenchmarkSchemaError as exc:
        capture = {}
        schema_status = "fail"
        schema_error = str(exc)
    blockers: list[str] = []
    if schema_status != "pass":
        blockers.append("schema_failed")
    comparison = capture.get("comparison_baseline")
    if not isinstance(comparison, dict):
        comparison = {}
    if comparison.get("status") not in {"reviewed_release", "reviewed_release_valid"}:
        blockers.append("release_review_not_promoted")
    if capture.get("correctness") not in {None, "ok"}:
        blockers.append("correctness_not_ok")
    if capture and not _is_reference_backend(capture) and not _gpu_events_complete(capture):
        blockers.append("gpu_events_missing_or_incomplete")
    target_variant = capture.get("target_variant") if isinstance(capture.get("target_variant"), dict) else {}
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    if not _is_reference_backend(capture) and target_variant.get("target_id") not in {None, "gfx1100"} and device.get("gcn_arch") != "gfx1100":
        blockers.append("not_gfx1100")
    disposition = "promote locally" if not blockers else "keep experimental"
    if "correctness_not_ok" in blockers or "schema_failed" in blockers:
        disposition = "drop/deprioritize"
    return {
        "capture": str(path),
        "kind": _capture_kind(capture),
        "semantics": capture.get("semantics"),
        "backend": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "median_end_to_end_us": _median_us(capture),
        "schema_status": schema_status,
        "schema_error": schema_error,
        "gpu_events_complete": _gpu_events_complete(capture) if capture else False,
        "disposition": disposition,
        "blockers": blockers or ["same_target_release_review_required_before_claim_update"],
    }


def _run_report(name: str, command: list[str], out_dir: Path) -> dict[str, Any]:
    log_path = out_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(result.stdout, encoding="utf-8")
    return {"name": name, "command": command, "returncode": result.returncode, "passed": result.returncode == 0, "log_path": str(log_path)}


def _result_compare_precheck(baseline_path: Path, candidate_path: Path) -> tuple[bool, str]:
    try:
        baseline = result_compare.load_result(baseline_path)
        candidate = result_compare.load_result(candidate_path)
        report = result_compare.compare(baseline, candidate, baseline_path, candidate_path)
    except SystemExit as exc:
        return False, str(exc)
    if not report["matching_contract"]:
        return False, "contract_metadata_differs"
    if not report["gpu_compatible"]:
        return False, "gpu_compatibility_metadata_differs"
    return True, "comparable"


def run_post_reports(captures: list[Path], out_dir: Path, shape_cache: Path | None) -> list[dict[str, Any]]:
    if not captures:
        return []
    reports: list[dict[str, Any]] = []
    reports.append(_run_report("benchmark_schema", [sys.executable, "tools/benchmark_schema.py", *map(str, captures)], out_dir))
    gpu_captures = [
        path
        for path in captures
        if (capture := _valid_capture(path)) is not None and not _is_reference_backend(capture)
    ]
    if gpu_captures:
        reports.append(
            _run_report(
                "gpu_event_report",
                [sys.executable, "tools/gpu_event_report.py", "--require-events", *map(str, gpu_captures)],
                out_dir,
            )
        )
    exact_export = [path for path in captures if "exact-wide" in str(path).lower() or "export" in str(path).lower()]
    if exact_export:
        reports.append(
            _run_report(
                "export_selector_report",
                [sys.executable, "tools/export_selector_report.py", "--out-dir", str(out_dir / "export-selector"), *map(str, exact_export)],
                out_dir,
            )
        )
    workspace = [path for path in captures if "resident-lifetime-arena" in str(path).lower()]
    if workspace:
        reports.append(
            _run_report(
                "resident_workspace_report",
                [sys.executable, "tools/resident_workspace_report.py", "--out-dir", str(out_dir / "workspace-arena"), *map(str, workspace)],
                out_dir,
            )
        )
    tile = [path for path in captures if "k-block-tile-variants" in str(path).lower()]
    if tile:
        reports.append(
            _run_report(
                "tile_shape_report",
                [sys.executable, "tools/tile_shape_report.py", "--out-dir", str(out_dir / "tile-k-block"), *map(str, tile)],
                out_dir,
            )
        )
    baseline_by_key: dict[tuple[Any, ...], Path] = {}
    for path in captures:
        capture = _valid_capture(path)
        if capture is None:
            continue
        export = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
        reconstruction = capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
        key = (
            capture.get("semantics"),
            capture.get("m"),
            capture.get("n"),
            capture.get("k"),
            export.get("name"),
            export.get("output_layout"),
            export.get("selector_status_policy"),
            export.get("d2h_policy"),
            export.get("final_output_mode"),
            export.get("limb_count"),
            reconstruction.get("name"),
        )
        if capture.get("backend_selected") in {"direct-hip", "hip-direct", "hip"}:
            baseline_by_key[key] = path
    for path in captures:
        capture = _valid_capture(path)
        if capture is None:
            continue
        export = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
        reconstruction = capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
        key = (
            capture.get("semantics"),
            capture.get("m"),
            capture.get("n"),
            capture.get("k"),
            export.get("name"),
            export.get("output_layout"),
            export.get("selector_status_policy"),
            export.get("d2h_policy"),
            export.get("final_output_mode"),
            export.get("limb_count"),
            reconstruction.get("name"),
        )
        baseline = baseline_by_key.get(key)
        if baseline is not None and baseline != path:
            comparable, reason = _result_compare_precheck(baseline, path)
            if not comparable:
                reports.append(
                    {
                        "name": f"result_compare_{path.stem}",
                        "command": [sys.executable, "tools/result_compare.py", str(baseline), str(path), "--json"],
                        "returncode": None,
                        "passed": True,
                        "log_path": None,
                        "skipped_reason": reason,
                    }
                )
                continue
            reports.append(
                _run_report(
                    f"result_compare_{path.stem}",
                    [sys.executable, "tools/result_compare.py", str(baseline), str(path), "--json"],
                    out_dir,
                )
            )
    if shape_cache is not None and shape_cache.exists():
        reports.append(
            _run_report(
                "shape_family_shadow_report",
                [
                    sys.executable,
                    "tools/shape_family_shadow_report.py",
                    "--cache",
                    str(shape_cache),
                    "--out-dir",
                    str(out_dir / "shape-family"),
                ],
                out_dir,
            )
        )
    else:
        reports.append(
            {
                "name": "shape_family_shadow_report",
                "command": [],
                "returncode": None,
                "passed": True,
                "log_path": None,
                "skipped_reason": "reviewed autotune cache not provided",
            }
        )
    isa_dir = out_dir / "isa"
    for backend, build_tree in [
        ("direct-hip", REPO_ROOT / "build" / "windows-msvc-hip-release"),
        ("ck", REPO_ROOT / "build" / "windows-msvc-ck-release"),
        ("rocwmma", REPO_ROOT / "build" / "windows-msvc-rocwmma-release"),
    ]:
        if build_tree.exists():
            reports.append(
                _run_report(
                    f"gpu_isa_report_{backend}",
                    [
                        sys.executable,
                        "tools/gpu_isa_report.py",
                        "--build-tree",
                        str(build_tree),
                        "--backend",
                        backend,
                        "--target",
                        "gfx1100",
                        "--out-dir",
                        str(isa_dir),
                    ],
                    out_dir,
                )
            )
    return reports


def write_summary(
    out_dir: Path,
    command_results: list[dict[str, Any]],
    captures: list[Path],
    report_results: list[dict[str, Any]],
) -> dict[str, str]:
    rows = [_decision_for_capture(path) for path in captures]
    payload = {
        "schema": "rns8_gfx1100_pending_validation_summary_v1",
        "target": "gfx1100",
        "policy": "windows_rdna3_local_evidence_only_no_cache_or_readme_claim_update",
        "command_results": command_results,
        "report_results": report_results,
        "capture_count": len(captures),
        "decision_rows": rows,
    }
    json_path = out_dir / "validation-summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "validation-summary.md"
    lines = [
        "# gfx1100 Pending Validation Summary",
        "",
        "- Scope: Windows RX 7900 XTX / gfx1100 local evidence only.",
        "- Cache, README, and durable performance claims are intentionally unchanged by this driver.",
        "",
        "| Kind | Semantics | Shape | Backend | Median us | Events | Disposition | Blockers |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        shape = row["shape"]
        lines.append(
            "| `{kind}` | `{semantics}` | `{shape}` | `{backend}` | {median} | `{events}` | `{disposition}` | `{blockers}` |".format(
                kind=row["kind"],
                semantics=row.get("semantics"),
                shape=f"{shape.get('m')}x{shape.get('n')}x{shape.get('k')}",
                backend=row.get("backend"),
                median=row.get("median_end_to_end_us") if row.get("median_end_to_end_us") is not None else "",
                events=row["gpu_events_complete"],
                disposition=row["disposition"],
                blockers=", ".join(row["blockers"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def previous_command_results(out_dir: Path) -> list[dict[str, Any]]:
    summary_path = out_dir / "validation-summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    results = payload.get("command_results")
    return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-captures", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--max-new-captures", type=int)
    parser.add_argument("--capture-timeout-seconds", type=int)
    parser.add_argument("--shape-family-cache", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = command_plan(args)
    write_command_plan(commands, args.out_dir)
    if args.list_only:
        print(args.out_dir / "command-plan.md")
        return 0
    command_results: list[dict[str, Any]] = []
    if not args.skip_captures:
        for item in commands:
            command_results.append(run_command(item))
    else:
        command_results = previous_command_results(args.out_dir)
    captures = discover_captures(args.out_dir)
    report_results = run_post_reports(captures, args.out_dir, args.shape_family_cache)
    outputs = write_summary(args.out_dir, command_results, captures, report_results)
    print(outputs["markdown"])
    return 0 if all(item.get("passed", False) for item in command_results if item.get("returncode") is not None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
