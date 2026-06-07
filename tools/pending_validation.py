#!/usr/bin/env python3
"""Reusable pending-performance validation driver core."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture
import result_compare


REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_REVIEWED_RELEASE_STATUSES = {
    "reviewed_release_same_contract_baseline",
    "reviewed_release",
    "reviewed_release_valid",
}


@dataclass(frozen=True)
class PlannedCommand:
    name: str
    command: list[str]
    log_path: Path


@dataclass(frozen=True)
class IsaReportSpec:
    name: str
    backend: str
    build_tree: Path
    target: str


@dataclass(frozen=True)
class PendingValidationConfig:
    target: str
    target_ids: frozenset[str]
    policy: str
    plan_schema: str
    summary_schema: str
    command_plan_title: str
    summary_title: str
    summary_scope_lines: tuple[str, ...]
    default_out_dir: Path
    bench_for: Mapping[str, Path]
    build_presets: tuple[str, ...]
    scenarios: tuple[str, ...]
    primary_bench_backend: str
    review_mode: str = "release"
    build_command_prefix: tuple[str, ...] = ("cmake", "--build", "--preset")
    build_command_uses_python: bool = False
    isa_reports: tuple[IsaReportSpec, ...] = ()
    repo_root: Path = REPO_ROOT


def _path_text(path: Path) -> str:
    return str(path)


def sweep_command(
    config: PendingValidationConfig,
    scenario: str,
    out_dir: Path,
    *,
    warmups: int,
    repeats: int,
    seed: int,
    skip_existing: bool,
    max_new_captures: int | None,
    capture_timeout_seconds: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "tools/benchmark_sweep.py",
        "--scenario",
        scenario,
        "--review-mode",
        config.review_mode,
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--out-root",
        _path_text(out_dir / "captures" / scenario),
    ]
    if skip_existing:
        command.append("--skip-existing")
    for backend, bench in config.bench_for.items():
        command.extend(["--bench-for", f"{backend}={bench}"])
    command.extend(["--bench", _path_text(config.bench_for[config.primary_bench_backend])])
    if max_new_captures is not None:
        command.extend(["--max-new-captures", str(max_new_captures)])
    if capture_timeout_seconds is not None:
        command.extend(["--capture-timeout-seconds", str(capture_timeout_seconds)])
    return command


def build_command(config: PendingValidationConfig, preset: str) -> list[str]:
    command = [*config.build_command_prefix, preset]
    if config.build_command_uses_python:
        return [sys.executable, *command]
    return command


def command_plan(config: PendingValidationConfig, args: argparse.Namespace) -> list[PlannedCommand]:
    commands: list[PlannedCommand] = []
    refresh_scenarios = set(args.refresh_scenario or [])
    if not args.skip_build:
        for preset in config.build_presets:
            commands.append(
                PlannedCommand(
                    name=f"build_{preset}",
                    command=build_command(config, preset),
                    log_path=args.out_dir / "logs" / f"build-{preset}.log",
                )
            )
    for scenario in config.scenarios:
        commands.append(
            PlannedCommand(
                name=f"sweep_{scenario}",
                command=sweep_command(
                    config,
                    scenario,
                    args.out_dir,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    seed=args.seed,
                    skip_existing=scenario not in refresh_scenarios,
                    max_new_captures=args.max_new_captures,
                    capture_timeout_seconds=args.capture_timeout_seconds,
                ),
                log_path=args.out_dir / "logs" / f"sweep-{scenario}.log",
            )
        )
    return commands


def write_command_plan(
    config: PendingValidationConfig,
    commands: list[PlannedCommand],
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": config.plan_schema,
        "target": config.target,
        "policy": config.policy,
        "commands": [
            {"name": item.name, "command": item.command, "log_path": str(item.log_path)}
            for item in commands
        ],
    }
    json_path = out_dir / "command-plan.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "command-plan.md"
    lines = [
        f"# {config.command_plan_title}",
        "",
        "| Step | Command | Log |",
        "|---|---|---|",
    ]
    for item in commands:
        lines.append(f"| `{item.name}` | `{' '.join(item.command)}` | `{item.log_path}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def run_command(config: PendingValidationConfig, item: PlannedCommand) -> dict[str, Any]:
    item.log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        item.command,
        cwd=config.repo_root,
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
        if path.name.endswith(".failed.json"):
            continue
        captures.append(path)
    return captures


def discover_review_reports(out_dir: Path) -> list[Path]:
    capture_root = out_dir / "captures"
    if not capture_root.exists():
        return []
    return sorted(capture_root.glob("*/review_report.json"))


def _path_key(repo_root: Path, path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate.absolute()
    key = str(resolved).replace("\\", "/")
    return key.lower() if sys.platform.startswith("win") else key


def _review_capture_keys(repo_root: Path, capture_path: str, report_path: Path) -> set[str]:
    text = capture_path.replace("\\", "/")
    raw = Path(text)
    paths = [raw]
    if not raw.is_absolute():
        paths.append(repo_root / raw)
        paths.append(report_path.parent / raw)
    return {_path_key(repo_root, path) for path in paths}


def load_review_index(
    review_reports: list[Path],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, dict[str, Any]]:
    reviewed: dict[str, dict[str, Any]] = {}
    for report_path in review_reports:
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for group_index, group in enumerate(data.get("groups", [])):
            if not isinstance(group, dict):
                continue
            fastest = group.get("fastest_promotable")
            fastest_backend = fastest.get("backend") if isinstance(fastest, dict) else None
            fastest_kernel = fastest.get("selected_kernel") if isinstance(fastest, dict) else None
            group_review = {
                "review_report": str(report_path),
                "review_report_group_index": group_index,
                "review_mode": group.get("review_mode") or data.get("review_mode"),
                "release_review_satisfied": group.get("release_review_satisfied"),
                "missing_required_baselines": group.get("missing_required_baselines") or [],
                "duplicate_backends": group.get("duplicate_backends") or [],
                "required_baselines": group.get("required_baselines") or [],
                "contract_key": group.get("contract_key"),
                "fastest_promotable_backend": fastest_backend,
                "fastest_promotable_kernel": fastest_kernel,
            }
            for candidate in group.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                capture_path = candidate.get("capture")
                if not isinstance(capture_path, str) or not capture_path:
                    continue
                entry = {
                    **group_review,
                    "candidate": candidate,
                    "review_report_promotable": candidate.get("promotable") is True,
                    "review_cache_write_status": candidate.get("cache_write_status"),
                    "review_promotion_reason": candidate.get("promotion_reason"),
                    "review_promotion_blockers": candidate.get("promotion_blockers") or [],
                    "review_speedup_vs_direct_hip": candidate.get("speedup_vs_direct_hip"),
                    "review_speedup_vs_vector_alu": candidate.get("speedup_vs_vector_alu"),
                    "review_median_end_to_end_us": candidate.get("median_end_to_end_us"),
                }
                for key in _review_capture_keys(repo_root, capture_path, report_path):
                    reviewed[key] = entry
    return reviewed


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
    if isinstance(events, dict) and events.get("requested") and events.get("complete") is False:
        return False
    timings = capture.get("gpu_event_timings_us")
    return isinstance(timings, dict) and bool(timings)


def _target_matches(config: PendingValidationConfig, capture: dict[str, Any]) -> bool:
    if _is_reference_backend(capture) or not config.target_ids:
        return True
    target_variant = capture.get("target_variant") if isinstance(capture.get("target_variant"), dict) else {}
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    candidates = {
        target_variant.get("target_id"),
        target_variant.get("target_arch"),
        target_variant.get("arch"),
        device.get("gcn_arch"),
        device.get("target_arch"),
    }
    return any(isinstance(item, str) and item in config.target_ids for item in candidates)


def _claim_blockers(capture: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for object_name in ("export_variant", "reconstruction_variant"):
        item = capture.get(object_name)
        if isinstance(item, dict) and item.get("promotion_eligible") is False:
            reason = item.get("promotion_blocker") or item.get("stale_entry_reason") or "non_promoting_variant"
            blockers.append(f"{object_name}:{reason}")
    tile_variant = capture.get("tile_shape_variant")
    if isinstance(tile_variant, dict) and tile_variant.get("k_block_policy") not in {None, "auto"}:
        blockers.append("tile_shape_variant:non_default_k_block_policy_requires_same_target_counter_review")
    arena = capture.get("workspace_arena")
    if isinstance(arena, dict) and arena.get("enabled"):
        if arena.get("measured_repeat_allocation_free") is not True:
            blockers.append("workspace_arena:measured_repeat_allocation_not_free")
        repeat_delta = arena.get("measured_repeat_allocation_delta")
        if isinstance(repeat_delta, dict) and (
            int(repeat_delta.get("allocate_calls") or 0) != 0
            or int(repeat_delta.get("free_calls") or 0) != 0
            or int(repeat_delta.get("allocated_bytes") or 0) != 0
        ):
            blockers.append("workspace_arena:measured_repeat_allocation_delta_nonzero")
    return sorted(set(blockers))


def _review_blockers(config: PendingValidationConfig, review_entry: dict[str, Any] | None) -> list[str]:
    if review_entry is None:
        return [f"{config.review_mode}_review_not_found"]
    blockers: list[str] = []
    if review_entry.get("review_mode") != config.review_mode:
        blockers.append(f"review_mode_not_{config.review_mode}")
    if config.review_mode == "release" and review_entry.get("release_review_satisfied") is not True:
        blockers.append("release_review_not_satisfied")
    missing = [
        str(item)
        for item in review_entry.get("missing_required_baselines", [])
        if isinstance(item, str) and item
    ]
    if missing:
        blockers.extend(f"missing_required_baseline:{item}" for item in missing)
    duplicates = [
        str(item)
        for item in review_entry.get("duplicate_backends", [])
        if isinstance(item, str) and item
    ]
    if duplicates:
        blockers.extend(f"duplicate_backend_record:{item}" for item in duplicates)
    candidate_blockers = [
        str(item)
        for item in review_entry.get("review_promotion_blockers", [])
        if isinstance(item, str) and item
    ]
    if candidate_blockers:
        blockers.extend(f"review_blocker:{item}" for item in candidate_blockers)
    if review_entry.get("review_report_promotable") is not True and not candidate_blockers:
        reason = review_entry.get("review_promotion_reason")
        blockers.append(f"review_report_not_promotable:{reason}" if reason else "review_report_not_promotable")
    return blockers


def _raw_capture_promoted(capture: dict[str, Any]) -> bool:
    comparison = capture.get("comparison_baseline")
    if not isinstance(comparison, dict):
        return False
    metadata = capture.get("backend_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return (
        comparison.get("status") in RAW_REVIEWED_RELEASE_STATUSES
        and (metadata.get("performance_validated") is True or comparison.get("speedup_claimed") is True)
    )


def decision_for_capture(
    config: PendingValidationConfig,
    path: Path,
    review_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    if capture and review_entry is None and not _raw_capture_promoted(capture):
        blockers.append(f"{config.review_mode}_review_not_found")
    elif review_entry is not None:
        blockers.extend(_review_blockers(config, review_entry))
    if capture.get("correctness") not in {None, "ok"}:
        blockers.append("correctness_not_ok")
    if capture and not _is_reference_backend(capture) and not _gpu_events_complete(capture):
        blockers.append("gpu_events_missing_or_incomplete")
    if capture and not _target_matches(config, capture):
        blockers.append(f"not_{config.target}")
    disposition = "promote locally" if not blockers else "keep experimental"
    if "correctness_not_ok" in blockers or "schema_failed" in blockers:
        disposition = "drop/deprioritize"
    claim_blockers = _claim_blockers(capture) if capture else []
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
        "blockers": sorted(set(blockers)),
        "review_report": review_entry.get("review_report") if review_entry else None,
        "review_report_promotable": review_entry.get("review_report_promotable") if review_entry else _raw_capture_promoted(capture),
        "review_cache_write_status": review_entry.get("review_cache_write_status") if review_entry else None,
        "review_promotion_reason": review_entry.get("review_promotion_reason") if review_entry else None,
        "review_promotion_blockers": review_entry.get("review_promotion_blockers") if review_entry else [],
        "review_release_review_satisfied": review_entry.get("release_review_satisfied") if review_entry else None,
        "review_missing_required_baselines": review_entry.get("missing_required_baselines") if review_entry else [],
        "review_speedup_vs_direct_hip": review_entry.get("review_speedup_vs_direct_hip") if review_entry else None,
        "review_speedup_vs_vector_alu": review_entry.get("review_speedup_vs_vector_alu") if review_entry else None,
        "fastest_promotable_backend": review_entry.get("fastest_promotable_backend") if review_entry else None,
        "fastest_promotable_kernel": review_entry.get("fastest_promotable_kernel") if review_entry else None,
        "claim_blockers": claim_blockers,
    }


def _run_report(config: PendingValidationConfig, name: str, command: list[str], out_dir: Path) -> dict[str, Any]:
    log_path = out_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        cwd=config.repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "log_path": str(log_path),
    }


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


def _direct_baseline_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    export = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    reconstruction = capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
    return (
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


def run_post_reports(
    config: PendingValidationConfig,
    captures: list[Path],
    out_dir: Path,
    shape_cache: Path | None,
) -> list[dict[str, Any]]:
    if not captures:
        return []
    reports: list[dict[str, Any]] = []
    reports.append(_run_report(config, "benchmark_schema", [sys.executable, "tools/benchmark_schema.py", *map(str, captures)], out_dir))
    gpu_captures = [
        path
        for path in captures
        if (capture := _valid_capture(path)) is not None and not _is_reference_backend(capture)
    ]
    if gpu_captures:
        reports.append(
            _run_report(
                config,
                "gpu_event_report",
                [sys.executable, "tools/gpu_event_report.py", "--require-events", *map(str, gpu_captures)],
                out_dir,
            )
        )
    exact_export = [path for path in captures if "exact-wide" in str(path).lower() or "export" in str(path).lower()]
    if exact_export:
        reports.append(
            _run_report(
                config,
                "export_selector_report",
                [sys.executable, "tools/export_selector_report.py", "--out-dir", str(out_dir / "export-selector"), *map(str, exact_export)],
                out_dir,
            )
        )
    workspace = [path for path in captures if "resident-lifetime-arena" in str(path).lower()]
    if workspace:
        reports.append(
            _run_report(
                config,
                "resident_workspace_report",
                [sys.executable, "tools/resident_workspace_report.py", "--out-dir", str(out_dir / "workspace-arena"), *map(str, workspace)],
                out_dir,
            )
        )
    tile = [path for path in captures if "k-block-tile-variants" in str(path).lower()]
    if tile:
        reports.append(
            _run_report(
                config,
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
        if capture.get("backend_selected") in {"direct-hip", "hip-direct", "hip"}:
            baseline_by_key[_direct_baseline_key(capture)] = path
    for path in captures:
        capture = _valid_capture(path)
        if capture is None:
            continue
        baseline = baseline_by_key.get(_direct_baseline_key(capture))
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
                    config,
                    f"result_compare_{path.stem}",
                    [sys.executable, "tools/result_compare.py", str(baseline), str(path), "--json"],
                    out_dir,
                )
            )
    if shape_cache is not None and shape_cache.exists():
        reports.append(
            _run_report(
                config,
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
    for spec in config.isa_reports:
        if spec.build_tree.exists():
            reports.append(
                _run_report(
                    config,
                    f"gpu_isa_report_{spec.name}",
                    [
                        sys.executable,
                        "tools/gpu_isa_report.py",
                        "--build-tree",
                        str(spec.build_tree),
                        "--backend",
                        spec.backend,
                        "--target",
                        spec.target,
                        "--out-dir",
                        str(out_dir / "isa"),
                    ],
                    out_dir,
                )
            )
    return reports


def write_summary(
    config: PendingValidationConfig,
    out_dir: Path,
    command_results: list[dict[str, Any]],
    captures: list[Path],
    report_results: list[dict[str, Any]],
) -> dict[str, str]:
    review_reports = discover_review_reports(out_dir)
    review_index = load_review_index(review_reports, repo_root=config.repo_root)
    rows = [
        decision_for_capture(config, path, review_index.get(_path_key(config.repo_root, path)))
        for path in captures
    ]
    payload = {
        "schema": config.summary_schema,
        "target": config.target,
        "target_ids": sorted(config.target_ids),
        "policy": config.policy,
        "command_results": command_results,
        "report_results": report_results,
        "review_report_count": len(review_reports),
        "review_reports": [str(path) for path in review_reports],
        "capture_count": len(captures),
        "decision_rows": rows,
    }
    json_path = out_dir / "validation-summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "validation-summary.md"
    lines = [
        f"# {config.summary_title}",
        "",
        *[f"- {line}" for line in config.summary_scope_lines],
        f"- Review reports consumed: `{len(review_reports)}`.",
        "",
        "| Kind | Semantics | Shape | Backend | Median us | Review | Speedup | Events | Disposition | Blockers | Claim blockers |",
        "|---|---|---|---|---:|---|---:|---|---|---|---|",
    ]
    for row in rows:
        shape = row["shape"]
        lines.append(
            "| `{kind}` | `{semantics}` | `{shape}` | `{backend}` | {median} | `{review}` | {speedup} | `{events}` | `{disposition}` | `{blockers}` | `{claim_blockers}` |".format(
                kind=row["kind"],
                semantics=row.get("semantics"),
                shape=f"{shape.get('m')}x{shape.get('n')}x{shape.get('k')}",
                backend=row.get("backend"),
                median=row.get("median_end_to_end_us") if row.get("median_end_to_end_us") is not None else "",
                review=row.get("review_promotion_reason") or row.get("review_cache_write_status") or row.get("review_report_promotable"),
                speedup=row.get("review_speedup_vs_direct_hip") if row.get("review_speedup_vs_direct_hip") is not None else "",
                events=row["gpu_events_complete"],
                disposition=row["disposition"],
                blockers=", ".join(row["blockers"]),
                claim_blockers=", ".join(row["claim_blockers"]),
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


def _result_item_passed(item: dict[str, Any]) -> bool:
    if item.get("returncode") is None:
        return item.get("passed") is True and bool(item.get("skipped_reason"))
    return item.get("passed") is True


def validation_run_passed(
    command_results: list[dict[str, Any]],
    report_results: list[dict[str, Any]],
) -> bool:
    return all(_result_item_passed(item) for item in command_results) and all(
        _result_item_passed(item) for item in report_results
    )


def build_arg_parser(config: PendingValidationConfig, description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-dir", type=Path, default=config.default_out_dir)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-captures", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--max-new-captures", type=int)
    parser.add_argument("--capture-timeout-seconds", type=int)
    parser.add_argument(
        "--refresh-scenario",
        action="append",
        choices=list(config.scenarios),
        default=[],
        help="Rerun all captures for this scenario instead of passing --skip-existing. May be repeated.",
    )
    parser.add_argument("--shape-family-cache", type=Path)
    return parser


def run_cli(config: PendingValidationConfig, description: str | None = None) -> int:
    parser = build_arg_parser(config, description)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = command_plan(config, args)
    write_command_plan(config, commands, args.out_dir)
    if args.list_only:
        print(args.out_dir / "command-plan.md")
        return 0
    command_results: list[dict[str, Any]] = []
    if not args.skip_captures:
        for item in commands:
            command_results.append(run_command(config, item))
    else:
        command_results = previous_command_results(args.out_dir)
    captures = discover_captures(args.out_dir)
    report_results = run_post_reports(config, captures, args.out_dir, args.shape_family_cache)
    outputs = write_summary(config, args.out_dir, command_results, captures, report_results)
    print(outputs["markdown"])
    return 0 if validation_run_passed(command_results, report_results) else 1
