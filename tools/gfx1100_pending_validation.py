#!/usr/bin/env python3
"""Run the Windows gfx1100 pending-performance validation control pass."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pending_validation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "temp" / "gfx1100-pending-validation-20260606"

BENCH_FOR = {
    "hip-direct": REPO_ROOT / "build" / "windows-msvc-hip-release" / "rns8-bench.exe",
    "hipblaslt": REPO_ROOT / "build" / "windows-msvc-hipblaslt-release" / "rns8-bench.exe",
    "ck": REPO_ROOT / "build" / "windows-msvc-ck-release" / "rns8-bench.exe",
    "rocwmma": REPO_ROOT / "build" / "windows-msvc-rocwmma-release" / "rns8-bench.exe",
}

BUILD_PRESETS = (
    "windows-release",
    "windows-hipblaslt-release",
    "windows-ck-release",
    "windows-rocwmma-release",
)

SCENARIOS = (
    "exact-wide-export",
    "export-bound-limb-variants",
    "resident-lifetime-arena",
    "k-block-tile-variants",
)

CONFIG = pending_validation.PendingValidationConfig(
    target="gfx1100",
    target_ids=frozenset({"gfx1100"}),
    policy="windows_rdna3_local_evidence_only_no_cache_or_readme_claim_update",
    plan_schema="rns8_gfx1100_pending_validation_plan_v1",
    summary_schema="rns8_gfx1100_pending_validation_summary_v1",
    command_plan_title="gfx1100 Pending Validation Command Plan",
    summary_title="gfx1100 Pending Validation Summary",
    summary_scope_lines=(
        "Scope: Windows RX 7900 XTX / gfx1100 local evidence only.",
        "Cache, README, and durable performance claims are intentionally unchanged by this driver.",
    ),
    default_out_dir=DEFAULT_OUT_DIR,
    bench_for=BENCH_FOR,
    build_presets=BUILD_PRESETS,
    scenarios=SCENARIOS,
    primary_bench_backend="hip-direct",
    build_command_prefix=("tools/windows_dev.py", "cmake", "--build", "--preset"),
    build_command_uses_python=True,
    isa_reports=(
        pending_validation.IsaReportSpec(
            name="direct-hip",
            backend="direct-hip",
            build_tree=REPO_ROOT / "build" / "windows-msvc-hip-release",
            target="gfx1100",
        ),
        pending_validation.IsaReportSpec(
            name="ck",
            backend="ck",
            build_tree=REPO_ROOT / "build" / "windows-msvc-ck-release",
            target="gfx1100",
        ),
        pending_validation.IsaReportSpec(
            name="rocwmma",
            backend="rocwmma",
            build_tree=REPO_ROOT / "build" / "windows-msvc-rocwmma-release",
            target="gfx1100",
        ),
    ),
)

PlannedCommand = pending_validation.PlannedCommand


def sweep_command(
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
    return pending_validation.sweep_command(
        CONFIG,
        scenario,
        out_dir,
        warmups=warmups,
        repeats=repeats,
        seed=seed,
        skip_existing=skip_existing,
        max_new_captures=max_new_captures,
        capture_timeout_seconds=capture_timeout_seconds,
    )


def build_command(preset: str) -> list[str]:
    return pending_validation.build_command(CONFIG, preset)


def command_plan(args: argparse.Namespace) -> list[PlannedCommand]:
    return pending_validation.command_plan(CONFIG, args)


def write_command_plan(commands: list[PlannedCommand], out_dir: Path) -> dict[str, str]:
    return pending_validation.write_command_plan(CONFIG, commands, out_dir)


def run_command(item: PlannedCommand) -> dict[str, Any]:
    return pending_validation.run_command(CONFIG, item)


def discover_captures(out_dir: Path) -> list[Path]:
    return pending_validation.discover_captures(out_dir)


def discover_review_reports(out_dir: Path) -> list[Path]:
    return pending_validation.discover_review_reports(out_dir)


def load_review_index(review_reports: list[Path]) -> dict[str, dict[str, Any]]:
    return pending_validation.load_review_index(review_reports, repo_root=REPO_ROOT)


def run_post_reports(captures: list[Path], out_dir: Path, shape_cache: Path | None) -> list[dict[str, Any]]:
    return pending_validation.run_post_reports(CONFIG, captures, out_dir, shape_cache)


def write_summary(
    out_dir: Path,
    command_results: list[dict[str, Any]],
    captures: list[Path],
    report_results: list[dict[str, Any]],
) -> dict[str, str]:
    return pending_validation.write_summary(CONFIG, out_dir, command_results, captures, report_results)


def previous_command_results(out_dir: Path) -> list[dict[str, Any]]:
    return pending_validation.previous_command_results(out_dir)


def main() -> int:
    return pending_validation.run_cli(CONFIG, __doc__)


if __name__ == "__main__":
    raise SystemExit(main())
