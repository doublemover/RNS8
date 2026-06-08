#!/usr/bin/env python3
"""Run and review reproducible rns8-bench capture sweeps."""

from __future__ import annotations

from benchmark_sweep_lib.capture_metadata import (
    backend_id,
    capture_contract_key,
    capture_execution_mode,
    capture_pack_mode,
    capture_prepack_reuse_operands,
    capture_prepack_reuse_strategy,
    median_phase,
    normalized_target_id,
    selected_kernel,
)
from benchmark_sweep_lib.cli import main, parse_args
from benchmark_sweep_lib.commands import (
    backend_allowed_for,
    command_for,
    default_backends_for,
    default_sweep_command_entries,
    scenario_args_for_item,
    scenario_backends_for_item,
    scenario_catalog,
    selected_scenario_items,
    scenario_sweep_command_entries,
    sweep_command_entries,
    sweep_commands,
)
from benchmark_sweep_lib.config import (
    ADAPTIVE_WORKLOAD_CASES,
    BOUNDED_BACKENDS,
    DEFAULT_EXACT_WIDE_LIMB_COUNT,
    EXACT_WIDE_LIMB_VARIANTS,
    PHASES,
    PROMOTABLE_RELEASE_SHAPES,
    RELEASE_MIN_REPEATS,
    RELEASE_MIN_WARMUPS,
    SCENARIO_DATA_DIR,
    WRAP64_ROCWMMA_CANDIDATE_BACKEND,
    ScenarioItem,
    SweepCase,
    SweepCommand,
)
from benchmark_sweep_lib.execution import (
    annotate_scenario_metadata,
    autotune_cache_path,
    cli_backend,
    execute_sweep_entries,
    existing_capture_valid,
    parse_backend_bench,
    run_command,
    validate_paths,
)
from benchmark_sweep_lib.parsing import parse_case
from benchmark_sweep_lib.reports import write_scenario_manifest
from benchmark_sweep_lib.review import (
    attach_cache_write_status,
    cache_entry_from_capture,
    phase_ratios,
    promotion_blockers,
    required_baselines,
    reviewed_release_status_for_target,
    review_captures,
    write_promoted_cache_entries,
)
from benchmark_sweep_lib.scenarios import load_scenario_data_catalog, load_scenario_data_family


if __name__ == "__main__":
    raise SystemExit(main())
