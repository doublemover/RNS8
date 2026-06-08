#!/usr/bin/env python3
"""Self-test rns8-inspect hard-cut CLI diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run_command(exe: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(exe), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def expect_exit(result: subprocess.CompletedProcess[str], code: int, label: str) -> None:
    if result.returncode != code:
        raise AssertionError(
            f"{label}: expected exit {code}, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def expect_text(haystack: str, needle: str, label: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"{label}: expected {needle!r} in:\n{haystack}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inspect_exe", type=Path, help="path to rns8-inspect executable")
    args = parser.parse_args()
    inspect_exe = args.inspect_exe

    invalid = run_command(inspect_exe, "--backend", "not-a-backend")
    expect_exit(invalid, 2, "invalid backend")
    expect_text(invalid.stderr, "invalid backend string: not-a-backend", "invalid backend")
    expect_text(invalid.stderr, "unknown names are not routed to auto", "invalid backend")

    hipblaslt = run_command(inspect_exe, "--backend", "hipblaslt")
    if hipblaslt.returncode == 0:
        expect_text(hipblaslt.stdout, "capability_status: implemented_baseline_backend", "hipblaslt")
        expect_text(
            hipblaslt.stdout,
            "selected_kernel:   hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2",
            "hipblaslt",
        )
        expect_text(hipblaslt.stdout, "exact_validated:   1", "hipblaslt")
        expect_text(hipblaslt.stdout, "perf_validated:    0", "hipblaslt")
    else:
        expect_exit(hipblaslt, 1, "hipblaslt")
        expect_text(hipblaslt.stderr, "unsupported backend", "hipblaslt")
        expect_text(hipblaslt.stderr, "requested accelerator is evidence-only", "hipblaslt")
        expect_text(hipblaslt.stderr, "real exact correctness backend", "hipblaslt")

    ck = run_command(inspect_exe, "--backend", "ck")
    if ck.returncode == 0:
        expect_text(ck.stdout, "capability_status: implemented_opt_in_ck_backend", "ck")
        expect_text(
            ck.stdout,
            "selected_kernel:   ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
            "ck",
        )
        expect_text(ck.stdout, "exact_validated:   1", "ck")
        expect_text(ck.stdout, "perf_validated:    0", "ck")
        expect_text(
            ck.stdout,
            "isa_evidence:      ck_cshuffle_int8_matrix_isa_gate_no_int32_global_store_no_divide",
            "ck",
        )
    else:
        expect_exit(ck, 1, "ck")
        expect_text(ck.stderr, "unsupported backend", "ck")
        expect_text(ck.stderr, "requested accelerator is evidence-only", "ck")
        expect_text(ck.stderr, "real exact correctness backend", "ck")

    rocwmma = run_command(inspect_exe, "--backend", "rocwmma")
    if rocwmma.returncode == 0:
        expect_text(rocwmma.stdout, "capability_status: implemented_opt_in_rocwmma_backend", "rocwmma")
        expect_text(
            rocwmma.stdout,
            "selected_kernel:   rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2",
            "rocwmma",
        )
        expect_text(rocwmma.stdout, "exact_validated:   1", "rocwmma")
        expect_text(rocwmma.stdout, "perf_validated:    0", "rocwmma")
        expect_text(
            rocwmma.stdout,
            "isa_evidence:      rocwmma_i8_matrix_isa_gate_no_divide",
            "rocwmma",
        )
    else:
        expect_exit(rocwmma, 1, "rocwmma")
        expect_text(rocwmma.stderr, "unsupported backend", "rocwmma")
        expect_text(rocwmma.stderr, "requested accelerator is evidence-only", "rocwmma")
        expect_text(rocwmma.stderr, "real exact correctness backend", "rocwmma")

    cpu = run_command(inspect_exe, "--backend", "cpu-reference", "--json")
    expect_exit(cpu, 0, "cpu-reference json")
    expect_text(cpu.stdout, '"backend": "cpu-reference"', "cpu-reference json")
    expect_text(cpu.stdout, '"hip_available": 0', "cpu-reference json")
    selector_shadow = run_command(inspect_exe, "--backend", "cpu-reference", "--json", "--selector-shadow")
    expect_exit(selector_shadow, 0, "selector shadow json")
    expect_text(selector_shadow.stdout, '"selector_shadow": {', "selector shadow json")
    expect_text(
        selector_shadow.stdout,
        '"conservative_family_boundary": "exact_cache_selection_unchanged_family_advisory_only"',
        "selector shadow json",
    )
    expect_text(selector_shadow.stdout, '"selected_family": "cpu_reference_family"', "selector shadow json")

    with tempfile.TemporaryDirectory() as temp_dir:
        env = os.environ.copy()
        env["RNS8_AUTOTUNE_CACHE_PATH"] = str(Path(temp_dir) / "autotune.json")
        autotune = run_command(
            inspect_exe,
            "--backend",
            "cpu-reference",
            "--json",
            "--autotune-key",
            "unit-test-missing-autotune-key",
            env=env,
        )
        expect_exit(autotune, 0, "autotune json")
        expect_text(autotune.stdout, '"autotune_cache": {', "autotune json")
        expect_text(autotune.stdout, '"exact_hit": false', "autotune json")
        expect_text(autotune.stdout, '"runtime_target_id": "cpu"', "autotune json")
        expect_text(autotune.stdout, "missing_cache_using_cpu_reference", "autotune json")

        reviewed_key = (
            "backend=cpu-reference;semantics=bounded_i64;m=4;n=5;k=6;prefix=9;"
            "tile_m=0;tile_n=0;groups=1;adaptive_prefix=0;adaptive_skip=0;"
            "kernel=cpu_reference_rns_gemm_v1;epilogue=cpu_reference_crt_export"
        )
        Path(env["RNS8_AUTOTUNE_CACHE_PATH"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "entries": [
                        {
                            "key": reviewed_key,
                            "selected_backend": "cpu-reference",
                            "selected_kernel": "cpu_reference_rns_gemm_v1",
                            "target_id": "cpu",
                            "hip_sdk_or_library_version": "",
                            "semantic_contract": "bounded_i64",
                            "finite_modulus": 0,
                            "shape": {"m": 4, "n": 5, "k": 6},
                            "layout": "row_major",
                            "prefix_schedule_hash": "unit-prefix",
                            "k_block_size": 6,
                            "tile_m": 0,
                            "tile_n": 0,
                            "epilogue": "cpu_reference_crt_export",
                            "kernel_family": "cpu_reference_rns_gemm_v1",
                            "workspace_bytes": 0,
                            "measured_medians_us": {
                                "pack": 1.0,
                                "rns_gemm": 2.0,
                                "crt_export": 3.0,
                                "end_to_end": 6.0,
                            },
                            "performance_validated": True,
                            "validation_status": "reviewed_release_same_contract_fastest_windows_gfx1100",
                            "schema_version": 1,
                            "updated_utc": "2026-06-03T00:00:00Z",
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        autotune_hit = run_command(
            inspect_exe,
            "--backend",
            "cpu-reference",
            "--json",
            "--autotune-key",
            reviewed_key,
            env=env,
        )
        expect_exit(autotune_hit, 0, "autotune exact hit json")
        expect_text(autotune_hit.stdout, '"exact_hit": true', "autotune exact hit json")
        expect_text(autotune_hit.stdout, '"plan_packing": {', "autotune exact hit json")
        expect_text(autotune_hit.stdout, '"backend": "cpu-reference"', "autotune exact hit json")
        expect_text(autotune_hit.stdout, '"semantics": "bounded_i64"', "autotune exact hit json")
        expect_text(
            autotune_hit.stdout,
            '"prepack_cache_scope": "host_resident_no_prepack_cache"',
            "autotune exact hit json",
        )
        expect_text(
            autotune_hit.stdout,
            '"output_domain": "rns_residue_current"',
            "autotune exact hit json",
        )
        expect_text(
            autotune_hit.stdout,
            '"next_op_flags": 3',
            "autotune exact hit json",
        )
        expect_text(autotune_hit.stdout, '"plan_lowering": {', "autotune exact hit json")
        expect_text(autotune_hit.stdout, '"operation": "MatMul"', "autotune exact hit json")
        expect_text(
            autotune_hit.stdout,
            '"desired_output": "final_export_or_rns_chain"',
            "autotune exact hit json",
        )
        expect_text(
            autotune_hit.stdout,
            '"schedule_strategy": "minimum_proven_uniform_prefix_schedule"',
            "autotune exact hit json",
        )
        expect_text(
            autotune_hit.stdout,
            '"lowering_path": "MatMul[rns_residue_current] -> RnsResidueCurrent; FinalExport only at requested boundary"',
            "autotune exact hit json",
        )
        expect_text(
            autotune_hit.stdout,
            '"production_prepack_cache_available": false',
            "autotune exact hit json",
        )

    print("rns8-inspect CLI self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
