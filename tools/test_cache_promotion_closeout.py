#!/usr/bin/env python3
"""Self-test cache-promotion closeout wrapper gates."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import cache_promotion_closeout
import install_autotune_cache
import promotion_ledger
from benchmark_schema import validate_capture
from test_benchmark_sweep_support import bounded_capture


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def candidate_capture() -> dict:
    capture = bounded_capture("ck", 700)
    capture["backend_metadata"]["performance_validated"] = False
    capture["comparison_baseline"]["status"] = "reviewed_release_same_contract_baseline"
    capture["comparison_baseline"]["speedup_vs_baseline_median_end_to_end"] = 1.4
    prefix = int(capture.get("selected_prefix") or capture.get("prefix") or 1)
    capture["avg_per_modulus_gemm_estimate_us"] = float(capture["avg_rns_gemm_us"]) / float(prefix)
    capture["target_variant"] = capture.get("device", {}).get("gcn_arch", "gfx1100")
    kernel = capture.get("selected_kernel",
                         "ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3")
    capture["export_variant"] = {"selected_kernel": kernel}
    capture["exact_output_contract"] = {"kernel_identity": kernel}
    capture["reconstruction_variant"] = {"kernel_identity": kernel}
    if "gpu_event_phase_order" not in capture.get("timing_metadata", {}):
        capture.setdefault("timing_metadata", {})["gpu_event_phase_order"] = []
    return capture


def cache_entry_from_capture(capture: dict, *, selector_review_only: bool = False, target_id: str | None = None) -> dict:
    metadata = capture["backend_metadata"]
    resolved_target_id = target_id or capture["device"]["gcn_arch"]
    entry = {
        "schema_version": 1,
        "key": metadata["autotune_key"],
        "selected_backend": capture["backend_selected"],
        "selected_kernel": capture["selected_kernel"],
        "target_id": resolved_target_id,
        "hip_sdk_or_library_version": "fixture",
        "semantic_contract": capture["semantics"],
        "layout": "row_major",
        "prefix_schedule_hash": "fixture-prefix",
        "epilogue": metadata["epilogue_mode"],
        "kernel_family": capture["selected_kernel"],
        "validation_status": install_autotune_cache.reviewed_release_status_for_target(resolved_target_id),
        "performance_validated": True,
        "shape": {"m": capture["m"], "n": capture["n"], "k": capture["k"]},
        "k_block_size": capture["k_block_size"],
        "tile_m": capture["tile_m"],
        "tile_n": capture["tile_n"],
        "workspace_bytes": metadata.get("workspace_required_bytes", 0),
        "measured_medians_us": {
            "pack": capture["timing_summary_us"]["pack"]["median"],
            "rns_gemm": capture["timing_summary_us"]["rns_gemm"]["median"],
            "crt_export": capture["timing_summary_us"]["crt_export"]["median"],
            "end_to_end": capture["timing_summary_us"]["end_to_end"]["median"],
        },
        "finite_modulus": 0,
        "updated_utc": "2026-06-07T00:00:00Z",
    }
    if target_id is not None:
        entry["key"] = entry["key"].replace(f"target_id={capture['device']['gcn_arch']}", f"target_id={target_id}")
    if selector_review_only:
        selector_key = (
            "semantics=bounded_i64;backend=ck;target_id=gfx1100;prefix=9;limb_count=0;"
            "signedness=signed;output_layout=scalar_i64;status_policy=range_checked_status_buffer;"
            "d2h_policy=host_ld_padded;final_output_mode=final_host_output;"
            "selected_kernel=ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3"
        )
        selector_hash = install_autotune_cache.selector_key_hash(selector_key)
        entry["export_variant"] = "compact-d2h-export-candidate"
        entry["reconstruction_variant"] = "default_garner"
        entry["export_selector_key"] = selector_key
        entry["export_selector_hash"] = selector_hash
        entry["cache_scope"] = "selector_review_only_non_default"
        entry["key"] = (
            f"{entry['key']};export_variant={entry['export_variant']};"
            f"reconstruction_variant=default_garner;export_selector_hash={selector_hash}"
        )
    return entry


def review_report(path: Path, capture_path: Path) -> Path:
    return write_json(
        path,
        {
            "schema_version": 3,
            "review_mode": "release",
            "groups": [
                {
                    "candidates": [
                        {
                            "backend": "ck",
                            "capture": str(capture_path),
                            "promotable": True,
                            "promotion_blockers": [],
                            "speedup_vs_direct_hip": 1.4,
                        }
                    ]
                }
            ],
        },
    )


def variance_report(path: Path, capture_path: Path, *, ready: bool = True) -> Path:
    return write_json(
        path,
        {
            "schema": "rns8_perf_variance_report_v1",
            "entries": [
                {
                    "capture": str(capture_path),
                    "capture_key": promotion_ledger.path_key(capture_path),
                    "promotion_ready": ready,
                    "required_speedup_margin": 1.03,
                    "observed_max_relative_noise": 0.02,
                    "blockers": [] if ready else ["high_within_capture_variance"],
                }
            ],
        },
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        capture = candidate_capture()
        capture_path = tmp / "capture.json"
        validate_capture(capture, capture_path)
        write_json(capture_path, capture)
        cache_path = write_json(tmp / "candidate-cache.json", {"schema_version": 1, "entries": [cache_entry_from_capture(capture)]})
        review = review_report(tmp / "review.json", capture_path)
        variance = variance_report(tmp / "variance.json", capture_path)
        destination = tmp / "installed.json"

        summary = cache_promotion_closeout.build_closeout(
            sources=[cache_path],
            destination=destination,
            captures=[capture_path],
            out_dir=tmp / "closeout",
            review_reports=[review],
            variance_reports=[variance],
            install=False,
        )
        assert summary["complete"] is True, json.dumps(summary["blockers"], indent=2)
        assert summary["dry_run"] is True
        assert not destination.exists()

        second_cache_path = write_json(
            tmp / "candidate-cache-copy.json",
            {"schema_version": 1, "entries": [cache_entry_from_capture(capture)]},
        )
        multi_source_summary = cache_promotion_closeout.build_closeout(
            sources=[cache_path, second_cache_path],
            destination=destination,
            captures=[capture_path],
            out_dir=tmp / "multi-source-closeout",
            review_reports=[review],
            variance_reports=[variance],
            install=False,
        )
        assert multi_source_summary["complete"] is True
        assert multi_source_summary["candidate_entry_count"] == 2

        blocked_variance = cache_promotion_closeout.build_closeout(
            sources=[cache_path],
            destination=destination,
            captures=[capture_path],
            out_dir=tmp / "blocked-variance",
            review_reports=[review],
            variance_reports=[variance_report(tmp / "variance-blocked.json", capture_path, ready=False)],
            install=False,
        )
        assert blocked_variance["complete"] is False
        assert any("variance" in blocker for blocker in blocked_variance["blockers"])

        selector_cache = write_json(
            tmp / "selector-cache.json",
            {"schema_version": 1, "entries": [cache_entry_from_capture(capture, selector_review_only=True)]},
        )
        selector_summary = cache_promotion_closeout.build_closeout(
            sources=[selector_cache],
            destination=destination,
            captures=[capture_path],
            out_dir=tmp / "selector",
            review_reports=[review],
            variance_reports=[variance],
            install=False,
        )
        assert selector_summary["complete"] is False
        assert "selector_review_only_cache_entry_not_runtime_installable" in selector_summary["install_error"]

        cdna_entry = cache_entry_from_capture(capture, target_id="gfx942")
        cdna_cache = write_json(tmp / "cdna-cache.json", {"schema_version": 1, "entries": [cdna_entry]})
        cdna_summary = cache_promotion_closeout.build_closeout(
            sources=[cdna_cache],
            destination=destination,
            captures=[capture_path],
            out_dir=tmp / "cdna",
            review_reports=[review],
            variance_reports=[variance],
            install=False,
        )
        assert cdna_summary["complete"] is False
        assert (
            "promotion_ledger_missing_target_validation_gate" in cdna_summary["install_error"]
            or "missing_promotion_ledger_entry" in cdna_summary["install_error"]
        )

    print("cache promotion closeout self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())