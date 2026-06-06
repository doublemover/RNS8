#!/usr/bin/env python3
"""Self-test export selector report grouping and blockers."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import export_selector_report
from test_benchmark_schema import add_helper_lane_fields, as_exact_wide_capture, expect_valid
from test_starfoundry_reports import starfoundry_capture


def write_capture(path: Path, capture: dict) -> None:
    path.write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")


def reviewable_fixed_limb_capture() -> dict:
    capture = add_helper_lane_fields(as_exact_wide_capture(expect_valid("v4_bounded_i64_ck.json")))
    selected_kernel = "ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2"
    capture["selected_kernel"] = selected_kernel
    capture["backend_selected"] = "ck"
    capture["backend_requested"] = "ck"
    capture["target_id"] = "gfx1100"
    capture["correctness"] = "ok"
    selected_prefix = capture.get("selected_prefix", capture["prefix"])
    capture["exact_output_contract"] = {
        "requested_final_output": "exact_wide_limb_host",
        "limb_count": 4,
        "status_policy": "structurally_elided",
        "output_domain_after_measured_repeats": "exact_wide_limb_host",
        "final_checksum_export_after_repeats": False,
    }
    capture["export_variant"] = {
        "name": "exact-wide-fixed-limb-export",
        "source": "reviewable_exact_wide_fixed_limb_selector",
        "selector_source": "rns8_internal_export_plan",
        "selector_key": (
            "semantics=exact_wide_signed;backend=ck;target_id=gfx1100;prefix=9;"
            "limb_count=4;signedness=signed;output_layout=fixed_u64_limbs;"
            "status_policy=none;d2h_policy=host_ld_padded;"
            f"final_output_mode=final_host_output;selected_kernel={selected_kernel}"
        ),
        "selector_policy": "semantic_prefix_limb_layout_status_d2h_backend_target",
        "semantic_contract": "exact_wide_signed",
        "backend": "ck",
        "target_id": "gfx1100",
        "prefix_contract": "prefix=9;min_selected=9;max_selected=9;groups=1",
        "signedness": "signed",
        "output_layout": "fixed_u64_limbs",
        "limb_count": 4,
        "status_policy": "required",
        "selector_status_policy": "none",
        "d2h_policy": "host_ld_padded",
        "final_output_mode": "final_host_output",
        "cache_visibility": "exact_shape_selector_metadata_only",
        "stale_entry_reason": "selector_key_mismatch_rejects_semantic_prefix_limb_layout_status_d2h_backend_target",
        "status_elision_reason": "exact_wide_requested_limb_count_covers_range_status",
        "requires_tile_metadata": False,
        "all_zero_tiled_output": False,
        "selected_kernel": selected_kernel,
        "constants_placement": "backend_default",
        "promotion_eligible": True,
        "promotion_blocker": None,
    }
    capture["reconstruction_variant"] = {
        "name": "default_garner",
        "family": "garner_fixed_prefix",
        "prefix_count": selected_prefix,
        "kernel_identity": selected_kernel,
        "controller": "benchmark_metadata_only",
        "promotion_eligible": True,
        "promotion_blocker": None,
    }
    return capture


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        good = starfoundry_capture()
        good_path = tmp / "good.json"
        write_capture(good_path, good)
        report = export_selector_report.build_report([good_path])
        assert report["schema"] == "rns8_export_selector_report_v1"
        row = report["groups"][0]["rows"][0]
        assert row["selector_key"]
        assert row["selected_kernel"] in row["selector_key"]
        assert row["promotion_eligible"] is True
        assert row["selector_promotion_eligible"] is True
        assert "missing_selector_key" not in row["promotion_blockers"]

        stale = copy.deepcopy(good)
        stale["export_variant"]["selector_key"] = None
        stale_path = tmp / "stale.json"
        write_capture(stale_path, stale)
        stale_report = export_selector_report.build_report([stale_path])
        stale_row = stale_report["groups"][0]["rows"][0]
        assert "missing_selector_key" in stale_row["promotion_blockers"]

        reviewable = reviewable_fixed_limb_capture()
        reviewable_path = tmp / "reviewable-fixed-limb.json"
        write_capture(reviewable_path, reviewable)
        reviewable_report = export_selector_report.build_report([reviewable_path])
        reviewable_row = reviewable_report["groups"][0]["rows"][0]
        assert reviewable_row["export_variant"] == "exact-wide-fixed-limb-export"
        assert reviewable_row["promotion_eligible"] is True
        assert reviewable_row["selector_promotion_eligible"] is True
        assert reviewable_row["promotion_blockers"] == []

        blocked = copy.deepcopy(reviewable)
        blocked["export_variant"]["name"] = "compact-d2h-export-candidate"
        blocked["export_variant"]["promotion_eligible"] = False
        blocked["export_variant"]["promotion_blocker"] = "experimental_export_variant"
        blocked_path = tmp / "blocked-compact.json"
        write_capture(blocked_path, blocked)
        blocked_report = export_selector_report.build_report([blocked_path])
        blocked_row = blocked_report["groups"][0]["rows"][0]
        assert blocked_row["promotion_eligible"] is False
        assert "experimental_export_variant" in blocked_row["promotion_blockers"]

    print("export selector report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
