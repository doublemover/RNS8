#!/usr/bin/env python3
"""Self-test for modulus-set autotune capture report."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import modulus_set_autotune_report
import modulus_set_search
from test_benchmark_schema import add_helper_lane_fields


FIXTURE = Path("tests") / "fixtures" / "benchmark_schema" / "v4_bounded_i64_adaptive_hip.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def set_median(capture: dict, value: float) -> dict:
    capture = copy.deepcopy(capture)
    int_value = int(value)
    capture["avg_end_to_end_us"] = float(int_value)
    capture["timing_summary_us"]["end_to_end"]["median"] = value
    capture["timing_summary_us"]["end_to_end"]["avg"] = float(int_value)
    capture["timing_summary_us"]["end_to_end"]["p95"] = float(int_value)
    capture["raw_timings_us"]["end_to_end"] = [int_value, int_value, int_value]
    return capture


def with_default_modulus_metadata(capture: dict) -> dict:
    capture = add_helper_lane_fields(copy.deepcopy(capture))
    capture["selected_prefix"] = capture["schedule_metadata"]["max_selected_prefix"]
    capture["requested_max_prefix"] = capture["prefix"]
    capture["contract_prefix_policy"] = "per_tile_minimum"
    capture["residue_planes_requested"] = capture["requested_max_prefix"]
    capture["residue_planes_selected"] = capture["selected_prefix"]
    capture["residue_planes_skipped"] = capture["requested_max_prefix"] - capture["selected_prefix"]
    capture["residue_plane_skip_fraction"] = capture["residue_planes_skipped"] / capture["requested_max_prefix"]
    capture["modulus_set"] = {
        "name": "default",
        "source": "rns8_default_modulus_ladder",
        "execution_ladder": "rns8_default_8bit_coprime_ladder",
        "experimental": False,
        "product_bits": 72,
        "prefix_count": capture["selected_prefix"],
        "pairwise_coprime_proof": "schema_declared_current_ladder_or_offline_search_report",
        "reducer_cost_hint": "backend_default",
        "cache_promotion_blocker": None,
    }
    capture["residue_count_policy"] = {
        "policy": capture["contract_prefix_policy"],
        "requested_prefix": capture["requested_max_prefix"],
        "selected_prefix": capture["selected_prefix"],
        "minimum_range_prefix": capture["schedule_metadata"]["min_required_prefix"],
        "redundant_residue_count": capture["selected_prefix"] - capture["schedule_metadata"]["min_required_prefix"],
        "autotune_scope": "current_exact_cache",
        "cache_promotion_blocker": None,
    }
    return capture


def as_experimental(capture: dict, name: str = "experimental:prefix5-byte-ladder-search") -> dict:
    capture = copy.deepcopy(capture)
    capture["modulus_set"] = {
        "name": name,
        "source": "benchmark_experimental_ladder",
        "execution_ladder": "rns8_default_8bit_coprime_ladder",
        "experimental": True,
        "runtime_selectable": False,
        "search_report_required": True,
        "product_bits": capture["modulus_set"]["product_bits"],
        "prefix_count": capture["selected_prefix"],
        "pairwise_coprime_proof": "schema_declared_current_ladder_or_offline_search_report",
        "reducer_cost_hint": "offline_search_required",
        "default_change_gate": "spec_cache_schema_proof_and_same_target_review_required",
        "cache_promotion_blocker": "experimental_modulus_set",
    }
    capture["residue_count_policy"]["autotune_scope"] = "evidence_only_non_promoting"
    capture["residue_count_policy"]["promotion_eligible"] = False
    capture["residue_count_policy"]["cache_promotion_blocker"] = "experimental_residue_count_policy"
    return capture


def write_capture(root: Path, name: str, capture: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        search = modulus_set_search.build_report(
            [("prefix5-byte-ladder-search", modulus_set_search.DEFAULT_MODULI[:9])],
            72,
        )
        default = set_median(with_default_modulus_metadata(load_fixture()), 200.0)
        candidate = set_median(as_experimental(default), 100.0)
        default_path = write_capture(root, "default.json", default)
        candidate_path = write_capture(root, "candidate.json", candidate)

        report = modulus_set_autotune_report.build_report([default_path, candidate_path], search)
        rows = {row["modulus_set"]: row for row in report["rows"]}
        assert rows["default"]["decision"] == "comparison_anchor"
        assert rows["experimental:prefix5-byte-ladder-search"]["decision"] == "ready_non_promoting_evidence"
        assert rows["experimental:prefix5-byte-ladder-search"]["speedup_vs_anchor"] == 2.0
        assert "runtime_ladder_not_selectable" in rows["experimental:prefix5-byte-ladder-search"]["blockers"]
        assert report["runtime_ladder_changed"] is False

        missing_anchor = copy.deepcopy(candidate)
        missing_anchor["seed"] = 999
        missing_anchor_path = write_capture(root, "missing-anchor.json", missing_anchor)
        missing_report = modulus_set_autotune_report.build_report([missing_anchor_path], search)
        missing_row = missing_report["rows"][0]
        assert missing_row["decision"] == "blocked"
        assert "missing_default_same_workload_anchor" in missing_row["blockers"]

        missing_search = copy.deepcopy(candidate)
        missing_search["modulus_set"]["name"] = "experimental:not-in-search-report"
        missing_search_path = write_capture(root, "missing-search.json", missing_search)
        missing_search_report = modulus_set_autotune_report.build_report(
            [default_path, missing_search_path],
            search,
        )
        missing_search_row = next(
            row for row in missing_search_report["rows"] if row["modulus_set"] == "experimental:not-in-search-report"
        )
        assert missing_search_row["decision"] == "blocked"
        assert "experimental_modulus_set_missing_search_candidate" in missing_search_row["blockers"]

    print("modulus-set autotune report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
