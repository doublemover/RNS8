#!/usr/bin/env python3
"""Self-test non-routing shape-family AUTO shadow recommendations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import shape_family_shadow_report
from test_autotune_cache_install import entry, write_cache


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        cache = tmp / "autotune.json"
        bounded = entry()
        finite = entry("-finite", finite_modulus=251)
        write_cache(cache, [bounded, finite])

        exact_query = shape_family_shadow_report.parse_query(
            "semantics=bounded_i64;m=512;n=512;k=512;target_id=gfx1100;layout=row_major"
        )
        family_query = shape_family_shadow_report.parse_query(
            "semantics=bounded_i64;m=768;n=768;k=768;target_id=gfx1100;layout=row_major"
        )
        missing_query = shape_family_shadow_report.parse_query(
            "semantics=finite_ring_u8;finite_modulus=253;m=512;n=512;k=512;target_id=gfx1100-finite"
        )
        report = shape_family_shadow_report.build_report(cache, [exact_query, family_query, missing_query])
        assert report["schema"] == "rns8_shape_family_shadow_report_v2"
        assert report["reviewed_cache_entry_count"] == 2
        assert report["family_count"] == 2
        assert "output_contract" in report["boundary_fields"]

        exact, family, missing = report["recommendations"]
        assert exact["would_recommend"] is True
        assert exact["recommendation_is_exact_cache_hit"] is True
        assert exact["runtime_routing_allowed"] is False
        assert exact["recommendation_boundary_status"] == "exact_reviewed_cache_entry"
        assert "exact_cache_hit_already_owned_by_current_AUTO_cache" in exact["promotion_blockers"]

        assert family["would_recommend"] is True
        assert family["recommendation_is_exact_cache_hit"] is False
        assert family["recommended_backend"] == "ck"
        assert family["runtime_routing_allowed"] is False
        assert family["recommendation_boundary_status"] == "same_boundary_family_shadow_representative"
        assert family["selector_explanation"] == "nearest same-family reviewed cache entry"
        assert family["representative_matrix"]["reviewed_entry_count"] == 1
        assert "exact_query_not_reviewed" in family["promotion_blockers"]
        assert "representative_matrix_requires_same_target_layout_contract_review" in family["promotion_blockers"]
        assert family["promotion_eligible"] is False

        assert missing["would_recommend"] is False
        assert missing["recommendation_boundary_status"] == "missing_same_boundary_family_reviewed_entry"
        assert "missing_same_family_reviewed_entry" in missing["promotion_blockers"]
        assert "same_family_entries_rejected_by_boundary" in missing["promotion_blockers"]
        assert missing["rejected_boundary_candidates"]
        assert "boundary_finite_modulus_mismatch" in missing["rejected_boundary_candidates"][0]["boundary_blockers"]

        cross_contract_query = shape_family_shadow_report.parse_query(
            "semantics=bounded_i64;m=768;n=768;k=768;target_id=gfx1100;layout=row_major;"
            "output_contract=exact_wide_signed_limbs"
        )
        cross_contract = shape_family_shadow_report.build_report(cache, [cross_contract_query])["recommendations"][0]
        assert cross_contract["would_recommend"] is False
        assert "missing_same_family_reviewed_entry" in cross_contract["promotion_blockers"]
        assert "same_family_entries_rejected_by_boundary" in cross_contract["promotion_blockers"]
        assert "boundary_output_contract_mismatch" in cross_contract["rejected_boundary_candidates"][0][
            "boundary_blockers"
        ]

        output = tmp / "report.json"
        output.write_text(json.dumps(report), encoding="utf-8")
        assert json.loads(output.read_text(encoding="utf-8"))["recommendations"][1]["recommended_backend"] == "ck"

    print("shape-family shadow report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
