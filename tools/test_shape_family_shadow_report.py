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
        assert report["schema"] == "rns8_shape_family_shadow_report_v1"
        assert report["reviewed_cache_entry_count"] == 2
        assert report["family_count"] == 2

        exact, family, missing = report["recommendations"]
        assert exact["would_recommend"] is True
        assert exact["recommendation_is_exact_cache_hit"] is True
        assert "exact_cache_hit_already_owned_by_current_AUTO_cache" in exact["promotion_blockers"]

        assert family["would_recommend"] is True
        assert family["recommendation_is_exact_cache_hit"] is False
        assert family["recommended_backend"] == "ck"
        assert "exact_query_not_reviewed" in family["promotion_blockers"]
        assert "family_policy_not_mechanically_enforced" in family["promotion_blockers"]
        assert family["promotion_eligible"] is False

        assert missing["would_recommend"] is False
        assert "missing_same_family_reviewed_entry" in missing["promotion_blockers"]

        output = tmp / "report.json"
        output.write_text(json.dumps(report), encoding="utf-8")
        assert json.loads(output.read_text(encoding="utf-8"))["recommendations"][1]["recommended_backend"] == "ck"

    print("shape-family shadow report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
