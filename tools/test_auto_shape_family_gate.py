#!/usr/bin/env python3
"""Self-test AUTO shape-family gate invariants."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import auto_shape_family_gate
import shape_family_shadow_report
from test_autotune_cache_install import entry, write_cache


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        cache = tmp / "autotune.json"
        bounded = entry()
        finite = entry("-finite", finite_modulus=251)
        write_cache(cache, [bounded, finite])

        queries = [
            shape_family_shadow_report.parse_query(
                "semantics=bounded_i64;m=512;n=512;k=512;target_id=gfx1100;layout=row_major"
            ),
            shape_family_shadow_report.parse_query(
                "semantics=bounded_i64;m=768;n=768;k=768;target_id=gfx1100;layout=row_major"
            ),
            shape_family_shadow_report.parse_query(
                "semantics=finite_ring_u8;finite_modulus=253;m=512;n=512;k=512;target_id=gfx1100-finite"
            ),
            shape_family_shadow_report.parse_query(
                "semantics=bounded_i64;m=768;n=768;k=768;target_id=gfx1100;layout=row_major;"
                "output_contract=exact_wide_signed_limbs"
            ),
        ]
        shadow = shape_family_shadow_report.build_report(cache, queries)
        shadow_path = tmp / "shape-family-shadow-report.json"
        write_json(shadow_path, shadow)

        report = auto_shape_family_gate.build_report(cache, [shadow_path], require_recommendations=True)
        assert report["schema"] == "rns8_auto_shape_family_gate_v1"
        assert report["rank36_gate_complete"] is True, json.dumps(report["blockers"], indent=2)
        assert report["runtime_exact_cache_guard"]["ready"] is True
        assert report["recommendation_count"] == 4
        assert report["exact_cache_hit_count"] == 1
        assert report["non_exact_recommendation_count"] == 1
        assert report["missing_same_boundary_count"] == 2
        assert report["boundary_rejected_recommendation_count"] == 2
        assert not report["blockers"]
        assert all(row["runtime_routing_allowed"] is False for row in report["recommendations"])
        assert all(row["promotion_eligible"] is False for row in report["recommendations"])

        routed_shadow = copy.deepcopy(shadow)
        routed_shadow["recommendations"][1]["runtime_routing_allowed"] = True
        routed_path = tmp / "routed-shadow.json"
        write_json(routed_path, routed_shadow)
        routed_report = auto_shape_family_gate.build_report(cache, [routed_path], require_recommendations=True)
        assert routed_report["rank36_gate_complete"] is False
        assert any("shape_family_runtime_routing_not_disabled" in item for item in routed_report["blockers"])

        stale_boundary_shadow = copy.deepcopy(shadow)
        stale_boundary_shadow["boundary_fields"] = [
            field for field in stale_boundary_shadow["boundary_fields"] if field != "output_contract"
        ]
        stale_boundary_path = tmp / "stale-boundary-shadow.json"
        write_json(stale_boundary_path, stale_boundary_shadow)
        stale_boundary_report = auto_shape_family_gate.build_report(
            cache,
            [stale_boundary_path],
            require_recommendations=True,
        )
        assert stale_boundary_report["rank36_gate_complete"] is False
        assert any("boundary_field_missing:output_contract" in item for item in stale_boundary_report["blockers"])

    print("AUTO shape-family gate self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
