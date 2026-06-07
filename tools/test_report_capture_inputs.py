#!/usr/bin/env python3
"""Self-test shared report capture directory input handling."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from benchmark_schema import BenchmarkSchemaError, load_capture
import export_selector_report
import finite_distribution_report
import layout_search_report
from report_capture_inputs import load_report_captures
import reconstruction_export_report
import resident_workspace_report
import tile_shape_report
import vector_to_rns_chain_report
import wrap64_direct_hip_tuning_report
import zero_skip_expansion_report


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        capture = load_capture(FIXTURE_DIR / "v4_bounded_i64_hipblaslt.json")
        capture_path = root / "capture.json"
        _write_json(capture_path, capture)
        _write_json(root / "review_report.json", {"schema_version": 3, "groups": []})
        _write_json(root / "scenario_manifest.json", {"scenarios": []})
        _write_json(root / "generated-report.json", {"schema": "not_a_capture", "rows": []})

        captures = load_report_captures([root])
        assert len(captures) == 1
        assert captures[0]["_path"] == str(capture_path)

        try:
            load_report_captures([root / "review_report.json"])
        except BenchmarkSchemaError:
            pass
        else:
            raise AssertionError("explicit non-capture JSON must fail strictly")

        layout_search_report.build_report(load_report_captures([root]))
        zero_skip_expansion_report.compare_zero_skip_expansion(load_report_captures([root]))
        vector_to_rns_chain_report.build_report(load_report_captures([root]))
        finite_distribution_report.build_report([root])
        export_selector_report.build_report([root])
        resident_workspace_report.build_report([root])
        tile_shape_report.build_report([root])
        wrap64_direct_hip_tuning_report.build_report([root])
        reconstruction_export_report.build_report([root])

    print("report capture input self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
