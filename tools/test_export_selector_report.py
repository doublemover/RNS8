#!/usr/bin/env python3
"""Self-test export selector report grouping and blockers."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import export_selector_report
from test_starfoundry_reports import starfoundry_capture


def write_capture(path: Path, capture: dict) -> None:
    path.write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")


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
        assert row["promotion_eligible"] is False
        assert "missing_selector_key" not in row["promotion_blockers"]

        stale = copy.deepcopy(good)
        stale["export_variant"]["selector_key"] = None
        stale_path = tmp / "stale.json"
        write_capture(stale_path, stale)
        stale_report = export_selector_report.build_report([stale_path])
        stale_row = stale_report["groups"][0]["rows"][0]
        assert "missing_selector_key" in stale_row["promotion_blockers"]

    print("export selector report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
