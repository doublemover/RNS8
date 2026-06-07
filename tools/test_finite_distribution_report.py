#!/usr/bin/env python3
"""Self-test finite distribution sensitivity report."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import finite_distribution_report
from benchmark_schema import load_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def _write(path: Path, capture: dict) -> Path:
    path.write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    base = load_capture(FIXTURE_DIR / "v4_finite_ring_u8_ck.json")
    full_uniform = copy.deepcopy(base)
    full_uniform["input_distribution"] = "u8_uniform_0_modulus_minus_1"
    binary = copy.deepcopy(base)
    binary["input_distribution"] = "u8_binary_0_1"

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        report = finite_distribution_report.build_report(
            [
                _write(root / "full.json", full_uniform),
                _write(root / "binary.json", binary),
            ]
        )

    assert report["schema"] == "rns8_finite_distribution_report_v1"
    assert report["capture_count"] == 2
    assert report["distribution_counts"] == {"binary": 1, "full_uniform": 1}
    by_distribution = {group["distribution_role"]: group for group in report["groups"]}
    assert by_distribution["full_uniform"]["disposition"] == "baseline_reference"
    assert by_distribution["binary"]["disposition"] == "keep experimental"
    assert "missing_cpu_reference" in by_distribution["binary"]["blockers"]
    assert "missing_direct_hip" in by_distribution["binary"]["blockers"]
    assert by_distribution["binary"]["full_uniform_winner"]["backend"] == "ck"

    print("finite distribution report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
