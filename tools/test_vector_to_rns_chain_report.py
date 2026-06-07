#!/usr/bin/env python3
"""Self-test vector/native-to-RNS chain reporting."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vector_to_rns_chain_report
from test_benchmark_schema_support_core import as_vector_to_rns_chain_capture, expect_valid


def make_capture(*, host_repack_control: bool, median: float, checksum: int = 12345) -> dict:
    capture = as_vector_to_rns_chain_capture(
        expect_valid("v4_bounded_i64_adaptive_hip.json"),
        "native_i64_to_rns_kernel",
        "vector_alu_i64_kernel",
        host_repack_control=host_repack_control,
    )
    capture["warmups"] = 3
    capture["repeats"] = 9
    capture["checksum"] = checksum
    for phase in ["pack", "rns_gemm", "crt_export", "end_to_end"]:
        capture["timing_summary_us"][phase]["median"] = median if phase == "end_to_end" else median / 4.0
    return capture


candidate = make_capture(host_repack_control=False, median=80.0)
control = make_capture(host_repack_control=True, median=120.0)
report = vector_to_rns_chain_report.build_report([candidate, control])
assert report["capture_count"] == 2
assert report["group_count"] == 1
row = report["rows"][0]
assert row["disposition"] == "local promote"
assert row["checksum_match"] is True
assert row["speedup_vs_host_repack_control"] == 1.5
assert row["control_host_repack_us"] == 3.0

missing = vector_to_rns_chain_report.build_report([candidate])
assert missing["rows"][0]["disposition"] == "keep experimental"
assert "missing_host_export_repack_control" in missing["rows"][0]["blockers"]

mismatch_control = copy.deepcopy(control)
mismatch_control["checksum"] = 777
mismatch = vector_to_rns_chain_report.build_report([candidate, mismatch_control])
assert mismatch["rows"][0]["disposition"] == "keep experimental"
assert "final_checksum_mismatch" in mismatch["rows"][0]["blockers"]

slow_candidate = make_capture(host_repack_control=False, median=140.0)
slow_report = vector_to_rns_chain_report.build_report([slow_candidate, control])
assert slow_report["rows"][0]["disposition"] == "drop/deprioritize"

with tempfile.TemporaryDirectory() as temp_dir:
    root = Path(temp_dir)
    (root / "review_report.json").write_text('{"schema_version": 3, "groups": []}\n', encoding="utf-8")
    (root / "capture.json").write_text(json.dumps(expect_valid("v4_bounded_i64_adaptive_hip.json")) + "\n", encoding="utf-8")
    loaded = [
        capture
        for path in vector_to_rns_chain_report.expand_inputs([root])
        if (capture := vector_to_rns_chain_report.load_validated_capture(path)) is not None
    ]
    assert len(loaded) == 1
    assert loaded[0]["benchmark"] == "rns8_bounded_gemm_persistent_rns"

print("vector-to-RNS chain report self-test: PASS")
