#!/usr/bin/env python3
"""Self-test temp-only GPU counter report assembly."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import gpu_counter_report


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def write_counter_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SQ_INSTS_VALU", "TCP_TCC_WRITE_REQ", "KernelName"])
        writer.writeheader()
        writer.writerow({"SQ_INSTS_VALU": "10", "TCP_TCC_WRITE_REQ": "3", "KernelName": "warmup"})
        writer.writerow({"SQ_INSTS_VALU": "30", "TCP_TCC_WRITE_REQ": "5", "KernelName": "repeat"})


def write_counter_json(path: Path) -> None:
    path.write_text(
        json.dumps({"records": [{"LDS_BANK_CONFLICT": 2, "WAIT_COUNT": 1}, {"LDS_BANK_CONFLICT": 4, "WAIT_COUNT": 3}]}),
        encoding="utf-8",
    )


def write_isa_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "object": "hip_direct_kernels.obj",
                "backend": "direct-hip",
                "target": "gfx1100",
                "reported_symbol_count": 2,
                "instruction_totals": {
                    "wmma": 0,
                    "mfma": 0,
                    "global_store": 4,
                    "lds_mentions": 1,
                    "wait_instructions": 2,
                },
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    capture = FIXTURE_DIR / "v4_bounded_i64_ck.json"
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        counter_csv = root / "counters.csv"
        counter_json = root / "counters.json"
        isa_summary = root / "isa-summary.json"
        out_dir = root / "reports"
        write_counter_csv(counter_csv)
        write_counter_json(counter_json)
        write_isa_summary(isa_summary)

        report = gpu_counter_report.report_for_capture(
            capture,
            [counter_csv, counter_json],
            [isa_summary],
            top_limit=3,
        )
        assert report["policy"] == gpu_counter_report.COUNTER_REPORT_POLICY
        assert report["capture"]["backend_selected"] == "ck"
        assert len(report["counter_inputs"]) == 2
        assert report["counter_inputs"][0]["numeric_metric_average"]["SQ_INSTS_VALU"] == 20.0
        assert report["counter_inputs"][1]["numeric_metric_average"]["LDS_BANK_CONFLICT"] == 3.0
        assert report["isa_summaries"][0]["instruction_totals"]["global_store"] == 4
        assert report["resource_summary"]["memory_pressure_signal"] == 4.0
        assert report["resource_summary"]["global_store_instruction_count"] == 4

        json_path = gpu_counter_report.write_json_report(report, out_dir, "self-test")
        md_path = gpu_counter_report.write_markdown_report(report, out_dir, "self-test")
        assert json_path.exists()
        assert md_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["capture"]["semantics"] == "bounded_i64"
        batch = gpu_counter_report.build_batch_report([report])
        assert batch["group_count"] == 1
        assert batch["groups"][0]["counter_evidence"] == "present"
        batch_paths = gpu_counter_report.write_batch_reports(batch, out_dir)
        assert Path(batch_paths["json"]).exists()
        assert Path(batch_paths["markdown"]).exists()

    print("gpu counter report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
