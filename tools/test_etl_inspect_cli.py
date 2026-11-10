"""Bounded real Windows helper identity/early-rejection checks, without an ETL."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_request import file_digest
from etl_inspector_identity import source_identity
from json_evidence import parse_evidence_json
from process_progress import run_capture, capture_succeeded

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args(argv)
    binary = args.binary.resolve(strict=True)
    before = file_digest(binary)
    result = run_capture(
        [str(binary), "--build-identity"],
        cwd=ROOT,
        label="etl-inspect-source-identity",
        timeout_seconds=15,
        max_output_bytes=16384,
        return_on_interrupt=True,
    )
    if not capture_succeeded(result):
        raise ValueError("native ETL inspector identity query did not complete")
    value = parse_evidence_json(result["stdout"])
    if (
        type(value) is not dict
        or value.get("schema") != "rns8_etl_inspector_identity_v1"
        or value.get("source_sha256") != source_identity()
        or type(value.get("process_id")) is not int
        or value["process_id"] != result["process_id"]
        or type(value.get("pointer_bytes")) is not int
        or value["pointer_bytes"] != 8
    ):
        raise ValueError("native ETL inspector was not compiled from the current source/ABI")
    # Invalid selectors must fail before canonicalizing/opening an input file.
    for text in ("0", "-1", "4294967296", "1x"):
        result = run_capture(
            [
                str(binary),
                "--input",
                str(ROOT / "temp/nonexistent-etl-selector-control.etl"),
                "--pid",
                text,
            ],
            cwd=ROOT,
            label="etl-inspect-invalid-pid",
            timeout_seconds=15,
            max_output_bytes=16384,
            return_on_interrupt=True,
        )
        if (
            type(result.get("returncode")) is not int
            or result["returncode"] != 2
            or not capture_succeeded(dict(result, returncode=0))
            or result["stdout"]
            or "positive uint32 PID required" not in result["stderr"]
        ):
            raise ValueError("invalid ETL PID did not reach the intended early-rejection assertion")
    if file_digest(binary) != before:
        raise ValueError("native ETL inspector changed during CLI checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
