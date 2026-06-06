from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_review_index, load_scenario_index, load_validated_captures
from .isa import load_isa_index
from .outputs import write_outputs
from .rows import build_database

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True, help="capture file or directory")
    parser.add_argument("--review-report", type=Path, action="append", default=[], help="benchmark_sweep review_report.json")
    parser.add_argument("--scenario-manifest", type=Path, action="append", default=[], help="scenario_manifest.json")
    parser.add_argument("--isa-report", type=Path, action="append", default=[], help="ISA summary file or directory")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="skip invalid JSON captures while recording skipped paths in the output summary",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("temp") / "evidence-database",
        help="ignored output directory for evidence_database.json, CSV, and Markdown",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invalid_captures: list[dict[str, str]] = []
    captures = load_validated_captures(
        args.capture,
        skip_invalid=args.skip_invalid,
        invalid_captures=invalid_captures,
    )
    scenario_index = load_scenario_index(args.scenario_manifest)
    review_index = load_review_index(args.review_report)
    isa_index = load_isa_index(args.isa_report)
    database = build_database(
        captures,
        scenario_index=scenario_index,
        review_index=review_index,
        isa_index=isa_index,
        invalid_captures=invalid_captures,
    )
    outputs = write_outputs(database, args.out_dir)
    print(json.dumps({"captures": len(captures), **outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
