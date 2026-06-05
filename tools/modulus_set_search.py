#!/usr/bin/env python3
"""Offline modulus ladder search for evidence-only RNS8 experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_OUT_DIR = Path("temp") / "modulus-set-search"


def is_coprime_set(values: list[int]) -> bool:
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if math.gcd(left, right) != 1:
                return False
    return True


def product_bits(values: list[int]) -> int:
    product = 1
    for value in values:
        product *= value
    return product.bit_length()


def reducer_cost_hint(value: int) -> str:
    if value in {251, 253, 255, 256}:
        return "hot_existing_reducer_or_shift_mask"
    if value < 256:
        return "single_byte_modulus_reciprocal_candidate"
    return "non_byte_modulus_not_current_kernel_candidate"


def candidate_report(moduli: list[int], name: str) -> dict:
    return {
        "name": name,
        "moduli": moduli,
        "prefix_count": len(moduli),
        "product_bits": product_bits(moduli),
        "pairwise_coprime": is_coprime_set(moduli),
        "reducer_cost_hints": {str(value): reducer_cost_hint(value) for value in moduli},
        "cache_promotion_blocker": "offline_modulus_set_search_not_reviewed",
    }


def parse_moduli(text: str) -> list[int]:
    values = [int(part.strip(), 0) for part in text.split(",") if part.strip()]
    if not values or any(value <= 1 for value in values):
        raise argparse.ArgumentTypeError("moduli must be comma-separated integers > 1")
    return values


def build_report(candidates: list[tuple[str, list[int]]], min_bits: int) -> dict:
    reports = [candidate_report(values, name) for name, values in candidates]
    return {
        "schema_version": 1,
        "policy": "offline_search_evidence_only_non_promoting",
        "minimum_product_bits": min_bits,
        "candidates": [
            {**item, "satisfies_min_bits": item["product_bits"] >= min_bits and item["pairwise_coprime"]}
            for item in reports
        ],
    }


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "modulus-set-search-report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="NAME=comma-separated moduli; repeatable",
    )
    parser.add_argument("--min-bits", type=int, default=72)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    candidates: list[tuple[str, list[int]]] = []
    for raw in args.candidate or ["default-hot=251,253,255,256"]:
        if "=" not in raw:
            raise SystemExit(f"--candidate must be NAME=moduli: {raw}")
        name, values = raw.split("=", 1)
        candidates.append((name, parse_moduli(values)))
    report = build_report(candidates, args.min_bits)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(write_report(report, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
