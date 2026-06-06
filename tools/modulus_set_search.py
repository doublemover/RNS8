#!/usr/bin/env python3
"""Offline modulus ladder search for evidence-only RNS8 experiments.

This tool deliberately does not change the runtime modulus ladder. It provides
the proof surface rank-53 work needs before any future runtime-selectable ladder
can be considered: pairwise-coprime checks, prefix product/range coverage,
reducer-cost hints, CRT constants, and small NTT-friendly-prime hints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_OUT_DIR = Path("temp") / "modulus-set-search"
DEFAULT_MODULI = [
    256,
    255,
    253,
    251,
    247,
    239,
    233,
    229,
    227,
    223,
    217,
    211,
    199,
    197,
    191,
    181,
    179,
    173,
    167,
    163,
    157,
    151,
    149,
    139,
    137,
    131,
    127,
    113,
]
HOT_REDUCER_MODULI = {251, 253, 255, 256}
DEFAULT_REQUIRED_BITS = {
    "bounded_i64_full_signed": 65,
    "bounded_u64_full_unsigned": 64,
    "exact_wide_signed_512": 137,
    "exact_wide_unsigned_512": 138,
    "exact_wide_signed_1024": 138,
    "exact_wide_unsigned_1024": 139,
    "exact_wide_signed_2048": 139,
    "exact_wide_unsigned_2048": 140,
}


def is_coprime_set(values: list[int]) -> bool:
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if math.gcd(left, right) != 1:
                return False
    return True


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def product_bits(values: list[int]) -> int:
    return product(values).bit_length()


def reducer_cost_hint(value: int) -> str:
    if value == 256:
        return "hot_shift_mask_mod256"
    if value == 255:
        return "hot_mersenne_fold_mod255"
    if value in {251, 253}:
        return "hot_existing_reducer_or_shift_mask"
    if is_prime(value):
        return "generic_prime_reciprocal_candidate"
    if value <= 256:
        return "single_byte_modulus_reciprocal_candidate"
    return "non_byte_modulus_not_current_kernel_candidate"


def ntt_friendly_lengths(value: int, max_length: int) -> list[int]:
    if not is_prime(value):
        return []
    lengths: list[int] = []
    length = 1
    while length <= max_length:
        if (value - 1) % (2 * length) == 0:
            lengths.append(length)
        length *= 2
    return lengths


def prefix_products(moduli: list[int], required_bits: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    running: list[int] = []
    for index, modulus in enumerate(moduli, start=1):
        running.append(modulus)
        bits = product_bits(running)
        rows.append(
            {
                "prefix": index,
                "last_modulus": modulus,
                "product_bits": bits,
                "satisfies": {
                    name: bits >= required
                    for name, required in required_bits.items()
                },
            }
        )
    return rows


def required_prefixes(moduli: list[int], required_bits: dict[str, int]) -> dict[str, int | None]:
    rows = prefix_products(moduli, required_bits)
    result: dict[str, int | None] = {}
    for name in required_bits:
        prefix = next((row["prefix"] for row in rows if row["satisfies"][name]), None)
        result[name] = prefix
    return result


def crt_prefix_constants(moduli: list[int]) -> list[dict]:
    constants: list[dict] = []
    prefix_product = 1
    for index, modulus in enumerate(moduli):
        if index == 0:
            constants.append(
                {
                    "prefix": 1,
                    "modulus": modulus,
                    "prior_product_mod_modulus": 1 % modulus,
                    "inverse_prior_product_mod_modulus": 1 % modulus,
                }
            )
        else:
            prior_mod = prefix_product % modulus
            inverse = pow(prior_mod, -1, modulus) if math.gcd(prior_mod, modulus) == 1 else None
            constants.append(
                {
                    "prefix": index + 1,
                    "modulus": modulus,
                    "prior_product_mod_modulus": prior_mod,
                    "inverse_prior_product_mod_modulus": inverse,
                }
            )
        prefix_product *= modulus
    return constants


def reducer_cost_summary(moduli: list[int]) -> dict:
    hints = [reducer_cost_hint(value) for value in moduli]
    return {
        "hot_reducer_count": sum(value in HOT_REDUCER_MODULI for value in moduli),
        "byte_modulus_count": sum(2 <= value <= 256 for value in moduli),
        "prime_count": sum(is_prime(value) for value in moduli),
        "generic_reciprocal_count": sum("reciprocal" in hint for hint in hints),
        "non_byte_modulus_count": sum(value > 256 for value in moduli),
    }


def ntt_summary(moduli: list[int], max_ntt_length: int) -> dict:
    per_modulus = {
        str(value): ntt_friendly_lengths(value, max_ntt_length)
        for value in moduli
        if ntt_friendly_lengths(value, max_ntt_length)
    }
    max_length = 0
    for lengths in per_modulus.values():
        if lengths:
            max_length = max(max_length, max(lengths))
    return {
        "max_power2_length": max_length,
        "per_modulus_power2_lengths": per_modulus,
    }


def candidate_disposition(item: dict, min_bits: int) -> str:
    if not item["pairwise_coprime"]:
        return "reject_not_pairwise_coprime"
    if item["product_bits"] < min_bits:
        return "reject_insufficient_range_bits"
    if item["reducer_cost_summary"]["non_byte_modulus_count"] > 0:
        return "keep_research_non_byte_modulus"
    return "candidate_ready_for_benchmark"


def candidate_report(moduli: list[int], name: str) -> dict:
    required_bits = dict(DEFAULT_REQUIRED_BITS)
    item = {
        "name": name,
        "moduli": moduli,
        "prefix_count": len(moduli),
        "product_bits": product_bits(moduli),
        "pairwise_coprime": is_coprime_set(moduli),
        "reducer_cost_hints": {str(value): reducer_cost_hint(value) for value in moduli},
        "reducer_cost_summary": reducer_cost_summary(moduli),
        "required_prefixes": required_prefixes(moduli, required_bits),
        "prefix_products": prefix_products(moduli, required_bits),
        "crt_prefix_constants": crt_prefix_constants(moduli),
        "ntt_friendliness": ntt_summary(moduli, max_ntt_length=128),
        "promotion_eligible": False,
        "runtime_selectable": False,
        "cache_promotion_blocker": "offline_modulus_set_search_not_runtime_selectable",
    }
    item["disposition"] = candidate_disposition(item, min_bits=72)
    return item


def parse_moduli(text: str) -> list[int]:
    values = [int(part.strip(), 0) for part in text.split(",") if part.strip()]
    if not values or any(value <= 1 for value in values):
        raise argparse.ArgumentTypeError("moduli must be comma-separated integers > 1")
    return values


def generated_candidates() -> list[tuple[str, list[int]]]:
    primes = [value for value in range(251, 1, -1) if is_prime(value)]
    ntt_primes = sorted(
        primes,
        key=lambda value: (max(ntt_friendly_lengths(value, 128) or [0]), value),
        reverse=True,
    )
    odd_coprime = [value for value in range(255, 1, -1) if value % 2 == 1]
    coprime_desc: list[int] = [256]
    for value in odd_coprime:
        if all(math.gcd(value, existing) == 1 for existing in coprime_desc):
            coprime_desc.append(value)
        if len(coprime_desc) >= len(DEFAULT_MODULI):
            break
    return [
        ("default", DEFAULT_MODULI),
        ("default-prefix9", DEFAULT_MODULI[:9]),
        ("default-prefix20", DEFAULT_MODULI[:20]),
        ("prime-ntt-front-byte", [256] + ntt_primes[:27]),
        ("coprime-desc-byte", coprime_desc),
        ("hot-four-anchor", DEFAULT_MODULI[:4]),
    ]


def build_report(
    candidates: list[tuple[str, list[int]]],
    min_bits: int,
    *,
    include_generated: bool = False,
) -> dict:
    all_candidates = list(candidates)
    if include_generated or not all_candidates:
        existing = {name for name, _values in all_candidates}
        for name, values in generated_candidates():
            if name not in existing:
                all_candidates.append((name, values))
    reports = [candidate_report(values, name) for name, values in all_candidates]
    for item in reports:
        item["satisfies_min_bits"] = item["product_bits"] >= min_bits and item["pairwise_coprime"]
        item["disposition"] = candidate_disposition(item, min_bits)
    return {
        "schema_version": 2,
        "policy": "offline_search_evidence_only_non_promoting",
        "minimum_product_bits": min_bits,
        "required_bits": DEFAULT_REQUIRED_BITS,
        "search_dimensions": [
            "candidate_rns_ladders",
            "modulus_ordering",
            "prefix_counts",
            "reducer_cost",
            "crt_prefix_constants",
            "ntt_friendly_prime_hints",
            "exact_range_products",
        ],
        "runtime_policy": "report_only_runtime_ladder_unchanged",
        "default_change_gate": "spec_cache_schema_proof_and_same_target_review_required",
        "candidates": reports,
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
    parser.add_argument("--generated-candidates", action="store_true")
    parser.add_argument("--min-bits", type=int, default=72)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    candidates: list[tuple[str, list[int]]] = []
    for raw in args.candidate or []:
        if "=" not in raw:
            raise SystemExit(f"--candidate must be NAME=moduli: {raw}")
        name, values = raw.split("=", 1)
        candidates.append((name, parse_moduli(values)))
    report = build_report(candidates, args.min_bits, include_generated=args.generated_candidates)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(write_report(report, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
