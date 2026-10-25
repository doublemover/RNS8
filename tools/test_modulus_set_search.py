#!/usr/bin/env python3
"""Self-test for offline modulus-set search reports."""

from __future__ import annotations

import modulus_set_search
import re
from pathlib import Path


def main() -> int:
    source = (Path(__file__).resolve().parents[1] / "src/core/internal.hpp").read_text(encoding="utf-8")
    ladder = re.search(r"kDefaultModuli\[[^]]+\]\s*=\s*\{([^}]+)\}", source)
    assert ladder
    actual = [int(value) for value in re.findall(r"\d+", ladder.group(1))]
    assert modulus_set_search.DEFAULT_MODULI == actual
    report = modulus_set_search.build_report(
        [
            ("default-prefix9", modulus_set_search.DEFAULT_MODULI[:9]),
            ("bad-shared-factor", [256, 128, 251]),
            ("tiny", [251, 253]),
        ],
        72,
    )

    assert report["schema_version"] == 3
    assert report["runtime_policy"] == "report_only_runtime_ladder_unchanged"
    assert "crt_prefix_constants" in report["candidates"][0]
    assert "ntt_friendly_prime_hints" in report["search_dimensions"]

    by_name = {item["name"]: item for item in report["candidates"]}
    default = by_name["default-prefix9"]
    assert default["pairwise_coprime"] is True
    assert default["satisfies_min_bits"] is True
    assert default["required_prefixes"]["bounded_i64_full_signed"] is not None
    assert default["reducer_cost_summary"]["hot_reducer_count"] >= 4
    assert default["crt_prefix_constants"][1]["inverse_prior_product_mod_modulus"] is not None

    bad = by_name["bad-shared-factor"]
    assert bad["pairwise_coprime"] is False
    assert bad["disposition"] == "reject_not_pairwise_coprime"

    tiny = by_name["tiny"]
    assert tiny["satisfies_min_bits"] is False
    assert tiny["disposition"] == "reject_insufficient_range_bits"

    generated = modulus_set_search.build_report([], 72, include_generated=True)
    generated_names = {item["name"] for item in generated["candidates"]}
    assert {"default", "prime-ntt-front-byte", "coprime-desc-byte"} <= generated_names

    default_full = next(item for item in generated["candidates"] if item["name"] == "default")
    assert default_full["required_prefixes"]["bounded_u64_full_unsigned"] == 9
    assert default_full["prefix_products"][7]["product"] == 13057518384392951040
    assert not default_full["prefix_products"][7]["satisfies"]["bounded_u64_full_unsigned"]
    assert generated["minimum_product_bits_is_semantic_proof"] is False
    for item in generated["candidates"]:
        for name, contract in generated["range_contracts"].items():
            expected = next((i for i in range(1, len(item["moduli"]) + 1)
                             if modulus_set_search.is_coprime_set(item["moduli"][:i]) and
                             modulus_set_search.product(item["moduli"][:i]) > contract["strict_product_threshold"]), None)
            assert item["required_prefixes"][name] == expected
    # Equality is insufficient. Equal bit lengths can lie on either side of B.
    profiles = {"unsigned": {"bound": 15, "signed": False},
                "signed": {"bound": 8, "signed": True}}
    assert modulus_set_search.required_prefixes([3, 5, 7], profiles) == {"unsigned": 3, "signed": 3}
    assert modulus_set_search.required_prefixes([4, 4, 7], profiles) == {"unsigned": None, "signed": None}
    assert modulus_set_search.required_prefixes([16, 17], profiles) == {"unsigned": 1, "signed": 2}
    assert generated["range_contracts"]["exact_wide_signed_512"]["strict_product_threshold"] == 512 * (1 << 127)
    assert generated["range_contracts"]["exact_wide_unsigned_512"]["strict_product_threshold"] == 512 * ((1 << 64) - 1) ** 2

    print("modulus-set search self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
