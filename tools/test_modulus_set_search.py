#!/usr/bin/env python3
"""Self-test for offline modulus-set search reports."""

from __future__ import annotations

import modulus_set_search


def main() -> int:
    report = modulus_set_search.build_report(
        [
            ("default-prefix9", modulus_set_search.DEFAULT_MODULI[:9]),
            ("bad-shared-factor", [256, 128, 251]),
            ("tiny", [251, 253]),
        ],
        72,
    )

    assert report["schema_version"] == 2
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

    print("modulus-set search self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
