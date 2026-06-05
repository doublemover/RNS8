#!/usr/bin/env python3
"""Focused semantic-contract validator smoke tests."""

from __future__ import annotations

import copy

from test_benchmark_schema import expect_invalid, expect_valid


def main() -> int:
    capture = expect_valid("v4_bounded_i64_ck.json")

    invalid_chain = copy.deepcopy(capture)
    invalid_chain["semantics"] = "finite_ring_u8"
    invalid_chain["residue_chain_length"] = 2
    expect_invalid(
        invalid_chain,
        "residue_chain_length > 1 captures must use bounded or exact-wide RNS semantics",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
