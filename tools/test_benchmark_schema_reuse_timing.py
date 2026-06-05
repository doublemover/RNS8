#!/usr/bin/env python3
"""Focused reuse-timing validator smoke tests."""

from __future__ import annotations

import copy

from test_benchmark_schema import expect_invalid, expect_valid


def main() -> int:
    capture = expect_valid("v4_bounded_i64_hipblaslt.json")

    invalid_reuse = copy.deepcopy(capture)
    invalid_reuse["reuse_packed_inputs"] = True
    invalid_reuse["pack_mode"] = "per_repeat_repack"
    invalid_reuse["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    expect_invalid(invalid_reuse, "pack_mode must describe a prepacked reuse mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
