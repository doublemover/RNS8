#!/usr/bin/env python3
"""Focused reuse-timing validator smoke tests."""

from __future__ import annotations

import copy

from test_benchmark_schema import expect_invalid, expect_valid, as_reused_pack_capture
from benchmark_schema import validate_capture


def main() -> int:
    capture = expect_valid("v4_bounded_i64_hipblaslt.json")

    invalid_reuse = copy.deepcopy(capture)
    invalid_reuse["reuse_packed_inputs"] = True
    invalid_reuse["pack_mode"] = "per_repeat_repack"
    invalid_reuse["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    expect_invalid(invalid_reuse, "pack_mode must describe a prepacked reuse mode")
    reused = as_reused_pack_capture(expect_valid("v4_bounded_i64_ck.json"))
    reused["prepack_setup_us"] = reused["avg_prepack_setup_us"] = 0.375
    reused["prepack_setup_breakdown_us"] = dict(pack_a=0.125, pack_b=0.25, runtime_cache=0.0, unclassified=0.0)
    validate_capture(reused)
    for value in (True, -0.25, float("inf"), float("nan")):
        invalid = copy.deepcopy(reused)
        invalid["prepack_setup_us"] = value
        expect_invalid(invalid, "finite nonnegative prepack_setup_us")
    invalid = copy.deepcopy(reused)
    invalid["avg_prepack_setup_us"] = 0.5
    expect_invalid(invalid, "avg_prepack_setup_us must match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
