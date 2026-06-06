#!/usr/bin/env python3
"""Focused GPU-event validator smoke tests."""

from __future__ import annotations

import copy

from test_benchmark_schema import expect_invalid, expect_valid


def main() -> int:
    capture = expect_valid("v4_bounded_i64_ck.json")

    invalid_events = copy.deepcopy(capture)
    invalid_events["gpu_event_timings_us"]["pack_h2d"] = [1.0]
    expect_invalid(invalid_events, "gpu_event_timings_us.pack_h2d length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
