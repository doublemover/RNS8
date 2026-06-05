#!/usr/bin/env python3
"""Focused output-layout policy validator smoke tests."""

from __future__ import annotations

import copy

from test_benchmark_schema import add_output_padding_fields, expect_invalid, expect_valid
from benchmark_schema import validate_capture


def main() -> int:
    capture = add_output_padding_fields(expect_valid("v4_bounded_i64_adaptive_hip.json"), padding=2)
    validate_capture(capture)

    invalid_layout = copy.deepcopy(capture)
    invalid_layout["timing_metadata"]["benchmark_output_destination_layout"] = "contiguous_row_major"
    expect_invalid(invalid_layout, "benchmark_output_destination_layout must be padded_row_major")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
