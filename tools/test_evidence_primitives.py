"""Strict boundary controls shared by calibration and qualification consumers."""

from __future__ import annotations

import math
import unittest

from evidence_scalars import demand, finite_number, integer, positive, text_choice
from json_evidence import parse_evidence_json, same_evidence_value
from verification_contract import checksum


class EvidencePrimitives(unittest.TestCase):
    def test_integer_bounds_reject_boolean_and_numeric_subclasses(self):
        class AuthoredInt(int):
            pass
        for value in (True, False, 1.0, "1", None, AuthoredInt(1)):
            self.assertFalse(integer(value, 0, 2))
        self.assertTrue(integer(0, 0, 2))
        self.assertTrue(integer(2, 0, 2))
        self.assertFalse(integer(3, 0, 2))

    def test_positive_timing_rejects_nonfinite_huge_and_non_numeric_values(self):
        for value in (True, False, 0, -1, math.inf, -math.inf, math.nan, "1", None, 1 << 16384):
            self.assertFalse(positive(value))
        self.assertTrue(positive(1))
        self.assertTrue(positive(0.25))

    def test_json_rejects_duplicate_constants_and_overflowing_exponents(self):
        for text in ('{"x":1,"x":2}', '{"nested":{"x":1,"x":2}}',
                     '[NaN]', '[Infinity]', '[-Infinity]', '[1e999]', '[-1e999]'):
            with self.assertRaises(ValueError):
                parse_evidence_json(text)
        value = parse_evidence_json('{"integer":18446744073709551615,"zero":0,"null":null}')
        self.assertEqual(value["integer"], (1 << 64) - 1)
        self.assertFalse(same_evidence_value(value["zero"], value["null"]))

    def test_finite_numbers_and_text_choices_reject_wrong_type_frontiers(self):
        for value in (True, [], {}, None, "1", 1 << 16384, math.nan, math.inf):
            self.assertFalse(finite_number(value))
        for value in (0, -1, 0.5):
            self.assertTrue(finite_number(value))
        choices = {"supported", "unsupported"}
        for value in ([], {}, None, True, 1, ["supported"]):
            self.assertFalse(text_choice(value, choices))
        self.assertTrue(text_choice("supported", choices))
        self.assertFalse(text_choice("unknown", choices))

    def test_structural_comparison_keeps_type_null_and_missing_distinct(self):
        for left, right in ((True, 1), (1, 1.0), ({}, {"x": None}),
                            ({"x": None}, {"x": 0}), ([1], [True])):
            self.assertFalse(same_evidence_value(left, right))
        self.assertTrue(same_evidence_value({"x": [1, None, "bytes"]}, {"x": [1, None, "bytes"]}))
        with self.assertRaisesRegex(ValueError, "authored reason"):
            demand(False, "authored reason")

    def test_empty_byte_checksum_retains_the_documented_64_bit_offset(self):
        self.assertEqual(checksum(b""), "cbf29ce484222325")


if __name__ == "__main__":
    unittest.main()
