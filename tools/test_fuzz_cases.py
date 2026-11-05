"""Authored wire-format/oracle controls; no fuzzer or library execution."""

from dataclasses import replace
import struct
import unittest

from fuzz_cases import Case, HEADER, decode, encode, expected, seeds


class FuzzCases(unittest.TestCase):
    def test_named_seeds_roundtrip_with_explicit_semantics(self):
        self.assertEqual(HEADER.size, 24)
        names, semantics, moduli, prefixes = set(), set(), set(), set()
        for name, case in seeds():
            self.assertNotIn(name, names)
            names.add(name)
            self.assertEqual(decode(encode(case)), case)
            semantics.add(case.semantic)
            if case.semantic == 2:
                moduli.add(case.modulus)
            else:
                prefixes.add(case.prefix)
            self.assertEqual(len(expected(case)), case.m * case.n)
        self.assertEqual(semantics, set(range(6)))
        self.assertEqual(moduli, set(range(2, 257)))
        self.assertTrue(set(range(1, 21)) <= prefixes)

    def test_reserved_lengths_dimensions_and_native_domains_fail_before_execution(self):
        case = Case(0, 1, 1, 1, (-1,), (1,))
        raw = encode(case)
        self.assertEqual(struct.unpack_from("<Q", raw, 24)[0], (1 << 64) - 1)
        for mutation in (raw[:-1], raw + b"x", b"R8F2" + raw[4:], raw[:18] + b"x" + raw[19:]):
            with self.assertRaises(ValueError):
                decode(mutation)
        for mutation in (
            replace(case, m=0),
            replace(case, k=1 << 32),
            replace(case, prefix=True),
            replace(case, a=(1 << 63,)),
            replace(case, a=(True,)),
            replace(case, semantic=2, prefix=0, modulus=256, a=(256,)),
        ):
            with self.assertRaises(ValueError):
                encode(mutation)

    def test_independent_finite_wrap_and_signed_oracles(self):
        self.assertEqual(expected(Case(0, 1, 1, 2, (-(1 << 63), (1 << 63) - 1), (1, 1))), (-1,))
        self.assertEqual(
            expected(Case(3, 1, 1, 1, ((1 << 64) - 1,), (2,), prefix=0)), ((1 << 64) - 2,)
        )
        self.assertEqual(expected(Case(2, 1, 1, 1, (255,), (255,), prefix=0, modulus=254)), (1,))
        self.assertEqual(expected(Case(4, 1, 1, 1, (-(1 << 63),), (-(1 << 63),))), (1 << 126,))


if __name__ == "__main__":
    unittest.main()
