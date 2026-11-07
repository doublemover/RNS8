"""Authored wire-format/oracle controls; no fuzzer or library execution."""

from dataclasses import replace
import struct
import unittest

from fuzz_cases import Case, HEADER, admitted, decode, encode, expected, metamorphic_seeds, seeds


class FuzzCases(unittest.TestCase):
    def test_mapped_metamorphic_relations_keep_independent_valid_intermediates(self):
        cases = dict(metamorphic_seeds())
        for semantic in range(6):
            prefix = f"metamorphic-{semantic}-"
            base = cases[prefix + "base"]
            reference = expected(base)
            self.assertEqual(
                expected(cases[prefix + "row-permutation"]), reference[3:] + reference[:3]
            )
            self.assertEqual(
                expected(cases[prefix + "column-permutation"]),
                tuple(reference[row * 3 + col] for row in range(2) for col in (2, 0, 1)),
            )
            for relation in ("common-k-permutation", "zero-k-padding", "physical-padding"):
                self.assertEqual(expected(cases[prefix + relation]), reference)
            if semantic not in (2, 3):
                self.assertEqual(expected(cases[prefix + "sufficient-prefix"]), reference)
            self.assertEqual(expected(cases[prefix + "identity-operand"]), base.a)
            self.assertEqual(expected(cases[prefix + "zero-operand"]), (0,) * 6)
            left, right = (
                expected(cases[prefix + "split-k-" + side]) for side in ("left", "right")
            )
            joined = tuple(a + b for a, b in zip(left, right))
            if semantic in (2, 3):
                modulus = base.modulus if semantic == 2 else 1 << 64
                joined = tuple(cell % modulus for cell in joined)
            self.assertEqual(joined, reference)
            for operations in range(4 if semantic in (2, 3) else 8):
                case = cases[f"history-{semantic}-{operations}"]
                self.assertTrue(admitted(case))
                self.assertEqual(case.operations, operations)
                self.assertEqual(expected(case), reference)

    def test_wide_zero_product_needs_independently_representable_inputs(self):
        for semantic, limit in ((4, 127), (5, 255)):
            for reverse in (False, True):
                a, b = ((0,), (limit + 1,)) if reverse else ((limit + 1,), (0,))
                rejected = Case(semantic, 1, 1, 1, a, b, prefix=1)
                self.assertEqual(expected(rejected), (0,))
                self.assertFalse(admitted(rejected))
                self.assertTrue(admitted(replace(rejected, prefix=20)))
                accepted = replace(
                    rejected, a=(0,) if reverse else (limit,), b=(limit,) if reverse else (0,)
                )
                self.assertTrue(admitted(accepted))
        # Bounded residue packing has a different contract: inputs need not be
        # uniquely reconstructible when the declared exact output bound fits.
        self.assertTrue(admitted(Case(0, 1, 1, 2, (256, -256), (1, 1), prefix=1)))
        self.assertFalse(admitted(Case(0, 1, 1, 1, (1 << 62,), (4,))))
        self.assertFalse(admitted(Case(5, 1, 1, 1, (200,), (200,), prefix=1)))

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
