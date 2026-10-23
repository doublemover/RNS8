"""Deterministic base-13 schoolbook oracle for scaled E2M1 integers.

The positional construction is described at https://arxiv.org/abs/2608.06812.
This implementation independently derives its digit map; it is not an AMD
floating-instruction qualification or an imported AWE coefficient catalogue.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import product
import json
from pathlib import Path

from arithmetic_certificate import ArithmeticCertificate, InstructionContract, ProductPlane, verify


E2M1_SCALED_ALPHABET = (0, 1, 2, 3, 4, 6, 8, 12, -1, -2, -3, -4, -6, -8, -12)
DIGITS = tuple(min((v for v in E2M1_SCALED_ALPHABET if v % 13 == r), key=lambda v: (abs(v), v))
               for r in range(13))

# Mathematical construction, independently enumerated below. Source document:
# https://github.com/FP4-is-All-you-Need/BLAS/blob/main/AWE/Solutions/SIX_PRODUCT.md
# Git blob 3c1b45655c4cf72a04cc5b9ddbace11e2b96cfe7, correction #2.
AWE_WEIGHTS = (8, 17, 48)
AWE_COEFFICIENTS = (-184, -391, 1104, -136, 384, 816)


@lru_cache(maxsize=1)
def _awe_lut() -> dict[int, tuple[int, ...]]:
    lut = {}
    for a, b, c in product(range(-4, 5), repeat=3):
        faces = (a, b, c, a - b, a + c, b + c)
        if max(map(abs, faces)) > 4:
            continue
        value = sum(w * x for w, x in zip(AWE_WEIGHTS, (a, b, c)))
        key = (max(map(abs, faces)), sum(x * x for x in faces), faces)
        if value not in lut or key < lut[value][0]:
            lut[value] = (key, faces)
    return {value: entry[1] for value, entry in lut.items()}


def awe_encode(value: int) -> tuple[int, ...]:
    if type(value) is not int or not -128 <= value <= 127:
        raise ValueError("AWE adapter accepts exactly the signed byte domain")
    return _awe_lut()[value]


def awe_certificate(modulus: int, safe_k: int = 65536) -> ArithmeticCertificate:
    if type(modulus) is not int or not 2 <= modulus <= 256:
        raise ValueError("expected byte modulus")
    encoded = [awe_encode(x - modulus if x >= (modulus + 1) // 2 else x)
               for x in range(modulus)]
    planes = tuple(ProductPlane(tuple(v[i] for v in encoded), tuple(v[i] for v in encoded), weight)
                   for i, weight in enumerate(AWE_COEFFICIENTS))
    instruction = InstructionContract("host-algebra-only", "scaled_e2m1_awe6_integer_model",
                                      -4, 4, -(1 << 24), 1 << 24, "unqualified",
                                      operand_alphabet=E2M1_SCALED_ALPHABET)
    return ArithmeticCertificate(modulus, planes, 1, safe_k, 0, 0, -(1 << 63), (1 << 63) - 1, instruction)


def encode(value: int, width: int = 3) -> tuple[int, ...]:
    if type(value) is not int or type(width) is not int or not 1 <= width <= 32:
        raise ValueError("encoding requires an integer and a bounded limb width")
    digits = []
    for _ in range(width):
        digit = DIGITS[value % 13]
        digits.append(digit)
        value = (value - digit) // 13
    if value:
        raise ValueError("input is outside the declared positional capacity")
    return tuple(digits)


def decode(digits: tuple[int, ...]) -> int:
    if any(d not in E2M1_SCALED_ALPHABET for d in digits):
        raise ValueError("stored digit is outside scaled E2M1")
    return sum(d * 13 ** i for i, d in enumerate(digits))


def schoolbook_certificate(modulus: int, safe_k: int = 65536) -> ArithmeticCertificate:
    if not 2 <= modulus <= 256:
        raise ValueError("expected byte modulus")
    centered = [x - modulus if x >= (modulus + 1) // 2 else x for x in range(modulus)]
    encoded = [encode(value) for value in centered]
    planes = tuple(ProductPlane(tuple(v[i] for v in encoded), tuple(v[j] for v in encoded), 13 ** (i + j))
                   for i in range(3) for j in range(3))
    # The integer model is deliberately separate from any hardware's internal
    # floating accumulation. Numeric scale 2 is part of the stored alphabet.
    instruction = InstructionContract("host-algebra-only", "scaled_e2m1_schoolbook_integer_model",
                                      -12, 12, -(1 << 24), 1 << 24, "unqualified",
                                      operand_alphabet=E2M1_SCALED_ALPHABET)
    return ArithmeticCertificate(modulus, planes, 1, safe_k, 0, 0, -(1 << 63), (1 << 63) - 1, instruction)


def generate_header(certificate: ArithmeticCertificate) -> str:
    result = verify(certificate)
    if not result["algebraically_verified"]:
        raise ValueError(result["proof_failures"])
    arrays = []
    for index, plane in enumerate(certificate.planes):
        for operand in ("lhs", "rhs"):
            values = ",".join(str(v) for v in getattr(plane, operand))
            arrays.append(f"inline constexpr int8_t {operand}_{index}[] = {{{values}}};")
    terms = " +\n      ".join(f"int64_t{{{p.weight}}} * lhs_{i}[a] * rhs_{i}[b]" for i, p in enumerate(certificate.planes))
    return ("// Generated algebraic reference only; no GPU dispatch.\n#pragma once\n#include <cstdint>\n#include <stdexcept>\n"
            "namespace rns8_encoding_reference {\n"
            f'inline constexpr char certificate_sha256[] = "{result["sha256"]}";\n'
            + "\n".join(arrays)
            + f"\ninline int64_t product(uint16_t a, uint16_t b) {{\n"
              f"  if (a >= {certificate.modulus} || b >= {certificate.modulus}) throw std::out_of_range(\"encoding input\");\n"
              f"  return ({terms}) / {certificate.denominator};\n}}\n}}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulus", type=int, required=True)
    parser.add_argument("--safe-k", type=int, default=65536)
    parser.add_argument("--construction", choices=("base13", "awe6"), default="base13")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header", type=Path)
    args = parser.parse_args()
    factory = awe_certificate if args.construction == "awe6" else schoolbook_certificate
    certificate = factory(args.modulus, args.safe_k)
    result = verify(certificate)
    result["source_domain"] = [-(args.modulus // 2), (args.modulus - 1) // 2]
    result["format"] = {"name": "E2M1", "numeric_scale": 2, "hardware_scale_bits": None}
    result["construction"] = args.construction
    if args.construction == "awe6":
        result["source"] = {"repository": "FP4-is-All-you-Need/BLAS",
                            "path": "AWE/Solutions/SIX_PRODUCT.md",
                            "git_blob_sha1": "3c1b45655c4cf72a04cc5b9ddbace11e2b96cfe7",
                            "encoding": "independently enumerated; no upstream verifier executed"}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.header:
        args.header.write_text(generate_header(certificate), encoding="utf-8")
    return 0 if result["algebraically_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
