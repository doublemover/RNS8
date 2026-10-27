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


def generate_header(certificate: ArithmeticCertificate, symbol: str | None = None) -> str:
    import re
    result = verify(certificate)
    if not result["algebraically_verified"]:
        raise ValueError(result["proof_failures"])
    c = certificate
    if (c.prior_min != 0 or c.prior_max != 0 or not 1 <= c.safe_k <= (1 << 32) - 1
            or c.recombination_min < -(1 << 63) or c.recombination_max >= (1 << 63)
            or c.instruction.accumulator_min < -(1 << 63) or c.instruction.accumulator_max >= (1 << 63)
            or not 1 <= c.denominator < (1 << 63)
            or any(not -(1 << 63) < p.weight < (1 << 63) for p in c.planes)
            or any(not -128 <= x <= 127 for p in c.planes for x in p.lhs + p.rhs)):
        raise ValueError("generated scalar implementation requires zero prior and proven byte/int64 intermediates")
    symbol = symbol or "encoding_" + result["sha256"]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", symbol) or "__" in symbol:
        raise ValueError("invalid generated C++ symbol")
    symbol = "rns8_" + symbol
    arrays = []
    for operand in ("lhs", "rhs"):
        rows = ["{" + ",".join(str(v) for v in getattr(p, operand)) + "}" for p in c.planes]
        arrays.append(f"  inline static constexpr int8_t {operand}[{len(c.planes)}][{c.modulus}] = {{" + ",".join(rows) + "};")
    weights = ",".join(str(p.weight) + "LL" for p in c.planes)
    return f"""// Generated exact scalar integer model; no floating instruction qualification.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace rns8_encoding_reference {{
struct {symbol} {{
  inline static constexpr char certificate_sha256[] = "{result['sha256']}";
  inline static constexpr uint32_t modulus = {c.modulus};
  inline static constexpr uint32_t safe_k = {c.safe_k};
  inline static constexpr uint32_t plane_count = {len(c.planes)};
  inline static constexpr int64_t weights[plane_count] = {{{weights}}};
{chr(10).join(arrays)}
  template<class T> static uint32_t canonical(T value) {{
    static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>);
    if constexpr (std::is_signed_v<T>) {{
      const auto r = value % static_cast<int64_t>(modulus);
      return static_cast<uint32_t>(r < 0 ? r + modulus : r);
    }} else return static_cast<uint32_t>(value % modulus);
  }}
  // All dot accumulators and ordered weighted partials are independently
  // covered by the embedded certificate. Divide BEFORE modular reduction.
  template<class T> static bool dot(const T* a, const T* b, std::size_t count, int8_t& output) {{
    static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>);
    if (count && (!a || !b)) return false;
    int64_t residue = 0;
    for (std::size_t offset = 0; offset < count;) {{
      const auto remaining = count - offset;
      const std::size_t length = remaining < safe_k ? remaining : safe_k;
      int64_t sums[plane_count]{{}};
      for (std::size_t q = 0; q < length; ++q) {{
        const auto x = canonical(a[offset + q]), y = canonical(b[offset + q]);
        for (uint32_t p = 0; p < plane_count; ++p)
          sums[p] += static_cast<int64_t>(lhs[p][x]) * rhs[p][y];
      }}
      int64_t numerator = 0;
      for (uint32_t p = 0; p < plane_count; ++p) numerator += weights[p] * sums[p];
      if (numerator % {c.denominator}LL) return false;
      residue = (residue + (numerator / {c.denominator}LL) % modulus) % modulus;
      offset += length;
    }}
    if (residue < 0) residue += modulus;
    if (residue >= (modulus + 1) / 2) residue -= modulus;
    output = static_cast<int8_t>(residue);
    return true;
  }}
}};
}} // namespace rns8_encoding_reference
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulus", type=int, required=True)
    parser.add_argument("--safe-k", type=int, default=65536)
    parser.add_argument("--construction", choices=("base13", "awe6"), default="base13")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--symbol")
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
    header = generate_header(certificate, args.symbol) if args.header else None
    if header is not None:
        import hashlib
        result["generated_source_sha256"] = hashlib.sha256(header.encode("utf-8")).hexdigest()
        result["generated_kind"] = "scalar_integer_dot_model"
        args.header.parent.mkdir(parents=True, exist_ok=True)
        args.header.write_text(header, encoding="utf-8", newline="\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0 if result["algebraically_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
