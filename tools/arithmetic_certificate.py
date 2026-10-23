"""Exact, offline arithmetic certificates; never qualifies an unmeasured opcode.

Encoding tables are indexed by canonical input residue. Integer/rational checking
is deliberately independent of device reducers and generated kernel arithmetic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class InstructionContract:
    target: str
    opcode: str
    operand_min: int
    operand_max: int
    accumulator_min: int
    accumulator_max: int
    compiler: str
    flags: tuple[str, ...] = ()
    operand_alphabet: tuple[int, ...] = ()
    exact_integer_products: bool = True
    measured_evidence_sha256: str | None = None


@dataclass(frozen=True)
class ProductPlane:
    lhs: tuple[int, ...]
    rhs: tuple[int, ...]
    weight: int = 1


@dataclass(frozen=True)
class ArithmeticCertificate:
    modulus: int
    planes: tuple[ProductPlane, ...]
    denominator: int
    safe_k: int
    prior_min: int
    prior_max: int
    recombination_min: int
    recombination_max: int
    instruction: InstructionContract
    semantics: str = "finite_ring"
    basis: tuple[int, ...] = ()
    output_bound: int | None = None


def certificate_hash(certificate: ArithmeticCertificate) -> str:
    payload = json.dumps(asdict(certificate), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _product_interval(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    products = [x * y for x in a for y in b]
    return min(products), max(products)


def verify(c: ArithmeticCertificate) -> dict:
    failures: list[str] = []
    instruction = c.instruction
    if type(c.modulus) is not int or not 2 <= c.modulus <= 256:
        failures.append("modulus_outside_byte_domain")
    if type(c.safe_k) is not int or c.safe_k < 1:
        failures.append("invalid_k_slab")
    if type(c.denominator) is not int or c.denominator <= 0:
        failures.append("invalid_denominator")
    if not c.planes:
        failures.append("missing_product_planes")
    if c.prior_min > c.prior_max or c.recombination_min > c.recombination_max:
        failures.append("invalid_intermediate_interval")
    if instruction.operand_min > instruction.operand_max or instruction.accumulator_min > instruction.accumulator_max:
        failures.append("invalid_instruction_interval")
    if not instruction.exact_integer_products:
        failures.append("instruction_exactness_unproven")
    for plane in c.planes:
        if len(plane.lhs) != c.modulus or len(plane.rhs) != c.modulus:
            failures.append("encoding_table_length")
        if type(plane.weight) is not int or any(type(x) is not int for x in plane.lhs + plane.rhs):
            failures.append("encoding_not_integral")
    if failures:
        return _result(c, failures, [])

    intervals = []
    running_min, running_max = c.prior_min, c.prior_max
    for index, plane in enumerate(c.planes):
        if any(x < instruction.operand_min or x > instruction.operand_max for x in plane.lhs + plane.rhs):
            failures.append(f"plane_{index}_operand_alphabet")
        if instruction.operand_alphabet and any(x not in instruction.operand_alphabet for x in plane.lhs + plane.rhs):
            failures.append(f"plane_{index}_stored_alphabet")
        lo, hi = _product_interval((min(plane.lhs), max(plane.lhs)), (min(plane.rhs), max(plane.rhs)))
        lo, hi = lo * c.safe_k, hi * c.safe_k
        if lo + c.prior_min < instruction.accumulator_min or hi + c.prior_max > instruction.accumulator_max:
            failures.append(f"plane_{index}_accumulator_overflow")
        weighted = sorted((lo * plane.weight, hi * plane.weight))
        if weighted[0] < c.recombination_min or weighted[1] > c.recombination_max:
            failures.append(f"plane_{index}_coefficient_overflow")
        running_min += weighted[0]
        running_max += weighted[1]
        if running_min < c.recombination_min or running_max > c.recombination_max:
            failures.append(f"plane_{index}_ordered_recombination_overflow")
        intervals.append({"plane": index, "product_sum": [lo, hi],
                          "ordered_recombination": [running_min, running_max]})

    # Exhaust all byte-domain pairs, independently from interval proofs.
    for a in range(c.modulus):
        for b in range(c.modulus):
            numerator = sum(p.weight * p.lhs[a] * p.rhs[b] for p in c.planes)
            decoded = Fraction(numerator, c.denominator)
            if decoded.denominator != 1:
                failures.append("nonintegral_recombination")
                break
            if decoded.numerator % c.modulus != a * b % c.modulus:
                failures.append("encoding_product_identity")
                break
        if failures and failures[-1] in {"nonintegral_recombination", "encoding_product_identity"}:
            break

    product = 1
    for modulus in c.basis:
        if type(modulus) is not int or not 2 <= modulus <= 256 or math.gcd(product, modulus) != 1:
            failures.append("basis_not_pairwise_coprime_byte_moduli")
            break
        product *= modulus
    if c.semantics in {"bounded_signed", "bounded_unsigned"}:
        if type(c.output_bound) is not int or c.output_bound < 0 or not c.basis:
            failures.append("missing_exact_range_contract")
        elif product <= c.output_bound * (2 if c.semantics == "bounded_signed" else 1):
            failures.append("insufficient_crt_range")
    elif c.semantics == "wrap64":
        failures.append("wrap64_requires_byte_limb_composition_certificate")
    elif c.semantics == "finite_field":
        if any(c.modulus % d == 0 for d in range(2, math.isqrt(c.modulus) + 1)):
            failures.append("field_modulus_not_prime")
    elif c.semantics != "finite_ring":
        failures.append("unknown_semantics")
    return _result(c, failures, intervals)


def _result(c: ArithmeticCertificate, failures: list[str], intervals: list[dict]) -> dict:
    return {"schema": "rns8_arithmetic_certificate_v1", "sha256": certificate_hash(c),
            "algebraically_verified": not failures, "proof_failures": sorted(set(failures)),
            "intervals": intervals, "simulator_verified": False,
            # An evidence string is a reference, not permission to self-certify.
            "instruction_measured": False, "end_to_end_qualified": False,
            "runtime_dispatch_eligible": False, "certificate": asdict(c)}


def centered_int8(modulus: int, safe_k: int = 65536) -> ArithmeticCertificate:
    values = tuple(x - modulus if x >= (modulus + 1) // 2 else x for x in range(modulus))
    instruction = InstructionContract("host-algebra-only", "signed_i8_product_i32_sum", -128, 127,
                                      -(1 << 31), (1 << 31) - 1, "unqualified")
    return ArithmeticCertificate(modulus, (ProductPlane(values, values),), 1, safe_k, 0, 0,
                                 -(1 << 63), (1 << 63) - 1, instruction)


def from_json(data: dict) -> ArithmeticCertificate:
    data = dict(data)
    data["planes"] = tuple(ProductPlane(tuple(p["lhs"]), tuple(p["rhs"]), p.get("weight", 1)) for p in data["planes"])
    instruction = dict(data["instruction"])
    instruction["flags"] = tuple(instruction.get("flags", ()))
    instruction["operand_alphabet"] = tuple(instruction.get("operand_alphabet", ()))
    data["instruction"] = InstructionContract(**instruction)
    data["basis"] = tuple(data.get("basis", ()))
    return ArithmeticCertificate(**data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(from_json(json.loads(args.certificate.read_text(encoding="utf-8"))))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if result["algebraically_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
