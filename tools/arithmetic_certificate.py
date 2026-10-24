"""Exact, offline arithmetic certificates; never qualifies an unmeasured opcode.

Encoding tables are indexed by canonical input residue. Integer/rational checking
is deliberately independent of device reducers and generated kernel arithmetic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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


def _schema_failures(c: ArithmeticCertificate) -> list[str]:
    if type(c) is not ArithmeticCertificate or type(c.instruction) is not InstructionContract:
        return ["invalid_certificate_structure"]
    i = c.instruction
    integers = (c.modulus, c.denominator, c.safe_k, c.prior_min, c.prior_max,
                c.recombination_min, c.recombination_max, i.operand_min,
                i.operand_max, i.accumulator_min, i.accumulator_max)
    failures = []
    if any(type(x) is not int for x in integers):
        failures.append("noninteger_proof_endpoint_or_parameter")
    if type(i.exact_integer_products) is not bool:
        failures.append("invalid_exactness_boolean")
    if any(type(x) is not str or not x for x in (i.target, i.opcode, i.compiler, c.semantics)):
        failures.append("invalid_proof_identity")
    if type(i.flags) is not tuple or any(type(x) is not str or not x for x in i.flags):
        failures.append("invalid_compiler_flags")
    digest = i.measured_evidence_sha256
    if digest is not None and (type(digest) is not str or len(digest) != 64
                               or any(x not in "0123456789abcdef" for x in digest)):
        failures.append("invalid_evidence_digest")
    if (type(c.basis) is not tuple or any(type(x) is not int for x in c.basis)
            or type(i.operand_alphabet) is not tuple
            or any(type(x) is not int for x in i.operand_alphabet)):
        failures.append("invalid_integer_alphabet_or_basis")
    if c.output_bound is not None and type(c.output_bound) is not int:
        failures.append("noninteger_output_bound")
    if type(c.planes) is not tuple or any(
            type(p) is not ProductPlane or type(p.lhs) is not tuple or type(p.rhs) is not tuple
            or type(p.weight) is not int or any(type(x) is not int for x in p.lhs + p.rhs)
            for p in c.planes):
        failures.append("invalid_product_plane_structure")
    return failures


def certificate_hash(certificate: ArithmeticCertificate) -> str:
    if _schema_failures(certificate):
        raise ValueError("cannot hash a malformed arithmetic certificate")
    payload = json.dumps({"checker_version": "integer-certificate-v2", "certificate": asdict(certificate)}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _product_interval(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    products = [x * y for x in a for y in b]
    return min(products), max(products)


def verify(c: ArithmeticCertificate) -> dict:
    malformed = _schema_failures(c)
    if malformed:
        return _result(c, malformed, [], malformed=True)
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
            # A common positive denominator permits exact integer division;
            # no rational allocation or gcd is needed for every byte pair.
            decoded, remainder = divmod(numerator, c.denominator)
            if remainder:
                failures.append("nonintegral_recombination")
                break
            if decoded % c.modulus != a * b % c.modulus:
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


def _result(c: ArithmeticCertificate, failures: list[str], intervals: list[dict], malformed: bool = False) -> dict:
    return {"schema": "rns8_arithmetic_certificate_v1", "sha256": None if malformed else certificate_hash(c),
            "checker_version": "integer-certificate-v2",
            "verdicts": {"algebraic_value_equality": "proven" if not failures else "unknown",
                         "floating_rewrite_equivalence": "unsupported",
                         "finite_exact_recoverability": "proven" if not failures else "unknown",
                         "physical_layout_correctness": "not_run", "target_execution": "not_run"},
            "verdict_scope": "declared-integer-model-and-bound-input-contract",
            "algebraically_verified": not failures, "proof_failures": sorted(set(failures)),
            "intervals": intervals, "simulator_verified": False,
            # An evidence string is a reference, not permission to self-certify.
            "instruction_measured": False, "end_to_end_qualified": False,
            "runtime_dispatch_eligible": False, "certificate": None if malformed else asdict(c)}


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
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    try:
        data = json.loads(args.certificate.read_text(encoding="utf-8"), parse_constant=reject_constant)
        if data.get("schema") == "rns8_arithmetic_composition_v1":
            from arithmetic_composition import Node, verify_composition
            result = verify_composition(tuple(Node(**node) for node in data["nodes"]))
        else:
            result = verify(from_json(data))
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        result = {"algebraically_verified": False, "runtime_dispatch_eligible": False,
                  "sha256": None, "proof_failures": ["invalid_certificate_input"], "error": str(error)}
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if result["algebraically_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
