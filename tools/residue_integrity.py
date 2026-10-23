"""Offline redundant-residue oracle. No runtime publication or fault correction."""
from __future__ import annotations

from dataclasses import dataclass
import math

from crt_research import balanced, make_plan


@dataclass(frozen=True)
class Identity:
    operation: str
    inputs: tuple[tuple[str, int], ...]
    epoch: int
    basis: tuple[int, ...]


@dataclass(frozen=True)
class CheckRecord:
    identity: Identity
    moduli: tuple[int, ...]
    residues: tuple[int, ...]
    input_path: str


def _validate_identity(identity: Identity) -> None:
    make_plan(identity.basis)
    if not identity.operation or type(identity.epoch) is not int or identity.epoch < 0:
        raise ValueError("operation and nonnegative epoch required")
    if not identity.inputs or any(not name or type(generation) is not int or generation < 0
                                 for name, generation in identity.inputs):
        raise ValueError("explicit input instance/generation identity required")


def _check_basis(basis: tuple[int, ...], checks: tuple[int, ...]) -> None:
    make_plan(basis + checks)
    if not checks:
        raise ValueError("no independent check channels remain")


def independent_dot_check(identity: Identity, a: tuple[int, ...], b: tuple[int, ...],
                          checks: tuple[int, ...]) -> CheckRecord:
    """Compute from original integers, never from a reconstructed candidate."""
    _validate_identity(identity)
    _check_basis(identity.basis, checks)
    if len(a) != len(b) or not a or any(type(x) is not int for x in a + b):
        raise ValueError("nonempty equal-sized integer operands required")
    residues = tuple(sum((x % q) * (y % q) for x, y in zip(a, b)) % q for q in checks)
    return CheckRecord(identity, checks, residues, "original_integer_dot")


def check_candidate(identity: Identity, residues: tuple[int, ...], check: CheckRecord,
                    *, signed: bool, magnitude_bound: int) -> dict:
    _validate_identity(identity)
    plan = make_plan(identity.basis)
    if type(signed) is not bool or type(magnitude_bound) is not int or magnitude_bound < 0:
        raise ValueError("explicit signedness and nonnegative exact bound required")
    if plan.product <= magnitude_bound * (2 if signed else 1):
        raise ValueError("computational basis alone does not prove uniqueness")
    failures = []
    if check.identity != identity:
        failures.append("stale_or_different_operation_identity")
    try:
        _check_basis(identity.basis, check.moduli)
    except ValueError:
        failures.append("check_basis_not_independent")
    if (len(check.residues) != len(check.moduli)
            or any(type(r) is not int or type(q) is not int or q < 2 or not 0 <= r < q
                   for r, q in zip(check.residues, check.moduli))):
        failures.append("invalid_check_residues")
    if check.input_path != "original_integer_dot":
        failures.append("independent_input_path_required")
    value = balanced(plan, residues)
    if signed and 2 * value >= plan.product:
        value -= plan.product
    if abs(value) > magnitude_bound:
        failures.append("candidate_outside_certified_bound")
    mismatches = [] if failures else [q for q, r in zip(check.moduli, check.residues) if value % q != r]
    if mismatches:
        failures.append("check_residue_mismatch")
    return {"status": "failed_integrity" if failures else "consistent_under_declared_checks",
            "candidate": value, "failures": failures, "mismatched_moduli": mismatches,
            "computational_range_product": plan.product,
            "check_product": math.prod(check.moduli) if not failures else None,
            "runtime_publication_authorized": False, "correction_attempted": False,
            "coverage": "Detects errors not divisible by all check moduli when inputs, identities and the check path are correct.",
            "limitations": ["Multiples of every check modulus are invisible.",
                            "Common erroneous input conversion can corrupt both paths coherently.",
                            "Identity metadata cannot authenticate an incorrectly labelled value.",
                            "No probabilistic or arbitrary-fault guarantee is claimed."]}
