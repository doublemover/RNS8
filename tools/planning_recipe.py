"""Typed offline provider boundary. No launches, cache writes or route promotion.

Reuse the arithmetic verifier and calibrated critical-path predictor. Providers
describe concrete kernels; the central gate checks identity, complete work and
bounded resources before costs can rank a recipe. Runtime binding is separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Protocol

from analytical_planner import Rate, Stage, predict
from arithmetic_certificate import ArithmeticCertificate, certificate_hash, verify

U64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class ProblemContract:
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    semantics: str
    basis: tuple[int, ...]
    output_boundary: str
    expression: str = "A*B"
    reuse_a: int = 1
    reuse_b: int = 1


@dataclass(frozen=True)
class ValueFacts:
    name: str
    instance: int
    generation: int
    max_magnitude: int
    materializations: tuple[str, ...]


@dataclass(frozen=True)
class KernelCapability:
    symbol: str
    certificate_sha256: str
    opcode: str
    layout: str
    instruction_evidence_sha256: str
    output_boundary: str = "resident_residues"


@dataclass(frozen=True)
class TargetCapabilities:
    target: str
    compiler: str
    artifact_sha256: str
    kernels: tuple[KernelCapability, ...]
    address_limit: int = U64_MAX


@dataclass(frozen=True)
class CostProfile:
    target: str
    compiler: str
    artifact_sha256: str
    rates: tuple[tuple[str, Rate], ...]
    kernel_symbols: tuple[str, ...]


@dataclass(frozen=True)
class TemporaryRegion:
    name: str
    producer: str
    consumers: tuple[str, ...]
    bytes: int
    alignment: int
    retained: bool = False


@dataclass(frozen=True)
class ResourcePlan:
    status: str  # success, unsupported, overflow, unknown
    regions: tuple[TemporaryRegion, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class CandidateRecipe:
    name: str
    contract: ProblemContract
    inputs: tuple[ValueFacts, ...]
    certificates: tuple[ArithmeticCertificate, ...]
    stages: tuple[Stage, ...]
    layout: str
    tile_m: int
    tile_n: int
    k_slab: int
    # Disjoint half-open flattened tile/plane/slab ranges. Arrival count alone
    # cannot establish that every required unit of work is covered.
    coverage: tuple[tuple[int, int], ...]
    resources: ResourcePlan
    required_materializations: tuple[str, str] = ("native_i64", "native_i64")


class Provider(Protocol):
    def enumerate(self, problem: ProblemContract, facts: tuple[ValueFacts, ...]) -> tuple[CandidateRecipe, ...]: ...


def _uint(value: object, positive: bool = False) -> bool:
    return type(value) is int and (1 if positive else 0) <= value <= U64_MAX


def _digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _identity(value: object) -> str:
    return hashlib.sha256(json.dumps(asdict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _facts(facts: tuple[ValueFacts, ...]) -> dict[str, ValueFacts]:
    if type(facts) is not tuple or any(type(f) is not ValueFacts for f in facts):
        raise ValueError("invalid value facts")
    by_name = {f.name: f for f in facts}
    if set(by_name) != {"A", "B"} or len(facts) != 2:
        raise ValueError("exactly one A and B fact is required")
    for f in facts:
        if (not _uint(f.instance, True) or not _uint(f.generation, True)
                or type(f.max_magnitude) is not int or f.max_magnitude < 0
                or type(f.materializations) is not tuple or not f.materializations
                or any(type(x) is not str or not x for x in f.materializations)
                or len(set(f.materializations)) != len(f.materializations)):
            raise ValueError("invalid generation/range/materialization facts")
    return by_name


def _problem(problem: ProblemContract) -> None:
    if type(problem) is not ProblemContract or any(not _uint(x, True) for x in
            (problem.m, problem.n, problem.k, problem.lda, problem.ldb, problem.ldc, problem.reuse_a, problem.reuse_b)):
        raise ValueError("invalid problem dimensions or reuse")
    if problem.lda < problem.k or problem.ldb < problem.n or problem.ldc < problem.n:
        raise ValueError("invalid problem strides")
    if problem.expression != "A*B":
        raise ValueError("unsupported expression; epilogues require a composed recipe")
    if problem.semantics not in {"finite_ring", "finite_field", "bounded_signed", "bounded_unsigned"}:
        raise ValueError("unsupported arithmetic semantics")
    if problem.output_boundary not in {"resident_residues", "native_integer"}:
        raise ValueError("unsupported output boundary")
    if (type(problem.basis) is not tuple or not problem.basis or len(problem.basis) > 20
            or any(type(p) is not int or not 2 <= p <= 256 for p in problem.basis)):
        raise ValueError("invalid ordered byte basis")
    product = 1
    for p in problem.basis:
        if math.gcd(product, p) != 1:
            raise ValueError("noncoprime basis")
        product *= p
    if problem.semantics in {"finite_ring", "finite_field"} and len(problem.basis) != 1:
        raise ValueError("finite semantics require one named modulus")


def checked_resources(plan: ResourcePlan, stages: tuple[Stage, ...], budget: int) -> dict:
    if type(plan) is not ResourcePlan or plan.status not in {"success", "unsupported", "overflow", "unknown"}:
        return {"status": "unknown", "reason": "invalid_workspace_query"}
    if plan.status != "success":
        return {"status": plan.status, "reason": plan.reason}
    if not _uint(budget):
        return {"status": "overflow", "reason": "invalid_budget"}
    names = {s.name for s in stages}
    stages_by_name = {s.name: s for s in stages}
    if len(names) != len(stages) or any(s.bytes_allocated for s in stages):
        return {"status": "unknown", "reason": "duplicate_stage_or_unowned_allocation"}
    seen, offsets, cursor = set(), [], 0
    allocation = {s.name: 0 for s in stages}
    retained = set()
    for region in plan.regions:
        if (type(region) is not TemporaryRegion or not region.name or region.name in seen
                or not _uint(region.bytes) or not _uint(region.alignment, True)
                or region.alignment & (region.alignment - 1) or type(region.retained) is not bool
                or region.producer not in names or len(set(region.consumers)) != len(region.consumers)
                or any(c not in names or region.producer not in stages_by_name[c].dependencies for c in region.consumers)):
            return {"status": "unknown", "reason": "invalid_region_or_lifetime"}
        seen.add(region.name)
        offset = (cursor + region.alignment - 1) & -region.alignment
        end = offset + region.bytes
        if end > U64_MAX:
            return {"status": "overflow", "reason": "aligned_workspace_overflow"}
        # A nonoverlapping arena is intentionally conservative. Padding belongs
        # to its producer too; it cannot disappear from the resource accounting.
        allocation[region.producer] += end - cursor
        offsets.append({"name": region.name, "offset": offset, "bytes": region.bytes})
        cursor = end
        if region.retained:
            retained.add(region.producer)
    if cursor > budget:
        return {"status": "unsupported", "reason": "workspace_budget", "required_bytes": cursor}
    priced_stages = tuple(replace(s, bytes_allocated=allocation[s.name], retain_output=s.retain_output or s.name in retained) for s in stages)
    return {"status": "success", "required_bytes": cursor, "regions": offsets, "stages": priced_stages}


def admit(recipe: CandidateRecipe, problem: ProblemContract, facts: tuple[ValueFacts, ...],
          target: TargetCapabilities, budget: int) -> dict:
    """Costs cannot affect mathematical, capability or resource admissibility."""
    _problem(problem)
    values = _facts(facts)
    if not recipe.name or recipe.contract != problem or recipe.inputs != facts:
        raise ValueError("recipe contract or generation binding mismatch")
    if not recipe.layout or any(not _uint(x, True) for x in (recipe.tile_m, recipe.tile_n, recipe.k_slab)):
        raise ValueError("invalid recipe tile or slab")
    if tuple(c.modulus for c in recipe.certificates) != problem.basis:
        raise ValueError("certificate coverage/order differs from basis")
    bound = problem.k * values["A"].max_magnitude * values["B"].max_magnitude
    if (len(recipe.required_materializations) != 2 or any(required not in values[name].materializations
            for name, required in zip(("A", "B"), recipe.required_materializations))):
        raise ValueError("required input materialization is unavailable")
    for certificate in recipe.certificates:
        proof = verify(certificate)
        if not proof["algebraically_verified"]:
            raise ValueError(f"arithmetic: {proof['proof_failures']}")
        if certificate.semantics != problem.semantics or recipe.k_slab > certificate.safe_k:
            raise ValueError("certificate semantics or safe slab mismatch")
        if problem.semantics.startswith("bounded_") and (certificate.basis != problem.basis or certificate.output_bound < bound):
            raise ValueError("insufficient generation-bound integer range")
        if certificate.instruction.target != target.target or certificate.instruction.compiler != target.compiler:
            raise ValueError("instruction target/compiler mismatch")
    total = ((problem.m + recipe.tile_m - 1) // recipe.tile_m
             * ((problem.n + recipe.tile_n - 1) // recipe.tile_n)
             * ((problem.k + recipe.k_slab - 1) // recipe.k_slab) * len(problem.basis))
    if total > U64_MAX:
        raise ValueError("work coverage overflow")
    end = 0
    for first, last in sorted(recipe.coverage):
        if not _uint(first) or not _uint(last) or first != end or last <= first:
            raise ValueError("missing or duplicate work coverage")
        end = last
    if end != total:
        raise ValueError("incomplete work coverage")
    if not _digest(target.artifact_sha256) or not _uint(target.address_limit, True):
        raise ValueError("unknown target artifact or address domain")
    for rows, cols, ld in ((problem.m, problem.k, problem.lda), (problem.k, problem.n, problem.ldb), (problem.m, problem.n, problem.ldc)):
        if ((rows - 1) * ld + cols) * 8 > target.address_limit:
            raise ValueError("address domain overflow")
    capabilities = {(c.symbol, c.certificate_sha256, c.layout): c for c in target.kernels}
    if len(capabilities) != len(target.kernels):
        raise ValueError("ambiguous duplicate kernel capability")
    consumed_stages = {name for stage in recipe.stages for name in stage.dependencies}
    for stage in recipe.stages:
        for certificate in recipe.certificates:
            capability = capabilities.get((stage.kernel_symbol, certificate_hash(certificate), recipe.layout))
            if (capability is None or capability.opcode != certificate.instruction.opcode
                    or not _digest(capability.instruction_evidence_sha256)
                    or capability.instruction_evidence_sha256 != certificate.instruction.measured_evidence_sha256):
                raise ValueError("unsupported or unmeasured kernel/certificate/layout combination")
            if capability.output_boundary not in {"resident_residues", "native_integer"}:
                raise ValueError("unknown kernel output boundary")
            if stage.name not in consumed_stages and capability.output_boundary != problem.output_boundary:
                raise ValueError("terminal kernel output boundary differs from requested result")
    resources = checked_resources(recipe.resources, recipe.stages, budget)
    if resources["status"] != "success":
        raise ValueError(f"workspace {resources['status']}: {resources['reason']}")
    return resources


def shortlist(problem: ProblemContract, facts: tuple[ValueFacts, ...], target: TargetCapabilities,
              profile: CostProfile, providers: tuple[Provider, ...], budget: int) -> dict:
    _problem(problem)
    _facts(facts)
    candidates, rejected, seen = [], [], set()
    for provider in providers:
        for recipe in provider.enumerate(problem, facts):
            if recipe.name in seen:
                raise ValueError("ambiguous duplicate recipe identity")
            seen.add(recipe.name)
            try:
                resources = admit(recipe, problem, facts, target, budget)
                if (profile.target, profile.compiler, profile.artifact_sha256) != (target.target, target.compiler, target.artifact_sha256):
                    raise ValueError("cost environment mismatch")
                if any(s.kernel_symbol not in profile.kernel_symbols for s in recipe.stages):
                    raise ValueError("missing kernel calibration")
                if len(dict(profile.rates)) != len(profile.rates):
                    raise ValueError("duplicate calibrated work term")
                prediction = predict(resources.pop("stages"), dict(profile.rates), recipe.certificates[0], budget, target.target)
                candidates.append({"name": recipe.name, "recipe_sha256": _identity(recipe),
                                   "prediction": prediction, "resources": resources})
            except ValueError as error:
                rejected.append({"name": recipe.name, "reason": str(error)})
    candidates.sort(key=lambda c: (c["prediction"]["latency_upper_ns"], c["resources"]["required_bytes"], c["name"]))
    return {"schema": "rns8_recipe_shortlist_v1", "problem_sha256": _identity(problem),
            "input_bindings": [dict(asdict(f), materializations=list(f.materializations)) for f in facts], "candidates": candidates,
            "rejected": sorted(rejected, key=lambda x: x["name"]), "promotion_eligible": False,
            "runtime_route_changed": False}


def bind_current(recipe: CandidateRecipe, current: tuple[ValueFacts, ...]) -> None:
    _facts(current)
    if recipe.inputs != current:
        raise ValueError("value identity, generation, proof or materialization changed; replan")


def profile_from_calibrations(calibrations: tuple, expected: dict) -> CostProfile:
    """Use the existing independent-run/ISA/reference evidence admission gate."""
    from calibration_manifest import planner_rates
    rates = planner_rates(calibrations, expected)
    identity = calibrations[0].identity
    return CostProfile(identity.target, identity.compiler, identity.artifact_sha256,
                       tuple(sorted(rates.items())), tuple(sorted({c.identity.kernel_symbol for c in calibrations})))
