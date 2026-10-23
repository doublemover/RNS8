"""CPU mock-provider fixtures; fabricated identities are never hardware evidence."""
from dataclasses import replace
from fractions import Fraction
import json

from analytical_planner import Rate, Stage, predict
from arithmetic_certificate import centered_int8, certificate_hash
from planning_recipe import (ProblemContract, ValueFacts, KernelCapability, TargetCapabilities,
                             CostProfile, TemporaryRegion, ResourcePlan, CandidateRecipe,
                             checked_resources, shortlist, bind_current, U64_MAX)


class MockProvider:
    def __init__(self, recipes):
        self.recipes = recipes

    def enumerate(self, problem, facts):
        return self.recipes


def main() -> int:
    problem = ProblemContract(17, 19, 33, 34, 21, 22, "finite_ring", (13,), "resident_residues")
    facts = (ValueFacts("A", 1, 4, 12, ("native_i64",)), ValueFacts("B", 2, 7, 12, ("native_i64",)))
    certificate = centered_int8(13)
    certificate = replace(certificate, instruction=replace(certificate.instruction,
        target="mock-target", compiler="mock-compiler", measured_evidence_sha256="a" * 64))
    stages = (Stage("compute", "mock_kernel", (), ("compute",), (("matrix", 100),), 0),)
    recipe = CandidateRecipe("matrix", problem, facts, (certificate,), stages, "plane_major", 16, 16, 16,
                             ((0, 12),), ResourcePlan("success", (TemporaryRegion("partial", "compute", (), 31, 16),)))
    other = replace(recipe, name="vector", stages=(replace(stages[0], work=(("vector", 100),)),))
    capability = KernelCapability("mock_kernel", certificate_hash(certificate), certificate.instruction.opcode, "plane_major", "a" * 64)
    target = TargetCapabilities("mock-target", "mock-compiler", "b" * 64, (capability,))
    rate = Rate(Fraction(1), Fraction(1), "operations", "mock-cohort")
    profile = CostProfile(target.target, target.compiler, target.artifact_sha256,
                          (("matrix", rate), ("vector", replace(rate, lower_units_per_ns=Fraction(2), upper_units_per_ns=Fraction(2)))), ("mock_kernel",))
    provider = MockProvider((recipe, other))
    result = shortlist(problem, facts, target, profile, (provider,), 31)
    assert [c["name"] for c in result["candidates"]] == ["vector", "matrix"]
    assert result == shortlist(problem, facts, target, profile, (provider,), 31)
    assert result == json.loads(json.dumps(result))
    assert not result["promotion_eligible"] and not result["runtime_route_changed"]
    faster_matrix = replace(profile, rates=(("matrix", replace(rate, lower_units_per_ns=Fraction(4), upper_units_per_ns=Fraction(4))), ("vector", rate)))
    changed = shortlist(problem, facts, target, faster_matrix, (provider,), 31)
    assert [c["name"] for c in changed["candidates"]] == ["matrix", "vector"]
    assert not changed["rejected"]  # Costs reorder, never change legal arithmetic.

    def rejects(broken, reason, *, cap=target, cost=profile, budget=31):
        output = shortlist(problem, facts, cap, cost, (MockProvider((broken,)),), budget)
        assert not output["candidates"], output
        assert reason in output["rejected"][0]["reason"], output

    rejects(replace(recipe, coverage=((0, 6), (7, 12))), "work coverage")
    rejects(replace(recipe, coverage=((0, 7), (6, 12))), "work coverage")
    rejects(replace(recipe, coverage=((0, 11),)), "work coverage")
    rejects(replace(recipe, k_slab=certificate.safe_k + 1), "safe slab")
    rejects(replace(recipe, inputs=(replace(facts[0], generation=5), facts[1])), "binding")
    rejects(replace(recipe, required_materializations=("packed", "native_i64")), "materialization")
    rejects(recipe, "unsupported", cap=replace(target, kernels=()))
    rejects(recipe, "unsupported", cap=replace(target, kernels=(replace(capability, instruction_evidence_sha256=""),)))
    rejects(recipe, "output boundary", cap=replace(target, kernels=(replace(capability, output_boundary="native_integer"),)))
    rejects(recipe, "duplicate kernel", cap=replace(target, kernels=(capability, capability)))
    native_problem = replace(problem, output_boundary="native_integer")
    native_recipe = replace(recipe, contract=native_problem)
    missing_export = shortlist(native_problem, facts, target, profile, (MockProvider((native_recipe,)),), 31)
    assert not missing_export["candidates"]
    assert "output boundary" in missing_export["rejected"][0]["reason"]
    # A separately attested terminal output policy can admit the native result;
    # changing only the requested label or cost cannot invent reconstruction.
    native_target = replace(target, kernels=(replace(capability, output_boundary="native_integer"),))
    assert shortlist(native_problem, facts, native_target, profile, (MockProvider((native_recipe,)),), 31)["candidates"]
    rejects(recipe, "target/compiler", cap=replace(target, target="different-target"))
    rejects(recipe, "cost environment", cost=replace(profile, artifact_sha256="c" * 64))
    rejects(recipe, "missing kernel calibration", cost=replace(profile, kernel_symbols=()))
    rejects(recipe, "missing or invalid calibrated", cost=replace(profile, rates=()))
    rejects(recipe, "workspace_budget", budget=30)
    rejects(recipe, "address domain", cap=replace(target, address_limit=1))
    rejects(replace(recipe, certificates=(replace(certificate, denominator=2),)), "arithmetic")
    for status in ("unsupported", "unknown", "overflow"):
        rejects(replace(recipe, resources=ResourcePlan(status, reason="backend query failed")), status)
    small = replace(other, resources=ResourcePlan("success"))
    limited = shortlist(problem, facts, target, profile, (MockProvider((recipe, small)),), 0)
    assert [c["name"] for c in limited["candidates"]] == ["vector"]
    assert limited["candidates"][0]["resources"]["required_bytes"] == 0
    bind_current(recipe, facts)
    for current in ((replace(facts[0], generation=5), facts[1]),
                    (replace(facts[0], instance=3), facts[1]),
                    (replace(facts[0], max_magnitude=13), facts[1])):
        try:
            bind_current(recipe, current)
        except ValueError:
            pass
        else:
            raise AssertionError("stale binding accepted")
    regions = (TemporaryRegion("a", "compute", (), 3, 1), TemporaryRegion("b", "compute", (), 5, 16))
    resources = checked_resources(ResourcePlan("success", regions), stages, 21)
    assert resources["required_bytes"] == 21 and resources["regions"][1]["offset"] == 16
    assert checked_resources(ResourcePlan("success", regions), stages, 20)["status"] == "unsupported"
    overflow = (replace(regions[0], bytes=U64_MAX), regions[1])
    assert checked_resources(ResourcePlan("success", overflow), stages, U64_MAX)["status"] == "overflow"
    assert checked_resources(ResourcePlan("success", (replace(regions[0], alignment=3),)), stages, 100)["status"] == "unknown"
    assert checked_resources(ResourcePlan("success", (replace(regions[0], consumers=("absent",)),)), stages, 100)["status"] == "unknown"
    # An internally valid CRT certificate must still cover the actual current
    # operand bound; changing a cost profile cannot repair insufficient range.
    bounded_problem = replace(problem, semantics="bounded_signed")
    bounded_cert = replace(certificate, semantics="bounded_signed", basis=(13,), output_bound=6)
    bounded_facts = (replace(facts[0], max_magnitude=1), replace(facts[1], max_magnitude=1))
    bounded_recipe = replace(recipe, contract=bounded_problem, inputs=bounded_facts, certificates=(bounded_cert,))
    output = shortlist(bounded_problem, bounded_facts, target, profile, (MockProvider((bounded_recipe,)),), 31)
    assert "generation-bound integer range" in output["rejected"][0]["reason"]
    for field, value in (("launch_ns", True), ("launch_ns", float("nan")), ("tiles", 1.5), ("bytes_allocated", False)):
        try:
            predict((replace(stages[0], **{field: value}),), dict(profile.rates), certificate, 100, target.target)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed stage admitted")
    print("typed recipe provider self-test: PASS (mock targets only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
