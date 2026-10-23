"""Brute-force cross-check of bounded weighted scheme search."""
import itertools
import math

from arithmetic_certificate import centered_int8
from modulus_scheme_optimizer import Form, optimize
from joint_scheme_optimizer import ScheduledForm, arithmetic_limits, joint_optimize
from dataclasses import replace


def joint_checks():
    # Independent labelled-bin enumeration deliberately includes duplicate
    # partitions; compare minima, not the optimizer's enumeration mechanics.
    forms = [ScheduledForm(Form(centered_int8(m, cap), 1, work, 1, 1, "toy"), "planar", 2, 4)
             for m, cap, work in ((4, 2, 1), (5, 4, 3), (7, 8, 2))]
    options = dict(slabs=(1, 2, 4, 8), max_group_planes=2, register_budget=4,
                   lds_budget=8, group_launch_ps=11, prefix_bit_ps=3)
    for bound in (0, 10, 50, 1000):
        for k in (1, 9):
            costs = []
            for count in range(1, len(forms) + 1):
                for order in itertools.permutations(range(len(forms)), count):
                    moduli = [forms[i].form.certificate.modulus for i in order]
                    if math.prod(moduli) <= 2 * bound or any(math.gcd(a, b) != 1 for a, b in itertools.combinations(moduli, 2)):
                        continue
                    prefix, prefix_cost = 1, 0
                    for m in moduli:
                        prefix *= m
                        prefix_cost += prefix.bit_length() * 3
                    for assignment in itertools.product(range(count), repeat=count):
                        groups = [tuple(order[j] for j in range(count) if assignment[j] == label)
                                  for label in range(count)]
                        groups = [g for g in groups if g]
                        if any(len(g) > 2 for g in groups):
                            continue
                        for slabs in itertools.product((1, 2, 4, 8), repeat=len(groups)):
                            if any(s > min(forms[i].form.certificate.safe_k for i in g) for s, g in zip(slabs, groups)):
                                continue
                            costs.append(prefix_cost + 2 * count + sum(
                                ((k + s - 1) // s) * (11 + sum(forms[i].form.slab_ps + 1 for i in g))
                                for g, s in zip(groups, slabs)))
            result = joint_optimize(forms, bound, True, k, **options)
            assert result["search_complete"]
            assert (result["best"]["cost_ps"] if result["best"] else None) == (min(costs) if costs else None)
            if result["best"]:
                assert result["lower_bound_ps"] == min(costs)
                schedule = result["best"]["schedule"]
                assert sorted(i for g in schedule for i in g["plane_ids"]) == list(range(len(result["best"]["ordered_moduli"])))
    limited = joint_optimize(forms, 10, False, 9, max_nodes=1, **options)
    assert not limited["search_complete"] and limited["lower_bound_ps"] == 0
    # Isolated cheapest form loses when its layout prevents sharing a launch.
    base = Form(centered_int8(5), 0, 1, 0, 0, "toy")
    alternative = replace(base, certificate=replace(base.certificate, safe_k=65535), slab_ps=2)
    coupled = [ScheduledForm(base, "different", 1, 0), ScheduledForm(alternative, "same", 1, 0),
               ScheduledForm(replace(base, certificate=centered_int8(7)), "same", 1, 0)]
    options.update(slabs=(1,), group_launch_ps=100, prefix_bit_ps=0)
    full = joint_optimize(coupled, 17, False, 1, **options)
    staged = joint_optimize(coupled, 17, False, 1, allowed_form_ids=(0, 2), **options)
    assert full["best"]["cost_ps"] == 103
    assert staged["best"]["cost_ps"] == 202
    assert set(full["best"]["ordered_form_ids"]) == {1, 2}
    # Intermediate arithmetic caps are separate from final range; reducing a
    # slab never admits a basis whose full product is insufficient.
    narrow = replace(centered_int8(5, 2), recombination_min=-10, recombination_max=10)
    limits = arithmetic_limits(narrow)
    assert limits["epilogue_composition_cap"] == 2
    assert limits["dot_accumulation_cap"] > 2
    assert limits["certified_slab_cap"] == 2


def main() -> int:
    joint_checks()
    forms = [Form(centered_int8(m), i + 1, 2 * i + 1, 1, 3, "synthetic-fixture-only")
             for i, m in enumerate((4, 5, 7, 9, 11))]
    for bound in (0, 8, 100, 1000, 10000):
        for signed in (False, True):
            possible = []
            for count in range(1, len(forms) + 1):
                for subset in itertools.combinations(forms, count):
                    moduli = [form.certificate.modulus for form in subset]
                    if all(math.gcd(a, b) == 1 for a, b in itertools.combinations(moduli, 2)):
                        if math.prod(moduli) > bound * (2 if signed else 1):
                            possible.append(sum(f.cost(65537) for f in subset))
            report = optimize(forms, bound, signed, 65537)
            assert report["search_complete"]
            assert report["feasible_upper_bound_ps"] == (min(possible) if possible else None)
            assert report["feasible"] == bool(possible)
    limited = optimize(forms, 100, False, 1, max_nodes=1)
    assert not limited["search_complete"]
    assert limited["lower_bound_ps"] == 0
    assert not limited["promotion_eligible"]
    print("modulus scheme optimizer self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
