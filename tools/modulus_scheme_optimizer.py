"""Bounded exact search over certified forms under a declared additive cost model.

Order-dependent and overlapping pipeline costs are deliberately not represented
by this objective. A measured optimum is only as applicable as its cost cohort.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from arithmetic_certificate import ArithmeticCertificate, certificate_hash, from_json, verify


@dataclass(frozen=True)
class Form:
    certificate: ArithmeticCertificate
    conversion_ps: int
    slab_ps: int
    epilogue_ps: int
    reconstruction_ps: int
    cost_cohort: str

    def cost(self, k: int) -> int:
        return self.conversion_ps + ((k + self.certificate.safe_k - 1) // self.certificate.safe_k) * (
            self.slab_ps + self.epilogue_ps) + self.reconstruction_ps


def optimize(forms: list[Form], bound: int, signed: bool, k: int, max_nodes: int = 1000000) -> dict:
    if (type(bound) is not int or bound < 0 or type(k) is not int or k <= 0
            or type(signed) is not bool or type(max_nodes) is not int or max_nodes < 1):
        raise ValueError("invalid exact output bound, K or search budget")
    if not forms or len(forms) > 24:
        raise ValueError("search domain must contain 1..24 explicitly supplied forms")
    if len({f.cost_cohort for f in forms}) != 1 or not forms[0].cost_cohort:
        raise ValueError("form costs must share an explicit calibration cohort")
    unique = set()
    for form in forms:
        if not verify(form.certificate)["algebraically_verified"]:
            raise ValueError("unverified arithmetic form")
        if any(type(cost) is not int or cost < 0 for cost in
               (form.conversion_ps, form.slab_ps, form.epilogue_ps, form.reconstruction_ps)):
            raise ValueError("costs must be known nonnegative integer picoseconds")
        identity = certificate_hash(form.certificate)
        if identity in unique:
            raise ValueError("duplicate form identity")
        unique.add(identity)
    target = bound * (2 if signed else 1)
    costs = [form.cost(k) for form in forms]
    best_cost = None
    best = None
    visited = 0
    exhausted = False

    def search(index: int, product: int, cost: int, selected: tuple[int, ...]) -> None:
        nonlocal best_cost, best, visited, exhausted
        if visited >= max_nodes:
            exhausted = True
            return
        visited += 1
        if product > target and selected:
            if best_cost is None or (cost, selected) < (best_cost, best):
                best_cost, best = cost, selected
            return  # Additional nonnegative-cost forms cannot improve latency.
        if index == len(forms) or (best_cost is not None and cost >= best_cost):
            return
        # Ignoring coprimality gives a safe optimistic upper bound on range.
        possible_product = product
        for form in forms[index:]:
            possible_product *= form.certificate.modulus
        if possible_product <= target:
            return
        modulus = forms[index].certificate.modulus
        if math.gcd(product, modulus) == 1:
            search(index + 1, product * modulus, cost + costs[index], selected + (index,))
        search(index + 1, product, cost, selected)

    search(0, 1, 0, ())
    chosen = [forms[i] for i in best] if best is not None else []
    products = []
    product = 1
    for form in chosen:
        product *= form.certificate.modulus
        products.append(product)
    return {"schema": "rns8_modulus_scheme_search_v1", "objective": "additive_pipeline_picoseconds",
            "order_policy": "input_order_additive_cost_is_order_invariant", "cost_cohort": forms[0].cost_cohort,
            "bound": bound, "signed": signed, "k": k, "strict_product_threshold": target,
            "visited_nodes": visited, "search_complete": not exhausted,
            "optimal_within_declared_domain": not exhausted and best is not None,
            "lower_bound_ps": best_cost if not exhausted and best is not None else 0,
            "feasible_upper_bound_ps": best_cost, "feasible": best is not None,
            "ordered_moduli": [f.certificate.modulus for f in chosen], "prefix_products": products,
            "form_hashes": [certificate_hash(f.certificate) for f in chosen],
            "domain_hashes": [certificate_hash(f.certificate) for f in forms],
            "runtime_ladder_changed": False, "promotion_eligible": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.input.read_text(encoding="utf-8"))
    forms = [Form(certificate=from_json(item["certificate"]), **item["costs"]) for item in request["forms"]]
    if "joint_schedule" in request:
        from joint_scheme_optimizer import ScheduledForm, joint_optimize
        scheduled = [ScheduledForm(form, **item["resources"]) for form, item in zip(forms, request["forms"])]
        result = joint_optimize(scheduled, request["bound"], request["signed"], request["k"],
                                max_nodes=request.get("max_nodes", 1000000), **request["joint_schedule"])
    else:
        result = optimize(forms, request["bound"], request["signed"], request["k"], request.get("max_nodes", 1000000))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.get("feasible", result.get("best") is not None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
