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

from arithmetic_certificate import (
    ArithmeticCertificate,
    from_json,
    verify,
    exact_form_admissible,
)


@dataclass(frozen=True)
class Form:
    certificate: ArithmeticCertificate
    conversion_ps: int
    slab_ps: int
    epilogue_ps: int
    reconstruction_ps: int
    cost_cohort: str

    def cost(self, k: int) -> int:
        return (
            self.conversion_ps
            + ((k + self.certificate.safe_k - 1) // self.certificate.safe_k)
            * (self.slab_ps + self.epilogue_ps)
            + self.reconstruction_ps
        )


class FormAdmissionError(ValueError):
    """Strict whole-domain rejection with retained per-form proof obligations."""

    def __init__(self, report: dict):
        self.report = report
        rejected = [row for row in report["forms"] if not row["admitted"]]
        super().__init__("unverified arithmetic forms: " + json.dumps(rejected, sort_keys=True))


def validate_problem(bound, signed, k, max_nodes):
    if (
        type(bound) is not int
        or bound < 0
        or type(k) is not int
        or k <= 0
        or type(signed) is not bool
        or type(max_nodes) is not int
        or max_nodes < 1
    ):
        raise ValueError("invalid exact output bound, K or search budget")


def admit_forms(forms: list[Form]) -> dict:
    if not forms or len(forms) > 24 or any(type(form) is not Form for form in forms):
        raise ValueError("search domain must contain 1..24 explicitly supplied forms")
    if (
        any(type(form.cost_cohort) is not str or not form.cost_cohort for form in forms)
        or len({f.cost_cohort for f in forms}) != 1
    ):
        raise ValueError("form costs must share an explicit calibration cohort")
    rows = []
    for index, form in enumerate(forms):
        proof = verify(form.certificate)
        rows.append(
            dict(
                form_id=index,
                certificate_sha256=proof["sha256"],
                checker_version=proof["checker_version"],
                verdicts=proof["verdicts"],
                proof_failures=proof["proof_failures"],
                admitted=exact_form_admissible(proof),
            )
        )
        if any(
            type(cost) is not int or cost < 0
            for cost in (form.conversion_ps, form.slab_ps, form.epilogue_ps, form.reconstruction_ps)
        ):
            raise ValueError("costs must be known nonnegative integer picoseconds")
    report = dict(
        schema="rns8_scheme_form_admission_v1",
        policy="strict_reject_entire_domain",
        forms=rows,
        accepted=all(row["admitted"] for row in rows),
        promotion_eligible=False,
        runtime_ladder_changed=False,
    )
    if not report["accepted"]:
        raise FormAdmissionError(report)
    identities = [row["certificate_sha256"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate form identity")
    return report


def optimize(forms: list[Form], bound: int, signed: bool, k: int, max_nodes: int = 1000000) -> dict:
    validate_problem(bound, signed, k, max_nodes)
    admission = admit_forms(forms)
    target = bound * (2 if signed else 1)
    costs = [form.cost(k) for form in forms]
    # The optimistic remaining range is independent of each chosen path.
    suffix_products = [1] * (len(forms) + 1)
    for index in range(len(forms) - 1, -1, -1):
        suffix_products[index] = suffix_products[index + 1] * forms[index].certificate.modulus
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
        possible_product = product * suffix_products[index]
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
    return {
        "schema": "rns8_modulus_scheme_search_v1",
        "objective": "additive_pipeline_picoseconds",
        "order_policy": "input_order_additive_cost_is_order_invariant",
        "cost_cohort": forms[0].cost_cohort,
        "arithmetic_admission": admission,
        "bound": bound,
        "signed": signed,
        "k": k,
        "strict_product_threshold": target,
        "visited_nodes": visited,
        "search_complete": not exhausted,
        "optimal_within_declared_domain": not exhausted and best is not None,
        "lower_bound_ps": best_cost if not exhausted and best is not None else 0,
        "feasible_upper_bound_ps": best_cost,
        "feasible": best is not None,
        "ordered_moduli": [f.certificate.modulus for f in chosen],
        "prefix_products": products,
        "form_hashes": [admission["forms"][i]["certificate_sha256"] for i in best]
        if best is not None
        else [],
        "domain_hashes": [row["certificate_sha256"] for row in admission["forms"]],
        "runtime_ladder_changed": False,
        "promotion_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.input.read_text(encoding="utf-8"))
    forms = [
        Form(certificate=from_json(item["certificate"]), **item["costs"])
        for item in request["forms"]
    ]
    try:
        if "joint_schedule" in request:
            from joint_scheme_optimizer import ScheduledForm, joint_optimize

            scheduled = [
                ScheduledForm(form, **item["resources"])
                for form, item in zip(forms, request["forms"])
            ]
            result = joint_optimize(
                scheduled,
                request["bound"],
                request["signed"],
                request["k"],
                max_nodes=request.get("max_nodes", 1000000),
                **request["joint_schedule"],
            )
        else:
            result = optimize(
                forms,
                request["bound"],
                request["signed"],
                request["k"],
                request.get("max_nodes", 1000000),
            )
    except FormAdmissionError as error:
        # Preserve rejected identities/verdicts in the ordinary file consumer;
        # a cheaper invalid form never becomes a certified incumbent.
        result = error.report
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.get("feasible", result.get("best") is not None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
