"""Shared exact form/cost admission for additive and joint scheme searches."""

from __future__ import annotations

from dataclasses import dataclass
import json

from arithmetic_certificate import ArithmeticCertificate, verify, exact_form_admissible


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
