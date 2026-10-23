"""Exact host prototypes and width certificates for alternative CRT DAGs.

No quotient estimate or device qualification is inferred from these prototypes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MergeNode:
    product: int
    left: int | None = None
    right: int | None = None
    inverse_left_mod_right: int = 0


@dataclass(frozen=True)
class CrtPlan:
    moduli: tuple[int, ...]
    product: int
    weights: tuple[int, ...]
    inverses: tuple[int, ...]
    direct_sum_max: int
    nodes: tuple[MergeNode, ...]
    root: int

    @property
    def direct_accumulator_bits(self) -> int:
        return max(1, self.direct_sum_max.bit_length())


def make_plan(moduli: tuple[int, ...]) -> CrtPlan:
    if not moduli or any(type(m) is not int or m < 2 for m in moduli):
        raise ValueError("CRT requires nonempty positive moduli")
    product = 1
    for m in moduli:
        if math.gcd(product, m) != 1:
            raise ValueError("CRT basis is not pairwise coprime")
        product *= m
    weights = tuple(product // m for m in moduli)
    inverses = tuple(pow(weight, -1, m) for weight, m in zip(weights, moduli))
    # Each v_i is canonical; their independently attainable maximum is m_i-1.
    direct_max = sum((m - 1) * weight for m, weight in zip(moduli, weights))
    nodes = [MergeNode(m) for m in moduli]
    level = list(range(len(nodes)))
    while len(level) > 1:
        next_level = []
        for begin in range(0, len(level), 2):
            if begin + 1 == len(level):
                next_level.append(level[begin])
                continue
            left, right = level[begin:begin + 2]
            a, b = nodes[left].product, nodes[right].product
            nodes.append(MergeNode(a * b, left, right, pow(a, -1, b)))
            next_level.append(len(nodes) - 1)
        level = next_level
    return CrtPlan(moduli, product, weights, inverses, direct_max, tuple(nodes), level[0])


def _canonical_residues(plan: CrtPlan, residues: tuple[int, ...]) -> None:
    if len(residues) != len(plan.moduli):
        raise ValueError("residue count differs from basis")
    if any(type(r) is not int or not 0 <= r < m for r, m in zip(residues, plan.moduli)):
        raise ValueError("residues must be canonical")


def direct(plan: CrtPlan, residues: tuple[int, ...], accumulator_bits: int | None = None) -> int:
    _canonical_residues(plan, residues)
    if accumulator_bits is not None and accumulator_bits < plan.direct_accumulator_bits:
        raise ValueError("direct CRT accumulator cannot hold its unreduced weighted sum")
    value = sum(((r * inv) % m) * weight
                for r, inv, m, weight in zip(residues, plan.inverses, plan.moduli, plan.weights))
    assert 0 <= value <= plan.direct_sum_max
    return value % plan.product


def balanced(plan: CrtPlan, residues: tuple[int, ...]) -> int:
    _canonical_residues(plan, residues)
    values = list(residues)
    for node in plan.nodes[len(residues):]:
        a, b = values[node.left], values[node.right]
        pa, pb = plan.nodes[node.left].product, plan.nodes[node.right].product
        # The un-reduced inverse multiplication can need twice pb's bit width.
        t = ((b - a) % pb) * node.inverse_left_mod_right % pb
        value = a + pa * t
        assert 0 <= value < node.product
        values.append(value)
    return values[plan.root]


def width_certificate(plan: CrtPlan) -> dict:
    merges = []
    for index, node in enumerate(plan.nodes):
        if node.left is None:
            continue
        pa, pb = plan.nodes[node.left].product, plan.nodes[node.right].product
        merges.append({"node": index, "children": [node.left, node.right],
                       "difference_min": -(pa - 1), "difference_max": pb - 1,
                       "inverse_multiply_max": (pb - 1) * node.inverse_left_mod_right,
                       "product_plus_left_max": pa * pb - 1,
                       "result_bits": (node.product - 1).bit_length()})
    return {"direct_sum_max": plan.direct_sum_max,
            "direct_accumulator_bits": plan.direct_accumulator_bits,
            "product_bits": plan.product.bit_length(), "merges": merges,
            "hardware_qualified": False, "quotient_hint_qualified": False}
