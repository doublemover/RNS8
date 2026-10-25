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


def direct(plan: CrtPlan, residues: tuple[int, ...], accumulator_bits: int | None = None,
           temporary_bits: int | None = None) -> int:
    """accumulator_bits covers the sum only; None temporaries use exact Python integers."""
    _canonical_residues(plan, residues)
    if accumulator_bits is not None and (type(accumulator_bits) is not int or accumulator_bits < plan.direct_accumulator_bits):
        raise ValueError("direct CRT accumulator cannot hold its unreduced weighted sum")
    required_temporary = max(max((m - 1) * inv, (m - 1) * weight).bit_length()
                             for m, inv, weight in zip(plan.moduli, plan.inverses, plan.weights))
    if temporary_bits is not None and (type(temporary_bits) is not int or temporary_bits < required_temporary):
        raise ValueError("direct CRT temporary cannot hold pre-reduction or weighted multiplication")
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
    direct_stages = []
    running = 0
    for index, (modulus, inverse, weight) in enumerate(zip(plan.moduli, plan.inverses, plan.weights)):
        multiply = (modulus - 1) * inverse
        weighted = (modulus - 1) * weight
        running += weighted
        direct_stages.append({"plane": index, "input_normalization": "canonical_unsigned_required",
                              "input_max": modulus - 1, "inverse_multiply_max": multiply,
                              "inverse_multiply_unsigned_bits": max(1, multiply.bit_length()),
                              "reduced_coefficient_max": modulus - 1,
                              "weighted_multiply_max": weighted,
                              "weighted_multiply_unsigned_bits": max(1, weighted.bit_length()),
                              "ordered_sum_max": running,
                              "ordered_sum_unsigned_bits": max(1, running.bit_length())})
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
            "direct_sum_accumulator_unsigned_bits": plan.direct_accumulator_bits,
            "direct_stages": direct_stages,
            "direct_final_reduction_input_max": plan.direct_sum_max,
            "direct_final_reduction_output_max": plan.product - 1,
            "product_bits": plan.product.bit_length(), "merges": merges,
            "hardware_qualified": False, "quotient_hint_qualified": False}


def quarter_certificate(plan: CrtPlan, bound: int, destination_bits: int, signed: bool = True) -> dict:
    """Exact independent coefficient/error certificate for ordered binary32 FMA.

    This is the centered quarter-range algorithm, not the reduced-coefficient
    direct algorithm above. Full W_i=M_i*y_i and all limb bounds are distinct.
    """
    from fractions import Fraction
    from floating_rewrite_oracle import round_binary32, rational
    if (type(bound) is not int or bound < 0 or type(destination_bits) is not int
            or not 1 <= destination_bits <= 2048 or type(signed) is not bool
            or len(plan.moduli)>20 or any(m>256 for m in plan.moduli)):
        raise ValueError("quarter certificate domain")
    if 4*bound>plan.product:
        raise ValueError("quarter admission requires 4B<=M, not only 2B<M")
    if bound >= 1 << (destination_bits-int(signed)):
        raise ValueError("quarter destination width")
    bits=tuple(round_binary32(Fraction(y,m)) for y,m in zip(plan.inverses,plan.moduli))
    coefficient_error=sum((m-1)*abs(rational(code)-Fraction(y,m)) for m,y,code in zip(plan.moduli,plan.inverses,bits))
    fma_error=Fraction(len(plan.moduli),4096)
    if coefficient_error+fma_error>=Fraction(1,64):
        raise ValueError("quotient error does not satisfy 1/64")
    digits=max(1,(bound.bit_length()+1+15)//16)
    # Conservative bound covers positive sum, q*M subtraction and carry even
    # without exploiting their cancellation. Every physical intermediate fits.
    limb_bound=20*255*65535+5121*65535+8192
    if limb_bound>=(1 << 31):
        raise ValueError("delayed signed32 bound")
    error=coefficient_error+fma_error
    return {"schema":"rns8_quarter_crt_model_v1","basis":list(plan.moduli),"bound":bound,
            "destination_bits":destination_bits,"signed":signed,"digits":digits,
            "full_weights":[w*y for w,y in zip(plan.weights,plan.inverses)],"fraction_bits":list(bits),
            "error_bound":[error.numerator,error.denominator],"limb_intermediate_absolute_bound":limb_bound,
            "fp_sequence":"prefix_order_binary32_RNE_FMA_then_RNE_integer",
            "target_execution":"not_run","runtime_dispatch_eligible":False}


def quarter_reconstruct(plan: CrtPlan, residues: tuple[int, ...], certificate: dict) -> int:
    from fractions import Fraction
    from floating_rewrite_oracle import round_binary32, rational
    _canonical_residues(plan,residues)
    expected=quarter_certificate(plan,certificate["bound"],certificate["destination_bits"],certificate["signed"])
    if certificate!=expected:raise ValueError("stale quarter certificate")
    z=0
    for r,bits in zip(residues,certificate["fraction_bits"]):
        z=round_binary32(Fraction(r)*rational(bits)+rational(z))
    value=rational(z);q,rem=divmod(value.numerator,value.denominator)
    q+=2*rem>value.denominator or (2*rem==value.denominator and q&1)
    carry=0;words=[]
    for digit in range(certificate["digits"]):
        column=sum(r*((weight>>(digit*16))&65535) for r,weight in zip(residues,certificate["full_weights"]))
        column-=q*((plan.product>>(digit*16))&65535);column+=carry
        carry,word=divmod(column,65536);words.append(word)
    result=sum(word<<(16*d) for d,word in enumerate(words))
    if words[-1]&32768:result-=1<<(16*len(words))
    if abs(result)>certificate["bound"] or (not certificate["signed"] and result<0):
        raise ValueError("source did not satisfy quarter range")
    if any(result%m!=r for r,m in zip(residues,plan.moduli)):
        raise ValueError("quotient/model certificate contradicted residues")
    return result
