"""Exact rational extension of arithmetic_certificate, not an AMD qualifier.

OCP MX v1 element decoding; FNUZ deliberately is not an alias of OCP FP8.
Every contract names storage, products, reduction order and outer accumulation.
The sufficient grid proof is conservative: failure means not admitted, not a
claim that every concrete input must round incorrectly.
"""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json

VERSION = "rns8-floating-grid-v1"
FORMATS = {"ocp_e2m1": (2, 1, 1), "ocp_e2m3": (2, 3, 1),
           "ocp_e3m2": (3, 2, 3), "ocp_e4m3": (4, 3, 7), "ocp_e5m2": (5, 2, 15)}


def power(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def decode_element(format_name, code):
    if format_name not in FORMATS:
        raise ValueError("unsupported_element_encoding")
    e, m, bias = FORMATS[format_name]
    if type(code) is not int or not 0 <= code < 1 << (1+e+m):
        raise ValueError("element_code_width")
    sign, exponent, mantissa = code >> (e+m), (code >> m) & ((1 << e)-1), code & ((1 << m)-1)
    if ((format_name == "ocp_e4m3" and exponent == 15 and mantissa == 7)
            or (format_name == "ocp_e5m2" and exponent == 31)):
        raise ValueError("nonfinite_element")
    return (-1 if sign else 1) * power((exponent if exponent else 1)-bias-m) * (mantissa + ((1 << m) if exponent else 0))


def decode_scale(code):
    if type(code) is not int or not 0 <= code <= 254:
        raise ValueError("nonfinite_or_invalid_e8m0")
    return power(code-127)


@dataclass(frozen=True)
class FloatingBlock:
    count: int
    scale_a: int
    scale_b: int
    # Multiply only the newly produced block by this exact power of two.
    rebase_exponent: int


@dataclass(frozen=True)
class FloatingContract:
    element_format: str
    blocks: tuple[FloatingBlock, ...]
    block_k: int = 32
    scale_format: str = "ocp_e8m0"
    scale_application: str = "widened_exponent_ldexp"
    accumulator_precision: int = 24
    accumulator_min_exponent: int = -126
    accumulator_max_exponent: int = 127
    rounding: str = "nearest_even"
    denormals: str = "gradual"
    reduction: str = "balanced"
    product_placement: str = "scaled_operands_then_product"
    outer_accumulation: str = "rebase_new_block_then_add"
    initial_accumulator: tuple[int, int] = (0, 1)
    output_grid_exponent: int = 0
    destination_bits: int = 64
    destination_signed: bool = True
    lowering_sha256: str = ""
    proof_artifact_sha256: str = ""


def schema_failures(f):
    if type(f) is not FloatingContract:
        return ["floating_contract_structure"]
    ints = (f.block_k, f.accumulator_precision, f.accumulator_min_exponent,
            f.accumulator_max_exponent, f.output_grid_exponent, f.destination_bits)
    if any(type(x) is not int for x in ints) or type(f.destination_signed) is not bool:
        return ["floating_noninteger_parameter"]
    if (type(f.blocks) is not tuple or not 1 <= len(f.blocks) <= 65536
            or any(type(b) is not FloatingBlock or any(type(x) is not int for x in
                   (b.count,b.scale_a,b.scale_b,b.rebase_exponent)) for b in f.blocks)):
        return ["floating_block_structure"]
    if (type(f.initial_accumulator) is not tuple or len(f.initial_accumulator) != 2
            or any(type(x) is not int for x in f.initial_accumulator) or f.initial_accumulator[1] <= 0):
        return ["floating_initial_accumulator"]
    if any(type(x) is not str or len(x) != 64 or any(ch not in "0123456789abcdef" for ch in x)
           for x in (f.lowering_sha256, f.proof_artifact_sha256)):
        return ["floating_missing_artifact_identity"]
    if (type(f.element_format) is not str or f.element_format not in FORMATS
            or f.scale_format != "ocp_e8m0" or f.scale_application != "widened_exponent_ldexp" or f.rounding != "nearest_even"
            or f.denormals not in ("gradual", "flush_to_zero")
            or f.reduction not in ("fold", "balanced")
            or f.product_placement not in ("scaled_operands_then_product", "raw_product_then_block_scale")
            or f.outer_accumulation != "rebase_new_block_then_add"):
        return ["unsupported_floating_lowering"]
    if (not 1 <= f.block_k <= 65536 or not 2 <= f.accumulator_precision <= 113
            or not -16382 <= f.accumulator_min_exponent < f.accumulator_max_exponent <= 16383
            or abs(f.output_grid_exponent) > 16384 or not 1 <= f.destination_bits <= 256
            or any(not 1 <= b.count <= f.block_k or not 0 <= b.scale_a <= 254
                   or not 0 <= b.scale_b <= 254 or abs(b.rebase_exponent) > 16384 for b in f.blocks)):
        return ["floating_parameter_domain"]
    return []


def _v2(value):
    value = abs(Fraction(value))
    if not value:
        return None
    n,d = value.numerator,value.denominator
    if d & (d-1):
        raise ValueError("nondyadic_grid")
    return (n & -n).bit_length()-1-(d.bit_length()-1)


def _grid(values):
    nonzero = [_v2(x) for x in values if x]
    return power(min(nonzero)) if nonzero else Fraction(1)


def _exact_grid(maximum, quantum, f):
    """All multiples of quantum in +/-maximum are finite and exact."""
    maximum, quantum = abs(Fraction(maximum)), Fraction(quantum)
    if not maximum:
        return True
    if quantum <= 0 or quantum.denominator & (quantum.denominator-1):
        return False
    floor = f.accumulator_min_exponent - (f.accumulator_precision-1 if f.denormals == "gradual" else 0)
    largest = (2-power(1-f.accumulator_precision))*power(f.accumulator_max_exponent)
    return quantum >= power(floor) and maximum <= largest and maximum/quantum <= 1 << f.accumulator_precision


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def lowering_hash(f):
    model=asdict(f);model.pop("lowering_sha256");model.pop("proof_artifact_sha256")
    return _digest({"checker":VERSION,"explicit_lowering":model})


def artifact_hash(c):
    artifact=asdict(c)
    artifact["instruction"]["floating"].pop("lowering_sha256")
    artifact["instruction"]["floating"].pop("proof_artifact_sha256")
    return _digest({"checker":VERSION,"bound_model":artifact})


def bind_artifacts(c):
    """Generate identities for a NEW proof input, never a validation receipt."""
    f=c.instruction.floating
    f=replace(f,lowering_sha256=lowering_hash(f),proof_artifact_sha256=artifact_hash(c))
    return replace(c,instruction=replace(c.instruction,floating=f))


def verify_floating(c):
    """Check all legal encoded operands, every partial and ordered outer sum.

    Logical tables/weights/denominator are checked by the existing integer
    checker. Here logical table entries equal decoded element * 2**grid.
    Each block rebases to a shared physical output grid BEFORE adding to the
    old accumulator; rescaling the old value is not this contract.
    """
    f = c.instruction.floating
    failures = schema_failures(f)
    if failures:
        return {"finite_exact_recoverability":"unsupported", "failures":failures}
    if f.lowering_sha256 != lowering_hash(f):
        failures.append("stale_floating_lowering_artifact")
    if f.proof_artifact_sha256 != artifact_hash(c):
        failures.append("stale_floating_proof_artifact")
    if sum(b.count for b in f.blocks) != c.safe_k:
        failures.append("floating_blocks_do_not_cover_k")
    initial = Fraction(*f.initial_accumulator)
    if c.prior_min != c.prior_max or initial*power(f.output_grid_exponent) != c.prior_min:
        failures.append("floating_initial_carry_mismatch")
    receipts = []
    for plane_index,p in enumerate(c.planes):
        prefix = f"floating_plane_{plane_index}_"
        if (len(p.lhs_codes) != len(p.lhs) or len(p.rhs_codes) != len(p.rhs)
                or type(p.grid_exponent) is not int or abs(p.grid_exponent) > 16384):
            failures.append(prefix+"encoding_structure");continue
        try:
            a = tuple(decode_element(f.element_format,x) for x in p.lhs_codes)
            b = tuple(decode_element(f.element_format,x) for x in p.rhs_codes)
        except ValueError as error:
            failures.append(prefix+str(error));continue
        if (tuple(x*power(p.grid_exponent) for x in a) != p.lhs
                or tuple(x*power(p.grid_exponent) for x in b) != p.rhs):
            failures.append(prefix+"decoded_grid_mismatch");continue
        ga,gb = _grid(a),_grid(b)
        ma,mb = max(map(abs,a)),max(map(abs,b))
        outer_abs,outer_grid = abs(initial),_grid((initial,))
        for index,block in enumerate(f.blocks):
            label = prefix+f"block_{index}_"
            sa,sb = decode_scale(block.scale_a),decode_scale(block.scale_b)
            if block.rebase_exponent != 2*p.grid_exponent-(block.scale_a+block.scale_b-254)-f.output_grid_exponent:
                failures.append(label+"outer_grid_alignment")
            # The storage/product assumptions are explicit. No infinity or
            # cancellation at the root can license an overflowing proper node.
            if f.product_placement == "scaled_operands_then_product":
                operands = ((ma*sa,ga*sa),(mb*sb,gb*sb))
                product_abs,product_grid = ma*mb*sa*sb,ga*gb*sa*sb
            else:
                operands = ((ma,ga),(mb,gb))
                product_abs,product_grid = ma*mb,ga*gb
            if any(not _exact_grid(m,g,f) for m,g in operands):
                failures.append(label+"stored_operand_not_exact")
            subtotal_abs = product_abs*block.count
            # A subtree has <= count leaves; bounding the full absolute sum
            # proves every fold/balanced partial, even when final signs cancel.
            if not _exact_grid(subtotal_abs,product_grid,f):
                failures.append(label+"product_or_partial_not_exact")
            if f.product_placement == "raw_product_then_block_scale":
                subtotal_abs*=sa*sb;product_grid*=sa*sb
                if not _exact_grid(subtotal_abs,product_grid,f):
                    failures.append(label+"scaled_block_not_exact")
            subtotal_abs*=power(block.rebase_exponent);product_grid*=power(block.rebase_exponent)
            if not _exact_grid(subtotal_abs,product_grid,f):
                failures.append(label+"rebased_block_not_exact")
            outer_grid=min(outer_grid,product_grid) if outer_abs else product_grid
            outer_abs+=subtotal_abs
            if not _exact_grid(outer_abs,outer_grid,f):
                failures.append(label+"outer_partial_not_exact")
        recovered_max=outer_abs*power(f.output_grid_exponent)
        if not _exact_grid(recovered_max,outer_grid*power(f.output_grid_exponent),f):
            failures.append(prefix+"integer_recovery_scale_not_exact")
        # Integer checker separately proves weighted recombination and exact
        # denominator. Check its actual destination interval, before reduction.
        receipts.append({"plane":plane_index,"absolute_sum":[outer_abs.numerator,outer_abs.denominator],
                         "grid":[outer_grid.numerator,outer_grid.denominator]})
    low = -(1 << (f.destination_bits-1)) if f.destination_signed else 0
    high = (1 << (f.destination_bits-int(f.destination_signed)))-1
    lo,hi = c.prior_min,c.prior_max
    for p in c.planes:
        products=[x*y for x in (min(p.lhs),max(p.lhs)) for y in (min(p.rhs),max(p.rhs))]
        weighted=sorted((min(products)*c.safe_k*p.weight,max(products)*c.safe_k*p.weight))
        lo+=weighted[0];hi+=weighted[1]
    if lo < low*c.denominator or hi > high*c.denominator:
        failures.append("floating_integer_destination_range")
    return {"checker_version":VERSION,"finite_exact_recoverability":"proven" if not failures else "unsupported",
            "failures":sorted(set(failures)),"plane_guards":receipts,
            "scope":"all_declared_operand_codes_and_partials_under_explicit_binary_model",
            "target_execution":"not_run","runtime_dispatch_eligible":False}


def from_json(data):
    data=dict(data);data["blocks"]=tuple(FloatingBlock(**b) for b in data["blocks"])
    data["initial_accumulator"]=tuple(data.get("initial_accumulator",(0,1)))
    return FloatingContract(**data)
