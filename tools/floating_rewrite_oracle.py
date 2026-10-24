"""Offline E2M1/E8M0 binary32 rewrite oracle; never hardware qualification.

Two evaluators: rational IEEE rounding and host binary64 operations rounded through
binary32 storage. Positive zero is the declared comparison policy. No FMA or FTZ.
"""
from fractions import Fraction
import hashlib
import json
import math
import struct

VERSION = "rns8-floating-rewrite-v1"
ALPHABET = tuple(Fraction(x, 2) for x in (0,1,2,3,4,6,8,12))
INF = 0x7f800000
NAN = 0x7fc00000


def pow2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def decode_element(code):
    if type(code) is not int or not 0 <= code < 16:
        raise ValueError("E2M1 requires four bits")
    return ALPHABET[code & 7] * (-1 if code & 8 else 1)


def decode_scale(code):
    if type(code) is not int or not 0 <= code <= 252:
        raise ValueError("qualified E8M0 code domain is 0..252")
    return pow2(code - 127)


def rational(bits):
    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 255
    if exponent == 255:
        return None
    mantissa = bits & 0x7fffff
    return sign * (Fraction(mantissa) * pow2(-149) if exponent == 0
                   else Fraction((1 << 23) + mantissa) * pow2(exponent - 150))


def _nearest_even(value):
    q, r = divmod(value.numerator, value.denominator)
    return q + (2*r > value.denominator or (2*r == value.denominator and q & 1))


def round_binary32(value):
    value = Fraction(value)
    if not value:
        return 0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < pow2(exponent):
        exponent -= 1
    if exponent < -126:
        mantissa = _nearest_even(value / pow2(-149))
        return (sign | mantissa) if mantissa else 0
    if exponent > 127:
        return sign | INF
    mantissa = _nearest_even(value / pow2(exponent - 23))
    if mantissa == 1 << 24:
        mantissa >>= 1
        exponent += 1
    if exponent > 127:
        return sign | INF
    return sign | ((exponent + 127) << 23) | (mantissa - (1 << 23))


def _soft(op, a, b):
    av, bv = rational(a), rational(b)
    if av is None or bv is None:
        # Exceptional behavior is shared only through IEEE category rules;
        # ordinary arithmetic and rounding use exact rationals above.
        if (a & 0x7fffffff) > INF or (b & 0x7fffffff) > INF:
            return NAN
        if op == "add":
            if av is None and bv is None and (a ^ b) >> 31:
                return NAN
            return a if av is None else b
        if av == 0 or bv == 0:
            return NAN
        return ((a ^ b) & 0x80000000) | INF
    return round_binary32(av + bv if op == "add" else av * bv)


def _host_bits(value):
    try:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return (0x80000000 if value < 0 else 0) | INF
    if bits & 0x7fffffff > INF:
        return NAN
    return 0 if bits & 0x7fffffff == 0 else bits


def _host(op, a, b):
    av = struct.unpack("<f", struct.pack("<I", a))[0]
    bv = struct.unpack("<f", struct.pack("<I", b))[0]
    return _host_bits(av + bv if op == "add" else av * bv)


def reduction_tree(k, topology):
    if type(k) is not int or not 1 <= k <= 65536:
        raise ValueError("bounded nonempty block required")
    if topology == "fold":
        root = 0
        for index in range(1, k):
            root = (root, index)
        return root
    if topology == "balanced":
        layer = list(range(k))
        while len(layer) > 1:
            layer = [(layer[i], layer[i+1]) if i+1 < len(layer) else layer[i]
                     for i in range(0, len(layer), 2)]
        return layer[0]
    raise ValueError("unsupported reduction topology")


def _reduce(values, tree, operation):
    # Iterative postorder avoids recursion limits for long left folds.
    stack, results, leaves = [(tree, False)], [], []
    while stack:
        node, visited = stack.pop()
        if type(node) is int:
            if not 0 <= node < len(values):
                raise ValueError("invalid leaf")
            leaves.append(node)
            results.append(values[node])
        elif type(node) is tuple and len(node) == 2:
            if visited:
                right, left = results.pop(), results.pop()
                results.append(operation("add", left, right))
            else:
                stack.extend(((node, True), (node[1], False), (node[0], False)))
        else:
            raise ValueError("malformed reduction tree")
    if sorted(leaves) != list(range(len(values))):
        raise ValueError("tree must contain each input exactly once")
    return results[0]


def evaluate(lhs, rhs, scale_a, scale_b, topology="balanced", denominator=1, destination_bits=64):
    if type(denominator) is not int or denominator <= 0 or type(destination_bits) is not int or not 1 <= destination_bits <= 256:
        raise ValueError("positive exact denominator and bounded integer destination required")
    if len(lhs) != len(rhs):
        raise ValueError("operand length mismatch")
    a, b = list(map(decode_element, lhs)), list(map(decode_element, rhs))
    sa, sb = decode_scale(scale_a), decode_scale(scale_b)
    tree = reduction_tree(len(a), topology)
    t = scale_a + scale_b - 254  # widened integers, never byte addition
    outputs = {}
    for name, operation in (("rational", _soft), ("host", _host)):
        direct = [operation("mul", operation("mul", round_binary32(x), round_binary32(sa)),
                            operation("mul", round_binary32(y), round_binary32(sb))) for x,y in zip(a,b)]
        raw = [operation("mul", round_binary32(x), round_binary32(y)) for x,y in zip(a,b)]
        root = _reduce(raw, tree, operation)
        if name == "rational":
            value = rational(root)
            hoisted = root if value is None else round_binary32(value * pow2(t))
        else:
            value = struct.unpack("<f", struct.pack("<I", root))[0]
            hoisted = _host_bits(math.ldexp(value, t))
        outputs[name] = {"direct": _reduce(direct, tree, operation), "hoisted": hoisted}
    exact = sum((x*y for x,y in zip(a,b)), Fraction()) * sa * sb
    observed = outputs["rational"]
    recovered = exact / denominator
    finite_exact = (all(rational(bits) == exact for bits in observed.values())
                    and recovered.denominator == 1
                    and -(1 << (destination_bits-1)) <= recovered < (1 << (destination_bits-1)))
    rewrite = ("unknown" if any((bits & 0x7fffffff) > INF for bits in observed.values())
               else "proven" if observed["direct"] == observed["hoisted"] else "refuted")
    # Sufficient guard uses the largest proper subtree for this topology.
    k = len(a)
    subtree = max(1, k-1 if topology == "fold" else (k+1)//2)
    upper = 129 - (144*subtree).bit_length() + 1
    guards = {"integer_product_domain": 144*k <= 1 << 24,
              "rewrite_band": -147 <= t <= upper,
              "finite_root": Fraction(144*k)*pow2(t-2) < pow2(128)}
    identity = {"version": VERSION, "lhs": list(lhs), "rhs": list(rhs),
                "scale_codes": [scale_a,scale_b], "topology": topology,
                "rounding": "binary32-RNE-separate", "denormals": "gradual",
                "zero_policy": "canonical-positive", "nan_policy": "canonical-unknown-equivalence",
                "denominator": denominator, "destination_bits": destination_bits,
                "outer_accumulation": "not-modeled"}
    return {"schema": VERSION, "sha256": hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
            "assumptions": identity, "outputs": outputs, "guards": guards,
            "evaluators_agree": outputs["rational"] == outputs["host"],
            "verdict_scope": "supplied-input-block-only",
            "verdicts": {"algebraic_value_equality": "proven",
                         "floating_rewrite_equivalence": rewrite,
                         "finite_exact_recoverability": "proven" if finite_exact else "refuted",
                         "physical_layout_correctness": "not_run", "target_execution": "not_run"},
            "runtime_dispatch_eligible": False}
