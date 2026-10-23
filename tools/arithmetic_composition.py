"""Exact physical-interval checker for a bounded, topologically ordered proof DAG.

Input contracts are assumptions requiring validation by the binding operation.
Integer equality, modular equality and physical scale are separate state.
This checker does not qualify any GPU instruction or runtime dispatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math


@dataclass(frozen=True)
class Node:
    name: str
    operation: str
    inputs: tuple[str, ...] = ()
    bits: int = 32
    signed: bool = True
    interval: tuple[int, int] | None = None
    multiple: int = 1
    count: int = 1
    modulus: int | None = None
    factor: int = 1
    scale: tuple[int, int] = (1, 1)


@dataclass(frozen=True)
class Domain:
    low: int
    high: int
    multiple: int
    modulus: int | None
    scale: Fraction


def verify_composition(nodes: tuple[Node, ...]) -> dict:
    if not nodes or len(nodes) > 4096:
        raise ValueError("bounded nonempty proof DAG required")
    domains: dict[str, Domain] = {}
    failures = []
    for node in nodes:
        try:
            if not node.name or node.name in domains or not 1 <= node.bits <= 256 or type(node.signed) is not bool:
                raise ValueError("invalid node identity or physical width")
            if any(name not in domains for name in node.inputs):
                raise ValueError("missing input or non-topological evaluation order")
            inputs = [domains[name] for name in node.inputs]
            op = node.operation
            if op == "input":
                if inputs or node.interval is None or len(node.interval) != 2 or node.interval[0] > node.interval[1]:
                    raise ValueError("invalid input interval")
                if any(type(x) is not int for x in node.interval) or type(node.multiple) is not int or node.multiple <= 0:
                    raise ValueError("integer source contract required")
                scale = Fraction(*node.scale)
                if scale <= 0 or (node.modulus is not None and (type(node.modulus) is not int or node.modulus < 2)):
                    raise ValueError("invalid source scale or modulus")
                result = Domain(*node.interval, node.multiple, node.modulus, scale)
            elif op in {"add", "multiply", "dot"}:
                if len(inputs) != (3 if op == "dot" and len(inputs) == 3 else 2):
                    raise ValueError("wrong arithmetic arity")
                a, b = inputs[:2]
                if a.modulus is not None and b.modulus is not None and a.modulus != b.modulus:
                    raise ValueError("incompatible modular domains")
                modulus = a.modulus if a.modulus is not None else b.modulus
                if op == "add":
                    if a.scale != b.scale:
                        raise ValueError("accumulator scale mismatch")
                    result = Domain(a.low + b.low, a.high + b.high, math.gcd(a.multiple, b.multiple), modulus, a.scale)
                else:
                    endpoints = [x * y for x in (a.low, a.high) for y in (b.low, b.high)]
                    count = node.count if op == "dot" else 1
                    if type(count) is not int or count <= 0:
                        raise ValueError("positive dot count required")
                    lo, hi = min(endpoints) * count, max(endpoints) * count
                    if op == "dot":
                        lo, hi = min(0, lo), max(0, hi)  # every partial sum, not only the final dot
                    result = Domain(lo, hi, a.multiple * b.multiple, modulus, a.scale * b.scale)
                    if len(inputs) == 3:
                        carry = inputs[2]
                        if carry.scale != result.scale or carry.modulus != result.modulus:
                            raise ValueError("carried accumulator domain/scale mismatch")
                        result = Domain(lo + carry.low, hi + carry.high, math.gcd(result.multiple, carry.multiple),
                                        modulus, result.scale)
            elif op in {"canonical_reduce", "centered_reduce", "exact_divide", "layout", "rescale"}:
                if len(inputs) != 1:
                    raise ValueError("unary node requires one input")
                a = inputs[0]
                if op in {"canonical_reduce", "centered_reduce"}:
                    q = node.modulus
                    if type(q) is not int or q < 2 or a.scale != 1 or a.modulus not in (None, q):
                        raise ValueError("invalid reduction domain")
                    lo, hi = (0, q - 1) if op == "canonical_reduce" else (-(q // 2), (q - 1) // 2)
                    result = Domain(lo, hi, math.gcd(a.multiple, q), q, Fraction(1))
                elif op == "exact_divide":
                    if (type(node.factor) is not int or node.factor <= 0 or a.modulus is not None
                            or a.scale != 1 or a.multiple % node.factor):
                        raise ValueError("exact division needs an unreduced divisible integer numerator")
                    result = Domain(a.low // node.factor, a.high // node.factor,
                                    a.multiple // node.factor, None, a.scale)
                elif op == "rescale":
                    if type(node.factor) is not int or node.factor <= 0 or a.modulus is not None:
                        raise ValueError("rescale requires a positive integer factor in an exact domain")
                    # Multiply physical values and divide scale: logical value is preserved.
                    result = Domain(a.low * node.factor, a.high * node.factor,
                                    a.multiple * node.factor, None, a.scale / node.factor)
                else:
                    result = a  # bit/value-preserving permutation
            else:
                raise ValueError("unsupported proof node")
            low = -(1 << (node.bits - 1)) if node.signed else 0
            high = (1 << (node.bits - int(node.signed))) - 1
            if result.low < low or result.high > high:
                raise ValueError("physical intermediate overflow")
            domains[node.name] = result
        except (ValueError, TypeError, ZeroDivisionError) as error:
            failures.append({"node": node.name, "obligation": str(error)})
            break
    identity = hashlib.sha256(json.dumps([asdict(n) for n in nodes], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": identity, "algebraically_verified": not failures, "failures": failures,
            "domains": {name: {"interval": [d.low, d.high], "multiple": d.multiple,
                                "modulus": d.modulus, "physical_scale": str(d.scale)} for name, d in domains.items()},
            "input_contracts_require_binding_validation": True,
            "instruction_measured": False, "runtime_dispatch_eligible": False}
