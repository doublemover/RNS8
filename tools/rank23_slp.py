"""Pinned rank-23 SLP import and exact algebra/lifetime planner. No execution evidence.

The certificate is mathematical data; upstream Python is never imported/executed.
Assignments are sequential (u22 and v12 are overwritten), not simultaneous.
"""
from dataclasses import dataclass, asdict
import argparse
import hashlib
import json
from pathlib import Path
import re

COMMIT = "473661e52ea9d6db7988e038212b4de9f34aa6e1"
BLOB = "7e5d2249242ddeff8dd88ff7da80b4d4e0689d06"
SOURCE = "Joshua-Stapleton/51-addition-alt-basis-3x3-rank23"
CERTIFICATE = Path(__file__).with_name("data") / "rank23-certificate.json"
VERSION = "rns8-rank23-slp-v1"


@dataclass(frozen=True)
class Node:
    left: int
    right: int
    subtract: bool


@dataclass(frozen=True)
class Plan:
    inputs: int
    nodes: tuple
    outputs: tuple
    steps: tuple
    output_slots: tuple
    slots: int
    source_additions: int


def compile_stage(lines, inputs, outputs):
    """Strict grammar, left-associative binary ops, structural CSE, lowest free slot.

    Inputs/outputs are names, aliases need no operation. Final outputs remain live
    until the whole stage ends; a destination may reuse a last-read operand slot.
    """
    if not inputs or len(set(inputs)) != len(inputs) or not outputs:
        raise ValueError("nonempty unique inputs and outputs required")
    names = {name: i for i, name in enumerate(inputs)}
    nodes, seen, additions = [], {}, 0
    for line in lines:
        if not isinstance(line, str):
            raise ValueError("SLP statement must be text")
        match = re.fullmatch(r"([a-z][0-9]+):=([a-z][0-9]+(?:[+-][a-z][0-9]+)*);", line)
        if not match:
            raise ValueError("invalid SLP grammar")
        destination, expression = match.groups()
        tokens = re.findall(r"[a-z][0-9]+|[+-]", expression)
        try:
            value = names[tokens[0]]
            for at in range(1, len(tokens), 2):
                other, subtract = names[tokens[at+1]], tokens[at] == "-"
                additions += 1
                # Only proven syntactic symmetry of integer addition is used.
                left, right = (value, other) if subtract else sorted((value, other))
                key = (left, right, subtract)
                if key not in seen:
                    seen[key] = len(inputs) + len(nodes)
                    nodes.append(Node(*key))
                value = seen[key]
            names[destination] = value
        except KeyError as exc:
            raise ValueError("read before assignment") from exc
    try:
        result = tuple(names[name] for name in outputs)
    except KeyError as exc:
        raise ValueError("missing stage output") from exc
    last = [-1] * (len(inputs) + len(nodes))
    for i, node in enumerate(nodes):
        last[node.left] = last[node.right] = i
    for value in result:
        last[value] = len(nodes)
    slots = list(range(len(inputs))) + [-1] * len(nodes)
    occupied = {slots[i] for i in range(len(inputs)) if last[i] >= 0}
    peak, steps = len(inputs), []
    for i, node in enumerate(nodes):
        left, right = slots[node.left], slots[node.right]
        for value in (node.left, node.right):
            if last[value] == i:
                occupied.discard(slots[value])
        slot = next(j for j in range(len(inputs)+len(nodes)) if j not in occupied)
        slots[len(inputs)+i] = slot
        if last[len(inputs)+i] >= i:
            occupied.add(slot)
        peak = max(peak, slot+1)
        steps.append((slot, left, right, node.subtract))
    return Plan(len(inputs), tuple(nodes), result, tuple(steps),
                tuple(slots[value] for value in result), peak, additions)


def evaluate(plan, values, modulus=None):
    if len(values) != plan.inputs or any(type(x) is not int for x in values):
        raise ValueError("integer stage input extent mismatch")
    if modulus is not None and (type(modulus) is not int or not 2 <= modulus <= 256):
        raise ValueError("byte modulus must be 2..256")
    def center(value):
        if modulus is None:
            return value
        value %= modulus
        return value-modulus if value >= (modulus+1)//2 else value
    storage = [0] * plan.slots
    storage[:plan.inputs] = map(center, values)
    for target, left, right, subtract in plan.steps:
        storage[target] = center(storage[left] + (-storage[right] if subtract else storage[right]))
    return [storage[i] for i in plan.output_slots]


def expansion(plan):
    columns = [evaluate(plan, [int(i == j) for i in range(plan.inputs)])
               for j in range(plan.inputs)]
    return [list(row) for row in zip(*columns)]


def matrix(value, rows, columns):
    if not isinstance(value, list) or len(value) != rows or any(
            not isinstance(row, list) or len(row) != columns or
            any(type(x) is not int or x not in (-1, 0, 1) for x in row) for row in value):
        raise ValueError("malformed certificate matrix")
    return value


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def verify_certificate(data):
    if not isinstance(data, dict) or data.get("dimension") != [3, 3, 3] or any(
            type(x) is not int for x in data["dimension"]) or type(data.get("rank")) is not int or data["rank"] != 23:
        raise ValueError("only the pinned 3x3 rank23 contract is supported")
    try:
        original = {key: matrix(data["original_matrices"][key], *shape)
                    for key, shape in (("U", (23,9)), ("V", (23,9)), ("W", (9,23)))}
        changed = {key: matrix(data["changed_matrices"][key], *shape)
                   for key, shape in (("U", (23,9)), ("V", (23,9)), ("W", (9,23)))}
        basis = {key: matrix(data["basis"][key], 9, 9)
                 for key in ("P", "Q", "R", "P_inv", "Q_inv", "R_inv")}
        stages = {key: compile_stage(data["slp"][key],
                    [prefix+str(i) for i in range(inputs)],
                    [output+str(i) for i in range(outputs)])
                  for key, prefix, output, inputs, outputs in
                  (("U","x","u",9,23),("V","y","v",9,23),("W","p","z",23,9))}
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete certificate") from exc
    identity = [[int(i == j) for j in range(9)] for i in range(9)]
    for key in ("P", "Q", "R"):
        if matmul(basis[key], basis[key+"_inv"]) != identity or matmul(basis[key+"_inv"], basis[key]) != identity:
            raise ValueError("basis inverse identity failed")
    for key, transform in (("U","P"),("V","Q")):
        if matmul(changed[key], basis[transform]) != original[key]:
            raise ValueError("input basis equation failed")
    if matmul(basis["R"], changed["W"]) != original["W"]:
        raise ValueError("output basis equation failed")
    if [stages[key].source_additions for key in ("U","V","W")] != [13,12,26]:
        raise ValueError("source kernel addition count differs from 51")
    for key in ("U", "V", "W"):
        if expansion(stages[key]) != changed[key]:
            raise ValueError("SLP expansion differs from changed matrix")
    for c in range(9):
        for a in range(9):
            for b in range(9):
                value = sum(original["U"][p][a]*original["V"][p][b]*original["W"][c][p] for p in range(23))
                if value != int(a//3 == c//3 and b%3 == c%3 and a%3 == b//3):
                    raise ValueError("integer matrix multiplication tensor failed")
    return stages


def full_plans(data):
    """All five basis additions are executable nodes, never free metadata."""
    verify_certificate(data)
    plans = {
        "U": compile_stage(["x8:=x8-x6;"]+data["slp"]["U"],
                           ["x"+str(i) for i in range(9)], ["u"+str(i) for i in range(23)]),
        "V": compile_stage(["y2:=y2+y0;", "y6:=y6-y7;", "y8:=y8+y5;"]+data["slp"]["V"],
                           ["y"+str(i) for i in range(9)], ["v"+str(i) for i in range(23)]),
        "W": compile_stage(data["slp"]["W"]+["z8:=z8+z2;"],
                           ["p"+str(i) for i in range(23)], ["z"+str(i) for i in range(9)])}
    for key in ("U", "V", "W"):
        if expansion(plans[key]) != data["original_matrices"][key]:
            raise ValueError("allocated full SLP/basis plan differs from original coefficients")
    return plans


def multiply(plans, a, b, modulus=None):
    u, v = evaluate(plans["U"], a, modulus), evaluate(plans["V"], b, modulus)
    return evaluate(plans["W"], [x*y for x,y in zip(u,v)], modulus)


def load_pinned(path=CERTIFICATE):
    raw = Path(path).read_bytes()
    # Git's certificate uses CRLF. Checked-in mathematical data uses LF.
    upstream = raw.replace(b"\r\n", b"\n").rstrip(b"\n").replace(b"\n", b"\r\n")
    digest = hashlib.sha1(b"blob "+str(len(upstream)).encode()+b"\0"+upstream).hexdigest()
    if digest != BLOB:
        raise ValueError("certificate differs from pinned upstream Git blob")
    return json.loads(raw)


def report(data):
    if data != load_pinned():
        raise ValueError("report source identity requires the pinned certificate")
    plans = full_plans(data)
    result = {"schema":VERSION, "repository":SOURCE, "commit":COMMIT, "git_blob":BLOB,
              "kernel_additions":51, "basis_additions":5,
              "verifier_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "plans":{key:asdict(value) for key,value in plans.items()},
              "proof":"exact_integer_basis_slp_and_729_tensor_coefficients",
              "device_execution":"not_run", "performance":None,
              "runtime_dispatch_eligible":False}
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, default=CERTIFICATE)
    args = parser.parse_args()
    print(json.dumps(report(load_pinned(args.certificate)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
