"""Strict pinned 55-addition artifact import, liveness lowering and tensor checker.

Code generation only parses/lowers data. --verify explicitly executes independent
exact arithmetic verification; generation never labels unexecuted checks passed.
"""
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "tools/data/rank23-october-55.txt"
HEADER = ROOT / "src/backend_common/rank23_october55_data.hpp"
BLOB = "f3b618ea06ec620180317e6054b3e901b24a0781"
COMMIT = "938ea3feddc3ebf44eace7a55222d0292a96c1dd"


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Node:
    left: int
    right: int
    left_sign: int
    right_sign: int


def parse(raw):
    text = raw.decode("ascii")
    section = None
    stages = {"U": [], "V": [], "W": []}
    names = {"U": [f"a{r}{c}" for r in range(1, 4) for c in range(1, 4)],
             "V": [f"b{r}{c}" for r in range(1, 4) for c in range(1, 4)],
             "W": [f"M{p}" for p in range(1, 24)]}
    refs = {key: {name: i for i, name in enumerate(value)} for key, value in names.items()}
    outputs = {"U": [f"P{i}" for i in range(1, 24)], "V": [f"Q{i}" for i in range(1, 24)],
               "W": [f"C{r}{c}" for r in range(1, 4) for c in range(1, 4)]}
    products = []
    for line in text.splitlines():
        if line in ("## A-side", "## B-side", "## C-side", "## products"):
            section = {"## A-side": "U", "## B-side": "V", "## C-side": "W", "## products": "products"}[line]
            continue
        if not line or line.startswith("#"): continue
        if section == "products":
            match = re.fullmatch(r"M([0-9]+) = P([0-9]+) \* Q([0-9]+)", line)
            if not match or len(set(match.groups())) != 1: raise ValueError("ordered product contract changed")
            products.append(int(match[1])); continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*) = (-?)([A-Za-z][A-Za-z0-9]*)(?: ([+-]) (-?)([A-Za-z][A-Za-z0-9]*))?", line)
        if section not in stages or not match: raise ValueError("unknown SLP grammar")
        target, minus_a, a, op, minus_b, b = match.groups()
        if target in refs[section] or a not in refs[section] or (b and b not in refs[section]):
            raise ValueError("duplicate name or read before definition")
        sa = -1 if minus_a else 1
        if op is None:
            if sa != 1: raise ValueError("unsupported signed output alias")
            refs[section][target] = refs[section][a]
        else:
            sb = (-1 if minus_b else 1) * (-1 if op == "-" else 1)
            refs[section][target] = len(names[section]) + len(stages[section])
            stages[section].append(Node(refs[section][a], refs[section][b], sa, sb))
    if products != list(range(1, 24)): raise ValueError("exactly 23 ordered products required")
    plans = {}
    for key in stages:
        out = [refs[key][name] for name in outputs[key]]
        plans[key] = {"inputs": len(names[key]), "nodes": [asdict(n) for n in stages[key]], "outputs": out}
    return plans


def allocate(plan):
    inputs, nodes, outputs = plan["inputs"], plan["nodes"], plan["outputs"]
    last = [-1] * (inputs + len(nodes))
    for i, node in enumerate(nodes): last[node["left"]] = last[node["right"]] = i
    for output in outputs: last[output] = len(nodes)
    locations = list(range(inputs)) + [-1] * len(nodes)
    occupied = {i for i in range(inputs) if last[i] >= 0}
    steps, peak = [], inputs
    for i, node in enumerate(nodes):
        left, right = locations[node["left"]], locations[node["right"]]
        for source in (node["left"], node["right"]):
            if last[source] == i: occupied.discard(locations[source])
        slot = next(j for j in range(inputs + len(nodes)) if j not in occupied)
        locations[inputs+i] = slot
        if last[inputs+i] >= i: occupied.add(slot)
        peak = max(peak, slot+1)
        steps.append(dict(node, target=slot, left=left, right=right))
    return {"inputs": inputs, "steps": steps, "slots": peak, "outputs": [locations[x] for x in outputs]}


def expand(plan):
    # Independent exact integer coefficient expansion: each source basis vector
    # is propagated through immutable nodes, not the allocated execution slots.
    width = plan["inputs"]
    values = [[int(i == j) for j in range(width)] for i in range(width)]
    l1 = 1
    for node in plan["nodes"]:
        value = [node["left_sign"] * a + node["right_sign"] * b
                 for a, b in zip(values[node["left"]], values[node["right"]])]
        values.append(value); l1 = max(l1, sum(abs(x) for x in value))
    return [values[x] for x in plan["outputs"]], l1


def evaluate(schedule, values, modulus=None):
    slots = list(values) + [0] * (schedule["slots"] - len(values))
    def center(value):
        if modulus is None: return value
        value %= modulus
        return value - modulus if value >= (modulus + 1) // 2 else value
    slots = [center(x) for x in slots]
    for node in schedule["steps"]:
        slots[node["target"]] = center(node["left_sign"] * slots[node["left"]] + node["right_sign"] * slots[node["right"]])
    return [slots[x] for x in schedule["outputs"]]


def verify(plans):
    matrices, norms = {}, {}
    for key, plan in plans.items():
        matrices[key], norms[key] = expand(plan)
        schedule = allocate(plan)
        for source in range(plan["inputs"]):
            actual = evaluate(schedule, [int(source == j) for j in range(plan["inputs"])])
            if actual != [row[source] for row in matrices[key]]: raise ValueError("allocated liveness schedule changed coefficients")
    if [len(plans[k]["nodes"]) for k in ("U", "V", "W")] != [13, 14, 28]: raise ValueError("55-addition count changed")
    if norms != {"U": 8, "V": 6, "W": 11}: raise ValueError("intermediate coefficient bounds changed")
    for output in range(9):
        for a in range(9):
            for b in range(9):
                actual = sum(matrices["U"][p][a] * matrices["V"][p][b] * matrices["W"][output][p] for p in range(23))
                expected = int(a//3 == output//3 and b%3 == output%3 and a%3 == b//3)
                if actual != expected: raise ValueError("ordered tensor coefficient mismatch")
    return {"tensor_coefficients": 729, "additions": [13, 14, 28], "intermediate_l1": norms,
            "expanded_coefficients_sha256": sha(matrices), "dag_sha256": sha(plans),
            "lowered_schedule_sha256": sha({k: allocate(p) for k, p in plans.items()}),
            "target_execution": "not_run", "performance_qualified": False}


def pinned(path=ARTIFACT):
    raw = Path(path).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != BLOB: raise ValueError("artifact does not match pinned Git blob")
    return raw


def header(raw, plans):
    lines = ["#pragma once", "// Generated from the unmodified Apache-2.0 mathematical artifact.",
             "// gsidebottom/logic commit " + COMMIT, "// Git blob " + BLOB,
             "// See third_party/rank23-october-55/{LICENSE,NOTICE}.",
             '// Artifact SHA256: ' + hashlib.sha256(raw).hexdigest(),
             '// DAG SHA256: ' + sha(plans),
             '// Lowered schedule SHA256: ' + sha({k: allocate(p) for k, p in plans.items()})]
    for key, plan in plans.items():
        lines += [f"constexpr slp::RawStage raw_{key.lower()}() {{", "  slp::RawStage raw{};",
                  f"  raw.inputs={plan['inputs']}; raw.outputs={len(plan['outputs'])}; raw.count={len(plan['nodes'])};"]
        for i, node in enumerate(plan["nodes"]):
            left, right, sa, sb = node["left"], node["right"], node["left_sign"], node["right_sign"]
            # Signed binary nodes use a widened sum then optional free negation.
            if sa == -1 and sb == 1: left, right, sa, sb = right, left, sb, sa
            subtract = sa != sb
            negate = sa == -1
            lines.append(f"  raw.assignments[{i}]={{{plan['inputs']+i},{left},{right},{str(subtract).lower()},{str(negate).lower()}}};")
        for i, value in enumerate(plan["outputs"]): lines.append(f"  raw.results[{i}]={value};")
        lines += ["  return raw;", "}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-header", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    raw = pinned(); plans = parse(raw)
    if args.write_header: HEADER.write_text(header(raw, plans))
    if args.verify: print(json.dumps(verify(plans), sort_keys=True, indent=2))
    if not args.write_header and not args.verify: parser.error("choose generation and/or explicit verification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
