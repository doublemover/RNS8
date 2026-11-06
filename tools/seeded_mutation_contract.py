"""Source-bound production mutations and conservative native test classification."""

from __future__ import annotations

import ast
import hashlib
import re

from process_progress import capture_succeeded


def transform(source: bytes, seed: dict) -> dict:
    if hashlib.sha256(source).hexdigest() != seed.get("source_sha256"):
        raise ValueError("mutation source changed; re-review the exact seed")
    text = source.decode("utf-8").replace("\r\n", "\n")
    tree = ast.parse(text)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == seed["function"]
    ]
    before, after = seed["before"], seed["after"]
    if len(functions) != 1 or not before or before == after or text.count(before) != 1:
        raise ValueError("unique exact production function/operation required")
    function = functions[0]
    start = text.index(before)
    line = 1 + text.count("\n", 0, start)
    if not function.lineno < line <= function.end_lineno:
        raise ValueError("mutation operation is outside the declared production function")
    if (
        "\n" in before
        or "\n" in after
        or text.splitlines()[line - 1].strip() != before
        or not any(
            isinstance(node, (ast.If, ast.Return)) and node.lineno == line
            for node in ast.walk(function)
        )
    ):
        raise ValueError("mutation must replace one complete production guard/return statement")
    changed = text[:start] + after + text[start + len(before) :]
    ast.parse(changed)
    return dict(
        baseline=text,
        mutant=changed,
        line=line,
        source_sha256=seed["source_sha256"],
        baseline_sha256=hashlib.sha256(text.encode()).hexdigest(),
        mutant_sha256=hashlib.sha256(changed.encode()).hexdigest(),
    )


def native_inventory(
    value, *, process_id: int, required_test: str, expected_discovery=None
) -> bool:
    if (
        not isinstance(value, dict)
        or value.get("schema") != "rns8_native_unittest_inventory_v1"
        or type(value.get("process_id")) is not int
        or value.get("process_id") != process_id
        or value.get("assertions_enabled") is not True
    ):
        return False
    discovered, selected, started, executed = (
        value.get(key) for key in ("discovered", "selected", "started", "executed")
    )
    if any(
        not isinstance(rows, list)
        or any(type(row) is not str for row in rows)
        or len(set(rows)) != len(rows)
        for rows in (discovered, selected, started, executed)
    ):
        return False
    if (
        not discovered
        or selected != discovered
        or set(started) != set(selected)
        or required_test not in executed
        or not set(executed).issubset(selected)
    ):
        return False
    if expected_discovery is not None and discovered != expected_discovery:
        return False
    outcomes = value.get("outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != set(selected):
        return False
    counts = value.get("counts")
    states = (
        "passed",
        "failed",
        "error",
        "skipped",
        "expected_failure",
        "unexpected_success",
        "not_run",
    )
    if any(
        not isinstance(value.get(key), list)
        or any(type(name) is not str for name in value[key])
        or len(set(value[key])) != len(value[key])
        for key in states
    ):
        return False
    for state in states:
        expected = []
        for name, row in outcomes.items():
            if (
                not isinstance(row, dict)
                or row.get("state") not in states
                or type(row.get("detail")) is not str
            ):
                return False
            if row["state"] == state:
                expected.append(name)
        if set(expected) != set(value[state]):
            return False
    if set(executed) != {
        name for name, row in outcomes.items() if row["state"] not in {"skipped", "not_run"}
    }:
        return False
    if not isinstance(counts, dict) or any(
        type(counts.get(key)) is not int or counts[key] != len(value[key])
        for key in (
            "discovered",
            "selected",
            "started",
            "executed",
            "passed",
            "failed",
            "error",
            "skipped",
            "expected_failure",
            "unexpected_success",
            "not_run",
        )
    ):
        return False
    return isinstance(value.get("fixture_failures"), list) and not value["fixture_failures"]


def site_observed(
    value, *, process_id: int, seed: dict, variant: str, sha256: str, line: int
) -> bool:
    observed = value.get("observed_test_ids") if isinstance(value, dict) else None
    return (
        isinstance(value, dict)
        and value.get("schema") == "rns8_production_mutation_observation_v1"
        and value.get("process_id") == process_id
        and value.get("seed") == seed["id"]
        and value.get("variant") == variant
        and value.get("overlay_sha256") == sha256
        and type(value.get("source_line")) is int
        and value["source_line"] == line
        and isinstance(observed, list)
        and all(type(name) is str for name in observed)
        and len(set(observed)) == len(observed)
        and seed["required_test"] in observed
    )


def classification(seed, capture, inventory, observation, *, variant_sha256, line, baseline=None):
    if baseline is not None and (
        not isinstance(baseline, dict)
        or baseline.get("status") != "passed"
        or not native_inventory(
            baseline, process_id=baseline.get("process_id"), required_test=seed["required_test"]
        )
        or set(baseline["passed"]) != set(baseline["selected"])
    ):
        return "baseline_failed"
    pid = capture.get("process_id")
    if capture.get("timed_out") is True:
        return "timeout"
    if (
        type(pid) is not int
        or pid <= 0
        or capture.get("timed_out") is not False
        or capture.get("capture_error") is not None
        or capture.get("limit_exceeded") is not None
        or (capture.get("interrupted") is not None and capture.get("interrupted") is not False)
    ):
        return "infrastructure_error"
    if not native_inventory(
        inventory,
        process_id=pid,
        required_test=seed["required_test"],
        expected_discovery=baseline.get("discovered") if baseline else None,
    ):
        return "infrastructure_error"
    if not site_observed(
        observation,
        process_id=pid,
        seed=seed,
        variant="mutant" if baseline else "baseline",
        sha256=variant_sha256,
        line=line,
    ):
        return "not_covered"
    if baseline is None:
        return (
            "baseline_passed"
            if capture_succeeded(capture)
            and inventory.get("status") == "passed"
            and set(inventory["passed"]) == set(inventory["selected"])
            else "baseline_failed"
        )
    outcome = inventory["outcomes"].get(seed["required_test"], {})
    if (
        type(capture.get("returncode")) is int
        and capture["returncode"] == 1
        and inventory.get("status") == "failed"
        and outcome.get("state") == "failed"
        and seed["assertion_marker"] in outcome.get("detail", "")
    ):
        return "killed_by_assertion"
    if (
        capture_succeeded(capture)
        and inventory.get("status") == "passed"
        and set(inventory["passed"]) == set(inventory["selected"])
    ):
        return "survived"
    diagnostic = str(capture.get("stderr", ""))
    if re.search(
        r"(?:ERROR: AddressSanitizer:|WARNING: ThreadSanitizer:|runtime error:)", diagnostic
    ):
        return "killed_by_sanitizer"
    return (
        "infrastructure_error"
        if outcome.get("state") == "error"
        else "unintended_assertion_failure"
    )
