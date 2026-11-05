"""Reconcile native CTest JSON/JUnit inventories without executing tests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest
from json_evidence import parse_evidence_json, same_evidence_value

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 64 * 1024**2
MAX_TESTS = 100_000


@dataclass(frozen=True)
class Test:
    name: str
    command: tuple[str, ...]
    properties_json: str
    disabled: bool
    labels: tuple[str, ...]

    def public(self):
        # Raw input receipts preserve replay details. Reports do not duplicate
        # arbitrary ENVIRONMENT properties or possibly sensitive argv values.
        return dict(
            name=self.name,
            command_sha256=canonical_digest(list(self.command)),
            properties_sha256=canonical_digest(parse_evidence_json(self.properties_json)),
            disabled=self.disabled,
            labels=list(self.labels),
        )


def text(value, name):
    if type(value) is not str or not value or len(value.encode("utf-8")) > 4096 or "\0" in value:
        raise ValueError("invalid test inventory " + name)
    return value


def discovery(value) -> dict[str, Test]:
    if (
        not isinstance(value, dict)
        or value.get("kind") != "ctestInfo"
        or not same_evidence_value(value.get("version"), dict(major=1, minor=0))
        or not isinstance(value.get("tests"), list)
        or not 1 <= len(value["tests"]) <= MAX_TESTS
    ):
        raise ValueError("nonempty native CTest JSON v1 inventory required")
    tests = {}
    for row in value["tests"]:
        if not isinstance(row, dict):
            raise ValueError("test discovery row must be an object")
        name = text(row.get("name"), "name")
        command = row.get("command")
        if name in tests or not isinstance(command, list) or not 1 <= len(command) <= 512:
            raise ValueError("duplicate test or absent native command")
        argv = tuple(text(arg, "argument") if arg != "" else "" for arg in command)
        text(argv[0], "executable")
        properties = row.get("properties")
        if not isinstance(properties, list) or len(properties) > 1024:
            raise ValueError("native test properties are missing or exceed budget")
        indexed = {}
        for item in properties:
            if not isinstance(item, dict) or set(item) != {"name", "value"}:
                raise ValueError("invalid native test property")
            key = text(item["name"], "property name")
            if key in indexed:
                raise ValueError("duplicate native test property")
            indexed[key] = item["value"]
        disabled = indexed.get("DISABLED", False)
        labels = indexed.get("LABELS", [])
        if type(disabled) is not bool or not isinstance(labels, list) or len(labels) > 1024:
            raise ValueError("invalid disabled/label property")
        labels = tuple(text(label, "label") for label in labels)
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate native test label")
        code = indexed.get("SKIP_RETURN_CODE")
        patterns = indexed.get("SKIP_REGULAR_EXPRESSION")
        if code is not None and (type(code) is not int or not 0 <= code <= 255):
            raise ValueError("invalid native skip return code")
        if patterns is not None and (
            not isinstance(patterns, list)
            or not patterns
            or any(type(pattern) is not str or not pattern for pattern in patterns)
        ):
            raise ValueError("invalid native skip expressions")
        tests[name] = Test(
            name, argv, json.dumps(indexed, sort_keys=True, separators=(",", ":")), disabled, labels
        )
    return tests


def junit(source: str) -> dict[str, dict]:
    if len(source.encode("utf-8")) > MAX_BYTES or "<!DOCTYPE" in source or "<!ENTITY" in source:
        raise ValueError("JUnit definitions or byte budget unsupported")
    try:
        root = ET.fromstring(source)
    except ET.ParseError as error:
        raise ValueError("malformed native CTest JUnit") from error
    if root.tag != "testsuite":
        raise ValueError("native CTest single-testsuite JUnit required")
    cases = list(root.findall("testcase"))
    if not 1 <= len(cases) <= MAX_TESTS or len(list(root.iter("testcase"))) != len(cases):
        raise ValueError("nonempty flat CTest testcases required")
    rows, counts = {}, dict(tests=len(cases), failures=0, disabled=0, skipped=0)
    for case in cases:
        name, state = text(case.get("name"), "JUnit name"), case.get("status")
        if name in rows or state not in {"run", "fail", "notrun", "disabled"}:
            raise ValueError("duplicate test or unsupported native CTest status")
        try:
            duration = float(case.get("time", ""))
        except ValueError as error:
            raise ValueError("invalid native CTest duration") from error
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid native CTest duration")
        failures, errors, skips = (case.findall(tag) for tag in ("failure", "error", "skipped"))
        if len(failures) != int(state == "fail") or errors or len(skips) != int(state == "notrun"):
            raise ValueError("CTest status contradicts failure/skip payload")
        if state == "fail":
            counts["failures"] += 1
        if state == "disabled":
            counts["disabled"] += 1
        if state == "notrun":
            counts["skipped"] += 1
        rows[name] = dict(
            state=state,
            time_seconds=duration,
            reason=skips[0].get("message", "")
            if skips
            else failures[0].get("message", "")
            if failures
            else "",
        )
    for key, expected in counts.items():
        raw = root.get(key, "")
        if not re.fullmatch(r"[0-9]+", raw) or int(raw) != expected:
            raise ValueError("native CTest declared " + key + " contradicts testcases")
    return rows


def reconcile(discovered, selected, report: str, *, required, allowed_skips=None) -> dict:
    all_tests, chosen, outcomes = discovery(discovered), discovery(selected), junit(report)
    required = list(required)
    if (
        not required
        or len(set(required)) != len(required)
        or any(type(name) is not str or not name for name in required)
    ):
        raise ValueError("explicit unique mandatory tests required")
    allowed = {} if allowed_skips is None else allowed_skips
    if not isinstance(allowed, dict) or any(
        name not in chosen or type(reason) is not str or not reason.strip()
        for name, reason in allowed.items()
    ):
        raise ValueError("allowed skips require selected exact names and reviewed reasons")
    for name, test in chosen.items():
        if name not in all_tests or test != all_tests[name]:
            raise ValueError("selected test command/properties differ from full discovery")
    blockers, states = [], dict(passed=[], failed=[], skipped=[], disabled=[], not_run=[])
    missing, extra = (
        sorted(chosen.keys() - outcomes.keys()),
        sorted(outcomes.keys() - chosen.keys()),
    )
    if missing:
        blockers.append("selected_tests_missing_from_junit")
    if extra:
        blockers.append("unselected_tests_in_junit")
    for name in sorted(chosen.keys() & outcomes.keys()):
        test, outcome = chosen[name], outcomes[name]
        state = outcome["state"]
        if test.disabled != (state == "disabled"):
            blockers.append("disabled_state_changed:" + name)
        if state == "run":
            states["passed"].append(name)
        elif state == "fail":
            states["failed"].append(name)
        elif state == "disabled":
            states["disabled"].append(name)
            if name not in allowed:
                blockers.append("unexpected_disabled:" + name)
        else:
            properties = parse_evidence_json(test.properties_json)
            reason = outcome["reason"]
            code = properties.get("SKIP_RETURN_CODE")
            legitimate = (type(code) is int and reason == "SKIP_RETURN_CODE=" + str(code)) or (
                reason == "SKIP_REGULAR_EXPRESSION_MATCHED"
                and bool(properties.get("SKIP_REGULAR_EXPRESSION"))
            )
            if legitimate:
                states["skipped"].append(name)
                if name not in allowed:
                    blockers.append("unexpected_skip:" + name)
            else:
                states["not_run"].append(name)
                blockers.append("test_not_run_without_native_skip_marker:" + name)
    if states["failed"]:
        blockers.append("application_tests_failed")
    if not states["passed"]:
        blockers.append("no_executed_passing_tests")
    for name in required:
        if name not in states["passed"]:
            blockers.append("mandatory_test_did_not_pass:" + name)
    result = dict(
        schema="rns8_native_test_execution_inventory_v1",
        status="failed" if blockers else "passed",
        discovered=[test.public() for test in all_tests.values()],
        selected=[test.public() for test in chosen.values()],
        excluded=sorted(all_tests.keys() - chosen.keys()),
        missing=missing,
        unexpected=extra,
        executed=sorted([*states["passed"], *states["failed"]]),
        **states,
        outcomes=outcomes,
        allowed_skips=allowed,
        required=required,
        blockers=blockers,
        scope="native_ctest_inventory_not_instrumentation_or_gpu_qualification",
        production_latency=False,
    )
    result["counts"] = {
        key: len(result[key])
        for key in (
            "discovered",
            "selected",
            "excluded",
            "missing",
            "unexpected",
            "executed",
            *states,
        )
    }
    return result


def bounded_read(path: Path) -> tuple[str, dict]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("test evidence must be a regular file")
    with path.open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("test evidence outside byte budget")
    return raw.decode("utf-8"), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("discovered", "selected", "junit", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--required", action="append", required=True)
    parser.add_argument("--allowed-skips", type=Path)
    args = parser.parse_args(argv)
    try:
        inputs, values = {}, {}
        for key in ("discovered", "selected", "junit", "allowed_skips"):
            path = getattr(args, key)
            if path is not None:
                source, inputs[key] = bounded_read(path)
                values[key] = source if key == "junit" else parse_evidence_json(source)
        result = reconcile(
            values["discovered"],
            values["selected"],
            values["junit"],
            required=args.required,
            allowed_skips=values.get("allowed_skips"),
        )
        result["inputs"] = inputs
        for reference in inputs.values():
            path = Path(reference["path"])
            if path.stat().st_size > MAX_BYTES or file_digest(path) != reference["sha256"]:
                raise ValueError("test inventory evidence changed during reconciliation")
        output = args.output.resolve()
        if output == ROOT / "temp" or not output.is_relative_to(ROOT / "temp"):
            raise ValueError("inventory reports require an ignored child of repository temp/")
        result["identity_sha256"] = canonical_digest(result)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return 0 if result["status"] == "passed" else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("test inventory: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
