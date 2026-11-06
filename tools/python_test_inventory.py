"""Record native unittest discovery and per-case execution without replacing it."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys
import unittest

from compiled_binary_identity import canonical_digest
from execution_inventory import bounded_read
from host_instrumentation_session import check_bookends, reference, write_json
from json_evidence import parse_evidence_json
from quality_inventory import ROOT


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        elif isinstance(item, unittest.TestCase):
            yield item
        else:
            raise ValueError("native unittest suite contains an unsupported object")


def discover(modules: list[str]):
    if not modules or len(set(modules)) != len(modules) or len(modules) > 128:
        raise ValueError("explicit unique bounded test modules required")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    references = []
    for name in modules:
        if not re.fullmatch(r"test_[A-Za-z0-9_]+", name):
            raise ValueError("native tools/test module name required")
        path = (ROOT / "tools" / (name + ".py")).resolve(strict=True)
        if not path.is_relative_to(ROOT / "tools"):
            raise ValueError("test module escaped tracked tools")
        before = reference(path)
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != path:
            raise ValueError("test module resolved to another source")
        suite.addTests(loader.loadTestsFromModule(module))
        check_bookends(before)
        references.append(before)
    if loader.errors:
        raise ValueError("native unittest discovery errors: " + "; ".join(loader.errors))
    selected = list(cases(suite))
    names = [test.id() for test in selected]
    if (
        not names
        or len(names) > 100000
        or len(set(names)) != len(names)
        or any(not name or len(name) > 4096 for name in names)
    ):
        raise ValueError("nonempty unique bounded native unittest IDs required")
    return selected, references


class Journal(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity, *, selected, **kwargs):
        super().__init__(stream, descriptions, verbosity, **kwargs)
        self.rows = {name: dict(state="not_run", detail="") for name in selected}
        self.started = []
        self.infrastructure = []

    def mark(self, test, state, detail=""):
        name = test.id()
        if name not in self.rows:
            self.infrastructure.append(dict(name=name, state=state, detail=detail[:8192]))
        else:
            if self.rows[name]["state"] in {"failed", "error", "unexpected_success"}:
                return  # Retain the first failing subtest/setup observation.
            self.rows[name] = dict(state=state, detail=detail[:8192])

    def startTest(self, test):
        self.started.append(test.id())
        super().startTest(test)

    def addSuccess(self, test):
        self.mark(test, "passed")
        super().addSuccess(test)

    def addFailure(self, test, error):
        self.mark(test, "failed", self._exc_info_to_string(error, test))
        super().addFailure(test, error)

    def addError(self, test, error):
        self.mark(test, "error", self._exc_info_to_string(error, test))
        super().addError(test, error)

    def addSkip(self, test, reason):
        self.mark(test, "skipped", reason)
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, error):
        self.mark(test, "expected_failure", self._exc_info_to_string(error, test))
        super().addExpectedFailure(test, error)

    def addUnexpectedSuccess(self, test):
        self.mark(test, "unexpected_success")
        super().addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, error):
        if error is not None:
            self.mark(
                test,
                "failed" if issubclass(error[0], test.failureException) else "error",
                self._exc_info_to_string(error, subtest),
            )
        super().addSubTest(test, subtest, error)

    def document(self, discovered, selected, modules):
        blockers = []
        if not discovered or not selected:
            blockers.append("empty_native_test_inventory")
        if self.infrastructure:
            blockers.append("unselected_native_fixture_or_subtest_outcome")
        if len(set(self.started)) != len(self.started) or set(self.started) != set(selected):
            blockers.append("selected_native_cases_did_not_execute_exactly_once")
        if any(row["state"] != "passed" for row in self.rows.values()):
            blockers.append("selected_native_cases_did_not_all_pass")
        if not self.wasSuccessful():
            blockers.append("native_unittest_result_unsuccessful")
        states = {
            state: sorted(name for name, row in self.rows.items() if row["state"] == state)
            for state in (
                "passed",
                "failed",
                "error",
                "skipped",
                "expected_failure",
                "unexpected_success",
                "not_run",
            )
        }
        executed = [
            name
            for name in self.started
            if name in self.rows and self.rows[name]["state"] not in {"not_run", "skipped"}
        ]
        return dict(
            schema="rns8_native_unittest_inventory_v1",
            status="failed" if blockers else "passed",
            discovered=discovered,
            selected=selected,
            excluded=sorted(set(discovered) - set(selected)),
            started=self.started,
            executed=executed,
            **states,
            outcomes=self.rows,
            modules=modules,
            fixture_failures=self.infrastructure,
            counts=dict(
                discovered=len(discovered),
                selected=len(selected),
                started=len(self.started),
                executed=len(executed),
                **{key: len(value) for key, value in states.items()},
            ),
            assertions_enabled=sys.flags.optimize == 0,
            process_id=os.getpid(),
            blockers=blockers,
            production_latency=False,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if sys.flags.optimize != 0:
            raise ValueError("Python assertions must be enabled; remove -O/PYTHONOPTIMIZE")
        output = args.output.resolve()
        if output == ROOT / "temp" or not output.is_relative_to(ROOT / "temp") or output.exists():
            raise ValueError("fresh ignored unittest inventory output required")
        tests, module_refs = discover(args.module)
        discovered = [test.id() for test in tests]
        names = discovered
        selection_ref = None
        if args.selected:
            source, selection_ref = bounded_read(args.selected)
            names = parse_evidence_json(source)
            if (
                not isinstance(names, list)
                or not names
                or any(type(name) is not str for name in names)
                or len(set(names)) != len(names)
                or not set(names).issubset(discovered)
            ):
                raise ValueError("explicit nonempty unique discovered unittest selection required")
        if args.discover_only:
            value = dict(
                schema="rns8_native_unittest_discovery_v1",
                discovered=discovered,
                selected=names,
                modules=module_refs,
                assertions_enabled=True,
                process_id=os.getpid(),
                executed=[],
                production_latency=False,
            )
        else:
            selection = set(names)
            chosen = [test for test in tests if test.id() in selection]

            def factory(stream, descriptions, verbosity, **kwargs):
                return Journal(stream, descriptions, verbosity, selected=names, **kwargs)

            result = unittest.TextTestRunner(verbosity=2, failfast=False, resultclass=factory).run(
                unittest.TestSuite(chosen)
            )
            value = result.document(discovered, names, module_refs)
        check_bookends(module_refs)
        if selection_ref:
            check_bookends(selection_ref)
            value["selection_input"] = selection_ref
        value["identity_sha256"] = canonical_digest(value)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, value)
        print(json.dumps(dict(inventory=str(output), status=value.get("status", "discovered"))))
        return 0 if value.get("status", "passed") == "passed" else 1
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        print("Python test inventory: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
