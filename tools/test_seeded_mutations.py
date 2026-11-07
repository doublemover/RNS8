"""Authored mutation orchestration/classification controls; no mutation subprocesses."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from seeded_mutation_contract import classification, transform
import seeded_mutation_session as session


def seed_fixture(source):
    return dict(
        id="guard-source",
        module="production",
        source="tools/production.py",
        function="gate",
        source_sha256=hashlib.sha256(source).hexdigest(),
        before="if value:",
        after="if False and value:",
        test_module="test_gate",
        required_test="test_gate.Case.test_reject",
        assertion_marker="wrong gate result",
        mandatory=True,
    )


def inventory_fixture(seed, *, pid=4321, state="passed"):
    name = seed["required_test"]
    value = dict(
        schema="rns8_native_unittest_inventory_v1",
        process_id=pid,
        assertions_enabled=True,
        status="passed" if state == "passed" else "failed",
        discovered=[name],
        selected=[name],
        started=[name],
        executed=[name],
        fixture_failures=[],
        outcomes={name: dict(state=state, detail="wrong gate result" if state == "failed" else "")},
    )
    for key in (
        "passed",
        "failed",
        "error",
        "skipped",
        "expected_failure",
        "unexpected_success",
        "not_run",
    ):
        value[key] = [name] if key == state else []
    value["counts"] = {
        key: len(value[key])
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
    }
    return value


def capture_fixture(rc=0):
    return dict(
        process_id=4321,
        returncode=rc,
        timed_out=False,
        interrupted=False,
        limit_exceeded=None,
        capture_error=None,
        stderr="",
        stdout="",
    )


def observation_fixture(seed, variant):
    return dict(
        schema="rns8_production_mutation_observation_v1",
        process_id=4321,
        seed=seed["id"],
        variant=variant,
        overlay_sha256="a" * 64,
        source_line=2,
        observed_test_ids=[seed["required_test"]],
    )


class MutationControls(unittest.TestCase):
    def test_actual_production_transform_requires_exact_source_function_and_operation(self):
        source = b"def gate(value):\n    if value:\n        return False\n    return True\n"
        seed = seed_fixture(source)
        changed = transform(source, seed)
        self.assertEqual(changed["line"], 2)
        self.assertEqual(changed["baseline"], source.decode())
        self.assertIn("if False and value:", changed["mutant"])
        self.assertNotEqual(changed["baseline_sha256"], changed["mutant_sha256"])
        for changed_seed in (
            {**seed, "source_sha256": "f" * 64},
            {**seed, "function": "absent"},
            {**seed, "before": "missing"},
        ):
            with self.assertRaises(ValueError):
                transform(source, changed_seed)
        with self.assertRaises(SyntaxError):
            transform(source, {**seed, "after": "if :"})

    def test_exact_class_method_scope_does_not_admit_nested_or_ambiguous_names(self):
        source = b"class Store:\n    def gate(self, value):\n        if value:\n            return False\n        return True\n"
        seed = {**seed_fixture(source), "function": "Store.gate"}
        self.assertEqual(transform(source, seed)["line"], 3)
        for scope in ("gate", "Other.gate", "Store.missing", "Store.gate.nested", "class.gate"):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                transform(source, {**seed, "function": scope})
        duplicated = source + source
        with self.assertRaises(ValueError):
            transform(duplicated, {**seed, "source_sha256": hashlib.sha256(duplicated).hexdigest()})
        repeated = source + b"\ndef other(value):\n    if value:\n        return False\n"
        self.assertEqual(
            transform(repeated, {**seed, "source_sha256": hashlib.sha256(repeated).hexdigest()})[
                "line"
            ],
            3,
        )
        nested = b"def gate(value):\n    def nested():\n        if value:\n            return False\n    return True\n"
        with self.assertRaises(ValueError):
            transform(nested, seed_fixture(nested))

    def test_named_return_argument_is_exact_and_cannot_change_other_arguments(self):
        source = b"def gate(value):\n    return dict(\n        covered=bool(value),\n        detail=value,\n    )\n"
        seed = {**seed_fixture(source), "before": "covered=bool(value),", "after": "covered=True,"}
        changed = transform(source, seed)
        self.assertEqual(changed["line"], 3)
        self.assertIn("covered=True,", changed["mutant"])
        for after in ("other=True,", "covered=True, detail=False,", "**value,"):
            with self.subTest(after=after), self.assertRaises(ValueError):
                transform(source, {**seed, "after": after})
        with self.assertRaises(ValueError):
            transform(source, {**seed, "before": "bool(value)", "after": "True"})

    def test_only_required_reached_assertion_after_passing_baseline_is_a_kill(self):
        seed = seed_fixture(b"fixture")
        baseline = inventory_fixture(seed)
        value = classification(
            seed,
            capture_fixture(),
            baseline,
            observation_fixture(seed, "baseline"),
            variant_sha256="a" * 64,
            line=2,
        )
        self.assertEqual(value, "baseline_passed")
        failed = inventory_fixture(seed, state="failed")

        def classify(capture=None, inventory=None, observation=None, base=None):
            return classification(
                seed,
                capture_fixture(1) if capture is None else capture,
                failed if inventory is None else inventory,
                observation_fixture(seed, "mutant") if observation is None else observation,
                variant_sha256="a" * 64,
                line=2,
                baseline=baseline if base is None else base,
            )

        self.assertEqual(classify(), "killed_by_assertion")
        self.assertEqual(classify(capture=capture_fixture(), inventory=baseline), "survived")
        self.assertEqual(
            classify(observation={**observation_fixture(seed, "mutant"), "observed_test_ids": []}),
            "not_covered",
        )
        self.assertEqual(classify(base=failed), "baseline_failed")
        self.assertEqual(classify(capture={**capture_fixture(1), "timed_out": True}), "timeout")
        self.assertEqual(
            classify(capture={**capture_fixture(1), "limit_exceeded": "output"}),
            "infrastructure_error",
        )
        self.assertEqual(
            classify(capture={**capture_fixture(1), "returncode": -11}, inventory={}),
            "infrastructure_error",
        )
        wrong = copy.deepcopy(failed)
        wrong["outcomes"][seed["required_test"]]["detail"] = "unrelated assertion"
        self.assertEqual(classify(inventory=wrong), "unintended_assertion_failure")
        wrong = copy.deepcopy(failed)
        wrong["counts"]["failed"] = 0
        self.assertEqual(classify(inventory=wrong), "infrastructure_error")
        self.assertEqual(
            classify(
                observation={**observation_fixture(seed, "mutant"), "overlay_sha256": "f" * 64}
            ),
            "not_covered",
        )

    def test_default_plan_never_imports_mutations_creates_overlays_or_runs_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "temp").mkdir()
            (root / "tools").mkdir()
            source = b"def gate(value):\n    if value:\n        return False\n    return True\n"
            (root / "tools/production.py").write_bytes(source)
            seed = seed_fixture(source)
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(dict(schema="rns8_python_production_mutations_v1", seeds=[seed]))
            )
            python = root / "python"
            python.write_bytes(b"never executed fixture")
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session, "REGISTRY", registry),
                mock.patch.object(session.Session, "run") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(
                    session.main(["--python", str(python), "--out", str(root / "temp/planned")]), 0
                )
                run.assert_not_called()
            plan = json.loads(stdout.getvalue())
            self.assertFalse(plan["runtime_validation_executed"])
            self.assertFalse(plan["original_checkout_mutated"])
            self.assertFalse((root / "temp/planned").exists())
            self.assertEqual((root / "tools/production.py").read_bytes(), source)

    def test_all_baselines_precede_mutants_and_failed_baseline_has_no_mutant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "tools").mkdir()
            (root / "metadata").mkdir()
            source = root / "tools/production.py"
            source.write_bytes(b"source fixture")
            registry = root / "registry.json"
            registry.write_text("{}")
            (root / "metadata/dependencies.json").write_text("{}")
            python = root / "python"
            python.write_bytes(b"never executed")
            seeds = [
                {**seed_fixture(source.read_bytes()), "id": name}
                for name in ("first-gate", "second-gate")
            ]
            calls = []

            def fake_variant(owned, args, seed, prepared, *, kind, baseline=None):
                calls.append((seed["id"], kind))
                status = (
                    "baseline_failed"
                    if seed["id"] == "second-gate"
                    else "baseline_passed"
                    if kind == "baseline"
                    else "killed_by_assertion"
                )
                return dict(classification=status, artifacts={}), inventory_fixture(
                    seed, state="failed" if status == "baseline_failed" else "passed"
                )

            args = SimpleNamespace(python=python, timeout_seconds=60, artifact_mib=128)
            output = root / "owned"
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session, "REGISTRY", registry),
                mock.patch.object(session, "source_inventory", return_value=[]),
                mock.patch.object(session, "variant", side_effect=fake_variant),
                mock.patch.dict(session.os.environ, {}, clear=True),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    session.execute(
                        args, dict(output=str(output)), seeds, {seed["id"]: {} for seed in seeds}
                    ),
                    1,
                )
            self.assertEqual(
                calls,
                [("first-gate", "baseline"), ("second-gate", "baseline"), ("first-gate", "mutant")],
            )
            value = json.loads((output / "report.json").read_text())
            self.assertEqual(value["counts"]["killed_by_assertion"], 1)
            self.assertEqual(value["counts"]["baseline_failed"], 1)
            self.assertEqual(value["attempted_mutants"], 1)


if __name__ == "__main__":
    unittest.main()
