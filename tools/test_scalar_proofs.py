"""Authored scalar proof/result controls; no verifier or compiler is launched."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from native_mutation_contract import change
from scalar_proof_contract import (
    FAMILIES,
    ROOT,
    complete_process,
    labels,
    proof_result,
    property_inventory,
    registry,
    replay_result,
)
from scalar_proof_session import execute


def process(value, code=0):
    return dict(
        returncode=code,
        stdout=json.dumps(value),
        stderr="",
        timed_out=False,
        interrupted=False,
        limit_exceeded=None,
        capture_error=None,
    )


def discovered(family="checked"):
    return {
        "check_case.assertion." + str(index): dict(
            name="check_case.assertion." + str(index),
            description=label,
            expression="independent specification",
            sourceLocation=dict(file="tests/proofs/scalar_arithmetic.cpp", line=str(index)),
            **{"class": "assertion"},
        )
        for index, label in enumerate(sorted(labels(family)), 1)
    }


def proof(inventory, *, fail=None):
    rows = [
        dict(
            property=key,
            description=row["description"],
            status="FAILURE" if key == fail else "SUCCESS",
        )
        for key, row in inventory.items()
    ]
    if fail:
        target = next(row for row in rows if row["property"] == fail)
        target["trace"] = [
            dict(
                stepType="output",
                outputID="rns8.bv.inputs",
                values=[
                    dict(name="integer", width=64, binary=f"{value:064b}")
                    for value in (7, 9, 11, 0, 1, 2)
                ],
            ),
            dict(stepType="failure", property=fail),
        ]
    return [dict(result=rows), dict(cProverStatus="failure" if fail else "success")]


class ScalarProofAdmission(unittest.TestCase):
    def test_native_replay_requires_the_intended_exact_oracle_failure(self):
        intended = ["rns8.bv.checked.failure_preserves_output"]
        good = process(dict(exact_reference_passed=True, failed_properties=[]))
        bad = process(dict(exact_reference_passed=False, failed_properties=intended), 1)
        self.assertTrue(replay_result(good, "checked"))
        self.assertTrue(replay_result(bad, "checked", intended=intended))
        self.assertFalse(replay_result(good, "checked", intended=intended))
        self.assertFalse(replay_result(bad, "checked"))
        for value in (
            dict(exact_reference_passed=0, failed_properties=intended),
            dict(exact_reference_passed=False, failed_properties=[]),
            dict(exact_reference_passed=False, failed_properties=["rns8.bv.checked.exact_value"]),
            dict(exact_reference_passed=False, failed_properties=intended * 2),
        ):
            self.assertFalse(replay_result(process(value, 1), "checked", intended=intended))

    def test_all_discovered_obligations_and_terminal_status_are_required(self):
        for family in FAMILIES:
            inventory = discovered(family)
            value = property_inventory(process([dict(properties=list(inventory.values()))]), family)
            self.assertEqual(value, inventory)
            self.assertEqual(
                proof_result(process(proof(value)), value, family)["verdict"], "proven"
            )
            for mutation in (
                "missing",
                "duplicate",
                "unknown",
                "unreachable",
                "description",
                "terminal",
                "exit",
            ):
                document = proof(value)
                code = 0
                if mutation == "missing":
                    document[0]["result"].pop()
                if mutation == "duplicate":
                    document[0]["result"].append(document[0]["result"][0])
                if mutation in {"unknown", "unreachable"}:
                    document[0]["result"][0]["status"] = (
                        "UNKNOWN" if mutation == "unknown" else "NOT_REACHABLE"
                    )
                if mutation == "description":
                    document[0]["result"][0]["description"] = "another property"
                if mutation == "terminal":
                    document[1]["cProverStatus"] = "inconclusive"
                if mutation == "exit":
                    code = 10
                with self.subTest(family=family, mutation=mutation):
                    self.assertEqual(
                        proof_result(process(document, code), value, family)["verdict"], "unknown"
                    )

    def test_refutation_requires_intended_property_and_concrete_in_domain_trace(self):
        inventory = discovered()
        key = next(iter(inventory))
        intended = [inventory[key]["description"]]
        value = proof(inventory, fail=key)
        receipt = proof_result(process(value, 10), inventory, "checked", intended=intended)
        self.assertEqual(receipt["verdict"], "refuted")
        self.assertEqual(receipt["counterexamples"][0]["inputs"], [7, 9, 11, 0, 1, 2])
        wrong = [label for label in labels("checked") if label not in intended]
        self.assertEqual(
            proof_result(process(value, 10), inventory, "checked", intended=wrong)["verdict"],
            "unknown",
        )
        for mutation in (
            "missing_trace",
            "missing_failure",
            "failure_before_inputs",
            "duplicate_failure",
            "duplicate_inputs",
            "symbolic",
            "wrong_width",
            "domain",
        ):
            doc = copy.deepcopy(value)
            trace = doc[0]["result"][0]["trace"]
            if mutation == "missing_trace":
                doc[0]["result"][0].pop("trace")
            if mutation == "missing_failure":
                trace.pop()
            if mutation == "failure_before_inputs":
                trace.reverse()
            if mutation == "duplicate_failure":
                trace.append(trace[-1])
            if mutation == "duplicate_inputs":
                trace.append(trace[0])
            if mutation == "symbolic":
                trace[0]["values"][0]["binary"] = "?" * 64
            if mutation == "wrong_width":
                trace[0]["values"][0]["width"] = 32
            if mutation == "domain":
                trace[0]["values"][4]["binary"] = f"{2:064b}"
            self.assertEqual(
                proof_result(process(doc, 10), inventory, "checked", intended=intended)["verdict"],
                "unknown",
            )

    def test_safety_and_unwind_failures_never_qualify_mutation_detection(self):
        inventory = discovered()
        safety = "main.division-by-zero.1"
        inventory[safety] = dict(
            name=safety, description="division by zero", **{"class": "division-by-zero"}
        )
        self.assertEqual(
            proof_result(process(proof(inventory, fail=safety), 10), inventory, "checked")[
                "verdict"
            ],
            "unknown",
        )
        document = proof(discovered())
        document[0]["result"].append(
            dict(property="main.unwind.0", description="unwinding assertion", status="SUCCESS")
        )
        self.assertEqual(
            proof_result(process(document), discovered(), "checked")["verdict"], "proven"
        )
        document[0]["result"][-1]["status"] = "FAILURE"
        document[1]["cProverStatus"] = "failure"
        self.assertEqual(
            proof_result(process(document, 10), discovered(), "checked")["verdict"], "unknown"
        )

    def test_partial_processes_and_false_empty_inventories_are_not_proofs(self):
        for field, value in (
            ("timed_out", True),
            ("interrupted", True),
            ("interrupted", 0),
            ("capture_error", "read failed"),
            ("limit_exceeded", "stdout_stderr_bytes"),
            ("returncode", True),
            ("returncode", None),
            ("returncode", -9),
        ):
            result = process(proof(discovered())) | {field: value}
            self.assertFalse(complete_process(result))
            self.assertEqual(proof_result(result, discovered(), "checked")["verdict"], "unknown")
        for rows in (
            [],
            list(discovered().values())[:-1],
            [*discovered().values(), next(iter(discovered().values()))],
        ):
            with self.assertRaises(ValueError):
                property_inventory(process([dict(properties=rows)]), "checked")
        with self.assertRaises(ValueError):
            property_inventory(
                process([dict(properties=list(discovered().values())), dict(messageType="ERROR")]),
                "checked",
            )

    def test_actual_registry_binds_unique_real_source_operations_and_rejects_drift(self):
        policy = registry()
        for seed in policy["seeds"]:
            source = (ROOT / seed["source"]).read_bytes()
            self.assertNotEqual(change(source, seed), source)
            with self.assertRaises(ValueError):
                change(source + b"\n", seed)

    def test_actual_driver_retains_capability_failure_and_stops_before_mutants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"

            class NoCapability:
                def __init__(self, *args):
                    self.stages = []

                def run(self, name, argv, **kwargs):
                    self.stages.append(dict(name=name, command=argv))
                    if name == "cbmc-version":
                        return process(None) | dict(stdout="6.8.0\n")
                    return process(None, 6) | dict(stderr="unsupported production source syntax")

            selected = dict(
                output=str(output),
                timeout_seconds=1,
                artifact_limit_bytes=1024**2,
                compiler_environment={},
                models=["LP64", "LLP64"],
                tool_invocations=dict(cbmc="/authored/cbmc"),
                policy=registry(),
            )
            with (
                patch("scalar_proof_session.Session", NoCapability),
                patch("scalar_proof_session.compiler_environment", return_value={}),
            ):
                value = execute(selected)
            self.assertEqual(value["status"], "unknown")
            self.assertEqual(len(value["rows"]), 6)
            self.assertTrue(all(row["verdict"] == "unknown" for row in value["rows"]))
            self.assertFalse(any(row["intended_negative_control"] for row in value["rows"]))
            self.assertTrue((output / "scalar-proof.json").is_file())
            self.assertFalse(value["production_eligible"])


if __name__ == "__main__":
    unittest.main()
