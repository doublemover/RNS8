"""Authored execution observations; no environment reads or child processes."""

from __future__ import annotations

import copy
import unittest

from execution_observation import (
    execution_observation,
    validate_execution_request,
    SCHEMA,
)
from execution_environment_contract import relevant_environment_key
from timing_eligibility import phase_blockers
from diagnostic_companion import production_evidence


def capture():
    arguments = ["--record-execution-environment", "--device", "0"]
    environment = {
        "HIP_VISIBLE_DEVICES": "4",
        "ROCP_TOOL_LIBRARIES": "/authored/tool.so",
    }
    return dict(
        execution_environment_witness=dict(
            schema=SCHEMA,
            state="stable",
            arguments=arguments,
            before=environment,
            after=copy.deepcopy(environment),
        )
    )


class ExecutionObservations(unittest.TestCase):
    def test_native_observation_matches_request_and_is_immutable(self):
        value = capture()
        witness = value["execution_environment_witness"]
        request = dict(arguments=witness["arguments"], environment=witness["before"])
        observed = validate_execution_request(value, request, required=True)
        self.assertEqual(observed.environment, witness["before"])
        copied = observed.environment
        copied["HIP_VISIBLE_DEVICES"] = "0"
        self.assertEqual(observed.environment["HIP_VISIBLE_DEVICES"], "4")
        value["execution_environment_witness"]["before"]["HIP_VISIBLE_DEVICES"] = "7"
        self.assertEqual(observed.environment["HIP_VISIBLE_DEVICES"], "4")

    def test_missing_changed_unrequested_and_resealed_caller_labels_cannot_bind(self):
        for mode in (
            "missing",
            "changed",
            "mismatch",
            "argv",
            "request_env",
            "not_requested",
            "extra",
        ):
            value = capture()
            witness = value["execution_environment_witness"]
            request = copy.deepcopy(
                dict(arguments=witness["arguments"], environment=witness["before"])
            )
            if mode == "missing":
                value.pop("execution_environment_witness")
            elif mode == "changed":
                witness["state"] = "changed"
            elif mode == "mismatch":
                witness["after"]["HIP_VISIBLE_DEVICES"] = "0"
            elif mode == "argv":
                request["arguments"].append("--different-work")
            elif mode == "request_env":
                request["environment"]["HIP_VISIBLE_DEVICES"] = "0"
            elif mode == "extra":
                witness["invented"] = True
            else:
                witness.update(state="not_requested", arguments=None, before=None, after=None)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_execution_request(value, request, required=True)

    def test_unknown_variables_types_and_budgets_fail_closed(self):
        for key, item in (
            ("UNRELATED_AUTH_TOKEN", "do not capture"),
            ("HIP_VISIBLE_DEVICES", True),
            ("RNS8_TOO_BIG", "a" * 4097),
        ):
            value = capture()
            value["execution_environment_witness"]["before"][key] = item
            value["execution_environment_witness"]["after"][key] = item
            with self.assertRaises(ValueError):
                execution_observation(value, required=True)
        self.assertTrue(relevant_environment_key("ROCPROFILER_TOOL_CONFIG"))
        self.assertTrue(relevant_environment_key("PERF_CONFIG"))
        self.assertFalse(relevant_environment_key("UNRELATED_AUTH_TOKEN"))
        value = dict(
            execution_environment_witness=dict(
                schema=SCHEMA,
                state="not_requested",
                arguments=None,
                before=None,
                after=None,
            )
        )
        self.assertIsNone(execution_observation(value))
        with self.assertRaises(ValueError):
            validate_execution_request(
                value,
                dict(arguments=["--record-execution-environment"], environment={}),
            )

    def test_requested_or_changed_witness_cannot_be_hidden_from_timing_admission(self):
        value = capture()
        witness = value["execution_environment_witness"]
        value["request_identity"] = dict(
            arguments=witness["arguments"], environment=witness["before"]
        )
        witness["state"] = "changed"
        self.assertIn("native_execution_observation_unqualified", phase_blockers(value))
        self.assertIn(
            "native_execution_observation_unqualified", production_evidence(value)["blockers"]
        )


if __name__ == "__main__":
    unittest.main()
