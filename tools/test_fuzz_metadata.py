"""Authored production JSON boundary and property-driver failure controls."""

from unittest.mock import patch
import copy
import json
import unittest

from benchmark_schema import BenchmarkSchemaError
from fuzz_metadata import FIXTURES, bounded_json, exercise, mutations


class MetadataProperties(unittest.TestCase):
    def test_positive_production_fixtures_and_wrong_type_registries(self):
        fixtures = sorted(FIXTURES.glob("*.json"))
        self.assertTrue(fixtures, "production metadata fixture inventory must not be empty")
        for path in fixtures:
            original = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(exercise(json.dumps(original).encode()), "accepted", path.name)
            for key in (
                "backend_selected",
                "backend_requested",
                "benchmark_execution_mode",
                "semantics",
                "pack_mode",
            ):
                for replacement in ([], {}, ["hip-direct"]):
                    value = copy.deepcopy(original)
                    value[key] = replacement
                    self.assertEqual(
                        exercise(json.dumps(value).encode()),
                        "schema_rejected",
                        (path.name, key, replacement),
                    )

    def test_unexpected_exceptions_and_api_disagreement_are_failures(self):
        with patch("fuzz_metadata.validation_errors", side_effect=TypeError("authored defect")):
            with self.assertRaisesRegex(TypeError, "authored defect"):
                exercise(b"{}")
        with (
            patch("fuzz_metadata.validation_errors", return_value=[]),
            patch(
                "fuzz_metadata.validate_capture",
                side_effect=BenchmarkSchemaError("authored rejection"),
            ),
        ):
            with self.assertRaisesRegex(AssertionError, "disagree"):
                exercise(b"{}")

        def mutate(value):
            value["changed"] = True
            raise BenchmarkSchemaError("authored rejection")

        with (
            patch("fuzz_metadata.validation_errors", return_value=["authored rejection"]),
            patch("fuzz_metadata.validate_capture", side_effect=mutate),
        ):
            with self.assertRaisesRegex(AssertionError, "mutated"):
                exercise(b"{}")

    def test_parser_budgets_and_literal_reproducer_bytes(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":1e999}', b"\xff", b"[" * 65 + b"0" + b"]" * 65):
            self.assertEqual(exercise(raw), "parse_or_budget_rejected")
        self.assertEqual(bounded_json(b'{"escaped":"[\\"{["}'), {"escaped": '["{['})
        self.assertEqual(exercise(b"[]"), "root_rejected")
        generated = list(mutations(b'{"schema_version":4,"nested":{"key":1}}'))
        self.assertEqual(generated[0][1], b'{"schema_version":4,"nested":{"key":1}}')
        self.assertTrue(any(row[0]["path"] == ["nested", "key"] for row in generated))
        self.assertTrue(all(len(row[1]) < 1024**2 for row in generated))


if __name__ == "__main__":
    unittest.main()
