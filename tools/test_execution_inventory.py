"""Authored native CTest producer fixtures; no project test processes spawned."""

from __future__ import annotations

import copy
import unittest
from xml.sax.saxutils import quoteattr

from execution_inventory import discovery, junit, reconcile


def inventory(*names, properties=None):
    return dict(
        kind="ctestInfo",
        version=dict(major=1, minor=0),
        tests=[
            dict(
                name=name,
                command=["/build/rns8_tests", name],
                properties=[
                    dict(name=key, value=value)
                    for key, value in (properties or {}).get(name, {}).items()
                ],
            )
            for name in names
        ],
    )


def report(*rows):
    counts = dict(tests=len(rows), failures=0, skipped=0, disabled=0)
    cases = []
    for name, state, reason in rows:
        if state != "run":
            counts[dict(fail="failures", notrun="skipped", disabled="disabled")[state]] += 1
        payload = ""
        if state in {"fail", "notrun"}:
            tag = "failure" if state == "fail" else "skipped"
            payload = f"<{tag} message={quoteattr(reason)}/>"
        cases.append(
            f'<testcase name={quoteattr(name)} status={quoteattr(state)} time="0.001">{payload}<system-out/></testcase>'
        )
    attributes = " ".join(f'{key}="{value}"' for key, value in counts.items())
    return f"<testsuite {attributes}>" + "".join(cases) + "</testsuite>"


class NativeInventory(unittest.TestCase):
    def test_full_selection_and_explicit_exclusion_denominator(self):
        full = inventory("mandatory", "optional", "excluded")
        selected = copy.deepcopy(full)
        selected["tests"].pop()
        value = reconcile(
            full,
            selected,
            report(("mandatory", "run", ""), ("optional", "run", "")),
            required=["mandatory"],
        )
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["counts"]["discovered"], 3)
        self.assertEqual(value["counts"]["executed"], 2)
        self.assertEqual(value["excluded"], ["excluded"])
        self.assertFalse(value["production_latency"])
        self.assertEqual(
            set(value["selected"][0]),
            {"name", "command_sha256", "properties_sha256", "disabled", "labels"},
        )

    def test_native_skip_code_and_disabled_are_unexecuted_and_explicit(self):
        full = inventory(
            "mandatory",
            "gpu",
            "disabled",
            properties={"gpu": {"SKIP_RETURN_CODE": 4}, "disabled": {"DISABLED": True}},
        )
        source = report(
            ("mandatory", "run", ""),
            ("gpu", "notrun", "SKIP_RETURN_CODE=4"),
            ("disabled", "disabled", ""),
        )
        value = reconcile(
            full,
            full,
            source,
            required=["mandatory"],
            allowed_skips={
                "gpu": "CPU-only build has no HIP",
                "disabled": "explicit feature exclusion",
            },
        )
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["executed"], ["mandatory"])
        self.assertEqual(value["skipped"], ["gpu"])
        self.assertEqual(value["disabled"], ["disabled"])
        blocked = reconcile(full, full, source, required=["mandatory"])
        self.assertIn("unexpected_skip:gpu", blocked["blockers"])
        self.assertIn("unexpected_disabled:disabled", blocked["blockers"])

    def test_startup_or_fixture_failure_is_not_an_allowed_skip(self):
        full = inventory(
            "mandatory",
            "optional",
            properties={
                "optional": {"SKIP_RETURN_CODE": 4, "SKIP_REGULAR_EXPRESSION": ["unsupported"]}
            },
        )
        for reason in (
            "Unable to find executable",
            "Fixture dependency failed",
            "SKIP_RETURN_CODE=77",
            "SKIP_RETURN_CODE=04",
            "SKIP_REGULAR_EXPRESSION",
            "SKIP_INSUFFICIENT_RESOURCES",
        ):
            with self.subTest(reason=reason):
                value = reconcile(
                    full,
                    full,
                    report(("mandatory", "run", ""), ("optional", "notrun", reason)),
                    required=["mandatory"],
                    allowed_skips={"optional": "reviewed GPU-only case"},
                )
                self.assertEqual(value["not_run"], ["optional"])
                self.assertEqual(value["status"], "failed")

    def test_exact_native_skip_expression_marker(self):
        full = inventory(
            "mandatory",
            "optional",
            properties={"optional": {"SKIP_REGULAR_EXPRESSION": ["unsupported"]}},
        )
        value = reconcile(
            full,
            full,
            report(
                ("mandatory", "run", ""), ("optional", "notrun", "SKIP_REGULAR_EXPRESSION_MATCHED")
            ),
            required=["mandatory"],
            allowed_skips={"optional": "documented unsupported case"},
        )
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["counts"]["skipped"], 1)

    def test_missing_extra_failed_disabled_and_mandatory_are_blockers(self):
        full = inventory("mandatory", "other")
        for source, blocker in (
            (report(("mandatory", "run", "")), "selected_tests_missing_from_junit"),
            (
                report(("mandatory", "run", ""), ("other", "run", ""), ("extra", "run", "")),
                "unselected_tests_in_junit",
            ),
            (
                report(("mandatory", "fail", "Failed"), ("other", "run", "")),
                "application_tests_failed",
            ),
            (
                report(("mandatory", "disabled", ""), ("other", "run", "")),
                "disabled_state_changed:mandatory",
            ),
        ):
            with self.subTest(blocker=blocker):
                value = reconcile(full, full, source, required=["mandatory"])
                self.assertEqual(value["status"], "failed")
                self.assertIn(blocker, value["blockers"])
        value = reconcile(
            full, inventory("other"), report(("other", "run", "")), required=["mandatory"]
        )
        self.assertIn("mandatory_test_did_not_pass:mandatory", value["blockers"])

    def test_selected_native_command_and_properties_must_match_discovery(self):
        full = inventory(
            "mandatory",
            properties={"mandatory": {"LABELS": ["cpu"], "ENVIRONMENT": ["TOKEN=private"]}},
        )
        for key, replacement in (("command", ["/another/binary"]), ("properties", [])):
            changed = copy.deepcopy(full)
            changed["tests"][0][key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differ"):
                reconcile(full, changed, report(("mandatory", "run", "")), required=["mandatory"])

    def test_empty_duplicate_and_malformed_discovery_refused(self):
        full = inventory("mandatory")
        malformed = [inventory(), inventory("x", "x")]
        for key, replacement in (
            ("command", []),
            ("properties", [{"name": "DISABLED", "value": 1}]),
            ("properties", [{"name": "SKIP_RETURN_CODE", "value": True}]),
            ("properties", [{"name": "SKIP_RETURN_CODE", "value": 256}]),
            ("properties", [{"name": "SKIP_REGULAR_EXPRESSION", "value": "unsupported"}]),
        ):
            changed = copy.deepcopy(full)
            changed["tests"][0][key] = replacement
            malformed.append(changed)
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                discovery(value)
        for required in ([], ["x", "x"], [None]):
            with self.subTest(required=required), self.assertRaises(ValueError):
                reconcile(full, full, report(("mandatory", "run", "")), required=required)
        full["tests"][0]["command"].append("")
        self.assertEqual(discovery(full)["mandatory"].command[-1], "")

    def test_native_junit_status_payload_counts_and_xml_safety(self):
        source = report(("mandatory", "run", ""))
        for malformed in (
            '<testsuite tests="0" failures="0" disabled="0" skipped="0"/>',
            source.replace('tests="1"', 'tests="2"'),
            source.replace('time="0.001"', 'time="nan"'),
            source.replace('status="run"', 'status="unknown"'),
            source.replace('status="run"', 'status="fail"'),
            source.replace("<system-out/>", "<skipped/><system-out/>"),
            source.replace("<system-out/>", "<error/><system-out/>"),
            "<testsuites>" + source + "</testsuites>",
            '<!DOCTYPE testsuite [<!ENTITY bad "untrusted">]>' + source,
        ):
            with self.subTest(source=malformed), self.assertRaises(ValueError):
                junit(malformed)


if __name__ == "__main__":
    unittest.main()
