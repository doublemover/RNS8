"""Authored forecast/quoted-cost policy controls; no real session executes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import session_budget as budget
import gpu_qualification as qualification


class SessionBudget(unittest.TestCase):
    def setUp(self):
        parent = qualification.ROOT / "temp/session-budget-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(dir=parent)))
        self.commands = [
            dict(name=name, argv=[str(self.root / "never-executed")], timeout_seconds=10)
            for name in ("first", "second")
        ]
        self.path = self.root / "forecast.json"
        self.forecast = dict(
            schema=budget.FORECAST_SCHEMA,
            stages=[
                dict(name=row["name"], lower_seconds="1", expected_seconds="5", upper_seconds="10")
                for row in self.commands
            ],
        )
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.forecast) + "\n")

    def plan(self, **changes):
        return budget.plan_budget(
            self.commands,
            **dict(total_seconds=100, reserve_seconds=5, forecast_path=self.path, **changes),
        )

    def test_complete_forecast_has_uncertainty_and_no_measurement_credit(self):
        value = self.plan()
        self.assertEqual(
            value["total_forecast_seconds"],
            dict(lower_seconds="7", expected_seconds="15", upper_seconds="25"),
        )
        self.assertEqual(value["blockers"], [])
        self.assertEqual(budget.replay_budget(value, self.commands), value)
        self.assertFalse(value["production_latency"])
        self.assertFalse(value["hardware_provisioned"])
        self.assertFalse(value["automatic_shutdown"])
        self.assertFalse(value["promotion_allowed"])

    def test_operator_quote_caps_owned_time_exactly_without_inventing_provider_prices(self):
        value = self.plan(
            price_per_hour="36", maximum_cost="1", currency="USD", already_incurred_cost="0.1"
        )
        self.assertEqual(value["effective_total_seconds"], 90)
        self.assertEqual(value["quote"]["affordable_whole_seconds"], "90")
        self.assertEqual(value["quote"]["forecast_upper_cost"], "0.35")
        self.assertEqual(budget.replay_budget(value, self.commands), value)
        per_device = self.plan(
            price_per_hour="18",
            maximum_cost="1",
            currency="USD",
            already_incurred_cost="0.1",
            price_scope="per-selected-device",
            selected_devices=2,
        )
        self.assertEqual(per_device["effective_total_seconds"], 90)
        self.assertEqual(per_device["quote"]["effective_hourly_rate"], "36")
        blocked = self.plan(price_per_hour="36", maximum_cost="0.01", currency="USD")
        self.assertTrue(blocked["blockers"])
        self.assertLess(blocked["effective_total_seconds"], blocked["reserve_seconds"])
        no_forecast = budget.plan_budget(
            self.commands,
            total_seconds=100,
            reserve_seconds=5,
            price_per_hour="36",
            maximum_cost="1",
            currency="USD",
        )
        self.assertTrue(
            any("complete operator stage forecast" in item for item in no_forecast["blockers"])
        )

    def test_forecast_requires_exact_names_unique_typed_ranges_and_bounded_values(self):
        original = copy.deepcopy(self.forecast)
        for mode in ("missing", "extra", "duplicate", "bool", "reverse", "oversized", "schema"):
            self.forecast = copy.deepcopy(original)
            if mode == "missing":
                self.forecast["stages"].pop()
            elif mode == "extra":
                self.forecast["stages"].append(dict(self.forecast["stages"][0], name="unknown"))
            elif mode == "duplicate":
                self.forecast["stages"][1]["name"] = "first"
            elif mode == "bool":
                self.forecast["stages"][0]["upper_seconds"] = True
            elif mode == "reverse":
                self.forecast["stages"][0]["expected_seconds"] = "11"
            elif mode == "oversized":
                self.forecast["stages"][0]["upper_seconds"] = "604801"
            else:
                self.forecast["schema"] = "unknown"
            self.write()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.plan()

    def test_resealed_budget_and_changed_forecast_cannot_resume_under_old_policy(self):
        value = self.plan(price_per_hour="36", maximum_cost="1", currency="USD")
        for key, changed in (("effective_total_seconds", 1000), ("reserve_seconds", True)):
            altered = copy.deepcopy(value)
            altered[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                budget.replay_budget(altered, self.commands)
        self.path.write_text(self.path.read_text() + "\n")
        with self.assertRaises(ValueError):
            budget.replay_budget(value, self.commands)

    def test_missing_quote_units_negative_or_nondecimal_values_are_rejected(self):
        for change in (
            dict(currency=None),
            dict(price_per_hour="0"),
            dict(maximum_cost="NaN"),
            dict(maximum_cost=0.5),
            dict(already_incurred_cost="2"),
            dict(price_scope="per-selected-device", selected_devices=0),
        ):
            options = dict(price_per_hour="36", maximum_cost="1", currency="USD")
            options.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan(**options)

    def test_stage_forecast_stops_runner_before_work_that_no_longer_fits(self):
        manifest = dict(
            identity_sha256="authored-session",
            devices=[],
            commands=self.commands,
            budgets=dict(total_seconds=100, export_reserve_seconds=5, minimum_free_bytes=1),
            session_budget=self.plan(),
        )
        ticks = iter((0, 0, 90, 90))

        def must_not_run(*args, **kwargs):
            self.fail("stage that cannot fit must never launch")

        self.assertEqual(
            qualification.execute_plan(
                manifest, self.root, {}, runner=must_not_run, clock=lambda: next(ticks, 90)
            ),
            1,
        )
        result = json.loads((self.root / "results.json").read_text())
        self.assertEqual(result["status"], "stage_forecast_does_not_fit")
        self.assertEqual(result["completed"], 0)
        self.assertFalse(budget.stage_fits(None, 0))
        self.assertTrue(budget.stage_fits(dict(upper_seconds="10"), 10))
        self.assertFalse(budget.stage_fits(dict(upper_seconds="10"), 9.99))


if __name__ == "__main__":
    unittest.main()
