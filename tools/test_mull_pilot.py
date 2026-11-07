"""Authored native Mull schema/tuple controls; no compiler or mutant runs."""

from __future__ import annotations

import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_request import file_digest
from mull_pilot import (
    COLUMNS,
    SOURCES,
    STATES,
    collect,
    compatibility,
    configuration,
    read_database,
)


class MullPilot(unittest.TestCase):
    def test_failed_or_interrupted_instrumented_baseline_restores_owned_environment(self):
        for failure in (None, InterruptedError("fixture interruption")):
            with (
                self.subTest(interrupted=failure is not None),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                environment = dict(ORIGINAL="unchanged")
                session = SimpleNamespace(output=root, environment=environment)
                args = SimpleNamespace(mull_plugin=root / "plugin with spaces")
                with patch(
                    "native_mutation_session.variant",
                    return_value=(dict(classification="baseline_failed"), None),
                    side_effect=failure,
                ) as build:
                    with self.assertRaises(InterruptedError if failure else ValueError):
                        collect(session, args, {}, ["literal case"], {})
                self.assertIs(session.environment, environment)
                self.assertNotIn("MULL_CONFIG", environment)
                self.assertIn("-fpass-plugin=", build.call_args.kwargs["compiler_flags"])
                self.assertFalse((root / "mull-native").exists())

    def test_exact_observed_plugin_tuple_and_bounded_production_filters(self):
        plugin = Path(tempfile.gettempdir()).resolve() / "installed prefix/lib/mull-ir-frontend-23"
        compiler = "clang version 23.1.5 (native fixture)"
        runner = "mull-runner 0.37.0\nPlugins:\n  " + str(plugin) + " (mull 0.37.0, LLVM 23.1.5)\n"
        self.assertFalse(compatibility(compiler, runner, plugin)["qualified"])
        for changed in (
            runner.replace("0.37.0", "0.36.0"),
            runner.replace("LLVM 23.1.5", "LLVM 23.1.4"),
            runner.replace(str(plugin), "/different/plugin"),
            runner + runner,
            "mull-runner 0.37.0\nPlugins:\n  none found",
        ):
            with self.assertRaises(ValueError):
                compatibility(compiler, changed, plugin)
        config = configuration(Path("/copied source"))
        self.assertEqual(len(config["includePaths"]), 2)
        self.assertEqual(config["parallelization"], dict(workers=1, executionWorkers=1))
        self.assertFalse(config["junkDetectionDisabled"])

    def fixture(self, root, statuses=tuple(STATES)):
        sources = {}
        for name in SOURCES:
            path = (root / name).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("int fixture;\n", encoding="utf-8")
            sources[str(path)] = dict(path=str(path), sha256=file_digest(path), relative_path=name)
        path = root / "native.sqlite"
        with sqlite3.connect(path) as connection:
            types = ["INT" if index in range(4, 11) else "TEXT" for index in range(len(COLUMNS))]
            connection.execute(
                "CREATE TABLE mutant ("
                + ",".join(name + " " + kind for name, kind in zip(COLUMNS, types))
                + ")"
            )
            connection.execute("CREATE TABLE information (key TEXT,value TEXT)")
            connection.execute("INSERT INTO information VALUES ('Mull Version','0.37.0')")
            for status in statuses:
                connection.execute(
                    "INSERT INTO mutant VALUES (" + ",".join("?" for _ in COLUMNS) + ")",
                    [
                        "mutant" + str(status),
                        "cxx_gt_to_ge",
                        next(iter(sources)),
                        "",
                        1,
                        1,
                        1,
                        2,
                        status,
                        1 if status == 1 else 0,
                        7,
                        "native output fixture",
                        "",
                        ">=",
                    ],
                )
        return path, sources

    def test_native_states_preserve_crash_timeout_unexecuted_and_review_missingness(self):
        with tempfile.TemporaryDirectory() as directory:
            path, sources = self.fixture(Path(directory))
            value = read_database(path, sources)
            self.assertEqual(len(value["results"]), len(STATES))
            self.assertEqual(set(value["counts"]), set(STATES.values()))
            self.assertFalse(value["qualified"])
            self.assertFalse(any(row["intended_assertion_kill"] for row in value["results"]))
            self.assertNotIn("mutation_score", value)

    def test_live_partial_wrong_producer_source_scope_and_copied_mutants_reject(self):
        for scenario in (
            "sidecar",
            "version",
            "duplicate",
            "outside",
            "status",
            "exit",
            "source",
            "empty",
            "schema",
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                path, sources = self.fixture(Path(directory), statuses=(1,))
                if scenario == "sidecar":
                    Path(str(path) + "-wal").write_bytes(b"partial")
                elif scenario == "source":
                    Path(next(iter(sources))).write_text(
                        "changed production source", encoding="utf-8"
                    )
                else:
                    with sqlite3.connect(path) as connection:
                        if scenario == "version":
                            connection.execute("UPDATE information SET value='0.36.0'")
                        elif scenario == "duplicate":
                            connection.execute("INSERT INTO mutant SELECT * FROM mutant")
                        elif scenario == "outside":
                            connection.execute("UPDATE mutant SET filename='/unrelated/test.cpp'")
                        elif scenario == "status":
                            connection.execute("UPDATE mutant SET execution_status=99")
                        elif scenario == "exit":
                            connection.execute("UPDATE mutant SET exit_status=0")
                        elif scenario == "empty":
                            connection.execute("DELETE FROM mutant")
                        else:
                            connection.execute("ALTER TABLE mutant ADD COLUMN unsupported TEXT")
                with self.assertRaises(ValueError):
                    read_database(path, sources)
        with tempfile.TemporaryDirectory() as directory:
            path, sources = self.fixture(Path(directory), statuses=(1,))
            wrong = copy.deepcopy(sources)
            wrong[next(iter(wrong))]["relative_path"] = "tests/pretend.cpp"
            with self.assertRaises(ValueError):
                read_database(path, wrong)


if __name__ == "__main__":
    unittest.main()
