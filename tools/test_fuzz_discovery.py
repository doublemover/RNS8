"""Inert controls for native fuzz object admission and bounded CPU orchestration."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import fuzz_discovery as discovery
import fuzz_library_inventory as inventory
import host_instrumentation_session as host


class FuzzDiscoveryControls(unittest.TestCase):
    def test_executed_feedback_requires_retained_library_sources_and_public_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "src").mkdir()
            source = root / "src/api.cpp"
            source.write_text("// line one\n// line two\n")
            retained = [host.reference(source)]
            location = source.as_posix()
            text = (
                f"COVERED_FUNC: hits: 17 edges: 3/8 rns8_create_plan {location}:1\n"
                f"COVERED_FUNC: hits: 8 edges: 2/3 production::admission(int) {location}:2\n"
            )
            result = inventory.library_feedback(text, retained, root=root)
            self.assertEqual(len(result["functions"]), 2)
            self.assertIsNone(result["source_branch_percentage"])
            for bad in (
                "cov: 200 ft: 999",
                text.replace("/src/", "/tests/"),
                text.replace("rns8_create_plan", "LLVMFuzzerTestOneInput"),
                text.replace("edges: 3/8", "edges: 1/8"),
                text.replace(":2\n", ":999\n"),
                text.replace("edges: 3/8", "edges: 9/8"),
            ):
                with self.assertRaises(ValueError):
                    inventory.library_feedback(bad, retained, root=root)
            with self.assertRaises(ValueError):
                inventory.library_feedback(text, [], root=root)

    def test_production_denominator_rejects_missing_or_aliased_translation_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary).resolve()
            compiler = build / "inert-compiler"
            compiler.write_text("never executed")
            sources = [build / name for name in ("first.cpp", "second.cpp")]
            for source in sources:
                source.write_text("int unused_declaration();")
            contexts = []
            for source in sources:
                record = dict(
                    file=str(source),
                    directory=str(build),
                    arguments=[
                        str(compiler),
                        "-fsanitize=fuzzer-no-link,address,undefined",
                        "-fno-sanitize-recover=undefined",
                        "-o",
                        "CMakeFiles/rns8_fuzz_library.dir/" + source.stem + ".o",
                        str(source),
                    ],
                )
                contexts.append(
                    SimpleNamespace(
                        kind="host",
                        source=source,
                        registered_targets=("rns8_fuzz_library",),
                        native_record=lambda record=record: dict(record),
                    )
                )
            metadata = dict(
                schema="rns8_fuzz_instrumentation_v1",
                configuration="Debug",
                scope="configured_not_executed",
                compiler=str(compiler),
                gpu_instrumented=False,
                external_dependencies_instrumented=False,
                sources=[str(source) for source in sources],
                library=str(build / "archive.a"),
            )
            (build / "rns8-fuzz-instrumentation-Debug.json").write_text(json.dumps(metadata))
            (build / "CMakeCache.txt").write_text("RNS8_ENABLE_HIP:BOOL=OFF\n")
            targets = dict(
                schema="rns8_first_party_analysis_targets_v1",
                targets=[
                    dict(
                        name=name,
                        configuration="Debug",
                        type="STATIC_LIBRARY" if index == 0 else "EXECUTABLE",
                        artifact=str(build / ("archive.a" if index == 0 else name)),
                    )
                    for index, name in enumerate(("rns8_fuzz_library", *inventory.TARGETS))
                ],
            )
            (build / "rns8-analysis-targets-Debug.json").write_text(json.dumps(targets))
            inputs = SimpleNamespace(configuration="Debug", compiler=compiler, contexts=contexts)
            with patch.object(inventory, "load_inputs", return_value=inputs):
                self.assertEqual(len(inventory.configured_objects(build, windows=False)[2]), 2)
                inputs.contexts = contexts[:1]
                with self.assertRaisesRegex(ValueError, "every actual"):
                    inventory.configured_objects(build, windows=False)
                inputs.contexts = [contexts[0], contexts[0]]
                with self.assertRaisesRegex(ValueError, "alias"):
                    inventory.configured_objects(build, windows=False)
                inputs.contexts = contexts
                contexts[0].registered_targets = ("ordinary",)
                with self.assertRaisesRegex(ValueError, "ownership"):
                    inventory.configured_objects(build, windows=False)

    def test_plan_does_not_inspect_build_start_compiler_or_write_output(self):
        with tempfile.TemporaryDirectory(dir=discovery.ROOT / "temp") as temporary:
            out = Path(temporary) / "output"
            args = [
                "--build",
                str(Path(temporary) / "missing-build"),
                "--nm",
                "missing-nm",
                "--symbolizer",
                "missing-symbolizer",
                "--out",
                str(out),
            ]
            with (
                patch.object(discovery, "execute") as execute,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(discovery.main(args), 0)
            execute.assert_not_called()
            self.assertFalse(out.exists())
            plan = json.loads(stdout.getvalue())
            self.assertFalse(plan["execute"])
            self.assertFalse(plan["gpu_enabled"])
            self.assertEqual(plan["targets"], list(inventory.TARGETS))

    def test_effective_flags_reject_harness_only_and_overridden_production_objects(self):
        linux = [
            "/compiler",
            "-fsanitize=fuzzer-no-link,address,undefined",
            "-fno-sanitize-recover=undefined",
        ]
        windows = ["C:/compiler", "/MT", "/fsanitize=address", "/clang:-fsanitize=fuzzer-no-link"]
        for argv, native in ((linux, False), (windows, True)):
            self.assertTrue(inventory.fuzz_flags(argv, windows=native)["configured_flags_verified"])
            for override in (
                "-fno-sanitize=all",
                "-fsanitize-ignorelist=omit.txt",
                "--config=hidden.cfg",
                "-flto",
                "-mllvm",
            ):
                with self.assertRaises(ValueError):
                    inventory.fuzz_flags([*argv, override], windows=native)
        for argv, native in (
            (["clang", "-fsanitize=address,undefined"], False),
            ([*windows, "/MD"], True),
            (windows[:-1], True),
            ([*linux, "-fsanitize=fuzzer"], False),
        ):
            with self.assertRaises(ValueError):
                inventory.fuzz_flags(argv, windows=native)
        self.assertTrue(inventory.feedback_symbols("__sanitizer_cov_trace_cmp8 U"))
        self.assertTrue(inventory.feedback_symbols("__sancov_lowest_stack U"))
        self.assertFalse(inventory.feedback_symbols("__asan_report_load8 U"))

    def test_discovery_needs_actual_completed_nonempty_feedback(self):
        text = "#1000 DONE   cov: 122 ft: 401 corp: 7/93b\nstat::number_of_executed_units: 1000\n"
        self.assertEqual(discovery.discovery_stats(text)["executed_units"], 1000)
        for bad in (
            "",
            "stat::number_of_executed_units: 1000\n",
            text.replace("1000\n", "0\n"),
            text.replace("cov: 122", "cov: 0"),
            text + "stat::number_of_executed_units: 1000\n",
        ):
            with self.assertRaises(ValueError):
                discovery.discovery_stats(bad)

    def test_named_corpora_cover_each_harness_and_distinguish_range_rejections(self):
        corpus, counts = discovery.corpus_inputs()
        self.assertEqual(set(corpus), set(inventory.TARGETS))
        self.assertGreater(counts["admitted"], 0)
        self.assertEqual(sum(counts["semantic_counts"].values()), len(corpus[inventory.TARGETS[1]]))
        self.assertEqual(set(counts["semantic_counts"]), {str(index) for index in range(6)})
        classified = discovery.named_seed_contract_counts(corpus, counts)
        self.assertEqual(classified[inventory.TARGETS[0]]["valid_plan_routes"], 127)
        self.assertEqual(classified[inventory.TARGETS[0]]["expected_rejection_routes"], 127)
        self.assertEqual(classified[inventory.TARGETS[1]]["valid_arithmetic_inputs"], 479)
        self.assertEqual(classified[inventory.TARGETS[2]]["valid_json_inputs"], 5)
        self.assertEqual(classified[inventory.TARGETS[2]]["parse_rejected_inputs"], 1)
        with self.assertRaisesRegex(ValueError, "classification differs"):
            discovery.named_seed_contract_counts(corpus, {**counts, "admitted": 1})
        for cases in corpus.values():
            self.assertTrue(cases)
            self.assertEqual(len(cases), len({name for name, _ in cases}))
        descriptors = dict(corpus[inventory.TARGETS[0]])
        self.assertEqual(descriptors["empty"], b"")
        self.assertEqual(len(descriptors["per-tile-semantic-0-invalid-14"]), 22)
        self.assertEqual(descriptors["per-tile-semantic-1-invalid-14"][12], 1)
        self.assertEqual(descriptors["per-tile-semantic-1-invalid-14"][-1], 14)
        self.assertEqual(descriptors["semantic-0-invalid-14"][-1], 14)
        self.assertEqual(len(descriptors["semantic-0-invalid-14"]), 14)
        self.assertEqual(len(descriptors["semantic-2-invalid-14"]), 6)
        self.assertEqual(len(descriptors["semantic-4-invalid-14"]), 5)
        for semantic in range(6):
            raw = descriptors[f"history-semantic-{semantic}"]
            self.assertEqual(raw[-23:-19], b"R8H1")
            self.assertEqual(
                list(raw[-19:]), [3, 4, 9, 0, 5, 3, 6, 5, 3, 7, 8, 4, 5, 3, 10, 11, 1, 10, 11]
            )

    def test_session_preserves_interruption_when_receipt_write_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = host.Session(Path(temporary), {}, 1, 1024)
            with (
                patch.object(host, "run_capture", return_value=dict(interrupted=True)),
                patch.object(host, "write_json", side_effect=OSError("disk full")),
            ):
                with self.assertRaises(InterruptedError):
                    session.run("interrupted", ["inert-never-launched"])

    def test_aggregate_deadline_prevents_another_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = discovery.DiscoverySession(Path(temporary), {}, 1, 1024)
            session.deadline = 0
            with patch.object(host, "run_capture") as run:
                with self.assertRaisesRegex(ValueError, "deadline"):
                    session.run("no-time", ["inert-never-launched"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
