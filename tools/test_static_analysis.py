"""Authored orchestration controls: no real analyzer, compiler or GPU launches."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis_diagnostics import diagnostic_document
from analysis_inputs import AnalysisInputs, Compilation
from benchmark_request import file_digest
from static_analysis import (
    ROOT,
    build_plan,
    control_header,
    execute,
    hip_projection,
    selected_tasks,
    verify_receipt,
)
from compiled_binary_identity import canonical_digest


class StaticAnalysis(unittest.TestCase):
    def setUp(self):
        (ROOT / "temp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "temp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.build = self.root / "build"
        self.build.mkdir()
        self.tool = self.root / "authored-native-analyzer"
        self.tool.write_bytes(b"AUTHORED NOT EXECUTABLE")
        self.source = self.root / "authored source λ.cpp"
        self.original = "int authored() { return 0; }\n"
        self.source.write_text(self.original)
        metadata = self.root / "authored-configured-input.json"
        metadata.write_text('{"authored_not_a_real_configuration":true}')
        record = dict(
            directory=str(self.build),
            file=str(self.source),
            arguments=[
                str(self.tool),
                "-include",
                str(self.root / "manifest.hpp"),
                "-c",
                str(self.source),
                "-o",
                str(self.build / "authored.obj"),
            ],
        )
        self.context = Compilation(
            "a" * 64,
            "host",
            self.source,
            self.build,
            json.dumps(record),
            (),
            (),
            (),
            ("authored_static",),
            False,
        )
        self.inputs = AnalysisInputs(
            self.build,
            "Debug",
            (self.context,),
            (),
            tuple((str(path), file_digest(path)) for path in (metadata, self.tool)),
            self.tool,
        )
        self.mutation, self.runs = None, 0

    def runner(self, argv, **kwargs):
        label = kwargs["label"]
        value = dict(
            command=argv,
            process_id=123,
            returncode=0,
            timed_out=False,
            interrupted=False,
            elapsed_seconds=0.01,
            stdout="authored output",
            stderr="",
        )
        if label == "analyzer-version":
            value["stdout"] = "LLVM version 23.1.4 (authored output)"
        elif label == "list-checks":
            value["stdout"] = "clang-analyzer-core.NullDereference"
        elif "--export-fixes=" in " ".join(argv):
            path = Path(
                next(
                    argument.split("=", 1)[1]
                    for argument in argv
                    if argument.startswith("--export-fixes=")
                )
            )
            source = Path(argv[argv.index("-p") + 2])
            if label == "parse-error-control":
                missing = next(argument for argument in argv if "missing-header-" in argument)
                row = dict(
                    DiagnosticName="clang-diagnostic-error",
                    Level="Error",
                    DiagnosticMessage=dict(
                        Message="authored missing header " + missing,
                        FilePath=str(source),
                        FileOffset=0,
                        Replacements=[],
                    ),
                )
                rows, value["returncode"] = [row], 1
            else:
                record = json.loads((path.parent / "compile_commands.json").read_text())[0]
                device = "--offload-device-only" in record.get("arguments", [])
                control, lines = control_header()
                line = lines["device" if device else "host"]
                offset = sum(
                    len(part) for part in control.read_bytes().splitlines(keepends=True)[: line - 1]
                )
                row = dict(
                    DiagnosticName="clang-analyzer-core.NullDereference",
                    Level="Warning",
                    DiagnosticMessage=dict(
                        Message="authored branch control",
                        FilePath=str(control),
                        FileOffset=offset,
                        Replacements=[],
                    ),
                )
                rows = [] if self.mutation == "missing_control" else [row]
                value["stderr"] = ". " + str(control) + "\n"
                if self.mutation == "finding":
                    rows.append(
                        dict(
                            DiagnosticName="clang-analyzer-core.NullDereference",
                            Level="Warning",
                            DiagnosticMessage=dict(
                                Message="authored program finding",
                                FilePath=str(source),
                                FileOffset=0,
                                Replacements=[],
                            ),
                        )
                    )
                elif self.mutation == "timeout":
                    value.update(returncode=None, timed_out=True)
                elif self.mutation == "interrupt":
                    value.update(
                        returncode=-2,
                        interrupted=True,
                        capture_error="authored interrupt",
                    )
                elif self.mutation == "source_change":
                    self.source.write_text("int authored() { return 1; }\n")
            path.write_text(json.dumps(dict(MainSourceFile=str(source), Diagnostics=rows)))
        elif label not in {
            "verify-config",
            "dump-config",
            "prepare-inputs",
        } and not label.startswith("hip-compiler-version-"):
            self.assertEqual(kwargs["cwd"], self.build)
            self.assertIn("-fsyntax-only", argv)
            self.assertNotIn("-c", argv)
            self.assertNotIn("-o", argv)
        return value

    def run_authored(self, *, lane="host", mutation=None):
        self.runs += 1
        self.source.write_text(self.original)
        self.mutation = mutation
        out = self.root / ("session-" + str(self.runs))
        with ExitStack() as stack:
            stack.enter_context(patch("static_analysis.load_inputs", return_value=self.inputs))
            stack.enter_context(patch("static_analysis.inventory", return_value=()))
            stack.enter_context(
                patch("static_analysis.importlib.metadata.version", return_value="0.19.1")
            )
            stack.enter_context(
                patch(
                    "static_analysis.read_diagnostics",
                    side_effect=lambda path, **kwargs: diagnostic_document(
                        json.loads(path.read_text()), **kwargs
                    ),
                )
            )
            plan = build_plan(
                self.build,
                out,
                lane=lane,
                tool=self.tool,
                timeout_seconds=10,
                total_seconds=120,
                minimum_free_bytes=1,
                max_artifact_bytes=32 * 1024**2,
            )
            self.assertFalse(out.exists())
            result = execute(plan, runner=self.runner, environment={})
        return result, out

    def verify_protocol_fixture(self, path):
        # Only the explicitly synthetic JSON producer fixture replaces the YAML
        # decoder. Native receipt verification always uses the locked decoder.
        with (
            patch("static_analysis.load_inputs", return_value=self.inputs),
            patch("static_analysis.importlib.metadata.version", return_value="0.19.1"),
            patch(
                "static_analysis.read_diagnostics",
                side_effect=lambda path, **kwargs: diagnostic_document(
                    json.loads(path.read_text()), **kwargs
                ),
            ),
        ):
            return verify_receipt(path)

    def test_control_and_native_records_bind_one_complete_host_pass(self):
        result, out = self.run_authored()
        self.assertEqual(result["status"], "selected_static_contexts_clean")
        self.assertEqual(result["analyzed_passes"], 1)
        self.assertEqual(result["syntax_checked_passes"], 0)
        self.assertEqual(len(result["preflight_controls"]), 1)
        self.assertFalse(result["hardware_qualification"])
        self.assertNotIn("stdout", result["results"][0]["process"])
        receipt = json.loads(Path(result["results"][0]["receipt"]["path"]).read_text())
        self.assertEqual(receipt["diagnostics"][0]["role"], "deliberate_branch_control")
        self.assertTrue(receipt["observed_first_party_headers"])
        self.assertTrue((out / "parse-error-control.stdout.log").exists())
        self.assertEqual(
            self.verify_protocol_fixture(out / "results.json")["identity_sha256"],
            result["identity_sha256"],
        )

    def test_retained_receipt_rejects_log_changes_and_resealed_false_coverage(self):
        result, out = self.run_authored()
        path = out / "results.json"
        original = path.read_bytes()
        result["analyzed_passes"] += 1
        result["identity_sha256"] = canonical_digest(
            {key: value for key, value in result.items() if key != "identity_sha256"}
        )
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "coverage counts"):
            self.verify_protocol_fixture(path)
        path.write_bytes(original)
        (out / "analyzer-version.stdout.log").write_text("changed authored output")
        with self.assertRaisesRegex(ValueError, "artifact.*changed"):
            self.verify_protocol_fixture(path)

    def test_resealed_derived_findings_commands_headers_and_controls_are_not_native_authority(self):
        for mutation in (
            "drop_finding",
            "hide_header",
            "change_command",
            "miss_parse_control",
            "omit_source",
            "forge_projection",
            "boolean_count",
        ):
            with self.subTest(mutation=mutation):
                result, out = self.run_authored(mutation="finding")
                summary = result["results"][0]
                full_path = Path(summary["receipt"]["path"])
                full = json.loads(full_path.read_text())

                def save(path, value, result=result):
                    path.write_text(json.dumps(value))
                    result["retained_artifacts"][str(path)] = file_digest(path)

                if mutation == "drop_finding":
                    full["diagnostics"] = full["diagnostics"][:1]
                    full["status"] = summary["status"] = "analyzed_clean"
                    summary["program_diagnostic_count"] = 0
                    result["status"] = "selected_static_contexts_clean"
                elif mutation == "hide_header":
                    full["observed_first_party_headers"] = []
                elif mutation == "change_command":
                    native_path = out / (summary["identity"] + ".process.json")
                    native = json.loads(native_path.read_text())
                    native["command"] = [
                        arg for arg in native["command"] if arg != "--extra-arg=-H"
                    ]
                    save(native_path, native)
                    full["process"] = native
                elif mutation == "miss_parse_control":
                    native_path = out / "parse-error-control.process.json"
                    native = json.loads(native_path.read_text())
                    native["returncode"] = 0
                    save(native_path, native)
                    result["preflight_controls"][0]["process"] = native
                elif mutation == "omit_source":
                    result["source_artifacts"].pop(str(self.source))
                elif mutation == "forge_projection":
                    plan_path = out / "plan.json"
                    plan = json.loads(plan_path.read_text())
                    plan["tasks"][0]["projected_native_record"]["arguments"] += [
                        "-DUNREVIEWED_PROJECTION=1"
                    ]
                    plan["identity_sha256"] = canonical_digest(
                        {key: value for key, value in plan.items() if key != "identity_sha256"}
                    )
                    save(plan_path, plan)
                    result["plan_identity_sha256"] = plan["identity_sha256"]
                else:
                    summary["program_diagnostic_count"] = True
                save(full_path, full)
                summary["receipt"]["sha256"] = file_digest(full_path)
                result["identity_sha256"] = canonical_digest(
                    {key: value for key, value in result.items() if key != "identity_sha256"}
                )
                (out / "results.json").write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    self.verify_protocol_fixture(out / "results.json")

    def test_missing_control_findings_timeouts_interrupts_and_changed_source_have_distinct_outcomes(
        self,
    ):
        expected = {
            "missing_control": "incomplete_or_unsupported",
            "finding": "findings_require_triage",
            "timeout": "incomplete_or_unsupported",
            "interrupt": "interrupted",
            "source_change": "blocked_or_incomplete",
        }
        for mutation, status in expected.items():
            result, _ = self.run_authored(mutation=mutation)
            self.assertEqual(result["status"], status, mutation)
            self.assertFalse(result["performance_promotion_allowed"])

    def hip_context(self):
        record = self.context.native_record()
        record["arguments"] += ["--offload-arch=gfx1100", "--offload-arch=gfx942"]
        return replace(
            self.context,
            identity="b" * 64,
            kind="hip",
            native_record_json=json.dumps(record),
            targets=("gfx1100", "gfx942"),
            environment=(("HIP_PATH", "authored SDK"),),
        )

    def test_hip_host_and_each_device_are_separate_real_driver_projections(self):
        context = self.hip_context()
        tasks = selected_tasks((context,), "hip-analyzer")
        self.assertEqual(
            [task.stage for task in tasks],
            ["hip-host-analyzer", "hip-device-analyzer", "hip-device-analyzer"],
        )
        for task in tasks[1:]:
            args = hip_projection(task)["arguments"]
            self.assertEqual(
                [arg for arg in args if arg.startswith("--offload-arch=")],
                ["--offload-arch=" + task.target],
            )
            self.assertIn("--offload-device-only", args)
            self.assertIn("-include", args)
        self.inputs = replace(self.inputs, contexts=(context,))
        result, out = self.run_authored(lane="hip-analyzer")
        self.assertEqual(self.verify_protocol_fixture(out / "results.json")["analyzed_passes"], 3)
        self.assertEqual(result["analyzed_passes"], 3)
        self.assertEqual(result["selected_contexts"], 1)
        receipts = [
            json.loads(Path(row["receipt"]["path"]).read_text()) for row in result["results"]
        ]
        self.assertEqual(sum(row["device_kernel_control_observed"] for row in receipts), 2)

    def test_hip_syntax_never_becomes_analyzer_coverage(self):
        self.inputs = replace(self.inputs, contexts=(self.hip_context(),))
        result, out = self.run_authored(lane="hip-syntax")
        self.assertEqual(
            self.verify_protocol_fixture(out / "results.json")["syntax_checked_passes"], 1
        )
        self.assertEqual(result["status"], "syntax_only_not_analyzer_coverage")
        self.assertEqual(result["analyzed_passes"], 0)
        self.assertEqual(result["syntax_checked_passes"], 1)
        self.assertEqual(result["preflight_controls"], [])


if __name__ == "__main__":
    unittest.main()
