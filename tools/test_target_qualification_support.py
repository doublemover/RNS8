"""Entirely authored offline receipts; dummy binaries are never executed."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import uuid

from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest, bind_document
from target_qualification import SCHEMA, PHASES, candidate_binding
from target_validation_report import capture_target
from timing_eligibility import capture_identity
from test_benchmark_sweep_support import bounded_capture
from test_capture_identity_support import (
    synthetic_identity,
    synthetic_static_isa_report,
)
from test_review_reference_support import add_synthetic_full_reference


def write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def reference(path: Path) -> dict:
    return dict(path=str(path.resolve()), sha256=file_digest(path))


class QualificationFixture:
    def __init__(self, root: Path, *, target="gfx942", seed=None):
        self.root = root.resolve()
        self.build = self.root / "build"
        self.build.mkdir(parents=True)
        self.binaries = {}
        self.code = b"AUTHORED SYNTHETIC CODE OBJECT; NOT EXECUTABLE"
        for name in ("rns8-bench", "rns8_tests", "rns8-verify"):
            binary = self.build / name
            binary.write_bytes(b"prefix" + self.code + b"suffix" + name.encode())
            self.binaries[name] = binary
        self.capture = bounded_capture("hip-direct", 300)
        self.capture.pop("_path", None)
        if seed is not None:
            self.capture["seed"] = seed
        self.capture["device"].update(gcn_arch=target, name="synthetic " + target)
        self.capture["configured_amdgpu_targets"] = target
        self.capture["compiler"] = dict(id="clang", version="synthetic")
        self.capture["hip_toolchain"].update(
            hip_root="/synthetic/rocm", hipcc_path="/synthetic/rocm/bin/hipcc"
        )
        key = self.capture["backend_metadata"]["autotune_key"]
        self.capture["backend_metadata"]["autotune_key"] = key.replace(
            "gfx1100", target
        )
        timing = self.capture["timing_metadata"]
        timing.update(
            host_instrumentation_state="hooks_disabled",
            gpu_event_timing=False,
            gpu_event_timing_status="not_requested_production_timing",
            gpu_event_timing_reason="explicit_production_hooks_disabled",
        )
        for name in (
            "gpu_event_phase_order",
            "gpu_event_timing_source",
            "gpu_event_timing_source_scope",
        ):
            timing[name] = None
        self.capture["gpu_event_timings_us"] = self.capture[
            "gpu_event_timing_summary_us"
        ] = None
        synthetic_identity(self.capture)
        self.capture["process_id"] = 111
        self.capture["request_identity"]["executable_sha256"] = file_digest(
            self.binaries["rns8-bench"]
        )
        request = self.capture["request_identity"]
        request["identity_sha256"] = canonical_digest(
            {key: value for key, value in request.items() if key != "identity_sha256"}
        )
        self.capture["artifact_capture"].update(
            executable_sha256=request["executable_sha256"],
            executable_path=str(self.binaries["rns8-bench"]),
        )
        add_synthetic_full_reference(self.capture)
        self.capture_path = write(self.root / "production.json", self.capture)
        self.row = capture_target(self.capture_path)
        self.binding = candidate_binding(self.capture_path, self.capture, self.row)
        self.binary_receipts = {}
        for name, binary in self.binaries.items():
            document = dict(
                schema="rns8_compiled_binary_identity_v1",
                build_manifest=self.capture["build_manifest"],
                library_build_manifest=self.capture["build_manifest"],
                library_abi_declaration_sha256="e" * 64,
            )
            raw = write(self.root / (name + ".identity.json"), document)
            self.binary_receipts[name] = bind_document(
                binary, raw, before_sha256=file_digest(binary)
            )
        self.make_profiler()

    def make_profiler(self):
        self.profiled = copy.deepcopy(self.capture)
        self.profiled["process_id"] = 222
        self.profiled["execution_id"] = str(uuid.uuid4())
        self.profiled["external_instrumentation"] = dict(
            schema="rns8_external_instrumentation_v1",
            kind="amd_profiler",
            tool="rocprofv3",
            mode="kernel_trace",
        )
        self.profiled_path = write(self.root / "profiled.json", self.profiled)
        symbol = "synthetic_selected_symbol"
        self.native = {
            "rocprofiler-sdk-tool": [
                dict(
                    metadata=dict(pid=222, init_time=1, fini_time=100),
                    agents=[
                        dict(id=dict(handle=3), name=self.capture["device"]["gcn_arch"])
                    ],
                    code_objects=[
                        dict(
                            code_object_id=7,
                            rocp_agent=dict(handle=3),
                            uri="file://"
                            + self.binaries["rns8-bench"].as_posix()
                            + "#offset=6&size="
                            + str(len(self.code)),
                        )
                    ],
                    kernel_symbols=[
                        dict(kernel_id=13, code_object_id=7, kernel_name=symbol)
                    ],
                    buffer_records=dict(
                        kernel_dispatch=[
                            dict(
                                start_timestamp=10,
                                end_timestamp=20,
                                correlation_id=dict(internal=17, external=0),
                                dispatch_info=dict(
                                    agent_id=dict(handle=3),
                                    queue_id=dict(handle=5),
                                    dispatch_id=11,
                                    kernel_id=13,
                                    grid_size=dict(x=64, y=1, z=1),
                                    workgroup_size=dict(x=16, y=1, z=1),
                                ),
                            )
                        ]
                    ),
                )
            ]
        }
        self.native_path = write(self.root / "native-trace.json", self.native)
        isa = synthetic_static_isa_report(self.profiled, {})
        isa["code_object_sha256"] = hashlib.sha256(self.code).hexdigest()
        isa["capture_binding"]["code_object_sha256"] = isa["code_object_sha256"]
        self.isa_path = write(self.root / "selected-isa.json", isa)
        self.tool = self.root / "rocprofv3-fixture"
        self.tool.write_bytes(b"authored tool identity, never executable")
        self.version_file = self.root / "profiler-version.txt"
        self.version_file.write_text("rocprofv3 1.4.1\n", encoding="utf-8")
        self.profiler_command = [
            str(self.tool),
            "--kernel-trace",
            "--output-format",
            "json",
            "--",
            str(self.binaries["rns8-bench"]),
            *self.profiled["request_identity"]["arguments"],
        ]
        self.dispatch = dict(
            schema="rns8_amd_profiler_dispatch_receipt_v1",
            adapter="rocprofiler-sdk-json-kernel-dispatch-v1",
            production_capture_sha256=capture_identity(self.capture),
            profiled_capture=reference(self.profiled_path),
            native_trace=reference(self.native_path),
            selected_static_isa=reference(self.isa_path),
            profiler_environment={},
            selected_dispatches=[dict(agent_id=3, queue_id=5, dispatch_id=11)],
            tool_executable=reference(self.tool),
            tool_version_output=reference(self.version_file),
            tool_version="1.4.1",
            acquisition_command=self.profiler_command,
        )
        self.dispatch_path = write(self.root / "dispatch-receipt.json", self.dispatch)

    def qualification(
        self, name="passing", *, failing=None, missing=(), supersedes=()
    ) -> Path:
        directory = self.root / name
        directory.mkdir()
        cache = directory / "CMakeCache.txt"
        cache.write_text("synthetic cache, no configuration was run", encoding="utf-8")
        commands, rows = [], []

        def stage(label, argv, artifacts, **extra):
            command = dict(name=label, argv=argv, **extra)
            stdout = directory / (label + ".stdout.log")
            stdout.write_text("authored process observation only", encoding="utf-8")
            row = dict(
                name=label,
                returncode=1 if label == failing else 0,
                timed_out=False,
                validation_errors=["authored failure"] if label == failing else [],
                artifacts={
                    str(path.resolve()): file_digest(path)
                    for path in [*artifacts, stdout]
                },
            )
            commands.append(command)
            rows.append(row)
            return row

        stage("build", ["cmake", "--build", str(self.build)], [cache])
        for tool in self.binaries:
            binary = self.binaries[tool]
            receipt = self.binary_receipts[tool]
            row = stage(
                "identity-" + tool,
                [str(binary), "--build-identity"],
                [binary, Path(receipt["identity_document"])],
                compiled_identity=True,
                build_dir=str(self.build),
            )
            row["compiled_binary_identity"] = receipt
        required = "qualification pack covers dispatch thresholds and plane tails"
        correctness = directory / "qualification.xml"
        correctness.write_text(
            '<testsuite><testcase name="' + required + '"/></testsuite>',
            encoding="utf-8",
        )
        stage(
            "qualification",
            [
                str(self.binaries["rns8_tests"]),
                "[qualification]",
                "--reporter",
                "junit",
                "--out",
                str(correctness),
            ],
            [self.binaries["rns8_tests"], correctness],
            junit=str(correctness),
            require_test=required,
        )
        ctest = directory / "ctest.xml"
        ctest.write_text(
            '<testsuite><testcase name="authored CTest entry"/></testsuite>',
            encoding="utf-8",
        )
        stage(
            "ctest",
            [
                "ctest",
                "--test-dir",
                str(self.build),
                "--no-tests=error",
                "--output-junit",
                str(ctest),
            ],
            [self.binaries["rns8_tests"], ctest],
            ctest_junit=str(ctest),
        )
        stage(
            "smoke",
            [str(self.binaries["rns8-verify"]), "--hip-smoke"],
            [self.binaries["rns8-verify"]],
        )
        stage(
            "release_capture",
            [
                str(self.binaries["rns8-bench"]),
                *self.capture["request_identity"]["arguments"],
            ],
            [self.binaries["rns8-bench"], self.capture_path],
            capture=str(self.capture_path),
            expected_backend=self.capture["backend_selected"],
            expected_target=self.capture["device"]["gcn_arch"],
        )
        profiler = stage(
            "profiler",
            self.profiler_command,
            [self.native_path, self.profiled_path, self.dispatch_path],
        )
        profiler["fresh_evidence_outputs"] = {
            str(path): file_digest(path)
            for path in (self.native_path, self.profiled_path)
        }
        manifest = dict(
            schema_version=2,
            command_working_directory=str(self.root),
            commands=commands,
        )
        manifest["identity_sha256"] = canonical_digest(manifest)
        manifest_path = write(directory / "manifest.json", manifest)
        run_id = str(uuid.uuid4())
        results = dict(
            schema="rns8_gpu_session_results_v2",
            run_id=run_id,
            manifest_identity_sha256=manifest["identity_sha256"],
            results=rows,
        )
        results_path = write(directory / "results.json", results)
        phases = {}
        for phase in PHASES:
            if phase in missing:
                continue
            tool = (
                "rns8_tests"
                if phase in ("build", "ctest")
                else "rns8-verify"
                if phase == "smoke"
                else "rns8-bench"
            )
            phases[phase] = dict(
                session_id=manifest["identity_sha256"],
                run_id=run_id,
                candidate_identity_sha256=self.binding["identity_sha256"],
                working_directory=str(self.root),
                session_manifest=reference(manifest_path),
                session_results=reference(results_path),
                stage=phase,
                binary_identity_stage="identity-" + tool,
            )
        if "profiler" in phases:
            phases["profiler"]["dispatch_receipt"] = reference(self.dispatch_path)
        if "ctest" in phases:
            phases["ctest"]["correctness_stage"] = "qualification"
        value = dict(
            schema=SCHEMA,
            session_id=manifest["identity_sha256"],
            run_id=run_id,
            candidate=self.binding,
            phases=phases,
            supersedes=list(supersedes),
        )
        value["identity_sha256"] = canonical_digest(value)
        return write(directory / "qualification.json", value)
