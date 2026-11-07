"""Authored detector/object/owned-session controls, using fixture processes only."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import host_instrumentation_session as session
from host_instrumentation_contract import (
    detector_result,
    detector_sites,
    native_argv,
    object_path,
    selected_flags,
    symbol_classes,
    valid_result,
)
from test_execution_inventory import inventory, report
from host_runtime_contract import (
    ARCHER_STARTUP,
    APPLICATION_SCOPE,
    WITNESS_PREFIX,
    application_witnesses,
    control_document,
    loaded_inventory,
)


def application_documents(executable, name="mandatory", *, token="a" * 64, windows_runtime=None):
    """Synthetic native protocol bytes; no native process or detector is run."""
    modules = dict(
        schema="rns8_host_loaded_modules_v1",
        process_id=7000,
        modules=[str(executable)],
        virtual_images=[],
        symbol_owners=dict(sanitizer=str(executable)),
        openmp_workers=0,
        scope=APPLICATION_SCOPE,
    )
    if windows_runtime:
        modules["modules"].append(str(windows_runtime))
        modules["symbol_owners"] = {}
    value = dict(
        schema="rns8_host_application_witness_v1",
        token=token,
        process_id=7000,
        parent_process_id=4321,
        executable=str(executable),
        arguments=[name],
        elapsed_ns=10,
        unwinding=False,
        address_feature=True,
        thread_feature=False,
        loaded_modules=modules,
    )
    return [
        dict(copy.deepcopy(value), phase="enter"),
        dict(copy.deepcopy(value), phase="leave", elapsed_ns=1000),
    ]


def application_report(documents, *, name="mandatory", state="run"):
    root = ET.fromstring(report((name, state, "SKIP_RETURN_CODE=4" if state == "notrun" else "")))
    root.find("testcase/system-out").text = "ordinary output\n" + "\n".join(
        WITNESS_PREFIX + json.dumps(value) for value in documents
    )
    return ET.tostring(root, encoding="unicode")


class ApplicationLoaderControls(unittest.TestCase):
    def test_native_child_bookends_bind_command_features_and_unique_module_files(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory).resolve() / "native-application"
            executable.write_bytes(b"synthetic instrumented executable")
            selected = inventory("mandatory")
            selected["tests"][0]["command"][0] = str(executable)
            docs = application_documents(executable)
            kwargs = dict(
                token="a" * 64, executables=[executable], lane="address-undefined", runtimes=[]
            )
            result = application_witnesses(application_report(docs), selected, capture(), **kwargs)
            self.assertEqual(result["cases"][0]["process_id"], 7000)
            self.assertEqual(result["ctest_process_id"], 4321)
            self.assertEqual(len(result["files"]), 1)
            self.assertEqual(result["scope"], APPLICATION_SCOPE)
            self.assertFalse(result["production_latency"])
            session.check_bookends(result)
            mutations = [
                ("token", "b" * 64),
                ("process_id", 4321),
                ("process_id", True),
                ("parent_process_id", 99),
                ("parent_process_id", True),
                ("arguments", []),
                ("executable", "relative"),
                ("elapsed_ns", True),
                ("elapsed_ns", 2**64),
                ("unwinding", True),
                ("address_feature", 1),
                ("thread_feature", True),
                ("schema", "detector_only"),
                ("unexpected", 1),
            ]
            for key, value in mutations:
                bad = copy.deepcopy(docs)
                bad[0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    application_witnesses(application_report(bad), selected, capture(), **kwargs)
            for bad in ([], docs[:1], docs + docs[:1], list(reversed(docs))):
                with self.assertRaises(ValueError):
                    application_witnesses(application_report(bad), selected, capture(), **kwargs)
            for key, value in (("elapsed_ns", 9), ("elapsed_ns", 10**9), ("process_id", 7001)):
                bad = copy.deepcopy(docs)
                bad[-1][key] = value
                with self.assertRaises(ValueError):
                    application_witnesses(application_report(bad), selected, capture(), **kwargs)
            for missing in ("modules", "process_id", "scope"):
                bad = copy.deepcopy(docs)
                bad[0]["loaded_modules"].pop(missing)
                with self.assertRaises(ValueError):
                    application_witnesses(application_report(bad), selected, capture(), **kwargs)
            with self.assertRaises(ValueError):
                application_witnesses(
                    application_report(docs), selected, capture(returncode=1), **kwargs
                )
            executable.write_bytes(b"changed after publication")
            with self.assertRaisesRegex(ValueError, "changed"):
                session.check_bookends(result)

    def test_windows_runtime_and_tsan_features_stay_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, runtime = root / "app.exe", root / "asan.dll"
            executable.write_bytes(b"synthetic app")
            runtime.write_bytes(b"selected runtime")
            selected = inventory("mandatory")
            selected["tests"][0]["command"][0] = str(executable)
            docs = application_documents(executable, windows_runtime=runtime)
            kwargs = dict(
                token="a" * 64,
                executables=[executable],
                lane="windows-address",
                runtimes=[session.reference(runtime)],
                windows_asan=runtime,
            )
            result = application_witnesses(application_report(docs), selected, capture(), **kwargs)
            self.assertEqual(len(result["files"]), 2)
            runtime.write_bytes(b"same name but different runtime")
            with self.assertRaises(ValueError):
                application_witnesses(application_report(docs), selected, capture(), **kwargs)
            docs = application_documents(executable)
            for item in docs:
                item.update(address_feature=False, thread_feature=True)
            kwargs = dict(token="a" * 64, executables=[executable], lane="thread", runtimes=[])
            self.assertEqual(
                len(
                    application_witnesses(application_report(docs), selected, capture(), **kwargs)[
                        "cases"
                    ]
                ),
                1,
            )
            docs[-1]["thread_feature"] = False
            with self.assertRaises(ValueError):
                application_witnesses(application_report(docs), selected, capture(), **kwargs)

    def test_skips_and_nonwitnessed_commands_cannot_gain_loader_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, other = root / "native", root / "other"
            executable.write_bytes(b"native fixture")
            other.write_bytes(b"other fixture")
            selected = inventory("mandatory", "skipped", "script")
            for row in selected["tests"]:
                row["command"][0] = str(other if row["name"] == "script" else executable)
            xml = ET.fromstring(
                report(
                    ("mandatory", "run", ""),
                    ("skipped", "notrun", "SKIP_RETURN_CODE=4"),
                    ("script", "run", ""),
                )
            )
            for element in xml.findall("testcase"):
                if element.get("name") != "script":
                    element.find("system-out").text = "\n".join(
                        WITNESS_PREFIX + json.dumps(value)
                        for value in application_documents(executable, element.get("name"))
                    )
            kwargs = dict(
                token="a" * 64, executables=[executable], lane="address-undefined", runtimes=[]
            )
            result = application_witnesses(
                ET.tostring(xml, encoding="unicode"), selected, capture(), **kwargs
            )
            self.assertEqual([row["test"] for row in result["cases"]], ["mandatory"])
            self.assertEqual(
                {row["test"] for row in result["excluded_cases"]}, {"skipped", "script"}
            )
            # A witness marker from an unselected native command is rejected.
            xml.findall("testcase")[-1].find("system-out").text = WITNESS_PREFIX + "{}"
            with self.assertRaises(ValueError):
                application_witnesses(
                    ET.tostring(xml, encoding="unicode"), selected, capture(), **kwargs
                )

    def test_distinct_application_commands_require_their_own_native_bookends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            names = ("api-case", "exact_cpu_allocation_faults")
            selected = inventory(*names)
            xml = ET.fromstring(report(*((name, "run", "") for name in names)))
            executables = []
            for index, (test, element) in enumerate(
                zip(selected["tests"], xml.findall("testcase"))
            ):
                executable = root / ("api" if index == 0 else "allocation-faults")
                executable.write_bytes(b"synthetic application " + bytes([index]))
                executables.append(executable)
                test["command"] = [str(executable)]  # Dedicated allocation main has no arguments.
                docs = application_documents(executable, test["name"])
                for document in docs:
                    document["arguments"] = []
                    document["process_id"] += index
                    document["loaded_modules"]["process_id"] += index
                element.find("system-out").text = "\n".join(
                    WITNESS_PREFIX + json.dumps(value) for value in docs
                )
            kwargs = dict(
                token="a" * 64, executables=executables, lane="address-undefined", runtimes=[]
            )
            result = application_witnesses(
                ET.tostring(xml, encoding="unicode"), selected, capture(), **kwargs
            )
            self.assertEqual({row["test"] for row in result["cases"]}, set(names))
            self.assertEqual(len(result["files"]), 2)
            self.assertEqual(result["excluded_cases"], [])
            # A parent/API-process snapshot cannot stand in for the second main.
            second = xml.findall("testcase")[1].find("system-out")
            original = second.text
            for replacement in ("", xml.findall("testcase")[0].find("system-out").text):
                second.text = replacement
                with self.assertRaises(ValueError):
                    application_witnesses(
                        ET.tostring(xml, encoding="unicode"), selected, capture(), **kwargs
                    )
            second.text = original
            selected["tests"].pop()
            xml.remove(xml.findall("testcase")[1])
            xml.set("tests", "1")
            with self.assertRaises(ValueError):
                application_witnesses(
                    ET.tostring(xml, encoding="unicode"), selected, capture(), **kwargs
                )

    def test_owned_ctest_sets_fresh_token_only_for_execution_and_retains_module_bookends(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            executable = output / "native"
            executable.write_bytes(b"owned fixture")
            selected = inventory("mandatory")
            selected["tests"][0]["command"][0] = str(executable)
            owned = session.Session(output, {}, 60, 1024**2)
            calls = []

            def fake(command, **kwargs):
                calls.append((command, kwargs["env"]))
                if "--show-only=json-v1" in command:
                    return capture(json.dumps(selected))
                token = kwargs["env"]["RNS8_HOST_RUNTIME_WITNESS_TOKEN"]
                (output / "app.xml").write_text(
                    application_report(application_documents(executable, token=token)),
                    encoding="utf-8",
                )
                return capture()

            with mock.patch.object(session, "run_capture", side_effect=fake):
                _, result = owned.tests(
                    "app",
                    output,
                    Path("/ctest"),
                    required=["mandatory"],
                    runtime_witness=dict(
                        executables=[executable], lane="address-undefined", runtimes=[]
                    ),
                )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(calls), 3)
            self.assertNotIn("RNS8_HOST_RUNTIME_WITNESS_TOKEN", calls[0][1])
            self.assertNotIn("RNS8_HOST_RUNTIME_WITNESS_TOKEN", owned.environment)
            self.assertEqual(len(calls[-1][1]["RNS8_HOST_RUNTIME_WITNESS_TOKEN"]), 64)
            self.assertIn("--test-output-size-passed", calls[-1][0])
            self.assertEqual(len(result["application_runtime_module_files"]), 1)
            executable.write_bytes(b"changed after native tests")
            with self.assertRaisesRegex(ValueError, "changed"):
                session.check_bookends(result)


def capture(stdout="", stderr="", returncode=0):
    return dict(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        process_id=4321,
        timed_out=False,
        interrupted=False,
        limit_exceeded=None,
        capture_error=None,
    )


class DetectorControls(unittest.TestCase):
    def test_only_intended_bound_nonzero_diagnostic_supplies_sensitivity(self):
        rows = (
            (
                "heap-overread",
                86,
                "==4321==ERROR: AddressSanitizer: heap-buffer-overflow\n #0 host_sanitizer_control.cpp:28",
            ),
            (
                "signed-overflow",
                87,
                "host_sanitizer_control.cpp:35 runtime error: signed integer overflow",
            ),
            (
                "race",
                88,
                "WARNING: ThreadSanitizer: data race (pid=4321)\n #0 host_sanitizer_control.cpp:50",
            ),
            (
                "openmp-race",
                88,
                "WARNING: ThreadSanitizer: data race (pid=4321)\n #0 host_sanitizer_control.cpp:80",
            ),
        )
        for mode, rc, diagnostic in rows:
            result = capture(stderr=diagnostic, returncode=rc)
            line = {"heap-overread": 28, "signed-overflow": 35, "race": 50, "openmp-race": 80}[mode]
            with self.subTest(mode=mode):
                value = detector_result(mode, result, source_line=line)
                self.assertEqual(value["status"], "passed")
                self.assertEqual(value["application_pass_count"], 0)
                for key, replacement in (
                    ("returncode", 0),
                    ("returncode", 1),
                    ("returncode", -11),
                    ("stderr", "generic process crash"),
                    ("process_id", None),
                    ("timed_out", True),
                    ("capture_error", "pipe failed"),
                    ("limit_exceeded", "output_bytes"),
                ):
                    changed = {**result, key: replacement}
                    self.assertNotEqual(
                        detector_result(mode, changed, source_line=line)["status"], "passed"
                    )
                self.assertEqual(
                    detector_result(mode, {**result, "returncode": 77}, source_line=line)["status"],
                    "unsupported",
                )
                self.assertNotEqual(
                    detector_result(mode, result, source_line=line + 1)["status"], "passed"
                )
                if mode != "signed-overflow":
                    caller_only = {**result, "stderr": diagnostic.replace("#0", "#2")}
                    self.assertNotEqual(
                        detector_result(mode, caller_only, source_line=line)["status"], "passed"
                    )
        wrong = capture(stderr=rows[0][2].replace("4321", "9999"), returncode=86)
        self.assertEqual(
            detector_result("heap-overread", wrong, source_line=28)["reason"],
            "asan_diagnostic_process_mismatch",
        )
        wrong = capture(stderr=rows[2][2].replace("4321", "9999"), returncode=88)
        self.assertEqual(
            detector_result("race", wrong, source_line=50)["reason"],
            "tsan_diagnostic_process_mismatch",
        )

    def test_source_site_markers_bind_current_native_control_lines(self):
        source = session.ROOT / "tests/controls/host_sanitizer_control.cpp"
        sites = detector_sites(source)
        lines = source.read_text(encoding="utf-8").splitlines()
        for mode, line in sites.items():
            self.assertIn("// rns8-detector-site:" + mode, lines[line - 1])
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.cpp"
            for text in ("", "// rns8-detector-site:race\n" * 2, "// rns8-detector-site:unknown"):
                bad.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    detector_sites(bad)

    def test_synchronized_control_needs_clean_exit_and_exact_typed_values(self):
        doc = dict(schema="rns8_host_detector_valid_v1", value=128, expected=128)
        self.assertEqual(valid_result(capture(), doc)["status"], "passed")
        for changed in (
            None,
            {**doc, "value": 127},
            {**doc, "value": 128.0},
            {**doc, "expected": True},
        ):
            self.assertEqual(valid_result(capture(), changed)["status"], "failed")
        self.assertEqual(valid_result(capture(returncode=1), doc)["status"], "failed")


class LoaderControls(unittest.TestCase):
    def test_only_single_native_document_and_expected_archer_startup_are_accepted(self):
        self.assertEqual(control_document(capture('{"control":1}')), {"control": 1})
        self.assertEqual(
            control_document(capture(ARCHER_STARTUP + '\n{"control":1}'), archer=True),
            {"control": 1},
        )
        for stdout, archer in (
            ("{}", True),
            (ARCHER_STARTUP + "\n{}", False),
            ("{}\n{}", False),
            ("unexpected\n{}", False),
            ("[]", False),
            (ARCHER_STARTUP + "\n" + ARCHER_STARTUP + "\n{}", True),
        ):
            with self.subTest(stdout=stdout), self.assertRaises(ValueError):
                control_document(capture(stdout), archer=archer)
        with self.assertRaises(ValueError):
            control_document(capture("{}", returncode=1))

    def test_native_modules_require_actual_process_and_selected_symbol_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, omp, archer, extra = [
                root / name for name in ("control", "libomp.so.5", "libarcher.so", "extra")
            ]
            for path in (executable, omp, archer, extra):
                path.write_bytes(b"owned fixture " + path.name.encode())
            document = dict(
                schema="rns8_host_loaded_modules_v1",
                process_id=4321,
                modules=[str(executable), str(omp), str(archer)],
                virtual_images=[],
                symbol_owners=dict(sanitizer=str(executable), openmp=str(omp)),
                openmp_workers=2,
                scope="control_process_loader_snapshot_not_all_application_processes",
            )
            kwargs = dict(executable=executable, runtimes=[], openmp=omp, archer=archer)
            value = loaded_inventory(document, capture(), **kwargs)
            self.assertTrue(value["archer_annotation_adapter"])
            self.assertEqual(len(value["files"]), 3)
            self.assertEqual(
                value["external_runtime_accesses_instrumented"],
                "not_established_by_loader_snapshot",
            )
            session.check_bookends(value)
            for changed in (
                {**document, "process_id": 9999},
                {**document, "process_id": True},
                {**document, "modules": [str(executable), str(omp)]},
                {**document, "modules": [str(omp), str(archer)]},
                {**document, "modules": [*document["modules"], str(executable)]},
                {**document, "modules": ["relative"]},
                {**document, "virtual_images": ["unidentified-vdso"]},
                {**document, "symbol_owners": dict(sanitizer=str(extra), openmp=str(omp))},
                {**document, "symbol_owners": dict(sanitizer=str(executable), openmp=str(archer))},
                {**document, "openmp_workers": 1},
                {**document, "openmp_workers": 2.0},
            ):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    loaded_inventory(changed, capture(), **kwargs)
            archer.write_bytes(b"changed after observation")
            with self.assertRaisesRegex(ValueError, "changed"):
                session.check_bookends(value)

    def test_windows_asan_requires_loaded_exact_runtime_not_a_same_named_dll(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, dll = root / "control.exe", root / "asan.dll"
            executable.write_bytes(b"executable")
            dll.write_bytes(b"runtime")
            doc = dict(
                schema="rns8_host_loaded_modules_v1",
                process_id=4321,
                modules=[str(executable), str(dll)],
                virtual_images=[],
                symbol_owners={},
                openmp_workers=0,
                scope="control_process_loader_snapshot_not_all_application_processes",
            )
            runtime = session.reference(dll)
            kwargs = dict(executable=executable, runtimes=[runtime], windows_asan=dll)
            self.assertEqual(len(loaded_inventory(doc, capture(), **kwargs)["files"]), 2)
            dll.write_bytes(b"different runtime")
            with self.assertRaisesRegex(ValueError, "selected compiler runtime"):
                loaded_inventory(doc, capture(), **kwargs)


class ObjectContracts(unittest.TestCase):
    def test_native_argv_preserves_real_command_and_rejects_hidden_expansions(self):
        args = ["/opt/clang++", "-I/some path", "-o", "a.o", "-c", "source.cpp"]
        self.assertEqual(native_argv(dict(arguments=args), windows=False), args)
        self.assertEqual(
            native_argv(
                dict(command="/opt/clang++ '-I/some path' -o a.o -c source.cpp"), windows=False
            ),
            args,
        )
        for row in (
            dict(arguments=args, command="shadow"),
            dict(command="clang++ @flags.rsp"),
            dict(command="clang++ -o a.o && replace a.o"),
            dict(arguments=[]),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                native_argv(row, windows=False)

    def test_object_output_is_single_bound_and_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory).resolve()
            row = dict(directory=str(build), output="out.o")
            self.assertEqual(
                object_path(row, ["clang++", "-o", "out.o"], build, windows=False), build / "out.o"
            )
            for args in (
                ["clang++"],
                ["clang++", "-o", "out.o", "-o", "other.o"],
                ["clang++", "-o", "../escape.o"],
            ):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    object_path(row, args, build, windows=False)
            with self.assertRaises(ValueError):
                object_path(
                    {**row, "output": "other.o"}, ["clang++", "-o", "out.o"], build, windows=False
                )
            self.assertEqual(
                object_path(
                    dict(directory=str(build)), ["clang-cl", "/Foout.obj"], build, windows=True
                ),
                build / "out.obj",
            )

    def test_selected_effective_flags_do_not_accept_disabled_or_mixed_objects(self):
        fixtures = {
            "address-undefined": ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"],
            "thread": ["-fsanitize=thread"],
            "windows-address": ["/fsanitize=address"],
            "coverage": ["-fprofile-instr-generate", "-fcoverage-mapping"],
        }
        for lane, args in fixtures.items():
            with self.subTest(lane=lane):
                self.assertTrue(selected_flags(args, lane)["configured_flags_verified"])
                for bad in (
                    [],
                    [*args, "-fno-sanitize=all"],
                    [*args, "-fsanitize-ignorelist=ignore.txt"],
                    [*args, "--config=hidden.cfg"],
                    [*args, "-fsanitize=memory"],
                ):
                    with self.assertRaises(ValueError):
                        selected_flags(bad, lane)
        with self.assertRaisesRegex(ValueError, "nonrecovering"):
            selected_flags(["-fsanitize=address,undefined"], "address-undefined")
        self.assertEqual(
            symbol_classes(
                "__asan_report_load4 U\n__ubsan_handle_add_overflow_abort U\n", "address-undefined"
            ),
            ["address", "undefined"],
        )
        self.assertEqual(symbol_classes("plain_symbol T 0", "thread"), [])
        self.assertEqual(symbol_classes("__llvm_prf_cnts D 0", "coverage"), ["profile"])


class SessionControls(unittest.TestCase):
    def test_total_deadline_caps_later_children_and_prevents_work_after_exhaustion(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(session.time, "monotonic", side_effect=[100, 102, 106]),
                mock.patch.object(session, "run_capture", return_value=capture()) as run,
            ):
                owned = session.Session(Path(directory), {}, 60, 1024**2, total_seconds=5)
                owned.run("first", ["inert fixture only"])
                self.assertEqual(run.call_args.kwargs["timeout_seconds"], 3)
                with self.assertRaisesRegex(ValueError, "total time budget exhausted"):
                    owned.run("second", ["must not launch"])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(len(owned.stages), 1)
        for value in (True, 0, -1, float("nan"), float("inf"), 86401, 10**1000):
            with self.assertRaises(ValueError):
                session.Session(Path("unused"), {}, 60, 1024**2, total_seconds=value)

    def test_shared_installed_stage_binds_archive_package_sources_and_case_inventory(self):
        for lane, mutation in (
            ("address-undefined", None),
            ("device-address", None),
            ("device-address", "duplicate"),
            ("address-undefined", "source_change"),
            ("address-undefined", "archive_change"),
        ):
            with (
                self.subTest(lane=lane, mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                build = root / "build"
                build.mkdir()
                compiler = root / "clang++"
                compiler.write_bytes(b"inert compiler fixture")
                archive = root / "librns8_static.a"
                archive.write_bytes(b"synthetic instrumented archive")
                native = session.Session(root, {}, 60, 1024**2)
                commands = []

                def invoke(
                    name,
                    command,
                    *,
                    mutation=mutation,
                    lane=lane,
                    commands=commands,
                    root=root,
                    archive=archive,
                    compiler=compiler,
                ):
                    commands.append((name, command))
                    if name == "install":
                        lib = root / "prefix/lib"
                        package = lib / "cmake/RNS8"
                        package.mkdir(parents=True)
                        (package / "RNS8Config.cmake").write_bytes(b"inert installed configuration")
                        (lib / archive.name).write_bytes(
                            b"wrong archive"
                            if mutation == "archive_change"
                            else archive.read_bytes()
                        )
                        examples = root / "prefix/share/RNS8/examples/downstream-cmake"
                        examples.mkdir(parents=True)
                        (examples / "CMakeLists.txt").write_bytes(b"inert consumer configuration")
                        for name in session.CONSUMER_SOURCES.values():
                            (examples / name).write_bytes(b"int main() { return 0; }\n")
                    if name == "consumer-build":
                        consumer = root / "consumer-build"
                        consumer.mkdir()
                        records = []
                        for name in session.CONSUMER_SOURCES.values():
                            source = root / "consumer-source" / name
                            obj = consumer / (name + ".o")
                            obj.write_bytes(b"synthetic object, not executable")
                            flags = (
                                ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
                                if lane == "address-undefined"
                                else [
                                    "-fsanitize=address",
                                    "-fno-sanitize-recover=address",
                                    "-shared-libsan",
                                    "-g",
                                ]
                            )
                            records.append(
                                dict(
                                    directory=str(consumer),
                                    file=str(source),
                                    arguments=[
                                        str(compiler),
                                        *flags,
                                        "-c",
                                        str(source),
                                        "-o",
                                        str(obj),
                                    ],
                                )
                            )
                        if mutation == "duplicate":
                            records[-1] = records[0]
                        (consumer / "compile_commands.json").write_text(json.dumps(records))
                    return capture()

                def cases(*args, lane=lane, mutation=mutation, root=root, **kwargs):
                    self.assertEqual(
                        kwargs["required"],
                        session.DEVICE_CONSUMER_CASES
                        if lane == "device-address"
                        else session.CONSUMERS,
                    )
                    self.assertEqual(kwargs["selection_names"], list(kwargs["required"]))
                    self.assertEqual(len(kwargs["runtime_witness"]["executables"]), 7)
                    if mutation == "source_change":
                        (root / "consumer-source/main.cpp").write_bytes(
                            b"changed source after compilation"
                        )
                    return {}, dict(status="passed", application_runtime_module_files=[])

                witness = dict(lane=lane, runtimes=[])
                if lane == "device-address":
                    witness["device_contract"] = dict(runtimes=dict(hsa=dict(path=str(archive))))
                args = SimpleNamespace(cmake=compiler, ctest=compiler, jobs=2)
                report = {}
                with (
                    mock.patch.object(
                        session,
                        "cache_values",
                        return_value=dict(
                            CMAKE_BUILD_TYPE="Debug",
                            nlohmann_json_DIR=str(root),
                            RNS8_HIP_ROOT=str(root),
                        ),
                    ),
                    mock.patch.object(native, "run", side_effect=invoke),
                    mock.patch.object(native, "tests", side_effect=cases) as test,
                ):
                    if mutation:
                        with self.assertRaises(ValueError):
                            session.installed_consumers(
                                native,
                                args,
                                report,
                                build=build,
                                output=root,
                                compiler=compiler,
                                static_library=session.reference(archive),
                                runtime_witness=witness,
                            )
                    else:
                        result = session.installed_consumers(
                            native,
                            args,
                            report,
                            build=build,
                            output=root,
                            compiler=compiler,
                            static_library=session.reference(archive),
                            runtime_witness=witness,
                        )
                        self.assertEqual(result["status"], "passed")
                        self.assertEqual(test.call_count, 1)
                        self.assertTrue(
                            any(
                                arg.startswith("-DRNS8_DIR=")
                                for _, argv in commands
                                for arg in argv
                            )
                        )
                        if lane == "device-address":
                            self.assertTrue(
                                any(
                                    arg == "-DRNS8_DOWNSTREAM_GPU_TESTS=ON"
                                    for _, argv in commands
                                    for arg in argv
                                )
                            )
                    if mutation in ("duplicate", "archive_change"):
                        test.assert_not_called()

    def test_interrupted_stage_preserves_capture_and_stops_before_following_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            owned = session.Session(output, {}, 60, 1024**2)
            result = {**capture(), "interrupted": True, "returncode": -1}
            with mock.patch.object(session, "run_capture", return_value=result) as run:
                with self.assertRaises(InterruptedError):
                    owned.run("interrupted", ["never executed fixture"])
                self.assertEqual(run.call_count, 1)
            self.assertTrue(owned.stages[0]["interrupted"])

    def test_ignored_native_filter_cannot_execute_unselected_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            native = inventory("mandatory", "unexpected")
            owned = session.Session(output, {}, 60, 1024**2)
            with mock.patch.object(
                session, "run_capture", return_value=capture(json.dumps(native))
            ) as run:
                with self.assertRaisesRegex(ValueError, "refusing execution"):
                    owned.tests(
                        "scope",
                        output,
                        Path("/ctest"),
                        required=["mandatory"],
                        selection_names=["mandatory"],
                    )
                self.assertEqual(run.call_count, 2)

    def test_environment_cannot_disable_checks_or_inject_runtime(self):
        env = session.child_environment(
            {
                "ASAN_OPTIONS": "halt_on_error=0",
                "PYTHONOPTIMIZE": "2",
                "COVERAGE_PROCESS_START": "secret",
                "OMP_TOOL_LIBRARIES": "unreviewed.so",
                "ARCHER_OPTIONS": "ignore_serial=1",
                "CXXFLAGS": "-fno-sanitize=all",
                "RNS8_HOST_RUNTIME_WITNESS_TOKEN": "inherited-not-owned",
                "PATH": "/safe",
            },
            Path("/llvm/symbolizer"),
            "address-undefined",
        )
        self.assertEqual(env["PYTHONOPTIMIZE"], "")
        self.assertNotIn("COVERAGE_PROCESS_START", env)
        self.assertNotIn("OMP_TOOL_LIBRARIES", env)
        self.assertNotIn("ARCHER_OPTIONS", env)
        self.assertNotIn("CXXFLAGS", env)
        self.assertNotIn("RNS8_HOST_RUNTIME_WITNESS_TOKEN", env)
        self.assertIn("halt_on_error=1", env["ASAN_OPTIONS"])
        for key in ("LD_PRELOAD", "LD_AUDIT", "CCC_OVERRIDE_OPTIONS", "CL"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                session.child_environment({key: "injected"}, Path("/llvm/symbolizer"), "thread")

    def test_plan_never_configures_runs_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "temp").mkdir()
            tool = root / "tool"
            tool.write_text("fixture", encoding="utf-8")
            out = root / "temp" / "planned"
            args = ["--preset", "cpu-tsan-release", "--out", str(out)]
            for name in ("cmake", "ctest", "nm", "symbolizer"):
                args.extend(["--" + name, str(tool)])
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session, "run_capture") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(session.main(args), 0)
                run.assert_not_called()
            value = json.loads(stdout.getvalue())
            self.assertFalse(out.exists())
            self.assertFalse(value["runtime_validation_executed"])
            self.assertIn("-DRNS8_BUILD_FUZZERS=OFF", value["configure"])
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session, "run_capture") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(session.main([*args, "--openmp-archer", str(tool)]), 0)
                run.assert_not_called()
            value = json.loads(stdout.getvalue())
            self.assertIn("-DRNS8_ENABLE_CPU_PARALLEL=ON", value["configure"])
            self.assertIn("-DRNS8_REQUIRE_OPENMP=ON", value["configure"])
            self.assertEqual(value["openmp_archer"], str(tool))
            self.assertFalse(out.exists())
            invalid = args.copy()
            invalid[invalid.index("cpu-tsan-release")] = "cpu-asan-ubsan-debug"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(session.main([*invalid, "--openmp-archer", str(tool)]), 2)

    def test_missing_and_changed_execution_evidence_never_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            executable = output / "test-binary"
            executable.write_text("unchanged", encoding="utf-8")
            native = inventory("mandatory")
            native["tests"][0]["command"] = [str(executable)]
            owned = session.Session(output, {}, 60, 1024**2)
            calls = []

            def fake(command, **kwargs):
                calls.append(command)
                if "--show-only=json-v1" in command:
                    return capture(json.dumps(native))
                (output / "app.xml").write_text(report(("mandatory", "run", "")), encoding="utf-8")
                return capture()

            with mock.patch.object(session, "run_capture", side_effect=fake):
                ref, value = owned.tests("app", output, Path("/ctest"), required=["mandatory"])
            self.assertEqual(value["status"], "passed")
            self.assertEqual(len(calls), 3)
            self.assertTrue(all("--no-tests=error" in call for call in calls))
            session.check_bookends(ref)
            executable.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                session.check_bookends(value)

    def test_owned_stage_failure_preserves_first_capture_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            owned = session.Session(output, {}, 60, 1024**2)
            failure = capture(stderr="first failure", returncode=3)
            with mock.patch.object(
                session, "run_capture", return_value=copy.deepcopy(failure)
            ) as run:
                with self.assertRaisesRegex(ValueError, "required stage failed"):
                    owned.run("compile", ["/compiler"])
                self.assertEqual(run.call_count, 1)
            retained = json.loads(Path(owned.stages[0]["receipt"]["path"]).read_text())
            self.assertEqual(retained["stderr"], "first failure")
            self.assertFalse(owned.stages[0]["succeeded"])


if __name__ == "__main__":
    unittest.main()
