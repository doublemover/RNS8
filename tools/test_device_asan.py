"""Deferred offline prerequisite, actual-command and artifact-custody controls."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import device_asan_preflight as session
from device_asan_contract import (
    HMM_KEYS,
    agent_observation,
    compiler_flags,
    dynamic_dependencies,
    host_prerequisites,
    hip_component_version,
    support,
    runtime_binding,
    CORRECTNESS_CASES,
)
from isa_common import bundle_target_for, extract_code_object
from test_capture_identity_support import synthetic_runtime_device_identity
from test_host_instrumentation import application_documents, application_report, capture
from test_execution_inventory import inventory
from host_runtime_contract import application_witnesses


def device_runtime_fixture(root):
    """Synthetic native protocol records only; not an executed HIP/HSA observation."""
    runtimes = {}
    for name in ("hip", "hsa", "asan"):
        path = root / (name + ".so")
        path.write_bytes(b"synthetic selected runtime " + name.encode())
        runtimes[name] = session.reference(path)
    value = dict(
        schema="rns8_device_asan_runtime_snapshot_v1",
        visible_hip_devices=1,
        current_hip_ordinal=0,
        runtime_version=71626385,
        driver_version=71626385,
        architecture="gfx942:sramecc+:xnack+",
        runtime_device_identity=synthetic_runtime_device_identity(),
        hsa_gpu_count=1,
        hsa_uuid="GPU-123456789abcdef0",
        hsa_name="gfx942",
        hsa_bdfid=4 << 8,
        hsa_domain=0,
        free_device_bytes=512 * 1024**2,
        total_device_bytes=1024 * 1024**2,
        hip_symbol_owner=runtimes["hip"]["path"],
        hsa_symbol_owner=runtimes["hsa"]["path"],
        scope="single_visible_hsa_uuid_and_pci_join_to_observed_hip_ordinal_zero",
    )
    contract = dict(
        target="gfx942:xnack+",
        agent_uuid=value["hsa_uuid"],
        runtimes=runtimes,
        minimum_free_bytes=256 * 1024**2,
    )
    loaded = dict(
        files=list(runtimes.values()), symbol_owners=dict(sanitizer=runtimes["asan"]["path"])
    )
    return value, contract, loaded


class OrdinaryRuntimeCustody(unittest.TestCase):
    def test_observed_agent_pci_target_and_actual_selected_runtime_owners_must_join(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value, contract, loaded = device_runtime_fixture(root)
            bound = runtime_binding(value, contract, loaded, entering=True)
            self.assertEqual(bound["pci_bdf"], "0000:04:00.0")
            self.assertEqual(bound["agent_uuid"], contract["agent_uuid"])
            for key, bad in (
                ("visible_hip_devices", 2),
                ("current_hip_ordinal", 1),
                ("hsa_gpu_count", True),
                ("runtime_version", 0),
                ("driver_version", True),
                ("hsa_bdfid", 2**28),
                ("hsa_domain", 65536),
                ("hsa_uuid", "GPU-ffffffffffffffff"),
                ("architecture", "gfx942:xnack-"),
                ("architecture", "gfx942:xnack+:xnack+"),
                ("architecture", "gfx950:xnack+"),
                ("architecture", "gfx942:sramecc+:sramecc-:xnack+"),
                ("hsa_name", "gfx950"),
                ("free_device_bytes", 1),
                ("total_device_bytes", 1),
                ("hip_symbol_owner", contract["runtimes"]["hsa"]["path"]),
                ("scope", "agent_advertisement_only"),
            ):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    runtime_binding(dict(value, **{key: bad}), contract, loaded, entering=True)
            for key, bad in (
                ("pci_bdf", "0000:05:00.0"),
                ("runtime_ordinal", 1),
                ("valid_fields", 2),
            ):
                changed = copy.deepcopy(value)
                changed["runtime_device_identity"][key] = bad
                with self.assertRaises(ValueError):
                    runtime_binding(changed, contract, loaded, entering=True)
            # A runtime's retained pool can change free bytes; do not invent an
            # exact free-memory restoration invariant after ordinary cleanup.
            self.assertEqual(
                runtime_binding(dict(value, free_device_bytes=1), contract, loaded, entering=False),
                bound,
            )
            for name in contract["runtimes"]:
                changed = copy.deepcopy(loaded)
                changed["files"] = [
                    row
                    for row in changed["files"]
                    if row["path"] != contract["runtimes"][name]["path"]
                ]
                with self.assertRaises(ValueError):
                    runtime_binding(value, contract, changed, entering=True)

    def test_device_bookends_are_separate_from_host_cleanliness_and_bind_each_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value, contract, _ = device_runtime_fixture(root)
            executable = root / "application"
            executable.write_bytes(b"synthetic instrumented application")
            selected = inventory("mandatory")
            selected["tests"][0]["command"][0] = str(executable)
            documents = application_documents(executable)
            for document in documents:
                document["device_runtime"] = copy.deepcopy(value)
                document["loaded_modules"]["modules"].extend(
                    row["path"] for row in contract["runtimes"].values()
                )
                document["loaded_modules"]["symbol_owners"]["sanitizer"] = contract["runtimes"][
                    "asan"
                ]["path"]
            kwargs = dict(
                token="a" * 64,
                executables=[executable],
                lane="device-address",
                runtimes=list(contract["runtimes"].values()),
                device_contract=contract,
            )
            result = application_witnesses(
                application_report(documents), selected, capture(), **kwargs
            )
            self.assertIn("device_runtime_binding", result["cases"][0])
            for changed in (
                {**kwargs, "device_contract": None},
                {**kwargs, "lane": "address-undefined"},
            ):
                with self.assertRaises(ValueError):
                    application_witnesses(
                        application_report(documents), selected, capture(), **changed
                    )
            bad = copy.deepcopy(documents)
            bad[-1]["device_runtime"]["runtime_version"] += 1
            with self.assertRaises(ValueError):
                application_witnesses(application_report(bad), selected, capture(), **kwargs)
            bad = copy.deepcopy(documents)
            bad[-1].pop("device_runtime")
            with self.assertRaises(ValueError):
                application_witnesses(application_report(bad), selected, capture(), **kwargs)
            bad = copy.deepcopy(documents)
            for document in bad:
                document.pop("device_runtime")
            host = dict(kwargs, lane="address-undefined")
            host.pop("device_contract")
            self.assertNotIn(
                "device_runtime_binding",
                application_witnesses(application_report(bad), selected, capture(), **host)[
                    "cases"
                ][0],
            )

    def test_owned_ordinary_stage_selects_only_fixed_safe_cases_and_restores_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, contract, loaded = device_runtime_fixture(root)
            ctest = root / "ctest"
            ctest.write_bytes(b"inert tool fixture")
            args = SimpleNamespace(
                ctest=ctest, compiler=ctest, target=contract["target"], minimum_free_mib=256
            )
            for fail in (False, True):
                report = dict(
                    cmake_version="cmake version 4.4.5",
                    agent=dict(uuid=contract["agent_uuid"]),
                    runtimes=contract["runtimes"],
                    compiled_objects=dict(
                        artifacts=dict(rns8_tests=dict(path=str(root / "native")))
                    ),
                    qualification={},
                )
                report["compiled_objects"]["artifacts"]["rns8_static"] = loaded["files"][0]
                owned = session.Session(root, {"ROCR_VISIBLE_DEVICES": "prior-value"}, 60, 1024**2)
                before = dict(owned.environment)

                def execute(*positional, owned=owned, fail=fail, **kwargs):
                    self.assertEqual(kwargs["required"], CORRECTNESS_CASES)
                    self.assertEqual(kwargs["selection_names"], list(CORRECTNESS_CASES))
                    self.assertEqual(kwargs["runtime_witness"]["lane"], "device-address")
                    self.assertEqual(
                        owned.environment["ROCR_VISIBLE_DEVICES"], contract["agent_uuid"]
                    )
                    if fail:
                        raise ValueError("first ordinary failure")
                    return {}, dict(
                        status="passed", application_runtime_module_files=loaded["files"]
                    )

                with (
                    mock.patch.dict(session.os.environ, {}, clear=True),
                    mock.patch.object(
                        owned, "run", return_value=dict(stdout="ctest version 4.4.5")
                    ),
                    mock.patch.object(owned, "tests", side_effect=execute) as run,
                    mock.patch.object(
                        session,
                        "installed_consumers",
                        return_value=dict(
                            status="passed", application_runtime_module_files=loaded["files"]
                        ),
                    ) as consumer,
                ):
                    if fail:
                        with self.assertRaisesRegex(ValueError, "first ordinary failure"):
                            session.ordinary_correctness(owned, args, report, root)
                        self.assertEqual(report["device_tests"], "attempted_incomplete")
                        consumer.assert_not_called()
                    else:
                        session.ordinary_correctness(owned, args, report, root)
                        self.assertEqual(report["device_tests"], "passed_bounded_subset")
                        self.assertEqual(
                            report["qualification"]["device_execution"],
                            "bounded_ordinary_correctness_passed",
                        )
                        self.assertEqual(consumer.call_count, 1)
                        self.assertEqual(
                            report["device_consumers"], "passed_seven_cpu_cases_two_hip_cases"
                        )
                    self.assertEqual(run.call_count, 1)
                self.assertEqual(owned.environment, before)


FLAGS = ["clang++", "-fsanitize=address", "-shared-libsan", "-g", "-fno-sanitize-recover=address"]


class DeviceEligibility(unittest.TestCase):
    def test_core_sdk_release_does_not_replace_actual_hip_component_identity(self):
        native = "HIP version: 7.16.26385-7da43026e4\nAMD clang version 23.0.0\n"
        self.assertEqual(hip_component_version(native, "7.16.26385"), "7.16.26385")
        for observed, header in (
            (native, "10.1.0"),
            (native, None),
            (native, "7.16.26384"),
            (native + native, "7.16.26385"),
            ("clang version 23.0.0", "7.16.26385"),
        ):
            with self.assertRaises(ValueError):
                hip_component_version(observed, header)

    def test_official_package_scope_never_becomes_execution_or_detector_pass(self):
        for system, release, target, expected in (
            ("Linux", "10.1.0", "gfx942:xnack+", "supported"),
            ("Linux", "10.1.0", "gfx950:xnack+", "supported"),
            ("Windows", "10.1.0", "gfx1100", "unsupported"),
            ("Linux", "10.1.0", "gfx90a:xnack+", "unsupported"),
            ("Linux", "10.1.0", "gfx942:xnack-", "unsupported"),
            ("Linux", "10.2.0", "gfx942:xnack+", "unsupported"),
        ):
            value = support(system, release, target)
            self.assertEqual(value["official_support"], expected)
            self.assertEqual(value["device_execution"], "not_run")
            self.assertFalse(value["production_latency"])

    def test_kernel_version_is_insufficient_without_running_config_and_xnack(self):
        config = "\n".join(key + "=y" for key in HMM_KEYS)
        self.assertEqual(
            host_prerequisites("6.8.0-native", config, {"HSA_XNACK": "1"})["HSA_XNACK"], "1"
        )
        for kernel, native, env in (
            ("5.5.0", config, {"HSA_XNACK": "1"}),
            ("6.8.0-microsoft-standard-WSL2", config, {"HSA_XNACK": "1"}),
            ("6.8.0", "", {"HSA_XNACK": "1"}),
            ("6.8.0", config, {}),
            ("6.8.0", config, {"HSA_XNACK": "0"}),
            ("6.8.0", config, {"HSA_XNACK": "1", "HSA_OVERRIDE_GFX_VERSION": "9.4.2"}),
            ("6.8.0", config + "\nCONFIG_HMM_MIRROR=n", {"HSA_XNACK": "1"}),
        ):
            with self.subTest(kernel=kernel, env=env), self.assertRaises(ValueError):
                host_prerequisites(kernel, native, env)
        for key in HMM_KEYS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                host_prerequisites(
                    "6.8.0", config.replace(key + "=y", key + "=n"), {"HSA_XNACK": "1"}
                )

    def test_native_agent_uuid_and_enabled_isa_must_belong_to_selected_agent(self):
        text = "Agent 0\n Name: CPU\nAgent 1\n Name: gfx942\n Uuid: GPU-1234abcdef567890\n Name: amdgcn-amd-amdhsa--gfx942:sramecc+:xnack+\nAgent 2\n Name: gfx950\n Uuid: GPU-abcd567890abcdef\n Name: amdgcn-amd-amdhsa--gfx950:xnack+\n"
        value = agent_observation(text, "gfx942:xnack+", 1)
        self.assertEqual(value["uuid"], "GPU-1234abcdef567890")
        for raw, index in (
            (text, 0),
            (text, 2),
            (text, 99),
            (text.replace("xnack+", "xnack-"), 1),
            (text, True),
            (text.replace("Uuid: GPU-1234abcdef567890", "Uuid: unavailable"), 1),
            (text + "Agent 1\n", 1),
            (text.replace("sramecc+:xnack+", "sramecc+:sramecc-:xnack+"), 1),
        ):
            with self.subTest(index=index), self.assertRaises(ValueError):
                agent_observation(raw, "gfx942:xnack+", index)

    def test_actual_commands_require_complete_host_and_device_flags(self):
        self.assertTrue(compiler_flags(FLAGS)["configured_flags_verified"])
        target = "gfx942:xnack+"
        self.assertTrue(
            compiler_flags([*FLAGS, "--offload-arch=" + target], target=target)[
                "configured_flags_verified"
            ]
        )
        for extra in (
            [],
            ["--offload-arch=gfx942:xnack-"],
            ["--offload-arch=" + target] * 2,
            ["--offload-arch=" + target, "--cuda-host-only"],
            ["--offload-arch=" + target, "-fno-sanitize=address"],
            ["--offload-arch=" + target, "-static-libsan"],
            ["--offload-arch=" + target, "-g0"],
            ["--offload-arch=" + target, "--config=hidden"],
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                compiler_flags([*FLAGS, *extra], target=target)
        for missing in FLAGS[1:]:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                compiler_flags([arg for arg in FLAGS if arg != missing])

    def test_shared_asan_must_be_a_direct_native_elf_dependency(self):
        native = "0x01 (NEEDED) Shared library: [libclang_rt.asan-x86_64.so]\n0x01 (NEEDED) Shared library: [libamdhip64.so.7]\n"
        self.assertEqual(
            dynamic_dependencies(native), ["libclang_rt.asan-x86_64.so", "libamdhip64.so.7"]
        )
        for raw in (
            "",
            "path/to/libclang_rt.asan-x86_64.so",
            native.replace("libclang_rt.asan-x86_64.so", "libc.so.6"),
            native + native,
        ):
            with self.assertRaises(ValueError):
                dynamic_dependencies(raw)


class ArtifactCustody(unittest.TestCase):
    def test_exact_bundle_selector_rejects_duplicate_and_wrong_feature_targets(self):
        selected = "hip-amdgcn-amd-amdhsa--gfx942:xnack+"
        self.assertEqual(bundle_target_for(["host-x86_64", selected], "gfx942:xnack+"), selected)
        for targets in (
            [selected, selected],
            [selected.replace("xnack+", "xnack-")],
            ["host-x86_64", "host-x86_64", selected],
        ):
            with self.assertRaises(RuntimeError):
                bundle_target_for(targets, "gfx942:xnack+")

    def test_shared_extractor_routes_every_command_through_owned_runner_and_never_rewrites_input(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original = root / "original.o"
            original.write_bytes(b"unchanged original object")
            calls = []

            def fake(command):
                calls.append(command)
                if "--list" in command:
                    return "host-x86_64\nhip-amdgcn-amd-amdhsa--gfx942:xnack+\n"
                return ""

            with mock.patch(
                "isa_common.run_command", side_effect=AssertionError("unowned command")
            ):
                result = extract_code_object(
                    "objcopy",
                    "bundler",
                    original,
                    "gfx942:xnack+",
                    root,
                    "candidate.fatbin",
                    run=fake,
                )
            self.assertEqual(result, root / "bundle_1.co")
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0][-2:], [str(original), str(root / "extracted-host-copy.obj")])
            self.assertEqual(original.read_bytes(), b"unchanged original object")

    def test_runtime_discovery_rejects_ambiguous_prefixes_and_external_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sdk = root / "sdk"
            (sdk / "lib").mkdir(parents=True)
            one = sdk / "lib/libamdhip64.so.7"
            one.write_bytes(b"runtime")
            self.assertEqual(session.runtime(sdk, "libamdhip64.so*", None), one)
            (sdk / "lib/libamdhip64.so.8").write_bytes(b"different")
            with self.assertRaises(ValueError):
                session.runtime(sdk, "libamdhip64.so*", None)
            self.assertEqual(session.runtime(sdk, "libamdhip64.so*", one), one)
            external = root / "external"
            external.write_bytes(b"not SDK owned")
            with self.assertRaises(ValueError):
                session.runtime(sdk, "libamdhip64.so*", external)

    def test_plan_never_runs_tools_configures_or_creates_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "temp").mkdir()
            sdk = root / "sdk"
            for names in session.TOOLS.values():
                path = sdk / names[0]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"inert tool fixture")
            cmake = root / "cmake"
            cmake.write_bytes(b"inert cmake/ninja")
            output = root / "temp/planned"
            args = [
                "--rocm-root",
                str(sdk),
                "--target",
                "gfx942:xnack+",
                "--agent-index",
                "1",
                "--out",
                str(output),
                "--compile",
                "--cmake",
                str(cmake),
                "--ninja",
                str(cmake),
            ]
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session.Session, "run") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(session.main(args), 0)
                run.assert_not_called()
            plan = json.loads(stdout.getvalue())
            self.assertFalse(output.exists())
            self.assertEqual(plan["device_tests"], "not_run")
            self.assertEqual(plan["device_faults"], "not_run_not_authorized")
            self.assertIn("-DRNS8_ENABLE_DEVICE_ASAN=ON", plan["configure"])
            self.assertIn("-DRNS8_ENABLE_CK=OFF", plan["configure"])
            self.assertIn("-DRNS8_ENABLE_HOST_COVERAGE=OFF", plan["configure"])
            self.assertEqual(plan["build_command"][-1], "--verbose")
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session.Session, "run") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(session.main([*args, "--correctness", "--ctest", str(cmake)]), 0)
                run.assert_not_called()
            planned = json.loads(stdout.getvalue())
            self.assertEqual(planned["correctness"]["cases"], list(CORRECTNESS_CASES))
            self.assertEqual(len(set(planned["correctness"]["cases"])), 7)
            self.assertTrue(planned["correctness"]["launch_requested_only_with_observe"])
            self.assertEqual(planned["device_tests"], "not_run")
            self.assertFalse(output.exists())
            no_compile = [arg for arg in args if arg != "--compile"]
            with (
                mock.patch.object(session, "ROOT", root),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    session.main([*no_compile, "--correctness", "--ctest", str(cmake)]), 2
                )


if __name__ == "__main__":
    unittest.main()
