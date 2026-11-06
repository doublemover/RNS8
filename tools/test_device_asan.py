"""Deferred offline prerequisite, actual-command and artifact-custody controls."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import device_asan_preflight as session
from device_asan_contract import (
    HMM_KEYS,
    agent_observation,
    compiler_flags,
    dynamic_dependencies,
    host_prerequisites,
    hip_component_version,
    support,
)
from isa_common import bundle_target_for, extract_code_object

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


if __name__ == "__main__":
    unittest.main()
