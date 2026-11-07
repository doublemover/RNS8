"""Inert Windows/Linux bootstrap and user-checkout preservation controls."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import bootstrap_rocm_accelerators as b
from check_dependencies_lib.resolution import selection_context

ROOT = Path(__file__).resolve().parents[1]


class BootstrapNative(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "temp/bootstrap-native-controls"
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def resolution(self):
        return {
            "request": {"host": "Windows"},
            "errors": [],
            "tools": {"host_cxx": {"path": "/native/clang++"}},
            "hip": {
                "selected": {
                    "canonical_root": str(self.root / "selected SDK"),
                    "paths": {
                        "compiler": str(self.root / "selected SDK/bin/hipcc.exe"),
                        "include": str(self.root / "selected SDK/include"),
                        "clang": str(self.root / "selected SDK/llvm/bin/clang++.exe"),
                        "device_libraries": str(self.root / "selected SDK/bitcode"),
                    },
                }
            },
        }

    def test_native_linux_never_requires_or_queries_msvc(self):
        for errors in ([], ["invalid explicit SDK"]):
            resolution = self.resolution()
            resolution["errors"] = errors
            with (
                patch.object(b.platform, "system", return_value="Linux"),
                patch.object(b, "resolve", return_value=resolution) as resolve,
                patch.object(
                    b.dependency_system,
                    "find_command",
                    side_effect=lambda name: "/selected/" + name,
                ),
                patch.object(
                    b, "find_visual_studio_installation", side_effect=AssertionError("MSVC query")
                ),
                patch.object(
                    b.dependency_discovery,
                    "hip_info_report",
                    side_effect=AssertionError("Windows GPU query"),
                ),
                patch.object(b, "run_command", return_value={"ok": True}) as execute,
            ):
                result = b.toolchain_report("gfx942:xnack-")
            self.assertEqual(result["ok"], not errors)
            self.assertEqual(result["host_platform"], "Linux")
            self.assertEqual(execute.call_args.args[0], ["/native/clang++", "--version"])
            self.assertFalse(resolve.call_args.args[0].require_vcpkg)
            self.assertEqual(
                resolve.call_args.args[0].overrides["RNS8_AMDGPU_TARGETS"], "gfx942:xnack-"
            )

    def test_windows_requires_msvc_but_not_matching_local_gpu(self):
        with (
            patch.object(b.platform, "system", return_value="Windows"),
            patch.object(b, "resolve", return_value=self.resolution()),
            patch.object(
                b.dependency_system, "find_command", side_effect=lambda name: "/selected/" + name
            ),
            patch.object(
                b, "find_visual_studio_installation", return_value=self.root / "VS with spaces"
            ),
            patch.object(
                b.dependency_discovery,
                "hip_info_report",
                return_value={"target": "gfx1100", "ok": True},
            ),
            patch.object(b, "run_command", return_value={"ok": True}) as execute,
        ):
            result = b.toolchain_report("gfx942")
        self.assertTrue(result["ok"])
        self.assertFalse(result["target_matches_hip_info"])
        self.assertFalse(result["target_match_is_toolchain_gate"])
        self.assertEqual(execute.call_args.args[0], ["where", "cl"])

    def test_selected_hip_environment_is_applied_before_windows_wrapping(self):
        active = self.resolution()
        compiler = active["hip"]["selected"]["paths"]["compiler"]
        for wrapped in (False, True):
            with (
                self.subTest(wrapped=wrapped),
                selection_context(active),
                patch.object(b.platform, "system", return_value="Windows"),
                patch.object(
                    b,
                    "command_in_msvc_environment",
                    return_value=(
                        "inert shell wrapper" if wrapped else [compiler, "--version"],
                        wrapped,
                    ),
                ),
                patch.object(
                    b.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, "inert output"),
                ) as execute,
                patch.dict(
                    b.os.environ,
                    {"HIP_PATH": "stale", "HIP_CLANG_PATH": "stale", "UNRELATED": "retained"},
                ),
            ):
                result = b.run_command([compiler, "--version"], cwd=self.root, use_msvc=True)
            self.assertTrue(result["ok"])
            for key, value in b.compiler_environment(active).items():
                self.assertEqual(execute.call_args.kwargs["env"][key], value)
            self.assertEqual(execute.call_args.kwargs["env"]["UNRELATED"], "retained")

    def test_init_preserves_dirty_foreign_and_uninspectable_sources(self):
        specs = {name: {"path": "deps/" + name} for name in ("ck", "rocwmma")}
        for scenario in (
            "clean pin",
            "dirty",
            "unknown status",
            "foreign url",
            "foreign files",
            "missing",
        ):
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory(dir=self.root) as folder,
            ):
                root = Path(folder)
                for name in specs:
                    checkout = root / "deps" / name
                    checkout.mkdir(parents=True)
                    if scenario not in {"foreign files", "missing"}:
                        (checkout / ".git").write_text("inert gitlink fixture", encoding="utf-8")
                    if scenario == "foreign files":
                        (checkout / "user.txt").write_text("preserve", encoding="utf-8")

                def report(name, scenario=scenario):
                    return {"url_matches": scenario != "foreign url", "sha_matches": True}

                status = (
                    (1, "cannot inspect")
                    if scenario == "unknown status"
                    else (0, " M user.hpp" if scenario == "dirty" else "")
                )
                with (
                    patch.dict(b.dependency_config.EXPECTED_ROCM_SUBMODULES, specs, clear=True),
                    patch.object(
                        b.dependency_discovery, "repo_local_dependency_report", side_effect=report
                    ),
                    patch.object(b.dependency_discovery, "git_text", return_value=status),
                    patch.object(b, "run_command", return_value={"ok": True}) as execute,
                ):
                    result = b.initialize_submodules(root)
                self.assertEqual(execute.call_count, 2 if scenario == "missing" else 0)
                self.assertEqual(result["ok"], scenario in {"clean pin", "missing"})
                if scenario == "foreign files":
                    for name in specs:
                        self.assertEqual(
                            (root / "deps" / name / "user.txt").read_text(encoding="utf-8"),
                            "preserve",
                        )

    def test_git_inventory_disables_optional_index_writes(self):
        with patch.object(
            b.dependency_discovery.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, ""),
        ) as execute:
            code, _ = b.dependency_discovery.git_text(["status", "--porcelain=v1"], cwd=self.root)
        self.assertEqual(code, 0)
        self.assertEqual(execute.call_args.kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")


if __name__ == "__main__":
    unittest.main()
