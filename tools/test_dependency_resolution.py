#!/usr/bin/env python3
"""Filesystem-only discovery fixtures; CMake script handoff never builds/probes."""
from __future__ import annotations
from dataclasses import replace
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from check_dependencies_lib import resolution as r
from check_dependencies_lib.readiness import optional_cpp_references
from check_dependencies_lib.system import find_command
import windows_dev

ROOT = Path(__file__).resolve().parents[1]


class Discovery(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "temp/dependency-resolution-tests"; scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.base = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.environment = {"PATH": "", "SystemDrive": str(self.base / "emptydrive"), "ProgramFiles": str(self.base / "emptyprograms")}

    def put(self, root, relative, data=b"fixture file; not a compatibility pass"):
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data); return path

    def sdk(self, name="SDK with spaces", host="Windows", version="7.1.0", device=True):
        root = self.base / name
        self.put(root, "bin/hipcc.exe" if host == "Windows" else "bin/hipcc")
        self.put(root, "bin/clang++.exe" if host == "Windows" else "llvm/bin/clang++")
        self.put(root, "include/hip/hip_runtime_api.h")
        if version:
            self.put(root, "include/hip/hip_version.h", "\n".join("#define HIP_VERSION_" + key + " " + value for key, value in zip(("MAJOR", "MINOR", "PATCH"), version.split("."))))
        self.put(root, "lib/amdhip64.lib" if host == "Windows" else "lib64/libamdhip64.so")
        if host == "Windows": self.put(root, "bin/amdhip64_" + (version.split(".")[0] if version else "7") + ".dll")
        if device: self.put(root, "amdgcn/bitcode/hip.bc")
        return root

    def vcpkg(self, name="vcpkg with spaces"):
        root = self.base / name
        self.put(root, "scripts/buildsystems/vcpkg.cmake", "# fixture; no package manager execution\n")
        self.put(root, "vcpkg.exe"); return root

    def request(self, host="Windows", overrides=None, features=None, hip=False, vcpkg=False):
        return r.Request(host, overrides or {}, features or {}, str(self.base / "source"), hip, vcpkg)

    def resolve(self, request, environment=None, defaults=()):
        original_guard = r.forbidden_windows_path
        def fixture_guard(value):
            # Portable layout fixtures use the real host's temporary directory.
            # Normalize just that fixture prefix for simulated Linux layout
            # tests; Windows-artifact substrings inside it remain detectable.
            text = str(value)
            base = str(self.base)
            if text.startswith(base): text = "/fixture" + text[len(base):].replace("\\", "/")
            return original_guard(text)
        with patch.object(r, "default_hip_roots", return_value=list(defaults)), patch.object(r, "forbidden_windows_path", side_effect=fixture_guard):
            return r.resolve(request, self.environment if environment is None else environment)

    def assert_clean(self, report): self.assertEqual(report["errors"], [], report["errors"])

    def test_missing_optional_stays_nonblocking_and_requested_hip_fails(self):
        report = self.resolve(self.request())
        self.assert_clean(report); self.assertIsNone(report["hip"]["selected"])
        self.assertTrue(self.resolve(self.request(hip=True))["errors"])

    def test_complete_explicit_sdk_records_paths_version_hashes_and_no_qualification(self):
        sdk = self.sdk()
        report = self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(sdk)}, hip=True))
        self.assert_clean(report)
        chosen = report["hip"]["selected"]
        self.assertEqual(chosen["version"], "7.1.0")
        self.assertEqual(Path(chosen["paths"]["compiler"]), sdk / "bin/hipcc.exe")
        self.assertRegex(chosen["file_sha256"]["header"], "^[0-9a-f]{64}$")
        self.assertEqual(report["hip"]["compatibility"], "not_run")
        self.assertFalse(any(report["feature_policy"].values()))

    def test_invalid_explicit_root_never_falls_back(self):
        good = self.sdk()
        env = {**self.environment, "HIP_PATH": str(good)}
        report = self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(self.base / "missing")}, hip=True), env)
        self.assertTrue(report["errors"]); self.assertIsNone(report["hip"]["selected"])
        self.assertIn("invalid explicit", report["errors"][0])

    def test_explicit_root_beats_invalid_environment_override(self):
        sdk = self.sdk()
        report = self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(sdk)}, hip=True),
                              {**self.environment, "RNS8_HIP_ROOT": str(self.base / "oldmissing")})
        self.assert_clean(report)

    def test_equally_eligible_versions_require_selection(self):
        one, two = self.sdk("release1"), self.sdk("release2", version="8.0.0")
        report = self.resolve(self.request(hip=True), defaults=(one, two))
        self.assertEqual(report["hip"]["status"], "ambiguous"); self.assertTrue(report["errors"])
        env = {**self.environment, "HIP_PATH": str(one), "ROCM_PATH": str(two)}
        report = self.resolve(self.request(hip=True), env)
        self.assertEqual(report["hip"]["status"], "ambiguous")
        report = self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(one)}, hip=True), env)
        self.assert_clean(report)

    def test_symlink_alias_deduplicates_without_highest_version_guessing(self):
        sdk = self.sdk(); alias = self.base / "alias"
        try: alias.symlink_to(sdk, target_is_directory=True)
        except OSError as error: self.skipTest("host cannot create directory symlink: " + str(error))
        report = self.resolve(self.request(hip=True), {**self.environment, "HIP_PATH": str(sdk), "ROCM_PATH": str(alias)})
        self.assert_clean(report); self.assertEqual(len([row for row in report["hip"]["candidates"] if not row["rejected"]]), 1)
        self.assertTrue(report["hip"]["selected"]["aliases"])

    def test_unknown_version_does_not_use_folder_label(self):
        sdk = self.sdk("ROCm-99.99", version=None)
        report = self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(sdk)}, hip=True))
        self.assert_clean(report); self.assertIsNone(report["hip"]["selected"]["version"])

    def test_partial_sdk_and_mixed_components_fail(self):
        incomplete = self.sdk("partial", device=False)
        self.assertTrue(self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(incomplete)}, hip=True))["errors"])
        one, two = self.sdk("one"), self.sdk("two")
        for variable, value in (("RNS8_HIP_INCLUDE_DIR", two / "include"), ("RNS8_HIP_LIBRARY", two / "lib/amdhip64.lib"),
                                ("RNS8_HIP_HIPCC", two / "bin/hipcc.exe"), ("RNS8_HIP_CLANG", two / "bin/clang++.exe")):
            with self.subTest(variable=variable):
                report = self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(one), variable: str(value)}, hip=True))
                self.assertTrue(report["errors"]); self.assertIsNone(report["hip"]["selected"])

    def test_runtime_version_and_binary_architecture_mismatch_fail(self):
        sdk = self.sdk()
        (sdk / "bin/amdhip64_7.dll").rename(sdk / "bin/amdhip64_6.dll")
        self.assertTrue(self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(sdk)}, hip=True))["errors"])
        (sdk / "bin/amdhip64_6.dll").rename(sdk / "bin/amdhip64_7.dll")
        elf = bytearray(64); elf[:4] = b"\x7fELF"; elf[5] = 1; elf[18:20] = (183).to_bytes(2, "little")
        self.put(sdk, "bin/amdhip64_7.dll", elf)
        report = self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(sdk), "CMAKE_SYSTEM_PROCESSOR": "AMD64"}, hip=True))
        self.assertTrue(report["errors"])

    def test_explicit_runtime_and_device_library_disambiguate(self):
        sdk = self.sdk(version=None, device=False)
        self.put(sdk, "bin/amdhip64_6.dll")
        one = self.put(sdk, "lib/clang/19/lib/amdgcn/bitcode/hip.bc").parent
        self.put(sdk, "lib/clang/20/lib/amdgcn/bitcode/hip.bc")
        base = {"RNS8_HIP_ROOT": str(sdk)}
        self.assertTrue(self.resolve(self.request(overrides=base, hip=True))["errors"])
        report = self.resolve(self.request(overrides={**base, "RNS8_HIP_RUNTIME": str(sdk / "bin/amdhip64_7.dll"),
            "RNS8_HIP_DEVICE_LIB_DIR": str(one)}, hip=True))
        self.assert_clean(report)

    def test_selected_commands_cannot_leak_to_path(self):
        sdk = self.sdk()
        report = self.resolve(self.request(overrides={"RNS8_HIP_ROOT": str(sdk)}, hip=True))
        with r.selection_context(report), patch("check_dependencies_lib.system.shutil.which", return_value="WRONG"):
            self.assertEqual(find_command("hipcc"), str(sdk / "bin/hipcc.exe"))
        report["hip"]["selected"] = None
        with r.selection_context(report), patch("check_dependencies_lib.system.shutil.which", return_value="WRONG"):
            self.assertIsNone(find_command("hipcc"))

    def test_vcpkg_toolchain_override_bootstrap_triplet_and_conflicts(self):
        one, two = self.vcpkg("one"), self.vcpkg("two")
        report = self.resolve(self.request(overrides={"CMAKE_TOOLCHAIN_FILE": str(one / "scripts/buildsystems/vcpkg.cmake")}, vcpkg=True))
        self.assert_clean(report); self.assertEqual(report["toolchain"]["provider"], "vcpkg")
        for overrides in ({"VCPKG_ROOT": str(self.base / "missing")},
                          {"VCPKG_ROOT": str(one), "VCPKG_TARGET_TRIPLET": "x64-linux"},
                          {"VCPKG_ROOT": str(one), "VCPKG_TARGET_TRIPLET": "arm64-windows", "CMAKE_SYSTEM_PROCESSOR": "x86_64"},
                          {"VCPKG_ROOT": str(two), "CMAKE_TOOLCHAIN_FILE": str(one / "scripts/buildsystems/vcpkg.cmake")}):
            with self.subTest(overrides=overrides): self.assertTrue(self.resolve(self.request(overrides=overrides, vcpkg=True))["errors"])
        (one / "vcpkg.exe").unlink()
        self.assertTrue(self.resolve(self.request(overrides={"VCPKG_ROOT": str(one)}, vcpkg=True))["errors"])

    def test_external_toolchain_is_preserved(self):
        toolchain = self.put(self.base, "custom toolchain.cmake", "# user toolchain\n")
        report = self.resolve(self.request(overrides={"CMAKE_TOOLCHAIN_FILE": str(toolchain)}, vcpkg=True))
        self.assert_clean(report); self.assertEqual(report["toolchain"]["provider"], "external")
        self.assertEqual(Path(report["cmake_bindings"]["CMAKE_TOOLCHAIN_FILE"]), toolchain)

    def test_native_linux_and_wsl_reject_windows_artifacts(self):
        for overrides in ({"CMAKE_TOOLCHAIN_FILE": str(self.vcpkg() / "scripts/buildsystems/vcpkg.cmake")},
                          {"CMAKE_PREFIX_PATH": "C:/vcpkg/installed/x64-windows"},
                          {"CMAKE_INCLUDE_PATH": "/mnt/c/vcpkg/installed/x64-windows/include"},
                          {"VCPKG_TARGET_TRIPLET": "x64-windows-static"}):
            with self.subTest(overrides=overrides): self.assertTrue(self.resolve(self.request("Linux", overrides))["errors"])
        report = self.resolve(self.request("Linux"))
        self.assert_clean(report); self.assertEqual(report["vcpkg"]["status"], "unsupported")
        self.assertTrue(self.resolve(self.request("Linux"), {**self.environment, "CMAKE_PREFIX_PATH": "/mnt/c/vcpkg/include"})["errors"])

    def test_native_layouts_multiarch_and_sysroot_only(self):
        sysroot = self.base / "cross sysroot"
        sdk = self.sdk("cross sysroot/opt/rocm-7.1", host="Linux")
        self.put(sysroot, "usr/include/x86_64-linux-gnu/gmp.h")
        self.put(sysroot, "usr/lib/x86_64-linux-gnu/libgmp.so")
        report = self.resolve(self.request("Linux", {"CMAKE_SYSROOT": str(sysroot), "CMAKE_FIND_ROOT_PATH_MODE_PACKAGE": "ONLY"},
                                            {"RNS8_ENABLE_GMP": True}, hip=True))
        self.assert_clean(report)
        self.assertEqual(report["hip"]["selected"]["canonical_root"], str(sdk))
        self.assertIn("multiarch", report["optional_components"]["gmp"]["selected"]["paths"]["include"].replace("x86_64-linux-gnu", "multiarch"))
        report = self.resolve(self.request("Linux", {"CMAKE_SYSROOT": str(self.base / "missing")}, hip=True))
        self.assertTrue(report["errors"])

    def test_optional_roots_and_component_overrides_never_silently_mix(self):
        one, two = self.base / "gmp one", self.base / "gmp two"
        for root in (one, two): self.put(root, "include/gmp.h"); self.put(root, "lib/gmp.lib")
        report = self.resolve(self.request(overrides={"RNS8_GMP_INCLUDE_DIR": str(one / "include"), "RNS8_GMP_LIBRARY": str(one / "lib/gmp.lib")}))
        self.assert_clean(report)
        self.assertEqual(Path(report["cmake_bindings"]["RNS8_GMP_LIBRARY"]), one / "lib/gmp.lib")
        bad = self.resolve(self.request(overrides={"RNS8_GMP_ROOT": str(one), "RNS8_GMP_LIBRARY": str(two / "lib/gmp.lib")}))
        self.assertTrue(bad["errors"])
        with r.selection_context(report):
            refs = optional_cpp_references({"gmp": "fake inventory version"})
            self.assertFalse(refs["gmp"]["ok"]); self.assertEqual(refs["gmp"]["compatibility"], "not_run")

    def test_source_revision_and_unknown_version_are_separate(self):
        root = self.base / "ck"
        self.put(root, "include/ck/ck.hpp"); self.put(root, ".git/HEAD", "a" * 40)
        report = self.resolve(self.request(overrides={"RNS8_CK_ROOT": str(root)}))
        self.assert_clean(report)
        selected = report["optional_components"]["ck"]["selected"]
        self.assertEqual(selected["revision"], "a" * 40); self.assertIsNone(selected["version"])

    def test_effective_identity_stable_after_cache_materialization_and_changes_with_evidence(self):
        sdk, vcpkg = self.sdk(), self.vcpkg()
        request = self.request(overrides={"RNS8_HIP_ROOT": str(sdk), "VCPKG_ROOT": str(vcpkg)}, features={"RNS8_ENABLE_HIP": True}, hip=True, vcpkg=True)
        first = self.resolve(request)
        second = self.resolve(replace(request, overrides={**request.overrides, **first["cmake_bindings"]}))
        self.assert_clean(first); self.assert_clean(second)
        self.assertEqual(first["identity_sha256"], second["identity_sha256"])
        self.put(sdk, "include/hip/hip_runtime_api.h", "changed header")
        self.assertNotEqual(first["identity_sha256"], self.resolve(request)["identity_sha256"])

    def test_invalid_requests_and_cache_reader(self):
        for value in ({"unknown": 1}, {"features": {"RNS8_ENABLE_HIP": "ON"}}, {"overrides": {"unknown": "x"}}, {"require_hip": 1}):
            with self.subTest(value=value), self.assertRaises(ValueError): r.Request.from_mapping(value)
        cache = self.put(self.base, "CMakeCache.txt", "RNS8_HIP_ROOT:PATH=/sdk path\nRNS8_ENABLE_HIP:BOOL=ON\nRNS8_GMP_LIBRARY:FILEPATH=RNS8_GMP_LIBRARY-NOTFOUND\n")
        overrides, features = r.read_cache(cache)
        self.assertEqual(overrides, {"RNS8_HIP_ROOT": "/sdk path"}); self.assertTrue(features["RNS8_ENABLE_HIP"])

    def test_cli_resolve_only_runs_no_probe_or_package_manager(self):
        sdk = self.sdk(host=platform.system())
        overrides = {"RNS8_HIP_ROOT": str(sdk)}
        if os.name == "nt": overrides["VCPKG_ROOT"] = str(self.vcpkg())
        request = self.request(platform.system(), overrides, hip=True)
        path = self.put(self.base, "request.json", json.dumps(r.asdict(request)))
        completed = subprocess.run([sys.executable, str(ROOT / "tools/check_dependencies.py"), "--resolve-only", "--json", "--resolution-request", str(path)],
                                   capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        report = json.loads(completed.stdout)
        self.assertEqual(report["schema"], r.SCHEMA); self.assertEqual(report["hip"]["runtime_validation"], "not_run")

    def test_cmake_preproject_handoff_script_without_project_or_build(self):
        cmake = shutil.which("cmake")
        if not cmake: self.skipTest("CMake unavailable; pure Python contract fixtures still run")
        sdk = self.sdk(host=platform.system()); cwd = self.base / "script build"; cwd.mkdir()
        command = [cmake, "-DRNS8_ENABLE_HIP=ON", "-DRNS8_HIP_ROOT=" + str(sdk), "-DPython3_EXECUTABLE=" + sys.executable]
        if os.name == "nt": command += ["-DVCPKG_ROOT=" + str(self.vcpkg())]
        script = self.put(self.base, "handoff.cmake", 'include("' + (ROOT / "cmake/rns8_discovery.cmake").as_posix() + '")\n'
            'file(WRITE "bound-paths.txt" "${RNS8_HIP_ROOT}\\n${RNS8_HIP_HIPCC}\\n${RNS8_HIP_COMPILER_ENVIRONMENT}\\n${RNS8_HIP_COMMON_FLAGS}\\n${CMAKE_PREFIX_PATH}")\n')
        # An existing cache can retain the broken path flags. The environment
        # handoff must clear them rather than forwarding them to hipcc.
        command += ["-DRNS8_HIP_COMMON_FLAGS=--hip-path=stale SDK with spaces", "-DCMAKE_PREFIX_PATH=" + str(sdk), "-P", str(script)]
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((cwd / "dependency-discovery/resolution.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(report["cmake_bindings"]["RNS8_HIP_ROOT"]), sdk)
        bound = (cwd / "bound-paths.txt").read_text(encoding="utf-8")
        self.assertNotIn("--hip-path", bound)
        self.assertNotIn("--hip-device-lib-path", bound)
        for name, value in report["compiler_environment"].items():
            self.assertIn(name + "=" + value, bound)
        if os.name == "nt":
            self.assertNotIn("\\", (cwd / "bound-paths.txt").read_text(encoding="utf-8"))
            self.assertTrue(all("\\" not in value for value in report["cmake_bindings"].values()))
            self.assertTrue(all("\\" not in value for value in report["compiler_environment"].values()))
        self.assertFalse((cwd / "CMakeCache.txt").exists())

    def test_cmake_linux_guard_preserves_ci_diagnostic(self):
        cmake = shutil.which("cmake")
        if not cmake: self.skipTest("CMake unavailable")
        cwd = self.base / "guard"; cwd.mkdir()
        result = subprocess.run([cmake, "-DRNS8_FORCE_LINUX_NATIVE_DISCOVERY_GUARD=ON",
            "-DCMAKE_PREFIX_PATH=/mnt/c/vcpkg/installed/x64-windows", "-DPython3_EXECUTABLE=" + sys.executable,
            "-P", str(ROOT / "cmake/rns8_discovery.cmake")], cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Linux/UNIX builds must use native system packages", result.stdout + result.stderr)
        self.assertNotIn('"candidates"', result.stderr)
        report = json.loads((cwd / "dependency-discovery/resolution.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "invalid")

    def test_hipblaslt_selected_root_does_not_import_alternate_package(self):
        cmake = shutil.which("cmake")
        if not cmake: self.skipTest("CMake unavailable")
        selected, alternate = self.base / "selected SDK", self.base / "alternate SDK"
        self.put(selected, "include/hipblaslt/hipblaslt.h")
        library = self.put(selected, "lib/hipblaslt.lib")
        self.put(alternate, "lib/cmake/hipblaslt/hipblaslt-config.cmake", 'message(FATAL_ERROR "imported alternate SDK package")\n')
        script = self.put(self.base, "hipblaslt.cmake", '\n'.join((
            'set(RNS8_DISCOVERY_IDENTITY_SHA256 fixture)',
            'set(RNS8_HIPBLASLT_ROOT "' + selected.as_posix() + '")',
            'set(RNS8_HIPBLASLT_INCLUDE_DIR "' + (selected / 'include').as_posix() + '")',
            'set(RNS8_HIPBLASLT_LIBRARY "' + library.as_posix() + '")',
            'set(RNS8_HIP_ROOT "' + alternate.as_posix() + '")',
            'set(CMAKE_PREFIX_PATH "' + alternate.as_posix() + '")',
            'function(rns8_append_windows_vcpkg_hints list_var)', 'endfunction()',
            'function(rns8_assert_no_linux_windows_vcpkg_paths)', 'endfunction()',
            'include("' + (ROOT / 'cmake/modules/FindRNS8HIPBLASLT.cmake').as_posix() + '")',
            'if(hipblaslt_FOUND)', 'message(FATAL_ERROR "unexpected package outside selected root")', 'endif()',
            'if(NOT _RNS8_HIPBLASLT_HINTS STREQUAL RNS8_HIPBLASLT_ROOT)',
            'message(FATAL_ERROR "alternate hints retained")', 'endif()',
        )))
        result = subprocess.run([cmake, "-P", str(script)], cwd=self.base, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrapper_dry_run_preserves_configure_arguments(self):
        args = ["windows_dev.py", "--dry-run", "cmake", "--preset", "windows-msvc-hip-debug", "-DRNS8_HIP_ROOT=D:/SDK with spaces"]
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", args), patch.object(windows_dev, "command_in_msvc_environment", side_effect=lambda command, **kwargs: (command, False)), \
             patch.object(windows_dev.subprocess, "run", side_effect=AssertionError("dry run executed a command")), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(windows_dev.main(), 0)
        self.assertIn("D:/SDK with spaces", out.getvalue()); self.assertIn("no CMake discovery", err.getvalue())


if __name__ == "__main__": unittest.main()
