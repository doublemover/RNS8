"""Offline SDK provisioning contracts; fixture files do not qualify GPU support."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import rocm_sdk as sdk
import init as init_entry


class SDKTests(unittest.TestCase):
    def setUp(self):
        scratch = sdk.ROOT / "temp/sdk-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def archive(self, entries=None):
        path = self.base / "sdk.tar.gz"
        files = entries if entries is not None else {
            "sdk/.info/version": b"10.1.0\n", "sdk/bin/hipcc.exe": b"fixture",
            "sdk/lib/llvm/bin/amdclang++.exe": b"fixture",
            "sdk/include/hip/hip_runtime_api.h": b"fixture",
            "sdk/include/hip/hip_version.h": b"#define HIP_VERSION_MAJOR 7\n#define HIP_VERSION_MINOR 16\n#define HIP_VERSION_PATCH 26385\n",
            "sdk/lib/amdhip64.lib": b"fixture", "sdk/bin/amdhip64_7.dll": b"fixture",
            "sdk/lib/llvm/amdgcn/bitcode/hip.bc": b"fixture",
        }
        with tarfile.open(path, "w:gz") as archive:
            for name, data in files.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        return path

    def args(self, *values):
        parser = argparse.ArgumentParser()
        sdk.add_arguments(parser)
        parser.add_argument("--target")
        parser.add_argument("--dry-run", action="store_true")
        return parser.parse_args(values)

    def test_family_dedup_and_separate_build_roots(self):
        lanes = sdk.plan(["gfx1101", "gfx1100", "gfx1100", "gfx1201"], "Windows", "10.1.0", self.base)
        self.assertEqual(len(lanes), 2)
        self.assertEqual(lanes[0]["targets"], ["gfx1100", "gfx1101"])
        self.assertNotEqual(lanes[0]["destination"], lanes[1]["destination"])
        self.assertEqual(lanes[0]["url"], sdk.BASE_URL + "therock-dist-windows-gfx110X-all-10.1.0.tar.gz")

    def test_bad_platform_target_and_version_fail(self):
        for targets, host, version in ((["gfx1100"], "Darwin", "10.1.0"),
                (["gfx9999"], "Linux", "10.1.0"), (["gfx1100"], "Windows", "../latest"),
                ([], "Windows", "10.1.0")):
            with self.subTest(targets=targets, host=host, version=version), self.assertRaises(ValueError):
                sdk.plan(targets, host, version, self.base)

    def test_dry_run_never_touches_disk_or_network(self):
        prefix = self.base / "absent"
        args = self.args("--sdk-install", "--sdk-target", "gfx1100", "--sdk-target", "gfx1200",
                         "--sdk-prefix", str(prefix), "--sdk-build-root", str(prefix / "build"), "--sdk-build", "--dry-run")
        with patch.object(sdk, "official_inventory", side_effect=AssertionError("network")), \
                patch.object(sdk, "inspect", side_effect=AssertionError("probe")), \
                patch.object(sdk.subprocess, "run", side_effect=AssertionError("execution")):
            result = sdk.run(args)
        self.assertEqual(len(result["lanes"]), 2)
        self.assertFalse(prefix.exists())
        self.assertNotEqual(result["lanes"][0]["build_dir"], result["lanes"][1]["build_dir"])

    def test_tar_path_guards(self):
        for name in ("../escape", "/escape", "sdk/C:evil", "sdk/a\\b", "sdk/NUL", "sdk/name.", "sdk/name "):
            with self.subTest(name=name), self.assertRaises((ValueError, tarfile.TarError)):
                sdk.extract(self.archive({name: b"bad"}), self.base / "unpack", "Windows")
        self.assertFalse((self.base.parent / "escape").exists())

    def test_expansion_limit(self):
        archive = self.archive({"sdk/test": b"12345"})
        with patch.object(sdk, "MAX_EXTRACT", 4), self.assertRaises(ValueError):
            sdk.extract(archive, self.base / "unpack", "Windows")
        with patch.object(sdk, "MAX_MEMBERS", 0), self.assertRaises(ValueError):
            sdk.extract(archive, self.base / "unpack", "Windows")

    def test_duplicate_case_alias_and_special_member(self):
        with self.assertRaises(ValueError):
            sdk.extract(self.archive({"sdk/A": b"a", "sdk/a": b"b"}), self.base / "unpack", "Windows")
        archive = self.base / "device.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            member = tarfile.TarInfo("sdk/device")
            member.type = tarfile.FIFOTYPE
            stream.addfile(member)
        with self.assertRaises(ValueError):
            sdk.extract(archive, self.base / "unpack", "Linux")

    def test_escaping_links_rejected(self):
        archive = self.base / "link.tar.gz"
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("sdk/link")
                member.type = kind
                member.linkname = "../../outside"
                stream.addfile(member)
            with self.subTest(kind=kind), self.assertRaises(tarfile.TarError):
                sdk.extract(archive, self.base / "unpack", "Linux")

    def test_internal_library_links_preserve_contents(self):
        archive = self.base / "libraries.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            member = tarfile.TarInfo("sdk/lib/libfixture.so.1")
            member.size = 7
            stream.addfile(member, io.BytesIO(b"fixture"))
            member = tarfile.TarInfo("sdk/lib/libfixture-copy.so.1")
            member.type = tarfile.LNKTYPE; member.linkname = "sdk/lib/libfixture.so.1"
            stream.addfile(member)
            member = tarfile.TarInfo("sdk/lib/libfixture.so")
            member.type = tarfile.SYMTYPE; member.linkname = "libfixture.so.1"
            stream.addfile(member)
        destination = self.base / "unpack"
        sdk.extract(archive, destination, sdk.platform.system())
        self.assertEqual((destination / "sdk/lib/libfixture-copy.so.1").read_bytes(), b"fixture")
        # tarfile may copy a link on Windows hosts lacking symlink privileges;
        # native Linux retains the relative library link through the data filter.
        self.assertEqual((destination / "sdk/lib/libfixture.so").read_bytes(), b"fixture")

    def test_provision_promotes_and_reuses_without_network(self):
        archive = self.archive()
        digest = sdk.sha256(archive)
        lane = sdk.plan(["gfx1100"], "Windows", "10.1.0", self.base / "install")[0]
        with patch.object(sdk, "official_inventory", return_value={lane["filename"]}), \
                patch.object(sdk, "download", return_value=(archive, digest)):
            root, receipt = sdk.provision(lane, self.base / "cache", digest)
        self.assertTrue((root / "lib/llvm/bin/amdclang++.exe").is_file())
        self.assertEqual(receipt["integrity"], "pinned_sha256")
        with patch.object(sdk, "official_inventory", side_effect=AssertionError("network")):
            self.assertEqual(sdk.provision(lane, self.base / "cache", digest)[0], root)
        with self.assertRaises(ValueError):
            sdk.provision(lane, self.base / "cache", "0" * 64)
        (root / ".info/version").write_text("10.0.0")
        with self.assertRaises(ValueError):
            sdk.provision(lane, self.base / "cache", digest)

    def test_unpublished_lane_fails_before_downloading(self):
        lane = sdk.plan(["gfx942"], "Windows", "10.1.0", self.base)[0]
        with patch.object(sdk, "official_inventory", return_value=set()), \
                patch.object(sdk, "download", side_effect=AssertionError("download")), self.assertRaises(ValueError):
            sdk.provision(lane, self.base / "cache")

    def test_bad_archive_is_not_published(self):
        archive = self.archive({"sdk/.info/version": b"10.1.0"})
        lane = sdk.plan(["gfx1100"], "Windows", "10.1.0", self.base / "install")[0]
        with patch.object(sdk, "official_inventory", return_value={lane["filename"]}), \
                patch.object(sdk, "download", return_value=(archive, sdk.sha256(archive))), self.assertRaises(ValueError):
            sdk.provision(lane, self.base / "cache")
        self.assertFalse(Path(lane["destination"]).exists())
        self.assertEqual(list(Path(lane["destination"]).parent.iterdir()), [])

    def test_download_hash_length_budget_and_cache_corruption(self):
        lane = sdk.plan(["gfx1100"], "Windows", "10.1.0", self.base)[0]
        class Response(io.BytesIO):
            url = lane["url"]
            headers = {"Content-Length": "3"}
        cache = self.base / "cache"
        with patch.object(sdk.urllib.request, "urlopen", return_value=Response(b"abc")):
            path, digest = sdk.download(lane, cache, None)
        self.assertEqual(sdk.download(lane, cache, digest), (path, digest))
        path.write_bytes(b"bad")
        with self.assertRaises(ValueError):
            sdk.download(lane, cache, None)
        path.unlink()
        for content, limit, expected in ((b"abc", 5, "0" * 64), (b"ab", 5, None), (b"abc", 2, None)):
            with self.subTest(content=content, limit=limit), \
                    patch.object(sdk.urllib.request, "urlopen", return_value=Response(content)), \
                    patch.object(sdk, "MAX_DOWNLOAD", limit), self.assertRaises(ValueError):
                sdk.download(lane, cache, expected)
            self.assertFalse(path.exists())
            self.assertFalse(list(cache.glob("*.part")))

    def test_driver_only_windows_detection_and_unknown_rejection(self):
        adapter = {"Name": "AMD Radeon RX 7900 XTX", "PNPDeviceID": "PCI\\VEN_1002&DEV_744C"}
        with patch.object(sdk.platform, "system", return_value="Windows"), \
                patch.object(sdk.shutil, "which", return_value=None), patch.object(sdk, "captured", return_value=json.dumps(adapter)):
            self.assertEqual(sdk.detect_targets(), ["gfx1100"])
            for name in ("Unknown future AMD GPU", "AMD Radeon RX 7900M"):
                adapter["Name"] = name
                with self.subTest(name=name), patch.object(sdk, "captured", return_value=json.dumps(adapter)), self.assertRaises(ValueError):
                    sdk.detect_targets()

    def test_entry_point_dry_run_json(self):
        result = subprocess.run([sys.executable, str(sdk.ROOT / "tools/bootstrap_rocm_accelerators.py"),
            "--sdk-install", "--sdk-target", "gfx1100", "--sdk-prefix", str(self.base / "sdk"), "--dry-run", "--json"],
            capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["schema"], "rns8_rocm_sdk_init_v1")
        self.assertFalse((self.base / "sdk").exists())

    def test_build_handoff_keeps_sdk_bindings_and_child_environment(self):
        root = self.base / "sdk with spaces"
        args = self.args("--sdk-root", str(root), "--sdk-target", "gfx1100", "--sdk-build",
            "--sdk-build-root", str(self.base / "build"), "--sdk-cmake-arg=-DRNS8_ENABLE_CK=ON",
            "--sdk-cmake-arg=-DRNS8_HIP_ROOT=unrelated")
        resolution = {"cmake_bindings": {"RNS8_HIP_ROOT": str(root),
            "RNS8_HIP_CLANG": str(root / "lib/llvm/bin/amdclang++.exe")},
            "compiler_environment": {"HIP_PATH": str(root), "ROCM_PATH": str(root)}}
        with patch.object(sdk, "inspect", return_value=resolution), \
                patch.object(sdk, "package_installation", return_value={}), patch.object(sdk.subprocess, "run") as run:
            result = sdk.run(args)
        self.assertEqual(run.call_count, 2)
        configure = run.call_args_list[0].args[0]
        choices = [value for value in configure if value.startswith("-DRNS8_HIP_ROOT=")]
        self.assertEqual(choices[-1], "-DRNS8_HIP_ROOT=" + str(root))
        self.assertIn("-DRNS8_CK_ROOT=" + str(root), configure)
        self.assertEqual(run.call_args.kwargs["env"]["HIP_PATH"], str(root))
        self.assertIn("windows-gfx110X-all-10.1.0" if sdk.platform.system() == "Windows" else "linux-gfx110X-all-10.1.0", result["lanes"][0]["build_dir"])

    def test_auto_reuses_exact_standard_sdk_without_network(self):
        sdk.extract(self.archive(), self.base / "unpack", "Windows")
        root = self.base / "unpack/sdk"
        lane = sdk.plan(["gfx1100"], "Windows", "10.1.0", self.base / "managed")[0]
        with patch.dict(sdk.os.environ, {}, clear=True), \
                patch.object(sdk, "default_hip_roots", return_value=[root]), \
                patch.object(sdk, "official_inventory", side_effect=AssertionError("network")):
            chosen, _ = sdk.existing_sdk(lane)
            self.assertEqual(chosen, root.resolve())
            (root / ".info/version").write_text("10.0.0")
            self.assertIsNone(sdk.existing_sdk(lane)[0])

    def test_init_entry_selects_pinned_python_and_forwards_without_shell(self):
        selected = self.base / "Python with spaces/python.exe"
        with patch.object(init_entry, "selected_python", return_value=selected), \
                patch.object(init_entry.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(init_entry.main(["--sdk-test", "--dry-run"]), 0)
        command = run.call_args.args[0]
        self.assertEqual(command[0], str(selected))
        self.assertIn("--sdk-auto", command)
        self.assertIn("--sdk-build", command)
        self.assertIn("--sdk-test", command)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_init_interpreter_is_exact_and_rejects_store_aliases(self):
        root = self.base / "source"
        root.mkdir(); (root / ".python-version").write_text("3.15.1")
        selected = root / ".venv/Scripts/python.exe"
        selected.parent.mkdir(parents=True); selected.write_bytes(b"fixture")
        with patch.object(init_entry.sys, "version_info", (3, 13, 3)), \
                patch.object(init_entry.sys, "platform", "win32"), \
                patch.object(init_entry.shutil, "which", return_value=None), \
                patch.dict(init_entry.os.environ, {}, clear=True), \
                patch.object(init_entry.subprocess, "run") as run:
            run.return_value.stdout = "Python 3.15.1\n"; run.return_value.stderr = ""
            self.assertEqual(init_entry.selected_python(root), selected)
            run.return_value.stdout = "Python 3.15.0\n"
            with self.assertRaises(ValueError):
                init_entry.selected_python(root)

    def test_auto_selects_prepared_pinned_packages_and_rejects_missing_files(self):
        root = self.base / "repo"
        local = self.base / "profile"
        installed = local / "RNS8/dependencies/installed-fixture"
        ports = {"boost-multiprecision": {"version": "1.92.0", "port_version": 0},
                 "nlohmann-json": {"version": "3.12.0", "port_version": 2},
                 "catch2": {"version": "3.16.1", "port_version": 0}}
        (root / "metadata").mkdir(parents=True)
        (root / "metadata/dependencies.json").write_text(json.dumps({"vcpkg": {"ports": ports}}))
        headers = ("boost/multiprecision/cpp_int.hpp", "nlohmann/json.hpp", "catch2/catch_test_macros.hpp")
        for name in headers:
            path = installed / "x64-windows/include" / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fixture")
        (installed / "vcpkg").mkdir()
        status = "\n\n".join(f"Package: {name}\nVersion: {item['version']}\nPort-Version: {item['port_version']}\nArchitecture: x64-windows\nStatus: install ok installed" for name, item in ports.items())
        (installed / "vcpkg/status").write_text(status)
        resolution = {"cmake_bindings": {"CMAKE_TOOLCHAIN_FILE": str(self.base / "vcpkg/scripts/buildsystems/vcpkg.cmake")}}
        with patch.object(sdk, "ROOT", root), patch.object(sdk.platform, "system", return_value="Windows"), \
                patch.dict(sdk.os.environ, {"LOCALAPPDATA": str(local)}, clear=True):
            found = sdk.package_installation(resolution, {})
            self.assertEqual(Path(found["bindings"]["VCPKG_INSTALLED_DIR"]), installed.resolve())
            (installed / "x64-windows/include" / headers[0]).unlink()
            with self.assertRaises(ValueError):
                sdk.package_installation(resolution, {})


if __name__ == "__main__":
    unittest.main()
