#!/usr/bin/env python3
"""Offline native-prefix publication and archive admission controls."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import archive_install as archives
import init_build_tools as native
import rocm_sdk as sdk


class NativeBuildTools(unittest.TestCase):
    def setUp(self):
        scratch = native.ROOT / "temp/native-build-tool-controls"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def artifact(self, host):
        platform = "windows-amd64" if host == "Windows" else "linux-x86_64"
        policy = json.loads((native.ROOT / "metadata/dependencies.json").read_text())
        return next(
            row for row in policy["bootstrap"]["native_artifacts"] if row["platform"] == platform
        )

    def archive(self, host):
        artifact = self.artifact(host)
        path = self.root / artifact["filename"]
        suffix = ".exe" if host == "Windows" else ""
        files = {
            **{f"bin/{name}{suffix}": b"offline fixture" for name in ("cmake", "ctest", "cpack")},
            "share/cmake-4.4/Modules/CMake.cmake": b"# offline module fixture\n",
        }
        if host == "Windows":
            with zipfile.ZipFile(path, "w") as output:
                for name, content in files.items():
                    output.writestr(artifact["prefix"] + "/" + name, content)
        else:
            with tarfile.open(path, "w:gz") as output:
                for name, content in files.items():
                    member = tarfile.TarInfo(artifact["prefix"] + "/" + name)
                    member.size, member.mode = (
                        len(content),
                        0o755 if name.startswith("bin/") else 0o644,
                    )
                    output.addfile(member, io.BytesIO(content))
        return artifact, path

    def test_install_reuse_and_complete_prefix_tamper(self):
        for host in ("Windows", "Linux"):
            with self.subTest(host=host):
                artifact, archive = self.archive(host)
                root = self.root / host
                with (
                    patch.object(native, "selected_artifact", return_value=artifact),
                    patch.object(
                        native,
                        "download",
                        side_effect=lambda row, path, source=archive: shutil.copyfile(source, path),
                    ) as download,
                ):
                    paths = native.install(root)
                    self.assertEqual(download.call_count, 1)
                    self.assertEqual(native.install(root), paths)
                    self.assertEqual(download.call_count, 1)
                    prefix = Path(paths["cmake"]).parent.parent
                    module = prefix / "share/cmake-4.4/Modules/CMake.cmake"
                    module.write_text("changed")
                    with self.assertRaisesRegex(ValueError, "integrity"):
                        native.install(root)
                    self.assertEqual(module.read_text(), "changed")

    def test_existing_foreign_prefix_and_failed_download_are_not_published(self):
        artifact = self.artifact("Windows")
        destination = native.installation(self.root, artifact)
        with (
            patch.object(native, "selected_artifact", return_value=artifact),
            patch.object(native, "download", side_effect=ValueError("hash mismatch")),
        ):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                native.install(self.root)
            self.assertFalse(destination.exists())
            self.assertEqual(list(self.root.iterdir()), [])
            destination.mkdir()
            sentinel = destination / "keep.txt"
            sentinel.write_text("unrelated")
            with self.assertRaises(OSError):
                native.install(self.root)
            self.assertEqual(sentinel.read_text(), "unrelated")

    def test_unsupported_platform_does_not_probe_or_download(self):
        for host, machine in (("Darwin", "x86_64"), ("Linux", "aarch64")):
            with (
                self.subTest(host=host, machine=machine),
                patch.object(native.platform, "system", return_value=host),
                patch.object(native.platform, "machine", return_value=machine),
                patch.object(native, "download", side_effect=AssertionError("network")),
                self.assertRaises(ValueError),
            ):
                native.install(self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_incomplete_archive_and_existing_archive_destination_fail_closed(self):
        artifact, archive = self.archive("Windows")
        incomplete = self.root / "incomplete.zip"
        with zipfile.ZipFile(incomplete, "w") as output:
            output.writestr(artifact["prefix"] + "/bin/cmake.exe", b"incomplete")
        root = self.root / "installation"
        with (
            patch.object(native, "selected_artifact", return_value=artifact),
            patch.object(
                native, "download", side_effect=lambda row, path: shutil.copyfile(incomplete, path)
            ),
            self.assertRaisesRegex(ValueError, "missing"),
        ):
            native.install(root)
        self.assertFalse(native.installation(root, artifact).exists())
        destination = self.root / "existing"
        destination.mkdir()
        sentinel = destination / "keep"
        sentinel.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "overlays"):
            archives.extract_zip(archive, destination, "Windows", max_bytes=1024, max_members=20)
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_zip_member_count_and_explicit_github_path_opt_in(self):
        _, archive = self.archive("Windows")
        with self.assertRaisesRegex(ValueError, "member"):
            archives.extract_zip(
                archive, self.root / "count", "Windows", max_bytes=1024, max_members=0
            )
        paths = dict(cmake=str(self.root / "native tools/bin/cmake.exe"))
        github_path = self.root / "github-path"
        with (
            patch.dict(native.os.environ, {"GITHUB_PATH": str(github_path)}),
            patch.object(native, "install", return_value=paths),
            patch("builtins.print"),
        ):
            self.assertEqual(native.main([]), 0)
            self.assertFalse(github_path.exists())
            self.assertEqual(native.main(["--github-path"]), 0)
        self.assertEqual(github_path.read_text().splitlines(), [str(Path(paths["cmake"]).parent)])

    def test_download_checks_exact_bytes_and_final_origin(self):
        artifact = self.artifact("Windows")
        artifact["sha256"] = hashlib.sha256(b"verified").hexdigest()

        class Response(io.BytesIO):
            def geturl(self):
                return "https://release-assets.githubusercontent.com/fixture"

        for content in (b"verified", b"tampered"):
            destination = self.root / content.decode()
            with patch.object(native.urllib.request, "urlopen", return_value=Response(content)):
                if content == b"verified":
                    native.download(artifact, destination)
                    self.assertEqual(destination.read_bytes(), content)
                else:
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        native.download(artifact, destination)
        with (
            patch.object(Response, "geturl", return_value="http://github.com/fixture"),
            patch.object(native.urllib.request, "urlopen", return_value=Response(b"verified")),
            self.assertRaisesRegex(ValueError, "release hosts"),
        ):
            native.download(artifact, self.root / "bad-origin")

    def test_zip_rejects_paths_aliases_links_and_budgets(self):
        for index, entries in enumerate(
            (
                [("../escape", b"x")],
                [("sdk/NUL", b"x")],
                [("sdk/A", b"x"), ("sdk/a", b"y")],
                [("sdk/C:bad", b"x")],
                [("sdk/large", b"12345")],
            )
        ):
            archive = self.root / f"bad-{index}.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for name, content in entries:
                    output.writestr(name, content)
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                archives.extract_zip(
                    archive, self.root / f"out-{index}", "Windows", max_bytes=4, max_members=5
                )
        archive = self.root / "link.zip"
        with zipfile.ZipFile(archive, "w") as output:
            link = zipfile.ZipInfo("sdk/link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            output.writestr(link, "target")
        with self.assertRaises(ValueError):
            archives.extract_zip(
                archive, self.root / "link", "Windows", max_bytes=1024, max_members=5
            )

    def test_sdk_selects_native_cmake_and_selected_interpreter_ninja(self):
        paths = dict(cmake="native/cmake", ctest="native/ctest", cpack="native/cpack")
        with (
            patch.object(sdk, "install_build_tools", return_value=paths) as install,
            patch.object(sdk.shutil, "which", return_value="python-tools/ninja") as which,
        ):
            self.assertEqual(
                sdk.build_tools(),
                dict(cmake=paths["cmake"], ctest=paths["ctest"], ninja="python-tools/ninja"),
            )
        install.assert_called_once_with()
        self.assertEqual(which.call_args.args[0], "ninja")
        with (
            patch.object(sdk.shutil, "which", return_value=None),
            patch.object(sdk, "install_build_tools", side_effect=AssertionError("install")),
            self.assertRaisesRegex(ValueError, "Ninja"),
        ):
            sdk.build_tools()


if __name__ == "__main__":
    unittest.main()
