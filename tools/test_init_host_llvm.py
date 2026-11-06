"""Authored offline archive/prefix controls; fixture tools are never executed."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import init_host_llvm as llvm


class HostLLVM(unittest.TestCase):
    def setUp(self):
        parent = llvm.ROOT / "temp/host-llvm-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def archive(self, host):
        policy = json.loads((llvm.ROOT / "metadata/dependencies.json").read_text())
        key = "windows-amd64" if host == "Windows" else "linux-x86_64"
        artifact = next(
            row for row in policy["bootstrap"]["host_llvm_artifacts"] if row["platform"] == key
        )
        # Minimal format headers for filesystem admission only, never native tools.
        header = bytearray(70)
        if host == "Windows":
            header[:2] = b"MZ"
            header[60:64] = (64).to_bytes(4, "little")
            header[64:70] = b"PE\0\0\x64\x86"
        else:
            header[:6] = b"\x7fELF\x02\x01"
            header[18:20] = (62).to_bytes(2, "little")
        names = (
            (*llvm.TOOLS, "clang-cl", "lld-link") if host == "Windows" else (*llvm.TOOLS, "ld.lld")
        )
        files = {
            "bin/" + name + (".exe" if host == "Windows" else ""): bytes(header) for name in names
        }
        files["lib/clang/23/include/stddef.h"] = b"/* fixture builtin header */"
        archive = self.root / artifact["filename"]
        with tarfile.open(archive, "w:xz") as output:
            for name, content in files.items():
                member = tarfile.TarInfo(artifact["prefix"] + "/" + name)
                member.size, member.mode = len(content), 0o755 if name.startswith("bin/") else 0o644
                output.addfile(member, io.BytesIO(content))
        return artifact, archive

    def test_plan_does_not_create_directories_download_or_select_hip(self):
        artifact, _ = self.archive("Windows")
        output, root = io.StringIO(), self.root / "not-created"
        with (
            patch.object(llvm, "selected_artifact", return_value=artifact),
            patch.object(llvm, "install", side_effect=AssertionError("install")),
            patch.object(llvm, "discover", side_effect=AssertionError("discover")),
            redirect_stdout(output),
        ):
            self.assertEqual(llvm.main(["--root", str(root)]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["mode"], "plan_only")
        self.assertEqual(plan["status"], "not_executed")
        self.assertFalse(plan["selects_hip_compiler"])
        self.assertFalse(root.exists())

    def test_xz_install_reuse_and_full_prefix_tamper(self):
        for host in ("Windows", "Linux"):
            with self.subTest(host=host):
                artifact, archive = self.archive(host)
                root = self.root / host
                with (
                    patch.object(llvm, "selected_artifact", return_value=artifact),
                    patch.object(
                        llvm,
                        "download",
                        side_effect=lambda row, path, archive=archive: shutil.copyfile(
                            archive, path
                        ),
                    ) as download,
                ):
                    self.assertIsNone(llvm.discover(root))
                    paths = llvm.install(root)
                    self.assertEqual(llvm.discover(root), paths)
                    self.assertEqual(llvm.install(root), paths)
                    self.assertEqual(download.call_count, 1)
                    prefix = Path(paths["root"])
                    receipt = json.loads((prefix.parent / llvm.RECEIPT).read_text())
                    self.assertEqual(receipt["qualification"], "installed_not_project_qualified")
                    header = prefix / "lib/clang/23/include/stddef.h"
                    header.write_text("changed")
                    with self.assertRaisesRegex(ValueError, "integrity"):
                        llvm.install(root)
                    self.assertEqual(download.call_count, 1)
                    self.assertEqual(header.read_text(), "changed")

    def test_missing_resources_wrong_format_and_foreign_destination(self):
        artifact, archive = self.archive("Windows")
        root = self.root / "install"
        with (
            patch.object(llvm, "selected_artifact", return_value=artifact),
            patch.object(
                llvm, "download", side_effect=lambda row, path: shutil.copyfile(archive, path)
            ),
        ):
            paths = llvm.install(root)
            prefix = Path(paths["root"])
            header = prefix / "lib/clang/23/include/stddef.h"
            header.rename(header.with_suffix(".saved"))
            with self.assertRaisesRegex(ValueError, "builtin-header"):
                llvm.binaries(prefix, artifact)
            header.with_suffix(".saved").rename(header)
            Path(paths["clang"]).write_bytes(b"wrong binary format")
            with self.assertRaisesRegex(ValueError, "format/architecture"):
                llvm.binaries(prefix, artifact)
        target = llvm.native_archive.destination(self.root / "foreign", "llvm", artifact)
        target.mkdir(parents=True)
        sentinel = target / "keep"
        sentinel.write_text("unrelated")
        with (
            patch.object(llvm, "selected_artifact", return_value=artifact),
            patch.object(llvm, "download", side_effect=AssertionError("download")),
            self.assertRaises(OSError),
        ):
            llvm.install(target.parent)
        self.assertEqual(sentinel.read_text(), "unrelated")

    def test_failed_download_and_unsupported_host_remain_unpublished(self):
        artifact, _ = self.archive("Linux")
        root = self.root / "failed"
        with (
            patch.object(llvm, "selected_artifact", return_value=artifact),
            patch.object(llvm, "download", side_effect=ValueError("hash mismatch")),
            self.assertRaisesRegex(ValueError, "hash mismatch"),
        ):
            llvm.install(root)
        self.assertEqual(list(root.iterdir()), [])
        for host, machine in (("Darwin", "x86_64"), ("Linux", "aarch64")):
            with (
                patch.object(llvm.platform, "system", return_value=host),
                patch.object(llvm.platform, "machine", return_value=machine),
                self.assertRaises(ValueError),
            ):
                llvm.selected_artifact()


if __name__ == "__main__":
    unittest.main()
