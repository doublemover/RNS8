"""Offline controls for inventory, non-mutating plans, locks and archive members."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from init_quality import archive_executable
from quality import command_plan, selected_tools
from quality_inventory import ROOT, SourceFile, batches, classify, inventory
from quality_policy import load_quality_lock


class Quality(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "metadata/first-party-inventory.json").read_text())

    def test_roles_include_hip_fragments_spaces_unicode_and_exclude_immutable_outputs(
        self,
    ):
        rows = classify(
            [
                "src/a space/λ.hip",
                "benchmarks/µ.inc",
                "tools/report.py",
                "scripts/probe.sh",
                ".github/workflows/ci.yml",
                "third_party/vendor/a.cpp",
                "tools/data/certificate.json",
                "include/rns8/generated/metadata_registry.hpp",
            ],
            self.policy,
        )
        self.assertEqual({row.role for row in rows}, {"cpp", "python", "bash", "workflow"})
        self.assertEqual(sum(row.role == "cpp" for row in rows), 2)
        for name in (
            "../src/a.cpp",
            "/src/a.cpp",
            "src\\a.cpp",
            "examples/unknown.inc",
        ):
            with self.assertRaises(ValueError):
                classify([name], self.policy)
        with self.assertRaises(ValueError):
            classify(["src/a.cpp", "src/a.cpp"], self.policy)

    def test_nul_inventory_is_not_shell_split_and_empty_role_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata").mkdir()
            (root / "metadata/first-party-inventory.json").write_text(json.dumps(self.policy))
            paths = [
                "src/a space λ.hip",
                "tools/a.py",
                "scripts/a.sh",
                ".github/workflows/a.yml",
            ]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("authored")
            with patch(
                "quality_inventory.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 0, ("\0".join(paths) + "\0").encode(), b""
                ),
            ):
                self.assertEqual(len(inventory(root)), 4)
            for data in (b"", b"src/a.cpp", b"src/a.cpp\0"):
                with patch(
                    "quality_inventory.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, data, b""),
                ):
                    with self.assertRaises(ValueError):
                        inventory(root)

    def test_batches_are_bounded_and_paths_are_never_requoted(self):
        paths = ["src/λ space " + str(i) + ".hip" for i in range(25)]
        groups = batches(paths, maximum_bytes=160)
        self.assertGreater(len(groups), 1)
        self.assertEqual([path for group in groups for path in group], paths)
        with self.assertRaises(ValueError):
            batches(["x" * 100], maximum_bytes=10)

    def test_check_and_fix_plans_keep_semantic_lint_non_mutating(self):
        rows = (
            SourceFile("src/a.hip", "cpp"),
            SourceFile("tools/a.py", "python"),
            SourceFile("scripts/a.sh", "bash"),
            SourceFile(".github/workflows/a.yml", "workflow"),
        )
        tools = {
            name: "authored path/" + name
            for name in ("ruff", "clang-format", "actionlint", "shellcheck")
        }
        check = command_plan(rows, tools)
        fix = command_plan(rows, tools, fix=True)
        self.assertIn("--dry-run", check[0].argv)
        self.assertIn("--Werror", check[0].argv)
        self.assertNotIn("-i", check[0].argv)
        self.assertIn("-i", fix[0].argv)
        self.assertTrue(any("--check" in row.argv for row in check if row.role == "python-format"))
        self.assertFalse(any("--fix" in row.argv or "--unsafe-fixes" in row.argv for row in fix))
        self.assertIn(
            "--source-path=SCRIPTDIR",
            next(row.argv for row in check if row.role == "bash"),
        )
        self.assertTrue(all(isinstance(row.argv, tuple) for row in check))

    def test_lock_rejects_changed_hash_version_and_requirements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "metadata/dependencies.json",
                "metadata/quality-tools.json",
                "scripts/requirements-quality-tools.txt",
                "scripts/requirements-type-tools.txt",
                "scripts/requirements-analysis-tools.txt",
            ):
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            lock = load_quality_lock(root)
            self.assertEqual(len(lock["tools"]), 4)
            path = root / "metadata/quality-tools.json"
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "shared dependency policy"):
                load_quality_lock(root)
            path.write_bytes(original)
            requirements = root / "scripts/requirements-quality-tools.txt"
            requirements.write_text("ruff==0.0.0 --hash=sha256:" + "0" * 64)
            with self.assertRaisesRegex(ValueError, "requirements differ"):
                load_quality_lock(root)

    def test_missing_or_wrong_tool_versions_are_errors_without_running_real_tools(self):
        lock = load_quality_lock(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, "missing"):
                selected_tools(lock, tool_bin=directory)
            for row in lock["tools"]:
                (directory / (row["name"] + (".exe" if os.name == "nt" else ""))).write_bytes(
                    b"authored never executed"
                )
            with patch(
                "quality.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "wrong 0.0.0", ""),
            ):
                with self.assertRaisesRegex(ValueError, "locked version"):
                    selected_tools(lock, tool_bin=directory)

    def test_zip_member_is_exact_regular_and_never_extracted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tool.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../do-not-extract", "untrusted")
                archive.writestr("tool.exe", b"AUTHORED")
            self.assertEqual(archive_executable(path, "tool.exe"), b"AUTHORED")
            self.assertFalse((path.parent.parent / "do-not-extract").exists())
            with zipfile.ZipFile(path, "w") as archive:
                link = zipfile.ZipInfo("tool.exe")
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(link, "target")
            with self.assertRaises(ValueError):
                archive_executable(path, "tool.exe")

    def test_tar_member_rejects_links_missing_duplicates_and_expansion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tool.tar.gz"
            for kind in ("regular", "link", "duplicate", "large"):
                with tarfile.open(path, "w:gz") as archive:
                    member = tarfile.TarInfo("tool")
                    if kind == "link":
                        member.type, member.linkname = tarfile.SYMTYPE, "target"
                    elif kind == "large":
                        member.size = 1024**3
                    else:
                        member.size = 8
                    archive.addfile(member, io.BytesIO(b"AUTHORED") if member.size == 8 else None)
                    if kind == "duplicate":
                        archive.addfile(member, io.BytesIO(b"AUTHORED"))
                if kind == "regular":
                    self.assertEqual(archive_executable(path, "tool"), b"AUTHORED")
                else:
                    with self.assertRaises(ValueError):
                        archive_executable(path, "tool")


if __name__ == "__main__":
    unittest.main()
