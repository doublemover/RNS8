"""Owned artifact metadata reuse without dropping link, size or scan guards."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from artifact_inventory import regular_files
from process_progress import _artifact_bytes


class ArtifactInventory(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "temp"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_scandir_metadata_is_read_once_and_path_predicates_are_not_repeated(self):
        class Entry:
            calls = 0

            def __fspath__(self):
                return str(self_path)

            def stat(self, *, follow_symlinks):
                if follow_symlinks or self.calls:
                    raise AssertionError("metadata must be non-following and read once")
                self.calls += 1
                return SimpleNamespace(st_mode=stat.S_IFREG, st_size=123)

        self_path = self.root / "file"
        entry = Entry()
        with (
            patch("artifact_inventory.os.scandir", return_value=nullcontext([entry])),
            patch.object(Path, "is_file", side_effect=AssertionError("extra stat")),
            patch.object(Path, "is_dir", side_effect=AssertionError("extra stat")),
            patch.object(Path, "is_symlink", side_effect=AssertionError("extra stat")),
            patch.object(Path, "is_junction", side_effect=AssertionError("extra stat")),
        ):
            self.assertEqual(list(regular_files([self.root], seconds=1)), [(self_path, 123)])
        self.assertEqual(entry.calls, 1)

    def test_real_nested_files_and_live_byte_budget(self):
        nested = self.root / "nested"
        nested.mkdir()
        (self.root / "a").write_bytes(b"123")
        (nested / "b").write_bytes(b"12345")
        self.assertEqual(
            sorted(regular_files([self.root], seconds=1)),
            sorted([(self.root / "a", 3), (nested / "b", 5)]),
        )
        self.assertEqual(_artifact_bytes([self.root], 100), 8)
        self.assertGreater(_artifact_bytes([self.root], 2), 2)

    def test_absence_is_only_tolerated_by_a_live_monitor(self):
        missing = self.root / "missing"
        self.assertEqual(list(regular_files([missing], seconds=1, missing_ok=True)), [])
        with self.assertRaisesRegex(ValueError, "absent"):
            list(regular_files([missing], seconds=1))

    def test_symlink_and_windows_junction_metadata_fail_before_enumeration(self):
        for mode, tag in ((stat.S_IFLNK, 0), (stat.S_IFDIR, 0xA0000003)):
            with (
                self.subTest(mode=mode, tag=tag),
                patch.object(
                    Path, "lstat", return_value=SimpleNamespace(st_mode=mode, st_reparse_tag=tag)
                ),
                patch("artifact_inventory.os.scandir") as scan,
                self.assertRaisesRegex(ValueError, "link or junction"),
            ):
                list(regular_files([self.root], seconds=1))
            scan.assert_not_called()

    def test_nonregular_files_deadlines_and_entry_limits_remain_fail_closed(self):
        with (
            patch.object(Path, "lstat", return_value=SimpleNamespace(st_mode=stat.S_IFIFO)),
            self.assertRaisesRegex(ValueError, "nonregular"),
        ):
            list(regular_files([self.root], seconds=1))
        with (
            patch("artifact_inventory.time.monotonic", side_effect=[0, 2]),
            self.assertRaisesRegex(ValueError, "scan time"),
        ):
            list(regular_files([self.root], seconds=1))
        (self.root / "file").write_bytes(b"data")
        with self.assertRaisesRegex(ValueError, "entry budget"):
            list(regular_files([self.root], seconds=1, maximum_entries=1))

    def test_malformed_limits_do_not_scan(self):
        for options in (
            dict(seconds=float("nan")),
            dict(seconds=0),
            dict(seconds=True),
            dict(seconds=1, maximum_entries=True),
            dict(seconds=1, missing_ok=1),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                list(regular_files([self.root], **options))


if __name__ == "__main__":
    unittest.main()
