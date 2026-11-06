"""Authored native coverage producer/denominator and no-execution plan controls."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from benchmark_request import file_digest
from coverage_contract import llvm_export, mapped_sources
import host_coverage_session as coverage


def fixture(root):
    names = (
        "src/core/range.cpp",
        "src/core/included.inc",
        "src/backend_hip_direct/device.hip",
        "include/api.h",
        "tests/consumer.cpp",
        "src/generated.inc",
    )
    sources = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fixture source\n", encoding="utf-8")
        sources.append(dict(path=str(path), sha256=file_digest(path)))
    metrics = {
        key: dict(count=2, covered=1, percent=50.0)
        for key in ("functions", "lines", "regions", "branches")
    }
    rows = [
        dict(
            filename=str(root / name),
            summary=copy.deepcopy(metrics),
            branches=[[1, 1, 1, 3, 1, 0, 0, 0, 4]],
            expansions=[],
            segments=[[1, 1, 1, True, True, False]],
        )
        for name in (names[0], names[1], names[4], names[5])
    ]
    value = dict(
        type="llvm.coverage.json.export",
        version="3.1.0",
        data=[
            dict(
                files=rows,
                functions=[dict(name="fixture", count=1, filenames=[str(root / names[0])])],
            )
        ],
    )
    kwargs = dict(
        root=root,
        sources=sources,
        compiled_sources={root / names[0], root / names[4]},
        generated={names[5]},
        required_modules=(names[0],),
    )
    return value, kwargs


class NativeCoverage(unittest.TestCase):
    def test_mapping_discovery_retains_actual_native_spelling_and_rejects_empty_or_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value, kwargs = fixture(root)
            admitted = {Path(row["path"]) for row in kwargs["sources"]}
            self.assertEqual(len(mapped_sources(value, admitted)), 4)
            for changed in ({**value, "data": []}, {**value, "data": [dict(files=[])]}):
                with self.assertRaises(ValueError):
                    mapped_sources(changed, admitted)
            with self.assertRaises(ValueError):
                mapped_sources(value, set())
            value["data"][0]["files"].append(copy.deepcopy(value["data"][0]["files"][0]))
            with self.assertRaises(ValueError):
                mapped_sources(value, admitted)

    def test_production_generated_test_and_unmapped_denominators_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value, kwargs = fixture(root)
            result = llvm_export(value, **kwargs)
            self.assertEqual(result["status"], "collected")
            self.assertEqual(result["missing_mapping_count"], 2)
            self.assertEqual(result["totals"]["production"]["branches"], dict(count=4, covered=2))
            self.assertEqual(result["totals"]["test_or_tool"]["branches"], dict(count=2, covered=1))
            self.assertEqual(result["totals"]["generated"]["branches"], dict(count=2, covered=1))
            rows = {row["relative_path"]: row for row in result["files"]}
            self.assertEqual(
                rows["src/backend_hip_direct/device.hip"]["state"],
                "translation_unit_not_selected_in_this_host_build",
            )
            self.assertEqual(
                rows["include/api.h"]["state"], "header_or_fragment_without_observed_mapping"
            )
            self.assertFalse(rows["src/core/included.inc"]["branches"][0]["false_covered"])
            self.assertIsNone(result["reviewed_percentage_threshold"])
            self.assertEqual(result["mutation_detection"], "not_run")

    def test_explicit_unmapped_source_row_is_not_evidence_disabled_gpu_code_compiled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value, kwargs = fixture(root)
            value["data"][0]["files"].append(
                dict(
                    filename=str(root / "src/backend_hip_direct/device.hip"),
                    summary={
                        key: dict(count=0, covered=0, percent=0)
                        for key in ("functions", "regions", "lines", "branches")
                    },
                    branches=[],
                    segments=[],
                    expansions=[],
                )
            )
            result = llvm_export(value, **kwargs)
            row = next(
                row
                for row in result["files"]
                if row["relative_path"] == "src/backend_hip_direct/device.hip"
            )
            self.assertIsNone(row["metrics"])
            self.assertEqual(row["state"], "translation_unit_not_selected_in_this_host_build")

    def test_source_drift_schema_empty_and_missing_mapping_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value, kwargs = fixture(root)
            for edit in (
                lambda d: d.update(version="2.0.1"),
                lambda d: d["data"][0].update(files=[]),
                lambda d: d["data"][0].update(functions=[]),
                lambda d: d["data"][0]["files"].append(copy.deepcopy(d["data"][0]["files"][0])),
                lambda d: d["data"][0]["files"][0].update(filename="relative.cpp"),
                lambda d: d["data"][0]["files"][0].pop("segments"),
            ):
                changed = copy.deepcopy(value)
                edit(changed)
                with self.assertRaises(ValueError):
                    llvm_export(changed, **kwargs)
            result = llvm_export(value, **{**kwargs, "required_modules": ("src/missing.cpp",)})
            self.assertEqual(result["status"], "failed")
            (root / "src/core/range.cpp").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                llvm_export(value, **kwargs)

    def test_impossible_native_counts_and_stripped_branch_details_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value, kwargs = fixture(root)
            for key, replacement in (
                ("count", True),
                ("count", -1),
                ("covered", 3),
                ("percent", 100),
                ("notcovered", 0),
            ):
                changed = copy.deepcopy(value)
                changed["data"][0]["files"][0]["summary"]["branches"][key] = replacement
                with self.subTest(key=key), self.assertRaises(ValueError):
                    llvm_export(changed, **kwargs)
            for branches in (
                None,
                [[1, 1, 1, 3, True, 0, 0, 0, 4]],
                [[1, 1, 1, 3, -1, 0, 0, 0, 4]],
                [[1, 1, 1, 3, 1, 0]],
                [[3, 1, 1, 1, 1, 0, 0, 0, 4]],
            ):
                changed = copy.deepcopy(value)
                changed["data"][0]["files"][0]["branches"] = branches
                with self.subTest(branches=branches), self.assertRaises(ValueError):
                    llvm_export(changed, **kwargs)

    def test_unknown_external_files_are_visible_and_not_in_production_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value, kwargs = fixture(root)
            external = copy.deepcopy(value["data"][0]["files"][0])
            external["filename"] = str(root / "third_party/external.hpp")
            value["data"][0]["files"].append(external)
            result = llvm_export(value, **kwargs)
            self.assertEqual(
                result["excluded"],
                [
                    dict(
                        path=external["filename"],
                        reason="outside_selected_first_party_source_denominator",
                    )
                ],
            )
            self.assertEqual(result["totals"]["production"]["branches"]["count"], 4)


class CoverageSession(unittest.TestCase):
    def test_missing_empty_unowned_and_oversized_profile_sets_cannot_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, "nonempty"):
                coverage.profiles(root)
            path = root / "123_0-4321.profraw"
            path.write_bytes(b"profile fixture")
            self.assertEqual(len(coverage.profiles(root)), 1)
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                coverage.profiles(root)
            path.rename(root / "not-owned.profraw")
            with self.assertRaisesRegex(ValueError, "filenames"):
                coverage.profiles(root)

    def test_native_coverage_plan_is_inert_and_uses_dedicated_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "temp").mkdir()
            tool = root / "tool"
            tool.write_text("fixture", encoding="utf-8")
            out = root / "temp/planned"
            args = ["--out", str(out)]
            for name in ("cmake", "ctest", "compiler", "nm", "llvm-cov", "llvm-profdata"):
                args.extend(["--" + name, str(tool)])
            with (
                mock.patch.object(coverage, "ROOT", root),
                mock.patch.object(coverage.Session, "run") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(coverage.main(args), 0)
                run.assert_not_called()
            plan = json.loads(stdout.getvalue())
            self.assertFalse(out.exists())
            self.assertIn("cpu-coverage-debug", plan["configure"])
            self.assertFalse(plan["runtime_validation_executed"])


if __name__ == "__main__":
    unittest.main()
