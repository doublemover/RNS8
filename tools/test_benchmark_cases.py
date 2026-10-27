"""Catalog identity, coverage, nested records and output containment checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_cases import ROOT, catalog_documents, expand_value, materialize, safe_relative


class CatalogTests(unittest.TestCase):
    def fixture(self):
        data = b'{\n  "m": 3,\n  "flag": false\n}\n'
        return dict(schema="rns8_benchmark_catalog_v1", family="fixture", groups=[dict(
            keys=["m", "flag"], defaults={"flag": False}, columns=["m"],
            rows=[["sample.json", hashlib.sha256(data).hexdigest(), 3]])]), data

    def test_exact_bytes_and_mutated_identity(self):
        catalog, data = self.fixture()
        self.assertEqual(list(catalog_documents(catalog)), [(Path("fixture/sample.json"), data)])
        catalog["groups"][0]["rows"][0][-1] = 4
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            list(catalog_documents(catalog))

    def test_nested_records_retain_missing_fields_types_and_key_order(self):
        value = {"$rns8_records": dict(layouts=[["m", "flag"], ["flag", "n"]],
            defaults={"flag": False}, rows=[[0, 3], [1, 1]])}
        self.assertEqual(expand_value(value), [{"m": 3, "flag": False}, {"flag": False, "n": 1}])
        self.assertEqual(list(expand_value(value)[1]), ["flag", "n"])
        value["$rns8_records"]["rows"][0].append(0)
        with self.assertRaises(ValueError):
            expand_value(value)

    def test_paths_duplicate_columns_and_reserved_names_fail(self):
        for name in ("../escape.json", "/root.json", "a\\b.json", "C:/case.json", "a//b.json", "a/CON.json", "a/../b.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_relative(name)
        catalog, _ = self.fixture()
        catalog["groups"][0]["columns"] = ["m", "m"]
        with self.assertRaises(ValueError):
            list(catalog_documents(catalog))

    def test_complete_repository_corpus_and_idempotent_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "cases"
            report = materialize(output)
            self.assertEqual((report["cases"], report["catalogs"], report["retained_files"]), (2190, 25, 38))
            files = sorted(output.rglob("*.json"))
            self.assertEqual(len(files), 2190)
            before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files}
            materialize(output)
            self.assertEqual(before, {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files})
            victim = files[0]
            victim.write_bytes(b"damaged")
            materialize(output)
            self.assertEqual(victim.read_bytes(), before[victim][1])

    def test_source_destination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temp/"):
            materialize(ROOT / "benchmarks")

    def test_concurrent_identical_publication_survives_a_windows_reader(self):
        for identical in (True,False):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);catalogs=root/'benchmarks/cases';catalogs.mkdir(parents=True)
                value,_=self.fixture();(catalogs/'fixture.json').write_text(json.dumps(value))
                def occupied_reader(source,target):
                    Path(target).write_bytes(Path(source).read_bytes() if identical else b'wrong winner')
                    raise PermissionError('simulated replacement denied while another reader is open')
                with patch('benchmark_cases.os.replace',side_effect=occupied_reader):
                    if identical:self.assertEqual(materialize(root/'temp/cases',root)['cases'],1)
                    else:
                        with self.assertRaises(PermissionError):materialize(root/'temp/cases',root)
                self.assertFalse(list((root/'temp/cases/fixture').glob('tmp*')))

    def test_default_sweep_catalog_contains_materialized_wide_cases(self):
        from benchmark_sweep_lib.commands import scenario_catalog
        from benchmark_sweep_lib.config import scenario_data_dir
        self.assertTrue((scenario_data_dir() / "exact_wide_magnitude.json").is_file())
        items = [item for family in scenario_catalog().values() for item in family]
        self.assertTrue(any(item.case.input_profile == "wide-three-limb-mixed" for item in items))

    def test_duplicate_cases_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalogs = root / "benchmarks/cases"
            catalogs.mkdir(parents=True)
            value, _ = self.fixture()
            for name in ("a", "b"):
                (catalogs / (name + ".json")).write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                materialize(root / "temp/cases", root)
            self.assertFalse((root / "temp/cases").exists())


if __name__ == "__main__":
    unittest.main()
