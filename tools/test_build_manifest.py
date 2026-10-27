"""Provenance regressions use an isolated repository, never the user's checkout."""
import json
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
from build_manifest import generate, dependency_tree_identity, manifest_provenance_complete, manifest_dependencies_compatible

class BuildManifestTest(unittest.TestCase):
    def test_source_identity_changes_without_committing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def git(*args):
                return subprocess.check_output(["git","-C",str(root),*args], stderr=subprocess.DEVNULL)
            git("init")
            (root/"src").mkdir(); source=root/"src"/"a.cpp"; source.write_text("int a=1;")
            (root/"CMakeLists.txt").write_text("project(test)")
            config=root/"configuration.txt"; config.write_text("config=Release")
            git("add","."); git("-c","user.name=Test","-c","user.email=test@example.invalid","commit","-m","A")
            a=generate(root,config)
            self.assertFalse(a["source_dirty"])
            source.write_text("int a=2;")
            b=generate(root,config)
            self.assertEqual(a["source_commit"],b["source_commit"])
            self.assertTrue(b["source_dirty"])
            self.assertNotEqual(a["source_digest"],b["source_digest"])
            # Saved artifact metadata remains immutable when checkout changes.
            saved=json.dumps(a)
            git("add","."); git("-c","user.name=Test","-c","user.email=test@example.invalid","commit","-m","B")
            self.assertNotEqual(a["source_commit"],generate(root,config)["source_commit"])
            self.assertEqual(json.loads(saved),a)

    def test_external_header_edits_additions_and_deletions_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "project"
            source.mkdir()
            (source / "CMakeLists.txt").write_text("project(test)")
            headers = root / "external"
            headers.mkdir()
            header = headers / "vendor.hpp"
            header.write_text("int vendor=1;")
            config = source / "config.txt"
            config.write_text(f"dependency_tree.vendor={headers}\n")
            first = generate(source, config)
            header.write_text("int vendor=2;")
            second = generate(source, config)
            self.assertEqual(first["source_digest"], second["source_digest"])
            self.assertNotEqual(first["external_dependencies"], second["external_dependencies"])
            extra = headers / "extra.inc"
            extra.write_bytes(b"binary\x00header")
            third = generate(source, config)
            self.assertEqual(third["external_dependencies"]["vendor"]["files"], 2)
            self.assertNotEqual(second["external_dependencies"], third["external_dependencies"])
            extra.unlink()
            self.assertEqual(second, generate(source, config))
            config.write_text(f"dependency_tree.vendor={headers}\ndependency_tree.vendor={headers}\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                generate(source, config)

    def test_dependency_identity_fails_closed_for_missing_empty_or_cyclic_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing"):
                dependency_tree_identity(root / "missing")
            with self.assertRaisesRegex(ValueError, "empty"):
                dependency_tree_identity(root)
            (root / "header.hpp").write_text("int x;")
            try:
                (root / "cycle").symlink_to(root, target_is_directory=True)
            except OSError:
                return  # Windows may not grant unprivileged symlink creation.
            with self.assertRaisesRegex(ValueError, "cycle"):
                dependency_tree_identity(root)

    def test_release_admission_requires_bound_external_dependencies(self):
        import copy
        from test_capture_identity_support import synthetic_identity
        capture = {}
        synthetic_identity(capture)
        manifest = capture["build_manifest"]
        self.assertTrue(manifest_provenance_complete(manifest, "ck"))
        missing = copy.deepcopy(manifest)
        missing["configuration"] = "ck=OFF"
        missing["external_dependencies"] = {}
        self.assertTrue(manifest_provenance_complete(missing, "cpu-reference"))
        self.assertFalse(manifest_provenance_complete(missing, "ck"))
        self.assertTrue(manifest_dependencies_compatible([manifest, copy.deepcopy(manifest)]))
        changed = copy.deepcopy(manifest)
        changed["external_dependencies"]["CK"]["sha256"] = "f" * 64
        self.assertTrue(manifest_provenance_complete(changed))
        self.assertFalse(manifest_dependencies_compatible([manifest, changed]))
        for mutate in (
            lambda m: m.update(version=1),
            lambda m: m["external_dependencies"].pop("CK"),
            lambda m: m["external_dependencies"]["CK"].update(files=True),
            lambda m: m["external_dependencies"]["CK"].update(sha256="z" * 64),
            lambda m: m.update(configuration="ck=ON"),
        ):
            changed = copy.deepcopy(manifest)
            mutate(changed)
            self.assertFalse(manifest_provenance_complete(changed))

    @unittest.skipUnless(os.name == "nt", "COFF link identity requires MSVC")
    def test_mixed_object_link_is_rejected(self):
        from msvc_env import run_in_msvc_environment
        scratch = Path(__file__).resolve().parents[1] / "temp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            (root/"CMakeLists.txt").write_text("project(witness)")
            config=root/"config.txt"
            generator=Path(__file__).with_name("build_manifest.py").resolve()
            for name in ("a","b"):
                config.write_text("config="+name)
                subprocess.run([sys.executable,str(generator),"--source",str(root),"--config",str(config),"--header",str(root/(name+".hpp"))],check=True)
            (root/"a.cpp").write_text('#include "a.hpp"\nint f(){return 1;}')
            (root/"b.cpp").write_text('#include "b.hpp"\nint f(); int main(){return f();}')
            result=run_in_msvc_environment(["cl", "/nologo", "/EHsc", "a.cpp", "b.cpp", "/Fe:mixed.exe"], cwd=root, capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("LNK2038",result.stdout)

if __name__=="__main__": unittest.main()
