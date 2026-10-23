"""Provenance regressions use an isolated repository, never the user's checkout."""
import json
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
from build_manifest import generate

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
