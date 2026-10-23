"""Provenance regressions use an isolated repository, never the user's checkout."""
import json
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

if __name__=="__main__": unittest.main()
