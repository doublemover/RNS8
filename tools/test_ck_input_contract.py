"""Offline selected-header WMMA signedness contracts; no GPU qualification."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Contract(unittest.TestCase):
    def test_signed_unsigned_mixed_and_unknown_headers(self):
        search = str(Path(sys.executable).parent / "Scripts")
        cmake = shutil.which("cmake", path=search) or shutil.which("cmake")
        self.assertIsNotNone(cmake)
        scratch = ROOT / "temp/ck-input-contract"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder)
            header = root / "wmma.hpp"
            script = root / "probe.cmake"
            script.write_text('include("' + (ROOT / "cmake/modules/RNS8CKInputContract.cmake").as_posix() + '")\n'
                + 'rns8_ck_wmma_input_contract("' + header.as_posix() + '" mode)\nmessage(STATUS "MODE=${mode}")\n')
            cases = (("bool neg_a = true,\nbool neg_b = true,", "1"),
                     ("bool neg_a = false,\nbool neg_b = false,", "0"),
                     ("bool neg_a = true,\nbool neg_b = false,", None),
                     ("bool neg_a = true,", None), ("unknown declaration", None))
            for source, expected in cases:
                with self.subTest(source=source):
                    header.write_text(source)
                    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True, check=False)
                    if expected is None:
                        self.assertNotEqual(result.returncode, 0)
                    else:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("MODE=" + expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
