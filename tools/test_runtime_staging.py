"""Runtime staging filesystem contracts; fake DLL bytes never execute."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeStaging(unittest.TestCase):
    def test_selected_files_replace_stale_and_reuse_identical_runtime(self):
        cmake = shutil.which("cmake", path=str(Path(sys.executable).parent / "Scripts")) or shutil.which("cmake")
        self.assertIsNotNone(cmake)
        scratch = ROOT / "temp/runtime-staging-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder)
            sdk = root / "SDK with spaces"; sdk.mkdir()
            output = root / "app with spaces"; output.mkdir()
            first = sdk / "amdhip64_7.dll"; first.write_bytes(b"selected fixture runtime")
            second = sdk / "rocm_kpack.dll"; second.write_bytes(b"selected fixture loader")
            stale = output / first.name; stale.write_bytes(b"old fixture")
            command = [cmake, "-DRNS8_RUNTIME_DLLS=" + str(first) + ";" + str(second),
                "-DRNS8_RUNTIME_DESTINATION=" + str(output), "-P", str(ROOT / "cmake/RNS8StageRuntime.cmake")]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(stale.read_bytes(), first.read_bytes())
            self.assertEqual((output / second.name).read_bytes(), second.read_bytes())
            timestamp = stale.stat().st_mtime_ns
            subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(stale.stat().st_mtime_ns, timestamp)
            second.unlink()
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Selected SDK runtime DLL is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
