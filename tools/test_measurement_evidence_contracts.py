"""Authored static artifact/symbol associations; never GPU measurements."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evidence_database_lib.isa import load_isa_index, lookup_isa_resources
from test_capture_identity_support import synthetic_identity, synthetic_static_isa_report
import gpu_isa_report
import isa_common


class StaticAssociation(unittest.TestCase):
    def test_extraction_keeps_the_inspected_artifact_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = root / "input.obj"
            obj.write_bytes(b"original COFF artifact")
            scratch = root / "scratch"
            scratch.mkdir()
            calls = []

            def command(argv):
                calls.append(argv)
                if argv[0] == "objcopy":
                    # Model LLVM's in-place rewrite when no output is given.
                    Path(argv[-1]).write_bytes(b"rewritten COFF artifact")
                return "hip-amdgcn-amd-amdhsa--gfx1100\n" if "--list" in argv else ""

            with patch.object(isa_common, "run_command", side_effect=command):
                isa_common.extract_code_object("objcopy", "bundler", obj, "gfx1100", scratch, "input.fatbin")
            self.assertEqual(obj.read_bytes(), b"original COFF artifact")
            self.assertEqual(calls[0][-2], str(obj))
            self.assertEqual(Path(calls[0][-1]).parent, scratch)
            alias = scratch / "extracted-host-copy.obj"
            with self.assertRaisesRegex(ValueError, "aliases"):
                isa_common.extract_code_object("objcopy", "bundler", alias, "gfx1100", scratch, "input.fatbin")

    def fixture(self):
        capture = {"backend_selected": "ck", "selected_kernel": "ck_selected",
                   "device": {"gcn_arch": "gfx1100"}}
        synthetic_identity(capture)
        report = synthetic_static_isa_report(capture, {"wmma": 7, "vgpr_count": 64,
            "matrix_instruction_histogram": {"v_wmma_i32_16x16x16_iu8": 7}})
        report["symbols"].append({"symbol": "unselected_export", "counts": {"wmma": 999, "vgpr_count": 200}})
        return capture, report

    def load(self, directory, reports):
        paths = []
        for i, report in enumerate(reports):
            path = Path(directory) / f"{i}-isa-summary.json"
            path.write_text(json.dumps(report)); paths.append(path)
        return load_isa_index(paths)

    def test_target_build_symbol_and_duplicate_boundaries(self):
        capture, report = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            other = deepcopy(report)
            other["capture_binding"]["artifact_sha256"] = "f" * 64
            index = self.load(directory, [report, other, deepcopy(report)])
            matched = lookup_isa_resources(index, "ck", "gfx1100", capture)
            self.assertTrue(matched["isa_association"]["qualified"])
            self.assertEqual(matched["isa_report_count"], 1)
            self.assertEqual(matched["isa_wmma_count"], 7)
            self.assertEqual(matched["isa_vgpr_count"], 64)
            self.assertFalse(matched["isa_association"]["dynamic_execution_proven"])
            for target in ("gfx942", "unknown", "gfx1100;gfx942", "gfx1100:xnack-"):
                missing = lookup_isa_resources(index, "ck", target, capture)
                self.assertFalse(missing["isa_association"]["qualified"])
                self.assertEqual(missing["isa_report_count"], 0)
            for field in ("artifact", "kernel", "build"):
                changed = deepcopy(capture)
                if field == "artifact": changed["artifact_capture"]["executable_sha256"] = "e" * 64
                elif field == "kernel": changed["selected_kernel"] = "another_kernel"
                else: changed["build_manifest"]["source_digest"] = "f" * 64
                self.assertFalse(lookup_isa_resources(index, "ck", "gfx1100", changed)["isa_association"]["qualified"])
            legacy = deepcopy(report); legacy.pop("capture_binding")
            inventory = lookup_isa_resources(self.load(directory, [legacy]), "ck", "gfx1100", capture)
            self.assertFalse(inventory["isa_association"]["qualified"])
            self.assertEqual(inventory["isa_inventory"]["isa_wmma_count"], 7)

    def test_conflicting_selected_symbol_never_summed(self):
        capture, report = self.fixture()
        conflict = deepcopy(report); conflict["symbols"][0]["counts"]["wmma"] = 8
        with tempfile.TemporaryDirectory() as directory:
            result = lookup_isa_resources(self.load(directory, [report, conflict]), "ck", "gfx1100", capture)
            self.assertFalse(result["isa_association"]["qualified"])
            self.assertIn("conflicting_same_object_symbol_reports", result["isa_association"]["reasons"])

    def test_binding_checks_captured_artifact_and_exact_symbol(self):
        capture, report = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "synthetic.exe"; artifact.write_bytes(b"fixture not executable")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            capture["artifact_capture"]["executable_sha256"] = digest
            report["object_sha256"] = digest
            metadata = {**capture, "target_id": "gfx1100", "build_manifest_sha256":
                hashlib.sha256(json.dumps(capture["build_manifest"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
            config = gpu_isa_report.IsaToolConfig(artifact, "gfx1100", "unused", "unused", "unused", Path(directory))
            gpu_isa_report.bind_selected_symbols(report, metadata, artifact, ["synthetic_selected_symbol"], config)
            self.assertEqual(report["capture_binding"]["artifact_sha256"], digest)
            with self.assertRaises(RuntimeError):
                gpu_isa_report.bind_selected_symbols(report, metadata, artifact, ["missing"], config)
            artifact.write_bytes(b"rebuilt fixture")
            with self.assertRaises(RuntimeError):
                gpu_isa_report.bind_selected_symbols(report, metadata, artifact, ["synthetic_selected_symbol"], config)

    def test_object_only_cli_reaches_report_without_build_tree(self):
        _, report = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            obj = Path(directory) / "ck_backend_kernels.obj"; obj.write_bytes(b"synthetic object")
            output = Path(directory) / "reports"
            argv = ["gpu_isa_report.py", "--object", str(obj), "--backend", "ck", "--target", "gfx1100",
                    "--llvm-objcopy", "unused", "--llvm-objdump", "unused", "--llvm-readobj", "unused",
                    "--clang-offload-bundler", "unused", "--out-dir", str(output)]
            with patch("sys.argv", argv), patch.object(gpu_isa_report, "report_object", return_value=report) as call:
                self.assertEqual(gpu_isa_report.main(), 0)
                self.assertEqual(call.call_count, 1)
            manifest = json.loads((output / "gpu-isa-report-manifest.json").read_text())
            self.assertEqual(manifest["expected_backends"], ["ck"])
            self.assertEqual(manifest["reported_backends"], ["ck"])


if __name__ == "__main__":
    unittest.main()
