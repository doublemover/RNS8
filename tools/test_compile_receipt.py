"""Authored owned-process bookkeeping controls; fake objects never qualify HIP."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import compile_receipt as receipt

FAKE = """import pathlib,sys
args=sys.argv
mode=args[args.index('--mode')+1]
if mode=='failed':
 print('authored compiler failure',file=sys.stderr);sys.exit(7)
if mode=='stale':sys.exit(0)
if mode=='changed':pathlib.Path(args[args.index('-c')+1]).write_text('changed')
pathlib.Path(args[args.index('-o')+1]).write_bytes(b'authored object bookkeeping only')
print('authored compiler output')
"""


class CompileReceipts(unittest.TestCase):
    def test_shared_compiler_environment_keeps_compile_overrides_and_excludes_unrelated_keys(self):
        from compiler_context import compiler_environment

        selected = dict(
            HIP_CLANG_PATH="SDK/compiler",
            HIPCC_COMPILE_FLAGS_APPEND="-O2",
            LLVM_PATH="SDK/llvm",
            CXXFLAGS="-g",
        )
        self.assertEqual(
            compiler_environment({**selected, "UNRELATED_API_TOKEN": "private"}), selected
        )
        for invalid in ({"PATH": 3}, {3: "value"}, {"": "value"}):
            with self.assertRaises(ValueError):
                compiler_environment(invalid)

    def context(self, root: Path, mode="healthy"):
        source = root / "native_variants.hip"
        source.write_text("// authored source, not compiled HIP\n")
        compiler = root / "authored_compiler.py"
        compiler.write_text(FAKE)
        output = root / "native.obj"
        path = root / "context.json"
        value = dict(
            schema=receipt.CONTEXT_SCHEMA,
            source_root=str(receipt.ROOT),
            build_root=str(root),
            directory=str(root),
            file=str(source),
            output=str(output),
            configuration="Release",
            targets=["gfx1100"],
            inputs=[str(source), str(compiler)],
            arguments=[
                sys.executable,
                str(compiler),
                "--mode",
                mode,
                "--offload-arch=gfx1100",
                "-c",
                str(source),
                "-o",
                str(output),
            ],
        )
        path.write_text(json.dumps(value))
        return path, source, output

    def run_compile(self, context):
        with patch.object(
            receipt.shutil,
            "disk_usage",
            return_value=type("Disk", (), {"free": receipt.MIN_FREE_BYTES + 1})(),
        ):
            return receipt.execute(context, timeout_seconds=10)

    def test_plan_does_not_launch_or_modify_existing_output(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            context, _, output = self.context(root)
            output.write_bytes(b"old")
            with patch.object(
                receipt, "run_capture", side_effect=AssertionError("plan launched compiler")
            ):
                plan = receipt.configured_plan(context)
            self.assertEqual(plan["status"], "planned")
            self.assertEqual(output.read_bytes(), b"old")
            self.assertFalse(Path(str(output) + ".compile.json").exists())
            for timeout in (True, 0, -1, float("nan"), float("inf"), 3601):
                with self.assertRaises(ValueError):
                    receipt.configured_plan(context, timeout_seconds=timeout)

    def test_success_requires_a_fresh_object_and_stable_inputs(self):
        for mode in ("healthy", "failed", "stale", "changed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                context, source, output = self.context(root, mode)
                output.write_bytes(b"stale old object")
                result = self.run_compile(context)
                self.assertEqual(result["status"], "complete" if mode == "healthy" else "failed")
                pointer = Path(str(output) + ".compile.json")
                native = Path(receipt.read_json(pointer)["receipt"])
                self.assertTrue(native.is_file())
                self.assertTrue(Path(result["streams"]["stderr"]["path"]).is_file())
                if mode == "healthy":
                    cost = receipt.load_compile_cost(
                        pointer, source=source, host_object=output, target="gfx1100"
                    )
                    self.assertGreater(cost["seconds"], 0)
                    self.assertFalse(result["hardware_qualification"])
                else:
                    with self.assertRaises(ValueError):
                        receipt.load_compile_cost(
                            pointer, source=source, host_object=output, target="gfx1100"
                        )
                    self.assertTrue(result["failure_reasons"])
                if mode == "stale":
                    self.assertFalse(output.exists())
                if mode == "failed":
                    self.assertEqual(result["process"]["returncode"], 7)
                if mode == "changed":
                    self.assertFalse(result["inputs_unchanged"])

    def test_receipt_context_artifact_log_and_target_tamper_cannot_supply_cost(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            context, source, output = self.context(root)
            record = self.run_compile(context)
            pointer = Path(str(output) + ".compile.json")
            path = Path(receipt.read_json(pointer)["receipt"])
            healthy = path.read_bytes()
            object_bytes = output.read_bytes()
            log = Path(record["streams"]["stdout"]["path"])
            log_bytes = log.read_bytes()
            for mode in ("object", "source", "log", "command", "duration", "target"):
                with self.subTest(mode=mode):
                    changed = copy.deepcopy(record)
                    if mode == "object":
                        output.write_bytes(b"changed object")
                    elif mode == "source":
                        source.write_text("changed source")
                    elif mode == "log":
                        log.write_bytes(b"changed log")
                    elif mode == "command":
                        changed["command"].append("--changed")
                    elif mode == "duration":
                        changed["process"]["elapsed_seconds"] = True
                    else:
                        changed["targets"] = ["gfx942"]
                    if mode in ("command", "duration", "target"):
                        changed.pop("identity_sha256")
                        receipt.write_json(path, receipt.seal(changed))
                    with self.assertRaises(ValueError):
                        receipt.load_compile_cost(
                            path, source=source, host_object=output, target="gfx1100"
                        )
                    path.write_bytes(healthy)
                    output.write_bytes(object_bytes)
                    log.write_bytes(log_bytes)
                    source.write_text("// authored source, not compiled HIP\n")

    def test_context_output_escape_and_target_disagreement_reject_before_execution(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            context, _, _ = self.context(root)
            healthy = receipt.read_json(context)
            for field, value in (
                ("output", str(root.parent / "escaped.obj")),
                ("targets", ["gfx942"]),
                ("arguments", []),
            ):
                changed = {**healthy, field: value}
                context.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    receipt.configured_plan(context)


if __name__ == "__main__":
    unittest.main()
