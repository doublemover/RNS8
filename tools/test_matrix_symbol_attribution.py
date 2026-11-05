"""Authored native-text controls; no compiler, object tool or GPU launches."""

from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import check_ck_kernel_isa as ck
import check_rocwmma_kernel_isa as wmma
import check_generated_native_wmma as generated
import gpu_isa_report
from isa_common import device_function_sizes, matrix_symbol_reports, selected_function_body


def native(symbol, instructions):
    return f"\n0000000000000000 <{symbol}>:\n{instructions}\n"


class MatrixSymbols(unittest.TestCase):
    def scan(self, module, target, bodies, selected):
        with (
            patch("isa_common.device_function_symbols", return_value=list(bodies)),
            patch(
                "isa_common.disassemble_code_object",
                side_effect=lambda tool, obj, arch, symbol: bodies[symbol],
            ),
        ):
            return module.scan_disassembly("authored-tool", Path("authored.co"), target, selected)

    def test_healthy_sibling_never_masks_empty_candidate_on_either_family(self):
        for module in (ck, wmma):
            for target in ("gfx1100", "gfx942"):
                opcode = module.required_matrix_mnemonics(target)[0]
                for empty in ("ordinary", "wrap"):
                    bodies = {
                        name: native(
                            name, "s_endpgm" if name == empty else opcode + " v0, v1, v2\ns_endpgm"
                        )
                        for name in ("ordinary", "wrap")
                    }
                    with self.assertRaisesRegex(RuntimeError, empty + ".*no_attributed"):
                        self.scan(module, target, bodies, list(bodies))
                bodies = {
                    name: native(name, opcode + " v0, v1, v2\ns_endpgm")
                    for name in ("ordinary", "wrap")
                }
                self.assertEqual(self.scan(module, target, bodies, list(bodies))[0], 2)

    def test_whole_object_text_and_mnemonic_comments_cannot_supply_selected_work(self):
        opcode = ck.RDNA_REQUIRED_MNEMONICS[0]
        bodies = {
            "empty": native("empty", "s_endpgm // " + opcode)
            + native("unrelated", opcode + " v0,v1,v2"),
            "unrelated": native("unrelated", opcode + " v0,v1,v2"),
        }
        for module in (ck, wmma):
            with self.assertRaisesRegex(RuntimeError, "no_attributed"):
                self.scan(module, "gfx1100", bodies, ["empty"])
        with self.assertRaisesRegex(RuntimeError, "missing/ambiguous"):
            selected_function_body(native("other", "s_endpgm"), "missing")

    def test_actual_checker_entrypoints_require_all_markers_and_each_specialization(self):
        for module in (ck, wmma):
            for target in ("gfx1100", "gfx942"):
                markers = (
                    module.required_symbol_markers(target)
                    if module is ck
                    else module.WMMA_SYMBOL_MARKERS
                )
                names = [marker + "_authored" for marker in markers]
                if module is ck:
                    names.append(names[0] + "_second_specialization")
                opcode = module.required_matrix_mnemonics(target)[0]
                config = SimpleNamespace(
                    host_object=Path("authored.obj"), target=target, objdump="authored-tool"
                )
                for missing_body in (None, names[-1]):
                    bodies = {
                        name: native(
                            name,
                            "s_endpgm" if name == missing_body else opcode + " v0,v1,v2\ns_endpgm",
                        )
                        for name in names
                    }
                    with (
                        patch.object(module, "parse_isa_tool_config", return_value=config),
                        patch.object(
                            module,
                            "extracted_device_code_object",
                            return_value=nullcontext(Path("authored.co")),
                        ),
                        patch("isa_common.device_function_symbols", return_value=names),
                        patch(
                            "isa_common.disassemble_code_object",
                            side_effect=lambda tool, obj, arch, symbol: bodies[symbol],
                        ),
                    ):
                        if missing_body:
                            with self.assertRaisesRegex(RuntimeError, "no_attributed"):
                                module.main()
                        else:
                            self.assertEqual(module.main(), 0)
                with patch("isa_common.device_function_symbols", return_value=["unrelated"]):
                    with self.assertRaisesRegex(RuntimeError, "missing expected"):
                        if module is ck:
                            module.ck_symbols("authored", Path("authored.co"), target)
                        else:
                            module.wmma_symbols("authored", Path("authored.co"))

    def test_only_resolved_reachable_helper_counts_and_unresolved_calls_fail(self):
        opcode = ck.RDNA_REQUIRED_MNEMONICS[0]
        bodies = {
            "entry": native("entry", "s_call_b64 s[0:1], 0 <helper>\ns_endpgm"),
            "helper": native("helper", opcode + " v0,v1,v2\ns_setpc_b64 s[0:1]"),
            "unrelated": native("unrelated", opcode + " v0,v1,v2"),
        }
        for module in (ck, wmma):
            self.assertEqual(self.scan(module, "gfx1100", bodies, ["entry"])[0], 1)
            changed = {
                **bodies,
                "entry": native("entry", opcode + " v0,v1,v2\ns_swappc_b64 s[0:1],s[2:3]"),
            }
            with self.assertRaisesRegex(RuntimeError, "unresolved_device_call"):
                self.scan(module, "gfx1100", changed, ["entry"])
        with (
            patch("isa_common.device_function_symbols", return_value=list(bodies)),
            patch(
                "isa_common.disassemble_code_object",
                side_effect=lambda tool, obj, arch, symbol: bodies[symbol],
            ),
        ):
            row = matrix_symbol_reports(
                "authored", Path("authored.co"), "gfx1100", ["entry"], [opcode]
            )[0]
        self.assertEqual({item["symbol"] for item in row["functions"]}, {"entry", "helper"})
        self.assertTrue(all(len(item["disassembly_sha256"]) == 64 for item in row["functions"]))

    def test_native_function_sizes_require_unambiguous_function_table_records(self):
        text = """0000000000000000 g     F .text 0000000000000040 entry
0000000000000040 l     F .text.helper 0000000000000020 .hidden helper
0000000000000060 g     O .rodata 0000000000000008 data
"""
        with patch("isa_common.run_command", return_value=text):
            sizes = device_function_sizes("authored", Path("authored.co"))
        self.assertEqual(
            sizes,
            {
                "entry": dict(address=0, bytes=64, section=".text"),
                "helper": dict(address=64, bytes=32, section=".text.helper"),
            },
        )
        for bad in (
            text + text.splitlines()[0],
            text.replace("0000000000000040 entry", "unknown entry"),
            text.replace(".hidden helper", ".unknown helper"),
        ):
            with patch("isa_common.run_command", return_value=bad), self.assertRaises(RuntimeError):
                device_function_sizes("authored", Path("authored.co"))

    def test_generated_family_inspection_attributes_each_entry_and_its_code_extent(self):
        opcode = "v_wmma_i32_16x16x16_iu8"
        manifest = dict(
            source_sha256="authored-source",
            variants=[
                dict(id=name, kernel_symbol=name, spec=dict(target="gfx1100"))
                for name in ("entry", "sibling")
            ],
        )
        resources = dict(
            wavefront_size=32,
            lds_bytes=0,
            vgpr_count=24,
            sgpr_count=16,
            scratch_bytes=0,
            vgpr_spill_count=0,
            sgpr_spill_count=0,
            max_flat_workgroup_size=32,
        )
        for change in (
            "healthy",
            "empty",
            "helper",
            "unresolved",
            "helper_store",
            "missing_size",
            "missing_metadata",
            "changed_object",
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                host = root / "host.obj"
                host.write_bytes(b"authored-host")
                code = root / "device.co"
                code.write_bytes(b"authored-code")
                config = SimpleNamespace(host_object=host, target="gfx1100", objdump="authored")
                bodies = {
                    name: native(name, opcode + " v0,v1,v2\ns_endpgm")
                    for name in ("entry", "sibling", "helper")
                }
                if change == "empty":
                    bodies["entry"] = native("entry", "s_endpgm // " + opcode) + bodies["sibling"]
                if change in ("helper", "helper_store"):
                    bodies["entry"] = native("entry", "s_call_b64 s[0:1],0 <helper>\ns_endpgm")
                if change == "unresolved":
                    bodies["entry"] += native("unrelated", opcode + " v0,v1,v2")
                if change == "unresolved":
                    bodies["entry"] = bodies["entry"].replace(
                        "s_endpgm", "s_swappc_b64 s[0:1],s[2:3]", 1
                    )
                if change == "helper_store":
                    bodies["helper"] = native(
                        "helper", opcode + " v0,v1,v2\nglobal_store_dword v0,v1,s[0:1]"
                    )
                sizes = {name: dict(bytes=64, address=0, section=".text") for name in bodies}
                metadata = {name: dict(resources) for name in bodies}
                if change == "missing_size":
                    sizes.pop("entry")
                if change == "missing_metadata":
                    metadata["entry"].pop("vgpr_spill_count")

                def disassemble(tool, obj, arch, symbol):
                    if change == "changed_object":
                        host.write_bytes(b"changed")
                    return bodies[symbol]

                with (
                    patch.object(generated, "verify_generated_source", return_value=manifest),
                    patch.object(
                        generated, "extracted_device_code_object", return_value=nullcontext(code)
                    ),
                    patch.object(generated, "device_function_symbols", return_value=list(bodies)),
                    patch.object(generated, "device_function_sizes", return_value=sizes),
                    patch.object(generated, "readobj_for_objdump", return_value="authored-readobj"),
                    patch.object(generated, "code_object_metadata", return_value=metadata),
                    patch("isa_common.device_function_symbols", return_value=list(bodies)),
                    patch("isa_common.disassemble_code_object", side_effect=disassemble),
                ):
                    if change == "changed_object":
                        with self.assertRaisesRegex(RuntimeError, "changed during"):
                            generated.inspect(config, root)
                        continue
                    report = generated.inspect(config, root)
                self.assertEqual(report["inspection_passed"], change in ("healthy", "helper"))
                self.assertFalse(report["promotion_eligible"])
                self.assertFalse(report["device_differential_executed"])
                if change in ("healthy", "helper"):
                    expected = 192 if change == "helper" else 128
                    self.assertEqual(
                        report["family_inventory"]["unique_reachable_code_bytes"], expected
                    )
                    self.assertEqual(report["symbols"][0]["attributed_matrix_instructions"], 1)
                if change == "missing_size":
                    self.assertIsNone(report["family_inventory"]["unique_reachable_code_bytes"])

    def test_general_isa_report_scans_only_selected_native_function(self):
        symbol = "rns8_ring_gemm_i8_i32_authored"
        disassembly = native(symbol, "s_endpgm") + native(
            "unrelated", "v_wmma_i32_16x16x16_iu8 v0,v1,v2"
        )
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            host = root / "host.obj"
            host.write_bytes(b"host")
            code = root / "device.co"
            code.write_bytes(b"code")
            config = SimpleNamespace(
                host_object=host,
                target="gfx1100",
                objdump="authored",
                objcopy="authored",
                bundler="authored",
            )
            with (
                patch.object(
                    gpu_isa_report, "extracted_device_code_object", return_value=nullcontext(code)
                ),
                patch.object(gpu_isa_report, "device_function_symbols", return_value=[symbol]),
                patch.object(gpu_isa_report, "code_object_metadata", return_value={}),
                patch.object(gpu_isa_report, "disassemble_code_object", return_value=disassembly),
            ):
                report = gpu_isa_report.report_object(config, "direct-hip", None, None)
        self.assertEqual(report["symbols"][0]["counts"]["matrix_instruction_count"], 0)
        self.assertEqual(len(report["symbols"][0]["disassembly_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
