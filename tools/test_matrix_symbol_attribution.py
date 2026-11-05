"""Authored native-text controls; no compiler, object tool or GPU launches."""
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import check_ck_kernel_isa as ck
import check_rocwmma_kernel_isa as wmma
from isa_common import matrix_symbol_reports, selected_function_body


def native(symbol, instructions):
    return f"\n0000000000000000 <{symbol}>:\n{instructions}\n"


class MatrixSymbols(unittest.TestCase):
    def scan(self, module, target, bodies, selected):
        with patch("isa_common.device_function_symbols", return_value=list(bodies)), patch("isa_common.disassemble_code_object", side_effect=lambda tool,obj,arch,symbol: bodies[symbol]):
            return module.scan_disassembly("authored-tool",Path("authored.co"),target,selected)

    def test_healthy_sibling_never_masks_empty_candidate_on_either_family(self):
        for module in (ck,wmma):
            for target in ("gfx1100","gfx942"):
                opcode=module.required_matrix_mnemonics(target)[0]
                for empty in ("ordinary","wrap"):
                    bodies={name:native(name,"s_endpgm" if name==empty else opcode+" v0, v1, v2\ns_endpgm") for name in ("ordinary","wrap")}
                    with self.assertRaisesRegex(RuntimeError,empty+".*no_attributed"):
                        self.scan(module,target,bodies,list(bodies))
                bodies={name:native(name,opcode+" v0, v1, v2\ns_endpgm") for name in ("ordinary","wrap")}
                self.assertEqual(self.scan(module,target,bodies,list(bodies))[0],2)

    def test_whole_object_text_and_mnemonic_comments_cannot_supply_selected_work(self):
        opcode=ck.RDNA_REQUIRED_MNEMONICS[0]
        bodies={"empty":native("empty","s_endpgm // "+opcode)+native("unrelated",opcode+" v0,v1,v2"),
                "unrelated":native("unrelated",opcode+" v0,v1,v2")}
        for module in (ck,wmma):
            with self.assertRaisesRegex(RuntimeError,"no_attributed"):
                self.scan(module,"gfx1100",bodies,["empty"])
        with self.assertRaisesRegex(RuntimeError,"missing/ambiguous"):
            selected_function_body(native("other","s_endpgm"),"missing")

    def test_actual_checker_entrypoints_require_all_markers_and_each_specialization(self):
        for module in (ck,wmma):
            for target in ("gfx1100","gfx942"):
                markers=module.required_symbol_markers(target) if module is ck else module.WMMA_SYMBOL_MARKERS
                names=[marker+"_authored" for marker in markers]
                if module is ck:
                    names.append(names[0]+"_second_specialization")
                opcode=module.required_matrix_mnemonics(target)[0]
                config=SimpleNamespace(host_object=Path("authored.obj"),target=target,objdump="authored-tool")
                for missing_body in (None,names[-1]):
                    bodies={name:native(name,"s_endpgm" if name==missing_body else opcode+" v0,v1,v2\ns_endpgm") for name in names}
                    with patch.object(module,"parse_isa_tool_config",return_value=config), patch.object(module,"extracted_device_code_object",return_value=nullcontext(Path("authored.co"))), patch("isa_common.device_function_symbols",return_value=names), patch("isa_common.disassemble_code_object",side_effect=lambda tool,obj,arch,symbol:bodies[symbol]):
                        if missing_body:
                            with self.assertRaisesRegex(RuntimeError,"no_attributed"):
                                module.main()
                        else:
                            self.assertEqual(module.main(),0)
                with patch("isa_common.device_function_symbols",return_value=["unrelated"]):
                    with self.assertRaisesRegex(RuntimeError,"missing expected"):
                        if module is ck:
                            module.ck_symbols("authored",Path("authored.co"),target)
                        else:
                            module.wmma_symbols("authored",Path("authored.co"))

    def test_only_resolved_reachable_helper_counts_and_unresolved_calls_fail(self):
        opcode=ck.RDNA_REQUIRED_MNEMONICS[0]
        bodies={"entry":native("entry","s_call_b64 s[0:1], 0 <helper>\ns_endpgm"),
                "helper":native("helper",opcode+" v0,v1,v2\ns_setpc_b64 s[0:1]"),
                "unrelated":native("unrelated",opcode+" v0,v1,v2")}
        for module in (ck,wmma):
            self.assertEqual(self.scan(module,"gfx1100",bodies,["entry"])[0],1)
            changed={**bodies,"entry":native("entry",opcode+" v0,v1,v2\ns_swappc_b64 s[0:1],s[2:3]")}
            with self.assertRaisesRegex(RuntimeError,"unresolved_device_call"):
                self.scan(module,"gfx1100",changed,["entry"])
        with patch("isa_common.device_function_symbols", return_value=list(bodies)), patch("isa_common.disassemble_code_object", side_effect=lambda tool,obj,arch,symbol:bodies[symbol]):
            row=matrix_symbol_reports("authored",Path("authored.co"),"gfx1100",["entry"],[opcode])[0]
        self.assertEqual({item["symbol"] for item in row["functions"]},{"entry","helper"})
        self.assertTrue(all(len(item["disassembly_sha256"])==64 for item in row["functions"]))


if __name__=="__main__":
    unittest.main()
