"""Declaration-discovery and mismatch controls; no compiler or library execution."""

from __future__ import annotations

from dataclasses import replace
import copy
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from abi_surface import collect_surface, surface_diff, declaration_sha256
from abi_probes import (
    declaration_assertions,
    library_inventory_source,
    metric_assertions,
    native_probe_source,
)
from abi_binary_types import complete_types
from check_public_abi import (
    baseline_metrics,
    reject_abi_flags,
    run_negative_consumers,
    tuple_key,
)


def enum(name="rns8_mode", *, fixed=True):
    node = dict(
        kind="EnumDecl",
        id="enum-id",
        name=name,
        inner=[
            dict(
                kind="EnumConstantDecl",
                name="RNS8_MODE",
                inner=[dict(kind="ConstantExpr", value="0")],
            )
        ],
    )
    if fixed:
        node["fixedUnderlyingType"] = dict(qualType="uint32_t")
    return node


def ast_fixture():
    return dict(
        kind="TranslationUnitDecl",
        inner=[
            enum(),
            dict(
                kind="CXXRecordDecl",
                name="rns8_record",
                tagUsed="struct",
                completeDefinition=True,
                inner=[dict(kind="FieldDecl", name="value", type=dict(qualType="uint64_t"))],
            ),
            dict(kind="RecordDecl", name="rns8_owner", tagUsed="struct"),
            dict(
                kind="TypedefDecl",
                name="rns8_callback",
                type=dict(qualType="void (*)(void *)"),
                inner=[dict(kind="PointerType", inner=[dict(kind="FunctionProtoType")])],
            ),
            dict(
                kind="FunctionDecl",
                name="rns8_query",
                mangledName="rns8_query",
                type=dict(qualType="void (rns8_record *)"),
                inner=[dict(kind="DLLImportAttr")],
            ),
        ],
    )


def surface():
    return collect_surface(
        ast_fixture(), {"rns8/rns8.h": "0" * 64}, {"RNS8_ABI_VERSION": "1u"}, cpp=True
    )


class PublicABIInventory(unittest.TestCase):
    def test_complete_ast_categories_and_signature_changes(self):
        value = surface()
        self.assertEqual(value.records["rns8_record"], [["value", "uint64_t"]])
        self.assertEqual(value.handles, ["rns8_owner"])
        self.assertEqual(value.callbacks["rns8_callback"], "void (*)(void *)")
        self.assertEqual(value.visibility["rns8_query"], ["DLLImportAttr"])
        for changed in (
            replace(value, enums={"rns8_mode": {"RNS8_MODE": 1}}),
            replace(value, records={"rns8_record": [["value", "double"]]}),
            replace(value, functions={"rns8_other": "void (rns8_record *)"}),
            replace(value, callbacks={"rns8_callback": "int (*)(void *)"}),
        ):
            self.assertTrue(surface_diff(value.to_json(), changed.to_json()))
            self.assertNotEqual(declaration_sha256(value), declaration_sha256(changed))
        # Content hashes bind a receipt, but comments/import modes are not a
        # changed ABI. Same-sized member types above still are a changed ABI.
        compatible = replace(value, headers={"rns8/rns8.h": "1" * 64}, visibility={})
        self.assertEqual(surface_diff(value.to_json(), compatible.to_json()), [])

    def test_fixed_backing_is_checked_on_declaration(self):
        ast = ast_fixture()
        ast["inner"][0] = enum(fixed=False)
        with self.assertRaises(ValueError):
            collect_surface(ast, {"rns8/rns8.h": "0" * 64}, {}, cpp=True)
        # C uses normal platform enum representation, checked by native probes.
        self.assertEqual(
            len(collect_surface(ast, {"rns8/rns8.h": "0" * 64}, {}, cpp=False).enums), 1
        )

    def test_anonymous_tag_public_typedef_is_discovered(self):
        ast = ast_fixture()
        ast["inner"][0].pop("name")
        ast["inner"].append(
            dict(
                kind="TypedefDecl",
                name="rns8_mode",
                type=dict(qualType="rns8_mode"),
                inner=[
                    dict(
                        kind="ElaboratedType",
                        ownedTagDecl=dict(id="enum-id", kind="EnumDecl"),
                    )
                ],
            )
        )
        self.assertIn(
            "rns8_mode",
            collect_surface(ast, {"rns8/rns8.h": "0" * 64}, {}, cpp=True).enums,
        )

    def test_unmodeled_declarations_fail_closed(self):
        for mutate in (
            "bitfield",
            "global",
            "non_c",
            "foreign_export",
            "missing_fold",
            "empty",
        ):
            ast = copy.deepcopy(ast_fixture())
            if mutate == "bitfield":
                ast["inner"][1]["inner"][0]["isBitfield"] = True
            elif mutate == "global":
                ast["inner"].append(dict(kind="VarDecl", name="rns8_global"))
            elif mutate == "non_c":
                ast["inner"][-1]["mangledName"] = "?rns8_query@@..."
            elif mutate == "foreign_export":
                ast["inner"][-1]["name"] = "foreign_query"
            elif mutate == "missing_fold":
                ast["inner"][0]["inner"][0]["inner"] = [dict(kind="IntegerLiteral", value="0")]
            else:
                ast["inner"] = []
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                collect_surface(ast, {"rns8/rns8.h": "0" * 64}, {}, cpp=True)

    def test_native_tuple_and_flags_cannot_be_assumed(self):
        self.assertEqual(
            tuple_key(
                "x86_64-pc-windows-msvc",
                {"sizeof:void_pointer": 8, "CHAR_BIT": 8, "sizeof:long": 4},
            ),
            "windows-x86_64-llp64",
        )
        with self.assertRaises(ValueError):
            baseline_metrics(dict(layout_baselines={}), "linux-aarch64-lp64")
        for flag in (
            "-fshort-enums",
            '"-fshort-enums"',
            "-fpack-struct",
            "-fpack-struct=1",
            "/Zp",
            "/Zp8",
            "-Zp1",
        ):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                reject_abi_flags(["clang " + flag + " source.c"])
        reject_abi_flags(["clang -O3 -std=c17 source.c", "cl /O2 /MD"])

    def test_generator_checks_members_callbacks_functions_and_all_metrics(self):
        value = surface()
        generated = declaration_assertions(value)
        self.assertIn("RNS8_ABI_MEMBER(rns8_record,value,", generated)
        self.assertIn("RNS8_ABI_TYPE(rns8_callback,", generated)
        self.assertIn("rns8_abi_reference_0=&rns8_query", generated)
        library = library_inventory_source(value)
        self.assertNotIn("rns8_abi_reference_", library)
        self.assertIn("offsetof(rns8_record,value)", library)
        with self.assertRaises(ValueError):
            metric_assertions(value, {})
        # The installed smoke cannot be silently generated for an old package.
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            root = Path(directory)
            (root / "rns8").mkdir()
            (root / "rns8/rns8.h").write_text("", encoding="utf-8")
            native_probe_source(value, root, None)

    def test_symbol_only_or_partial_debug_types_never_qualify(self):
        value = surface()
        xml = """<abi-corpus><abi-instr>
          <type-decl id="u64" name="uint64_t" size-in-bits="64"/>
          <type-decl id="void" name="void"/>
          <type-decl id="u32" name="uint32_t" size-in-bits="32"/>
          <class-decl id="rec" name="rns8_record" size-in-bits="64">
            <data-member layout-offset-in-bits="0"><var-decl name="value" type-id="u64"/></data-member>
          </class-decl>
          <enum-decl name="rns8_mode" size-in-bits="32"><underlying-type type-id="u32"/><enumerator name="RNS8_MODE" value="0"/></enum-decl>
          <function-decl name="rns8_query"><parameter type-id="rec"/><return type-id="void"/></function-decl>
        </abi-instr></abi-corpus>"""
        self.assertTrue(complete_types(xml, value)["type_information_complete"])
        for broken in (
            "<abi-corpus/>",
            xml.replace('size-in-bits="64"', 'size-in-bits="0"'),
            xml.replace('name="value"', 'name="shifted"'),
            xml.replace('value="0"', 'value="1"'),
            xml.replace('type-id="rec"', 'type-id="missing"'),
        ):
            with self.subTest(xml=broken), self.assertRaises(ValueError):
                complete_types(broken, value)

    def test_negative_build_requires_relevant_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = subprocess.CompletedProcess([], 1, "disk full", "")
            with (
                patch("check_public_abi.subprocess.run", return_value=failed),
                self.assertRaises(ValueError),
            ):
                run_negative_consumers(Path("cmake"), root, ["rns8_abi_negative_packing_c"], root)
            expected = subprocess.CompletedProcess([], 1, "static assertion failed: offsetof", "")
            with patch("check_public_abi.subprocess.run", return_value=expected):
                result = run_negative_consumers(
                    Path("cmake"), root, ["rns8_abi_negative_packing_c"], root
                )
            self.assertTrue(result["controls"]["rns8_abi_negative_packing_c"]["expected_rejection"])


if __name__ == "__main__":
    unittest.main()
