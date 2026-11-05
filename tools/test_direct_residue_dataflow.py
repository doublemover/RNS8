"""Authored source-policy drift controls; no compiler or GPU execution."""

import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

from direct_residue_dataflow import (
    CORE,
    ROOT,
    SCHEDULED,
    SOURCE_INPUTS,
    WRAPPERS,
    inspect_source,
    validate_policy_manifest,
)


class TiledSourceContracts(unittest.TestCase):
    def setUp(self):
        self.value = json.loads((ROOT / "metadata/kernels.yaml").read_text(encoding="utf-8"))[
            "direct_residue_tile_dataflow"
        ]

    def test_policy_denominator_and_source_binding(self):
        report = inspect_source(self.value)
        self.assertEqual(report["selected_wrappers"], 17)
        self.assertFalse(report["hardware_qualification"])
        self.assertEqual(len({row["symbol"] for row in report["policies"]}), 17)
        self.assertEqual(sum(row["outputs"] == 2 for row in report["policies"]), 1)
        admitted = validate_policy_manifest(self.value)
        self.value["policies"][0]["a"] = "changed after admission"
        self.assertEqual(admitted[0]["a"], "centered_i8")

    def test_policy_drift_empty_duplicates_and_incompatible_combinations_reject(self):
        for mutation in (
            "empty",
            "duplicate",
            "native_b_only",
            "selected_region",
            "float_tile",
            "scalar_slab_bool",
            "outputs_bool",
            "paired_packed",
            "scheduled_source",
            "pair_entry",
        ):
            value = copy.deepcopy(self.value)
            if mutation == "empty":
                value["policies"] = []
            elif mutation == "duplicate":
                value["policies"][-1] = copy.deepcopy(value["policies"][0])
            elif mutation == "native_b_only":
                value["policies"][0]["b"] = "native_u8"
            elif mutation == "selected_region":
                value["policies"][9]["region"] = True
            elif mutation == "float_tile":
                value["tile"][0] = 16.0
            elif mutation == "scalar_slab_bool":
                value["safe_signed_byte_slab"] = True
            elif mutation == "outputs_bool":
                value["policies"][0]["outputs"] = True
            elif mutation == "paired_packed":
                value["policies"][14]["dot"] = "packed"
            elif mutation == "scheduled_source":
                value["policies"][0]["source"] = SCHEDULED
            else:
                value["policies"][14]["schedule"] = "entry"
            with self.assertRaises(ValueError, msg=mutation):
                validate_policy_manifest(value)

    def test_changed_real_wrapper_tuple_shared_arrays_and_barriers_reject(self):
        sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in SOURCE_INPUTS}
        changes = (
            (WRAPPERS, "ResidueTileDot::configured", "ResidueTileDot::packed"),
            (WRAPPERS, "CenteredResidueTileInput{A,lda}", "CenteredResidueTileInput{A,ldb}"),
            (
                WRAPPERS,
                "a_tile[kRns8HipTileM][kRns8HipTileK]",
                "a_tile[kRns8HipTileM][kRns8HipTileK+1]",
            ),
            (CORE, "__syncthreads();", "/* omitted barrier */"),
            (SCHEDULED, "tile_A=A+static_cast<int>(entry.row_offset)*lda;", "tile_A=A;"),
            (
                SCHEDULED,
                "{static_cast<int>(entry.row_extent),static_cast<int>(entry.col_extent)",
                "{m,static_cast<int>(entry.col_extent)",
            ),
            (
                SCHEDULED,
                "kRns8HipTileNPadded,tile_extent,acc0,acc1",
                "kRns8HipTileN,tile_extent,acc0,acc1",
            ),
            (
                SCHEDULED,
                "if(output_zero_by_row_col)C[row*ldc+col]=0;",
                "/* missing scheduled zero store */",
            ),
            (SCHEDULED, "__syncthreads();", "/* omitted pair barrier */"),
        )
        for changed, old, new in changes:
            # Match complete lexical tokens with flexible formatting, and prove
            # the control actually edits the intended source before inspecting it.
            tokens = re.findall(r"\w+|::|[^\w\s]", old)
            pattern = re.compile(r"\s*".join(re.escape(token) for token in tokens))
            mutated, count = pattern.subn(
                lambda _, replacement=new: replacement, sources[changed], count=1
            )
            self.assertEqual(count, 1, changed + ":" + old)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name, text in sources.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(mutated if name == changed else text, encoding="utf-8")
                with self.assertRaises(ValueError, msg=changed + ":" + old):
                    inspect_source(self.value, root=root)


if __name__ == "__main__":
    unittest.main()
