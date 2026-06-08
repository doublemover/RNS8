#!/usr/bin/env python3
"""Self-test AMD matrix instruction report parsing, ranking, and artifact capture."""

from __future__ import annotations

from pathlib import Path
import tempfile

import amd_matrix_instruction_report as report


DETAIL_32 = """
Architecture: CDNA3
Instruction: V_MFMA_I32_32X32X16_I8
    Matrix Dimensions:
        M: 32
        N: 32
        K: 16
        blocks: 1
    Execution statistics:
        Ops: 32768
        Execution cycles: 32
        Ops/CU/cycle: 4096
        Can co-execute with VALU: True
        VALU co-execution cycles possible: 28
    Register usage:
        GPRs required for A: 2
        GPRs required for B: 2
        GPRs required for C: 16
        GPRs required for D: 16
    Register data types:
        Src0: int8 (Signed 8-bit integer)
        Src1: int8 (Signed 8-bit integer)
        Src2: int32 (Signed 32-bit integer)
        Vdst: int32 (Signed 32-bit integer)
    Register modifiers:
        Sparse A matrix: False
"""

DETAIL_16 = """
Architecture: CDNA3
Instruction: V_MFMA_I32_16X16X32_I8
    Matrix Dimensions:
        M: 16
        N: 16
        K: 32
        blocks: 1
    Execution statistics:
        Ops: 16384
        Execution cycles: 16
        Ops/CU/cycle: 4096
        Can co-execute with VALU: True
        VALU co-execution cycles possible: 12
    Register usage:
        GPRs required for A: 2
        GPRs required for B: 2
        GPRs required for C: 4
        GPRs required for D: 4
    Register modifiers:
        Sparse A matrix: False
"""

DETAIL_SPARSE = """
Architecture: CDNA3
Instruction: V_SMFMAC_I32_32X32X32_I8
    Matrix Dimensions:
        M: 32
        N: 32
        K: 32
        blocks: 1
    Execution statistics:
        Ops: 65536
        Execution cycles: 32
        Ops/CU/cycle: 8192
        Can co-execute with VALU: True
        VALU co-execution cycles possible: 24
    Register usage:
        GPRs required for A: 2
        GPRs required for B: 4
        GPRs required for D: 16
    Register modifiers:
        Sparse A matrix: True
"""

DETAIL_RDNA3_DENSE = """
Architecture: RDNA3
Instruction: V_WMMA_I32_16X16X16_IU8
    Matrix Dimensions:
        M: 16
        N: 16
        K: 16
    Execution statistics:
        Ops: 8192
        Execution cycles: 8
        Ops/WGP/cycle: 4096
        Can co-execute with VALU: False
    Wave32 register usage:
        GPRs required for A: 1
        GPRs required for B: 1
        GPRs required for C: 8
        GPRs required for D: 8
    Wave64 register usage:
        GPRs required for A: 1
        GPRs required for B: 1
        GPRs required for C: 4
        GPRs required for D: 4
    Register data types:
        Src0: IU8 (Signed/unsigned 8-bit integer)
        Src1: IU8 (Signed/unsigned 8-bit integer)
        Src2: int32 (Signed 32-bit integer)
        Vdst: int32 (Signed 32-bit integer)
    Register modifiers:
        OPSEL supported: True
        NEG bits supported: True
"""

DETAIL_RDNA4_SPARSE = """
Architecture: RDNA4
Instruction: V_SWMMAC_I32_16X16X32_IU8
    Matrix Dimensions:
        M: 16
        N: 16
        K: 32
    Execution statistics:
        Ops: 16384
        Execution cycles: 8
        Ops/WGP/cycle: 8192
        Can co-execute with VALU: False
    Wave32 register usage:
        GPRs required for A: 2
        GPRs required for B: 4
        GPRs required for D: 8
    Wave64 register usage:
        GPRs required for A: 1
        GPRs required for B: 2
        GPRs required for D: 4
    Register data types:
        Src0: IU8 (Signed/unsigned 8-bit integer)
        Src1: IU8 (Signed/unsigned 8-bit integer)
        Src2: A matrix compression indices
        Vdst: int32 (Signed 32-bit integer)
    Register modifiers:
        OPSEL supported: True
        NEG bits supported: True
"""


def _write_fake_calculator(path: Path) -> None:
    path.write_text(
        f"""#!/usr/bin/env python3
import sys
DETAIL = {DETAIL_RDNA4_SPARSE!r}
args = sys.argv[1:]
if "--version" in args:
    print("fake-matrix-calculator 1.0")
elif "--list-instructions" in args:
    print("Available instructions in the RDNA4 architecture:")
    print("    v_swmmac_i32_16x16x32_iu8")
elif "--detail-instruction" in args:
    print(DETAIL.strip())
elif "--register-layout" in args:
    print("matrix,0")
    print("0,v0{{0}}")
elif "--matrix-layout" in args:
    print("lane,v0")
    print("0,D[0][0]")
elif "--output-calculation" in args:
    print("D[0][0] = Src0_v0{{0}} * Src1_v0{{0}} + Src2_v0{{0}}")
else:
    print("unexpected args: " + " ".join(args), file=sys.stderr)
    raise SystemExit(2)
""",
        encoding="utf-8",
    )


def main() -> int:
    listed = report.parse_instruction_list(
        """
Available instructions in the CDNA3 architecture:
    v_mfma_i32_32x32x16_i8
    v_smfmac_i32_32x32x32_i8
"""
    )
    assert listed == ["v_mfma_i32_32x32x16_i8", "v_smfmac_i32_32x32x32_i8"]

    dense_32 = report.parse_detail("gfx942", "v_mfma_i32_32x32x16_i8", DETAIL_32)
    dense_16 = report.parse_detail("gfx942", "v_mfma_i32_16x16x32_i8", DETAIL_16)
    sparse = report.parse_detail("gfx942", "v_smfmac_i32_32x32x32_i8", DETAIL_SPARSE)
    rdna_dense = report.parse_detail("gfx1100", "v_wmma_i32_16x16x16_iu8", DETAIL_RDNA3_DENSE)
    rdna4_sparse = report.parse_detail("gfx1200", "v_swmmac_i32_16x16x32_iu8", DETAIL_RDNA4_SPARSE)
    assert report.DEFAULT_ARCHITECTURES == ("gfx942", "gfx1100", "gfx1200", "gfx1201")
    assert dense_32.ops_per_cu_cycle == 4096
    assert dense_32.ops_per_cycle == 4096
    assert dense_32.ops_per_cycle_metric == "ops_per_cu_cycle"
    assert dense_32.gprs_d == 16
    assert dense_16.gprs_d == 4
    assert dense_32.register_data_types["Src0"].startswith("int8")
    assert sparse.category == "sparse_i8_i32_matrix_core"
    assert sparse.sparse_a_matrix is True
    assert sparse.sparse_future_only is True
    assert rdna_dense.category == "dense_i8_i32_matrix_core"
    assert rdna_dense.operand_signedness == "signed_or_unsigned_selected_by_instruction_modifier"
    assert rdna_dense.wavefront_register_usage["32"]["d"] == 8
    assert rdna_dense.wavefront_register_usage["64"]["d"] == 4
    assert rdna_dense.gprs_d == 4
    assert rdna_dense.rdna_integer_modifier_constraints["NEG[2]"].startswith("must be zero")
    assert rdna_dense.rdna_integer_modifier_constraints["NEG[0]"].startswith("A operand")
    assert rdna4_sparse.instruction_family == "swmmac"
    assert rdna4_sparse.category == "sparse_i8_i32_matrix_core"
    assert rdna4_sparse.modifier_support["opsel_supported"] is True

    recommendations = report._dense_recommendations([dense_32, dense_16, sparse])
    assert recommendations[0]["instruction"] == "v_mfma_i32_16x16x32_i8"
    assert recommendations[0]["reason"] == "highest_dense_integer_throughput_lowest_accumulator_register_pressure"
    assert recommendations[0]["ops_per_cycle"] == 4096
    sparse_notes = report._sparse_notes([dense_32, dense_16, sparse, rdna4_sparse])
    assert sparse_notes[0]["eligibility"] == "future_sparse_only_requires_explicit_4_to_2_A_matrix_compression_contract"

    assert report.instruction_category("v_wmma_i32_16x16x16_iu4") == "dense_i4_i32_matrix_core"
    assert report.instruction_category("v_swmmac_i32_16x16x32_iu4") == "sparse_i4_i32_matrix_core"

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        calculator = root / "fake_matrix_calculator.py"
        _write_fake_calculator(calculator)
        generated = report.build_report(calculator, ["gfx1200"], out_dir=root, capture_layouts=True)
        arch = generated["architectures"][0]
        assert generated["schema_version"] == 2
        assert arch["sparse_integer_i32_instruction_count"] == 1
        artifacts = arch["sparse_integer_i32_instructions"][0]["layout_artifacts"]
        assert any(item["matrix"] == "compression" and item["kind"] == "register_layout_csv" for item in artifacts)
        assert any(item["kind"] == "d_output_calculation" for item in artifacts)
        for item in artifacts:
            assert (root / item["path"]).exists()

    print("amd matrix instruction report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
