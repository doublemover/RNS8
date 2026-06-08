#!/usr/bin/env python3
"""Self-test AMD matrix instruction report parsing and ranking."""

from __future__ import annotations

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
    assert dense_32.ops_per_cu_cycle == 4096
    assert dense_32.ops_per_cycle == 4096
    assert dense_32.ops_per_cycle_metric == "ops_per_cu_cycle"
    assert dense_32.gprs_d == 16
    assert dense_16.gprs_d == 4
    assert sparse.category == "sparse_i8_i32_matrix_core"
    assert sparse.sparse_a_matrix is True

    recommendations = report._dense_recommendations([dense_32, dense_16, sparse])
    assert recommendations[0]["instruction"] == "v_mfma_i32_16x16x32_i8"
    assert recommendations[0]["reason"] == "highest_dense_i8_throughput_lowest_accumulator_register_pressure"
    assert recommendations[0]["ops_per_cycle"] == 4096
    sparse_notes = report._sparse_notes([dense_32, dense_16, sparse])
    assert sparse_notes[0]["eligibility"] == "future_sparse_only_requires_explicit_4_to_2_A_matrix_compression_contract"

    print("amd matrix instruction report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
