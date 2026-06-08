#!/usr/bin/env python3
"""Self-test AMDGPU metadata extraction in gpu_isa_report."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import gpu_isa_report
import isa_common


def main() -> int:
    rocm_candidates = [
        str(path).replace("\\", "/")
        for path in isa_common.hip_sdk_tool_candidates(Path("/opt/rocm/bin/hipcc"), "llvm-objcopy")
    ]
    assert any(
        path.endswith("/opt/rocm/llvm/bin/llvm-objcopy") or path.endswith("/opt/rocm/llvm/bin/llvm-objcopy.exe")
        for path in rocm_candidates
    )

    readobj_output = """
AMDGPU Metadata: ---
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           8
    .group_segment_fixed_size: 4096
    .max_flat_workgroup_size: 256
    .name:           _ZN12_GLOBAL__N_121rns8_fixture_kernelEv
    .private_segment_fixed_size: 0
    .sgpr_count:     36
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_121rns8_fixture_kernelEv.kd
    .vgpr_count:     44
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .offset:         0
        .size:           8
    .group_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_119rns8_export_kernelEv
    .private_segment_fixed_size: 16
    .sgpr_count:     24
    .sgpr_spill_count: 1
    .symbol:         _ZN12_GLOBAL__N_119rns8_export_kernelEv.kd
    .vgpr_count:     20
    .vgpr_spill_count: 2
    .wavefront_size: 32
...
"""
    metadata = gpu_isa_report.parse_amdgpu_metadata(readobj_output)
    fixture = metadata["_ZN12_GLOBAL__N_121rns8_fixture_kernelEv"]
    assert fixture["vgpr_count"] == 44
    assert fixture["sgpr_count"] == 36
    assert fixture["lds_bytes"] == 4096
    assert fixture["scratch_bytes"] == 0
    assert fixture["max_flat_workgroup_size"] == 256
    assert fixture["wavefront_size"] == 32

    disassembly_counts = {
        "wmma": 0,
        "mfma": 0,
        "smfmac": 0,
        "swmmac": 0,
        "matrix_instruction_count": 0,
        "dense_integer_matrix_instruction_count": 0,
        "sparse_integer_matrix_instruction_count": 0,
        "mfma_dense_i8": 0,
        "smfmac_sparse_i8": 0,
        "wmma_dense_integer": 0,
        "swmmac_sparse_integer": 0,
        "matrix_instruction_histogram": {},
        "matrix_instruction_families": [],
        "global_store": 3,
        "lds_mentions": 2,
        "wait_instructions": 1,
        "instruction_lines": 10,
        "vgpr_count": None,
        "sgpr_count": None,
        "occupancy": None,
    }
    merged = gpu_isa_report.merge_resource_metadata(disassembly_counts, fixture)
    assert merged["global_store"] == 3
    assert merged["vgpr_count"] == 44
    assert merged["sgpr_count"] == 36
    assert merged["lds_bytes"] == 4096

    scanned = gpu_isa_report.scan_disassembly(
        """
v_mfma_i32_16x16x32_i8 v0, v1, v2, v3
v_smfmac_i32_32x32x32_i8 v0, v1, v2, v3
v_wmma_i32_16x16x16_iu8 v0, v1, v2, v3
v_swmmac_i32_16x16x32_iu8 v0, v1, v2, v3
global_store_dword v0, v1
s_waitcnt vmcnt(0)
"""
    )
    assert scanned["matrix_instruction_count"] == 4
    assert scanned["dense_integer_matrix_instruction_count"] == 2
    assert scanned["sparse_integer_matrix_instruction_count"] == 2
    assert scanned["mfma_dense_i8"] == 1
    assert scanned["smfmac_sparse_i8"] == 1
    assert scanned["wmma_dense_integer"] == 1
    assert scanned["swmmac_sparse_integer"] == 1
    assert scanned["matrix_instruction_histogram"]["v_swmmac_i32_16x16x32_iu8"] == 1
    assert scanned["matrix_instruction_families"] == ["mfma", "smfmac", "swmmac", "wmma"]
    assert isa_common.FORBIDDEN_MATRIX_ENGINE_RE.search("v_swmmac_i32_16x16x32_iu8")

    totals = gpu_isa_report.sum_symbol_reports(
        [
            {"counts": merged},
            {"counts": scanned},
            {
                "counts": gpu_isa_report.merge_resource_metadata(
                    disassembly_counts,
                    metadata["_ZN12_GLOBAL__N_119rns8_export_kernelEv"],
                )
            },
        ]
    )
    assert totals["global_store"] == 7
    assert totals["matrix_instruction_count"] == 4
    assert totals["dense_integer_matrix_instruction_count"] == 2
    assert totals["sparse_integer_matrix_instruction_count"] == 2
    assert totals["matrix_instruction_histogram"]["v_mfma_i32_16x16x32_i8"] == 1
    assert totals["vgpr_count"] == 44
    assert totals["sgpr_count"] == 36
    assert totals["lds_bytes"] == 4096
    assert totals["scratch_bytes"] == 16
    assert totals["occupancy"] is None

    with tempfile.TemporaryDirectory() as tmp_name:
        matrix_report = Path(tmp_name) / "amd-matrix-instruction-report.json"
        matrix_report.write_text(
            json.dumps(
                {
                    "architectures": [
                        {
                            "architecture_query": "gfx1200",
                            "dense_integer_i32_instructions": [
                                {"instruction": "v_wmma_i32_16x16x16_iu8", "architecture_query": "gfx1200"}
                            ],
                            "sparse_integer_i32_instructions": [
                                {"instruction": "v_swmmac_i32_16x16x32_iu8", "architecture_query": "gfx1200"}
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        candidates = gpu_isa_report.load_matrix_instruction_candidates(matrix_report, "gfx1200")
        assert candidates is not None
        assert candidates["candidate_instruction_count"] == 2
        linked = {"instruction_totals": {"matrix_instruction_histogram": scanned["matrix_instruction_histogram"]}}
        gpu_isa_report.correlate_matrix_instruction_report(linked, candidates)
        correlation = linked["matrix_instruction_calculator_evidence"]
        assert correlation["observed_in_calculator_candidates"]["v_wmma_i32_16x16x16_iu8"] is True
        assert correlation["observed_in_calculator_candidates"]["v_mfma_i32_16x16x32_i8"] is False

    print("gpu isa report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
