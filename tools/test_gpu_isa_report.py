#!/usr/bin/env python3
"""Self-test AMDGPU metadata extraction in gpu_isa_report."""

from __future__ import annotations

import gpu_isa_report


def main() -> int:
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

    totals = gpu_isa_report.sum_symbol_reports(
        [
            {"counts": merged},
            {
                "counts": gpu_isa_report.merge_resource_metadata(
                    disassembly_counts,
                    metadata["_ZN12_GLOBAL__N_119rns8_export_kernelEv"],
                )
            },
        ]
    )
    assert totals["global_store"] == 6
    assert totals["vgpr_count"] == 44
    assert totals["sgpr_count"] == 36
    assert totals["lds_bytes"] == 4096
    assert totals["scratch_bytes"] == 16
    assert totals["occupancy"] is None

    print("gpu isa report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
