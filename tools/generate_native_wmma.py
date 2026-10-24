"""Bounded standalone RDNA3 native WMMA family; never default-dispatch qualified."""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

from arithmetic_certificate import centered_int8, verify
from arithmetic_composition import Node, verify_composition
from analytical_planner import Stage
from planning_recipe import ResourcePlan, checked_resources
from modulus_set_search import DEFAULT_MODULI


@dataclass(frozen=True)
class Variant:
    target: str = "gfx1100"
    unsigned_inputs: bool = False
    group: int = 1
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    waves: int = 1
    pipeline_depth: int = 1
    vector_width: int = 1
    safe_k: int = 65536
    split_k: int = 1
    layout: str = "native_row_major_to_planar_centered_i8"
    output: str = "default_basis_prefix_residues"
    tails: str = "masked"
    lds_budget: int = 0


DEPENDENCIES = ("src/backend_hip_direct/hip_direct_device_common.cuh",
                "src/backend_hip_direct/hip_direct_gemm_native_kernels.cuh",
                "src/backend_hip_direct/hip_direct_native_wmma_kernels.cuh",
                "src/backend_common/finite_u8_reducer.hpp",
                "src/backend_common/garner_constants.hpp",
                "src/backend_common/native_wmma_queue.hpp",
                "src/backend_common/fragment_layout.hpp",
                "src/backend_common/address_domain.hpp",
                "tools/arithmetic_certificate.py", "tools/arithmetic_composition.py",
                "tools/planning_recipe.py", "tools/analytical_planner.py", "tools/modulus_set_search.py")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def composition(v, modulus):
    """Prove one slab with a centered carry; induction covers further slabs.

    Executable map source is bound separately; algebra does not prove device lane behavior.
    Replicated WMMA fragments are not injective layout transforms in this proof model;
    only the reduced values enter the slab proof, leaving physical layout unknown.
    The arithmetic certificate separately proves exact native modular encoding.
    """
    low, high = (0, (1 << 64) - 1) if v.unsigned_inputs else (-(1 << 63), (1 << 63) - 1)
    nodes = (
        Node("native_a", "input", bits=64, signed=not v.unsigned_inputs, interval=(low, high)),
        Node("native_b", "input", bits=64, signed=not v.unsigned_inputs, interval=(low, high)),
        Node("residue_a", "centered_reduce", ("native_a",), bits=8, modulus=modulus),
        Node("residue_b", "centered_reduce", ("native_b",), bits=8, modulus=modulus),
        Node("carry", "input", bits=8, interval=(-(modulus // 2), (modulus - 1) // 2), modulus=modulus),
        Node("slab", "dot", ("residue_a", "residue_b", "carry"), count=v.safe_k),
        Node("output", "centered_reduce", ("slab",), bits=8, modulus=modulus),
    )
    result = verify_composition(nodes)
    if not result["algebraically_verified"]:
        raise ValueError("native encoding/slab/carry/store composition failed")
    return result


def generate(directory, variants, root=None):
    root = root or Path(__file__).resolve().parents[1]
    variants = tuple(variants)
    if not variants or len(variants) > 6 or len(set(variants)) != len(variants):
        raise ValueError("expected 1..6 distinct bounded variants")
    targets = {v.target for v in variants}
    if len(targets) != 1 or not targets <= {"gfx1100", "gfx1101", "gfx1102"}:
        raise ValueError("one explicit RDNA3 target is required")
    for v in variants:
        baseline = Variant(target=v.target, unsigned_inputs=v.unsigned_inputs, group=v.group)
        if (type(v.unsigned_inputs) is not bool or type(v.group) is not int or v.group not in (1, 2, 4)
                or any(type(getattr(v, field)) is not int for field in
                       ("tile_m", "tile_n", "tile_k", "waves", "pipeline_depth", "vector_width", "safe_k", "split_k", "lds_budget"))
                or v != baseline):
            raise ValueError("unsupported tile/format/schedule/epilogue/resource combination")
    dependencies = {p: digest((root / p).read_bytes()) for p in DEPENDENCIES}
    generator = digest(Path(__file__).read_bytes())
    certificates = [verify(replace(centered_int8(m), prior_min=-255, prior_max=255)) for m in DEFAULT_MODULI[:20]]
    if not all(c["algebraically_verified"] for c in certificates):
        raise ValueError("unsafe native WMMA arithmetic")
    proof_hashes = [c["sha256"] for c in certificates]
    entries = []
    for v in variants:
        compositions = [composition(v, m) for m in DEFAULT_MODULI[:20]]
        # The concrete fused kernel owns no global temporary arena. Register
        # occupancy/spills remain separate compiled resource evidence.
        resource = checked_resources(ResourcePlan("success"),
            (Stage("fused", "generated_native", (), ("matrix",), (), 0),), 0)
        if resource["status"] != "success":
            raise ValueError("generated resource plan failed")
        identity = {"spec": asdict(v), "dependencies": dependencies, "generator_sha256": generator,
                    "arithmetic_certificates": proof_hashes, "ordered_basis": DEFAULT_MODULI[:20],
                    "fragment_layout": {"version": 1, "instruction": "rdna3_iu8_16x16x16_wave32_v1",
                        "definition_sha256": dependencies["src/backend_common/fragment_layout.hpp"],
                        "logical_axes": ["row16", "column16"], "input_slot_axes": ["lane32", "word4", "byte4"],
                        "output_slot_axes": ["lane32", "register8"], "input_replicas": 2,
                        "output_replicas": 1, "edge_mask": "logical_rows_columns", "hardware_qualified": False},
                    "composition_certificates": [c["sha256"] for c in compositions],
                    "resource_plan": {"status": resource["status"], "global_temporary_bytes": resource["required_bytes"]},
                    "compatibility": {"form": "centered_int8", "group": v.group,
                                      "slab": v.safe_k, "layout": v.layout, "output": v.output,
                                      "reconstruction": "none", "cross_feature_runtime_qualified": False}}
        variant_id = digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
        entries.append({**identity, "id": variant_id, "kernel_symbol": "rns8_native_" + variant_id,
                        "launch_symbol": "rns8_launch_" + variant_id})
    target = next(iter(targets))
    source = ["// Generated bounded research family. No scalar target fallback.",
              "#include <hip/hip_runtime.h>", "#include <cstdint>",
              '#include "backend_common/finite_u8_reducer.hpp"',
              '#include "backend_common/garner_constants.hpp"',
              '#include "backend_common/native_wmma_queue.hpp"',
              '#include "backend_common/fragment_layout.hpp"',
              '#include "backend_common/address_domain.hpp"',
              f"#if defined(__HIP_DEVICE_COMPILE__) && !defined(__{target}__)",
              '#error "generated target does not match compiler offload target"', "#endif",
              "#if defined(__HIP_DEVICE_COMPILE__) && !__has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)",
              '#error "generated matrix instruction is unavailable"', "#endif",
              "namespace {", "constexpr int kRns8DefaultModulusCount = 28;",
              "constexpr int rns8_default_moduli_host[28] = {" + ",".join(map(str, DEFAULT_MODULI)) + "};",
              '#include "backend_hip_direct/hip_direct_device_common.cuh"',
              '#include "backend_hip_direct/hip_direct_gemm_native_kernels.cuh"',
              '#include "backend_hip_direct/hip_direct_native_wmma_kernels.cuh"', "}"]
    header = ["#pragma once", "#include <cstdint>"]
    signature = "const void* a, const void* b, int8_t* c, int m, int n, int k, int lda, int ldb, int ldc, int prefix"
    for v, entry in zip(variants, entries):
        kernel, launch = entry["kernel_symbol"], entry["launch_symbol"]
        native = "uint64_t" if v.unsigned_inputs else "int64_t"
        source += [f'extern "C" __global__ __launch_bounds__(32) void {kernel}({signature}) {{',
                   f"  rns8_native_grouped_wmma_tile<{native}, {v.group}>(static_cast<const {native}*>(a),",
                   f"      static_cast<const {native}*>(b), c, m,n,k,lda,ldb,ldc,prefix,",
                   f"      static_cast<int64_t>(blockIdx.y)*16, static_cast<int64_t>(blockIdx.x)*16, blockIdx.z*{v.group});", "}",
                   f'extern "C" int {launch}({signature}) {{',
                   "  using namespace rns8::detail;",
                   "  if (!a || !b || !c || prefix < 1 || prefix > 20 ||",
                   "      !qualified_i32_matrix_address(m,k,lda,1,8) || !qualified_i32_matrix_address(k,n,ldb,1,8) ||",
                   "      !qualified_i32_matrix_address(m,n,ldc,prefix)) return static_cast<int>(hipErrorInvalidValue);",
                   "  const uint64_t cb = static_cast<uint64_t>(m)*ldc*prefix;",
                   "  if (!nonoverlapping_byte_ranges(a,static_cast<uint64_t>(m)*lda*8,c,cb) ||",
                   "      !nonoverlapping_byte_ranges(b,static_cast<uint64_t>(k)*ldb*8,c,cb)) return static_cast<int>(hipErrorInvalidValue);",
                   f"  hipLaunchKernelGGL({kernel}, dim3((n+15)/16,(m+15)/16,(prefix+{v.group}-1)/{v.group}),",
                   "      dim3(32),0,nullptr,a,b,c,m,n,k,lda,ldb,ldc,prefix);",
                   "  return static_cast<int>(hipGetLastError());", "}"]
        header.append(f'extern "C" int {launch}({signature});')
    header += ["namespace rns8::generated {",
               "struct NativeVariant { const char* id; uint32_t group; bool unsigned_inputs;",
               "  int (*launch)(const void*,const void*,int8_t*,int,int,int,int,int,int,int); };",
               "inline constexpr NativeVariant native_variants[] = {"]
    for v, entry in zip(variants, entries):
        header.append(f'  {{"{entry["id"]}",{v.group},{str(v.unsigned_inputs).lower()},&{entry["launch_symbol"]}}},')
    header += ["};", "}"]
    source_text, header_text = "\n".join(source) + "\n", "\n".join(header) + "\n"
    manifest = {"schema": "rns8_generated_native_wmma_v1", "variants": entries,
                "source_sha256": digest(source_text.encode()), "header_sha256": digest(header_text.encode()),
                "compiled": False, "qualified": False, "promotion_eligible": False,
                "toolchain_binding": "requires_actual_compiled_build_manifest_and_artifact",
                "resource_evidence": "launch_bounds_32_and_no_explicit_LDS_only_registers_unmeasured",
                "required_tests": ["signed_unsigned_full_width", "tails_and_padded_strides", "prefix_group_tails",
                                   "k_65535_65536_65537", "cpu_Boost_differential", "paired_pack_then_gemm",
                                   "paired_reverse_k", "launched_symbol_ISA_resources"]}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "native_variants.hip").write_bytes(source_text.encode())
    (directory / "native_variants.hpp").write_bytes(header_text.encode())
    (directory / "native_variants.json").write_bytes((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    generate(args.out_dir, [Variant(args.target, unsigned, group) for unsigned in (False, True) for group in (1, 2, 4)])
