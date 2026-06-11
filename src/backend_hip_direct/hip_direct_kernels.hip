#include <hip/hip_runtime.h>

#include <cstdint>

#include "../backend_common/finite_u8_reducer.hpp"

namespace {

struct rns8_plan_tile_schedule_entry {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t flags;
  uint64_t tile_row;
  uint64_t tile_col;
  int64_t row_offset;
  int64_t col_offset;
  int64_t row_extent;
  int64_t col_extent;
  uint32_t required_prefix;
  uint32_t selected_prefix;
  uint32_t group_index;
  uint32_t range_bit_length;
};

constexpr int kRns8DefaultModulusCount = 28;

constexpr int rns8_default_moduli_host[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

#include "hip_direct_device_common.cuh"

#include "hip_direct_pack_kernels.cuh"

#include "hip_direct_gemm_kernels.cuh"

#include "hip_direct_export_kernels.cuh"

}  // namespace

#include "hip_direct_pack_wrappers.inc"
#include "hip_direct_finite_wrappers.inc"
#include "hip_direct_rns_native_wrappers.inc"
#include "hip_direct_grouped_zero_wrappers.inc"
#include "hip_direct_bounded_finite_export_wrappers.inc"
#include "hip_direct_exact_unsigned_export_wrappers.inc"
#include "hip_direct_exact_signed_export_wrappers.inc"
#include "hip_direct_finite_grouped_export_wrappers.inc"


// === Phase 1a: Persistent small-shape GEMM wrapper ===
extern "C" int rns8_hip_direct_persistent_small_gemm_rns_device(
    const int8_t* a_residues,
    const int8_t* b_residues,
    int8_t* c_residues,
    int m,
    int n,
    int k,
    int prefix) {
  const int total_cells = m * n;
  const int block_size = 256;
  const int grid_size = (total_cells + block_size - 1) / block_size;
  rns8_persistent_small_gemm_rns_kernel<<<grid_size, block_size>>>(
      a_residues, b_residues, c_residues, m, n, k, prefix);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) return static_cast<int>(err);
  err = hipDeviceSynchronize();
  return static_cast<int>(err);
}



// === Fast Garner CRT export wrappers (prefixes 1-8) ===
#define RNS8_GARNER_EXPORT_WRAPPER_I64(Prefix) \
extern "C" int rns8_hip_direct_export_i64_garner_prefix##Prefix##_device( \
    const int8_t* r, int64_t* d, int rows, int cols, int ld, int64_t bound, int* status) { \
  const int block = 256; \
  const int64_t el = (int64_t)rows * cols; \
  const int grid = (int)((el + 2 * block - 1) / (2 * block)); \
  rns8_export_bounded_i64_garner_kernel<Prefix><<<grid, block>>>(r, d, rows, cols, ld, bound, status); \
  hipError_t e = hipGetLastError(); if (e != hipSuccess) return (int)e; \
  e = hipDeviceSynchronize(); return (int)e; \
}
RNS8_GARNER_EXPORT_WRAPPER_I64(1) RNS8_GARNER_EXPORT_WRAPPER_I64(2) RNS8_GARNER_EXPORT_WRAPPER_I64(3)
RNS8_GARNER_EXPORT_WRAPPER_I64(4) RNS8_GARNER_EXPORT_WRAPPER_I64(5) RNS8_GARNER_EXPORT_WRAPPER_I64(6)
RNS8_GARNER_EXPORT_WRAPPER_I64(7) RNS8_GARNER_EXPORT_WRAPPER_I64(8)
#undef RNS8_GARNER_EXPORT_WRAPPER_I64

#define RNS8_GARNER_EXPORT_WRAPPER_U64(Prefix) \
extern "C" int rns8_hip_direct_export_u64_garner_prefix##Prefix##_device( \
    const int8_t* r, uint64_t* d, int rows, int cols, int ld, uint64_t bound, int* status) { \
  const int block = 256; \
  const int64_t el = (int64_t)rows * cols; \
  const int grid = (int)((el + 2 * block - 1) / (2 * block)); \
  rns8_export_bounded_u64_garner_kernel<Prefix><<<grid, block>>>(r, d, rows, cols, ld, bound, status); \
  hipError_t e = hipGetLastError(); if (e != hipSuccess) return (int)e; \
  e = hipDeviceSynchronize(); return (int)e; \
}
RNS8_GARNER_EXPORT_WRAPPER_U64(1) RNS8_GARNER_EXPORT_WRAPPER_U64(2) RNS8_GARNER_EXPORT_WRAPPER_U64(3)
RNS8_GARNER_EXPORT_WRAPPER_U64(4) RNS8_GARNER_EXPORT_WRAPPER_U64(5) RNS8_GARNER_EXPORT_WRAPPER_U64(6)
RNS8_GARNER_EXPORT_WRAPPER_U64(7) RNS8_GARNER_EXPORT_WRAPPER_U64(8)
#undef RNS8_GARNER_EXPORT_WRAPPER_U64

extern "C" int rns8_hip_direct_export_i64_garner_select_device(
    const int8_t* r, int64_t* d, int rows, int cols, int ld, int64_t bound, int* status, int prefix) {
  switch (prefix) {
    case 1: return rns8_hip_direct_export_i64_garner_prefix1_device(r,d,rows,cols,ld,bound,status);
    case 2: return rns8_hip_direct_export_i64_garner_prefix2_device(r,d,rows,cols,ld,bound,status);
    case 3: return rns8_hip_direct_export_i64_garner_prefix3_device(r,d,rows,cols,ld,bound,status);
    case 4: return rns8_hip_direct_export_i64_garner_prefix4_device(r,d,rows,cols,ld,bound,status);
    case 5: return rns8_hip_direct_export_i64_garner_prefix5_device(r,d,rows,cols,ld,bound,status);
    case 6: return rns8_hip_direct_export_i64_garner_prefix6_device(r,d,rows,cols,ld,bound,status);
    case 7: return rns8_hip_direct_export_i64_garner_prefix7_device(r,d,rows,cols,ld,bound,status);
    case 8: return rns8_hip_direct_export_i64_garner_prefix8_device(r,d,rows,cols,ld,bound,status);
    default: return (int)hipErrorInvalidValue;
  }
}
extern "C" int rns8_hip_direct_export_u64_garner_select_device(
    const int8_t* r, uint64_t* d, int rows, int cols, int ld, uint64_t bound, int* status, int prefix) {
  switch (prefix) {
    case 1: return rns8_hip_direct_export_u64_garner_prefix1_device(r,d,rows,cols,ld,bound,status);
    case 2: return rns8_hip_direct_export_u64_garner_prefix2_device(r,d,rows,cols,ld,bound,status);
    case 3: return rns8_hip_direct_export_u64_garner_prefix3_device(r,d,rows,cols,ld,bound,status);
    case 4: return rns8_hip_direct_export_u64_garner_prefix4_device(r,d,rows,cols,ld,bound,status);
    case 5: return rns8_hip_direct_export_u64_garner_prefix5_device(r,d,rows,cols,ld,bound,status);
    case 6: return rns8_hip_direct_export_u64_garner_prefix6_device(r,d,rows,cols,ld,bound,status);
    case 7: return rns8_hip_direct_export_u64_garner_prefix7_device(r,d,rows,cols,ld,bound,status);
    case 8: return rns8_hip_direct_export_u64_garner_prefix8_device(r,d,rows,cols,ld,bound,status);
    default: return (int)hipErrorInvalidValue;
  }
}

// === VOPD Garner export selectors ===
extern "C" int rns8_hip_direct_export_i64_vopd_garner_device(
    const int8_t* residues, int64_t* dst, int rows, int cols, int ld,
    int64_t bound, int* status, int prefix) {
  const int block = 256;
  const int cp2 = 2;
  const int64_t el = (int64_t)rows * cols;
  const int grid = (int)((el + cp2 * block - 1) / (cp2 * block));
  switch (prefix) {
    case 1: rns8_export_bounded_i64_vopd_combined_kernel<1><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 2: rns8_export_bounded_i64_vopd_combined_kernel<2><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 3: rns8_export_bounded_i64_vopd_combined_kernel<3><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 4: rns8_export_bounded_i64_vopd_combined_kernel<4><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 5: rns8_export_bounded_i64_vopd_combined_kernel<5><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 6: rns8_export_bounded_i64_vopd_combined_kernel<6><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 7: rns8_export_bounded_i64_vopd_combined_kernel<7><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 8: rns8_export_bounded_i64_vopd_combined_kernel<8><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    default: return (int)hipErrorInvalidValue;
  }
  hipError_t e = hipGetLastError(); if (e != hipSuccess) return (int)e;
  e = hipDeviceSynchronize(); return (int)e;
}


// === Fused GEMM+export wrapper (Item C) ===
extern "C" int rns8_hip_direct_fused_gemm_export_i64_device(
    const int8_t* a_residues, const int8_t* b_residues, int64_t* dst,
    int m, int n, int k, int lda, int ldb, int ldc, int prefix, int64_t bound, int* status) {
  const int block = 256;
  const int cells_p2 = 2;
  const int64_t el = (int64_t)m * n;
  const int grid = (int)((el + cells_p2 * block - 1) / (cells_p2 * block));
  rns8_fused_gemm_export_i64_kernel<<<grid, block>>>(a_residues, b_residues, dst, m, n, k, lda, ldb, ldc, prefix, bound, status);
  hipError_t e = hipGetLastError(); if (e != hipSuccess) return (int)e;
  e = hipDeviceSynchronize(); return (int)e;
}

// Item E: Persistent small native GEMM wrapper
extern "C" int rns8_hip_direct_persistent_small_native_gemm_device(
    const int64_t* a_native, const int64_t* b_native, int64_t* c_native,
    int m, int n, int k, int lda, int ldb, int ldc, uint64_t bound) {
  const int block = 256;
  const int grid = (m * n + block - 1) / block;
  rns8_persistent_small_native_gemm_kernel<<<grid, block>>>(a_native, b_native, c_native, m, n, k, lda, ldb, ldc, bound);
  hipError_t e = hipGetLastError(); if (e != hipSuccess) return (int)e;
  e = hipDeviceSynchronize(); return (int)e;
}

// === Combined final-output Garner export selector ===
extern "C" int rns8_hip_direct_export_i64_combined_final_device(
    const int8_t* residues, int64_t* dst, int rows, int cols, int ld,
    int64_t bound, int prefix) {
  const int block = 256;
  const int cp2 = 2;
  const int64_t el = (int64_t)rows * cols;
  const int grid = (int)((el + cp2 * block - 1) / (cp2 * block));
  switch (prefix) {
    case 1: rns8_export_bounded_i64_combined_final_output_kernel<1><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, false); break;
    case 2: rns8_export_bounded_i64_combined_final_output_kernel<2><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, false); break;
    case 3: rns8_export_bounded_i64_combined_final_output_kernel<3><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, false); break;
    case 4: rns8_export_bounded_i64_combined_final_output_kernel<4><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, false); break;
    case 5: rns8_export_bounded_i64_combined_final_output_kernel<5><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, false); break;
    case 6: rns8_export_bounded_i64_combined_final_output_kernel<6><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, false); break;
    case 7: rns8_export_bounded_i64_combined_final_output_kernel<7><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, false); break;
    case 8: rns8_export_bounded_i64_combined_final_output_kernel<8><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, false); break;
    default: return (int)hipErrorInvalidValue;
  }
  hipError_t e = hipGetLastError(); if (e != hipSuccess) return (int)e;
  e = hipDeviceSynchronize(); return (int)e;
}



// === u64 VOPD + Combined final-output export selectors ===
extern "C" int rns8_hip_direct_export_u64_vopd_garner_device(
    const int8_t* residues, uint64_t* dst, int rows, int cols, int ld,
    uint64_t bound, int* status, int prefix) {
  const int block = 256; const int cp2 = 2;
  const int64_t el = (int64_t)rows * cols;
  const int grid = (int)((el + cp2 * block - 1) / (cp2 * block));
  switch (prefix) {
    case 1: rns8_export_bounded_u64_vopd_combined_kernel<1><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 2: rns8_export_bounded_u64_vopd_combined_kernel<2><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 3: rns8_export_bounded_u64_vopd_combined_kernel<3><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 4: rns8_export_bounded_u64_vopd_combined_kernel<4><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 5: rns8_export_bounded_u64_vopd_combined_kernel<5><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 6: rns8_export_bounded_u64_vopd_combined_kernel<6><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 7: rns8_export_bounded_u64_vopd_combined_kernel<7><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    case 8: rns8_export_bounded_u64_vopd_combined_kernel<8><<<grid, block>>>(residues, dst, status, rows, cols, ld, bound, false); break;
    default: return (int)hipErrorInvalidValue;
  }
  hipError_t e = hipGetLastError(); if (e != hipSuccess) return (int)e;
  e = hipDeviceSynchronize(); return (int)e;
}

extern "C" int rns8_hip_direct_export_u64_combined_final_device(
    const int8_t* residues, uint64_t* dst, int rows, int cols, int ld,
    uint64_t bound, int prefix) {
  const int block = 256; const int cp2 = 2;
  const int64_t el = (int64_t)rows * cols;
  const int grid = (int)((el + cp2 * block - 1) / (cp2 * block));
  switch (prefix) {
    case 1: rns8_export_bounded_u64_combined_final_output_kernel<1><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, true); break;
    case 2: rns8_export_bounded_u64_combined_final_output_kernel<2><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, true); break;
    case 3: rns8_export_bounded_u64_combined_final_output_kernel<3><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, true); break;
    case 4: rns8_export_bounded_u64_combined_final_output_kernel<4><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, true); break;
    case 5: rns8_export_bounded_u64_combined_final_output_kernel<5><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, true); break;
    case 6: rns8_export_bounded_u64_combined_final_output_kernel<6><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, true); break;
    case 7: rns8_export_bounded_u64_combined_final_output_kernel<7><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, true); break;
    case 8: rns8_export_bounded_u64_combined_final_output_kernel<8><<<grid, block>>>(residues, dst, nullptr, rows, cols, ld, bound, true); break;
    default: return (int)hipErrorInvalidValue;
  }
  hipError_t e = hipGetLastError(); if (e != hipSuccess) return (int)e;
  e = hipDeviceSynchronize(); return (int)e;
}


// === Native-to-RNS VOPD 8-wide pack wrapper (Item F) ===
// Launches one kernel per plane for VOPD-friendly 8-wide processing.
extern "C" int rns8_hip_direct_pack_native_i64_to_rns_vopd_device(
    const int64_t* src, int8_t* dst, int rows, int cols, int ld, int prefix,
    const int* moduli) {
  const int64_t elements = (int64_t)rows * cols;
  const int block = 256;
  const int64_t cells_per_block = (int64_t)block * 8;  // 8 cells per thread
  const int grid = (int)((elements + cells_per_block - 1) / cells_per_block);
  int code = 0;
  for (int plane = 0; plane < prefix && code == 0; ++plane) {
    rns8_pack_native_i64_to_rns_8wide_vopd_kernel<<<grid, block>>>(
        src, dst, rows, cols, ld, prefix, plane, moduli[plane]);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) code = (int)e;
  }
  if (code != 0) return code;
  hipError_t e = hipDeviceSynchronize();
  return e == hipSuccess ? 0 : (int)e;
}

// === INT4 research pack wrapper (Item G) ===
// Packs pairs of int8 values into 4-bit each, halving storage.
// Schema-gated behind RNS8_ENABLE_INT4_RESEARCH.
extern "C" int rns8_hip_direct_pack_i4_research_device(
    const int8_t* src, int8_t* dst, int rows, int cols, int ld, int prefix) {
  const int block = 256;
  const int64_t elements = (int64_t)rows * cols;
  const int64_t pairs = (elements + 1) / 2;
  const int64_t total = pairs * (int64_t)prefix;
  const int grid = (int)((total + block - 1) / block);
  rns8_pack_i4_research_kernel<<<grid, block>>>(src, dst, rows, cols, ld, prefix);
  hipError_t e = hipGetLastError();
  if (e != hipSuccess) return (int)e;
  e = hipDeviceSynchronize();
  return (int)e;
}
