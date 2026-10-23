#include <hip/hip_runtime.h>

#include <cstdint>
#include "../backend_common/garner_constants.hpp"

#include "../backend_common/finite_u8_reducer.hpp"
#include "../backend_common/residue_dot.hpp"
#include "../backend_common/residue_epilogue.hpp"

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
#include "hip_direct_lift_kernels.cuh"
#include "hip_direct_split_k_kernels.cuh"
#include "hip_direct_epilogue_kernels.cuh"

}  // namespace

#include "hip_direct_pack_wrappers.inc"
#include "hip_direct_split_k_wrappers.inc"
#include "hip_direct_finite_wrappers.inc"
#include "hip_direct_rns_native_wrappers.inc"
#include "hip_direct_grouped_zero_wrappers.inc"
#include "hip_direct_bounded_finite_export_wrappers.inc"
#include "hip_direct_exact_unsigned_export_wrappers.inc"
#include "hip_direct_exact_signed_export_wrappers.inc"


// Persistent small-shape GEMM wrapper
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



// === Native-to-RNS VOPD 8-wide pack wrapper (Item F) ===
// Launches one kernel per plane for VOPD-friendly 8-wide processing.
extern "C" int rns8_hip_direct_pack_native_i64_to_rns_vopd_device(
    const int64_t* src, int8_t* dst, int rows, int cols, int ld, int prefix,
    const uint16_t* moduli) {
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

extern "C" int rns8_hip_direct_lift_suffix_device(
    const int8_t* residues, int8_t* suffix, int cells, int source_prefix, int target_prefix,
    bool signed_value, uint64_t bound0, uint64_t bound1, uint64_t bound2, int* status) {
  if (!residues || !suffix || !status || cells <= 0 || cells > kRns8MaxInt - 256 ||
      source_prefix <= 0 || target_prefix > kRns8MaxSupportedPrefix || source_prefix >= target_prefix)
    return static_cast<int>(hipErrorInvalidValue);
  hipLaunchKernelGGL(rns8_exact_lift_suffix_kernel, dim3((cells + 255) / 256), dim3(256), 0, nullptr,
                    residues, suffix, cells, source_prefix, target_prefix, signed_value,
                    (rns8_u192_device{bound0, bound1, bound2}), status);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_gemm_rns_axpby(
    const int8_t* a, const int8_t* b, const int8_t* d, int8_t* c,
    int m, int n, int k, int lda, int ldb, int ldd, int ldc, int prefix, int64_t alpha, int64_t beta) {
  const auto blocks = static_cast<unsigned>((static_cast<int64_t>(m) * n + 255) / 256);
  hipLaunchKernelGGL(rns8_gemm_rns_axpby_kernel, dim3(blocks, prefix), dim3(256), 0, nullptr,
                    a, b, d, c, m, n, k, lda, ldb, ldd, ldc, alpha, beta);
  return static_cast<int>(hipGetLastError());
}
