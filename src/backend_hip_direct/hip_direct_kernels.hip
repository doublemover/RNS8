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

