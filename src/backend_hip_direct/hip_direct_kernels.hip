#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>
#include "../backend_common/garner_constants.hpp"

#include "../backend_common/finite_u8_reducer.hpp"
#include "../backend_common/residue_dot.hpp"
#include "../backend_common/residue_epilogue.hpp"
#include "../backend_common/residue_layout.hpp"
#include "../backend_common/direct_crt_constants.hpp"
#include "../backend_common/balanced_crt.hpp"
#include "../backend_common/strassen_program.hpp"
#include "../backend_common/rank23_program.hpp"
#include "../backend_common/int4_encoding.hpp"
#include "../backend_common/native_tile_bound.hpp"
#include "../backend_common/native_wmma_queue.hpp"
#include "../backend_common/ragged_residues.hpp"
#include "../backend_common/mixed_radix_lift.hpp"

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
#include "hip_direct_narrow_kernels.cuh"
#include "hip_direct_native_byte_kernels.cuh"
#include "hip_direct_layout_kernels.cuh"
#include "hip_direct_direct_crt_kernels.cuh"
#include "hip_direct_balanced_crt_kernels.cuh"
#include "hip_direct_strassen_kernels.cuh"
#include "hip_direct_integrity_kernels.cuh"
#include "hip_direct_int4_kernels.cuh"
#include "hip_direct_native_wmma_kernels.cuh"
#include "hip_direct_adaptive_native_kernels.cuh"
#include "hip_direct_bound_kernels.cuh"
#include "hip_direct_ragged_kernels.cuh"
#include "hip_direct_block_sparse_kernels.cuh"

}  // namespace

extern "C" int rns8_hip_native_byte_domain(const void* source, int rows, int cols,
    int ld, uint64_t* maximum, uint64_t* invalid, bool unsigned_inputs) {
  const dim3 grid((static_cast<uint64_t>(rows) * cols + 255) / 256);
  if (unsigned_inputs) {
    hipLaunchKernelGGL((rns8_native_byte_domain_kernel<uint64_t>), grid, dim3(256), 0, nullptr,
        static_cast<const uint64_t*>(source), rows, cols, ld,
        reinterpret_cast<unsigned long long*>(maximum), reinterpret_cast<unsigned long long*>(invalid));
  } else {
    hipLaunchKernelGGL((rns8_native_byte_domain_kernel<int64_t>), grid, dim3(256), 0, nullptr,
        static_cast<const int64_t*>(source), rows, cols, ld,
        reinterpret_cast<unsigned long long*>(maximum), reinterpret_cast<unsigned long long*>(invalid));
  }
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_native_byte_wmma(const void* a, const void* b, void* c,
    int m, int n, int k, int lda, int ldb, int ldc, bool unsigned_inputs) {
  const dim3 grid((static_cast<uint64_t>(n) + 15) / 16, (static_cast<uint64_t>(m) + 15) / 16);
  if (unsigned_inputs) {
    hipLaunchKernelGGL((rns8_native_byte_wmma_kernel<uint64_t>), grid, dim3(32), 0, nullptr,
        static_cast<const uint64_t*>(a), static_cast<const uint64_t*>(b), static_cast<uint64_t*>(c), m, n, k, lda, ldb, ldc);
  } else {
    hipLaunchKernelGGL((rns8_native_byte_wmma_kernel<int64_t>), grid, dim3(32), 0, nullptr,
        static_cast<const int64_t*>(a), static_cast<const int64_t*>(b), static_cast<int64_t*>(c), m, n, k, lda, ldb, ldc);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_integrity_check(const void* a,const void* b,const int8_t* candidate,
    int m,int n,int k,int lda,int ldb,int prefix,bool unsigned_inputs,const uint64_t* bound,uint32_t* failures) {
  const dim3 grid((static_cast<uint64_t>(m)*n+255)/256);
  const rns8_u192_device wide_bound{bound[0],bound[1],bound[2]};
  if(unsigned_inputs) {
    hipLaunchKernelGGL((rns8_integrity_check_kernel<uint64_t>),grid,dim3(256),0,nullptr,
        static_cast<const uint64_t*>(a),static_cast<const uint64_t*>(b),candidate,m,n,k,lda,ldb,prefix,false,wide_bound,failures);
  } else {
    hipLaunchKernelGGL((rns8_integrity_check_kernel<int64_t>),grid,dim3(256),0,nullptr,
        static_cast<const int64_t*>(a),static_cast<const int64_t*>(b),candidate,m,n,k,lda,ldb,prefix,true,wide_bound,failures);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_split_k_canonicalize(uint8_t* values, int64_t cells, int prefix, int fragments) {
  hipLaunchKernelGGL(rns8_split_k_canonicalize_kernel, dim3((cells + 255) / 256, prefix, fragments),
                    dim3(256), 0, nullptr, values, cells, prefix, fragments);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_strassen_transform(const int8_t* source,int8_t* output,
    int rows,int cols,int ld,int hr,int hc,int prefix,int product,bool lhs,bool rank23) {
  const dim3 grid((static_cast<uint64_t>(hr)*hc+255)/256,prefix);
  if (rank23) {
    hipLaunchKernelGGL((rns8_bilinear_transform_kernel<3>),grid,dim3(256),0,nullptr,
        source,output,rows,cols,ld,hr,hc,product,lhs);
  } else {
    hipLaunchKernelGGL((rns8_bilinear_transform_kernel<2>),grid,dim3(256),0,nullptr,
        source,output,rows,cols,ld,hr,hc,product,lhs);
  }
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_strassen_combine(const int8_t* product,int8_t* stage,int m,int n,int hm,int hn,int prefix,int index,bool rank23) {
  const dim3 grid((static_cast<uint64_t>(m)*n+255)/256,prefix);
  if (rank23) {
    hipLaunchKernelGGL((rns8_bilinear_combine_kernel<3>),grid,dim3(256),0,nullptr,
        product,stage,m,n,hm,hn,index);
  } else {
    hipLaunchKernelGGL((rns8_bilinear_combine_kernel<2>),grid,dim3(256),0,nullptr,
        product,stage,m,n,hm,hn,index);
  }
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_strassen_commit(const int8_t* stage,int8_t* output,int m,int n,int ldc,int prefix) {
  hipLaunchKernelGGL(rns8_strassen_commit_kernel,dim3((static_cast<uint64_t>(m)*n+255)/256,prefix),dim3(256),0,nullptr,
      stage,output,m,n,ldc);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_export_canonical_balanced_crt(
    const int8_t* residues,uint64_t* limbs,int cells,const rns8::detail::balanced_crt::Plan* plan) {
  hipLaunchKernelGGL(rns8_export_canonical_balanced_crt_kernel,dim3((cells+255)/256),dim3(256),0,nullptr,
      residues,limbs,cells,*plan);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_block_sparse_gemm(
    const int8_t* a,const uint32_t* ao,const uint32_t* ai,
    const int8_t* b,const uint32_t* bo,const uint32_t* bi,
    int8_t* c,int m,int n,int k,int ldc,int prefix) {
  const uint64_t tile_cols=(static_cast<uint64_t>(n)+15)/16;
  const uint64_t tiles=((static_cast<uint64_t>(m)+15)/16)*tile_cols;
  hipLaunchKernelGGL(rns8_block_sparse_gemm_kernel,dim3(tiles,prefix),dim3(256),0,nullptr,
      a,ao,ai,b,bo,bi,c,m,n,k,ldc,prefix,static_cast<int>(tile_cols));
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_materialize_ragged(
    const int8_t* source, const rns8::detail::RaggedResidueTile* tiles,
    int tile_rows, int tile_cols, int grid_cols, bool signed_value,
    int8_t* destination, rns8::detail::ResidueLayout target, int* status) {
  hipLaunchKernelGGL(rns8_materialize_ragged_kernel, dim3((target.rows * target.cols + 255) / 256),
      dim3(256), 0, nullptr, source, tiles, tile_rows, tile_cols, grid_cols, signed_value, destination, target, status);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_int4_capability(int* supported) {
  hipLaunchKernelGGL(rns8_int4_capability_kernel, dim3(1), dim3(1), 0, nullptr, supported);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_finite_int4_gemm(
    const int8_t* a, const int8_t* b, int8_t* c, int m, int n, int k,
    int lda, int ldb, int ldc, int modulus) {
  hipLaunchKernelGGL(rns8_finite_int4_wmma_kernel,
      dim3((static_cast<uint64_t>(n) + 15) / 16, (static_cast<uint64_t>(m) + 15) / 16),
      dim3(32), 0, nullptr, a, b, c, m, n, k, lda, ldb, ldc, modulus);
  return static_cast<int>(hipGetLastError());
}

#include "hip_direct_pack_wrappers.inc"
#include "hip_direct_native_wmma_wrappers.inc"
#include "hip_direct_adaptive_native_wrappers.inc"
#include "hip_direct_bound_wrappers.inc"
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
    bool signed_value, uint64_t bound0, uint64_t bound1, uint64_t bound2, int* status, bool mixed_radix) {
  if (!residues || !suffix || !status || cells <= 0 || cells > kRns8MaxInt - 256 ||
      source_prefix <= 0 || target_prefix > kRns8MaxSupportedPrefix || source_prefix >= target_prefix)
    return static_cast<int>(hipErrorInvalidValue);
  if (mixed_radix) {
    hipLaunchKernelGGL((rns8_exact_lift_suffix_kernel<true>), dim3((cells + 255) / 256), dim3(256), 0, nullptr,
                      residues, suffix, cells, source_prefix, target_prefix, signed_value,
                      (rns8_u192_device{bound0, bound1, bound2}), status);
  } else {
    hipLaunchKernelGGL((rns8_exact_lift_suffix_kernel<false>), dim3((cells + 255) / 256), dim3(256), 0, nullptr,
                      residues, suffix, cells, source_prefix, target_prefix, signed_value,
                      (rns8_u192_device{bound0, bound1, bound2}), status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_axpby_residues(
    const int8_t* x, const int8_t* d, int8_t* c,
    int m, int n, int ldx, int ldd, int ldc, int prefix, int64_t alpha, int64_t beta) {
  hipLaunchKernelGGL(rns8_axpby_residues_kernel, dim3((int64_t{m} * n + 255) / 256, prefix), dim3(256), 0, nullptr,
                    x, d, c, m, n, ldx, ldd, ldc, alpha, beta);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_gemm_rns_axpby(
    const int8_t* a, const int8_t* b, const int8_t* d, int8_t* c,
    int m, int n, int k, int lda, int ldb, int ldd, int ldc, int prefix, int64_t alpha, int64_t beta, void* stream) {
  const auto blocks = static_cast<unsigned>((static_cast<int64_t>(m) * n + 255) / 256);
  hipLaunchKernelGGL(rns8_gemm_rns_axpby_kernel, dim3(blocks, prefix), dim3(256), 0, static_cast<hipStream_t>(stream),
                    a, b, d, c, m, n, k, lda, ldb, ldd, ldc, alpha, beta);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_native_i64_abs_max(const int64_t* source, int rows, int cols, int ld, void* maximum) {
  const auto blocks = static_cast<unsigned>((static_cast<int64_t>(rows) * cols + 255) / 256);
  hipLaunchKernelGGL(rns8_native_i64_abs_max_kernel, dim3(blocks), dim3(256), 0, nullptr,
                    source, rows, cols, ld, static_cast<unsigned long long*>(maximum));
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_native_i64_narrow_gemm(
    const int64_t* a, const int64_t* b, int64_t* c, int m, int n, int k, int lda, int ldb, int ldc) {
  const auto blocks = static_cast<unsigned>((static_cast<int64_t>(m) * n + 255) / 256);
  hipLaunchKernelGGL(rns8_native_i64_narrow_gemm_kernel, dim3(blocks), dim3(256), 0, nullptr,
                    a, b, c, m, n, k, lda, ldb, ldc);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_convert_residue_layout(
    const int8_t* src, rns8::detail::ResidueLayout source, int8_t* dst, rns8::detail::ResidueLayout destination) {
  const auto blocks = static_cast<unsigned>((source.rows * source.cols * source.prefix + 255) / 256);
  hipLaunchKernelGGL(rns8_convert_residue_layout_kernel, dim3(blocks), dim3(256), 0, nullptr,
                    src, source, dst, destination);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_export_canonical_direct_crt(const int8_t* residues, uint64_t* limbs, int cells, int prefix) {
  hipLaunchKernelGGL(rns8_export_canonical_direct_crt_kernel, dim3((cells + 255) / 256), dim3(256), 0, nullptr,
                    residues, limbs, cells, prefix);
  return static_cast<int>(hipGetLastError());
}
