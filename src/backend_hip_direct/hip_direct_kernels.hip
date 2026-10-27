#include <hip/hip_runtime.h>

// These internal HIP objects serve both library variants; API exports live in host TUs.
#define RNS8_STATIC
#include "rns8/rns8.h"

#include <cstdint>
#include <type_traits>
#include "../backend_common/garner_constants.hpp"
#include "../backend_common/basis_word_arithmetic.hpp"

#include "../backend_common/finite_u8_reducer.hpp"
#include "../backend_common/residue_dot.hpp"
#include "../backend_common/residue_epilogue.hpp"
#include "../backend_common/split_k_partial.hpp"
#include "../backend_common/complex_residue_algebra.hpp"
#include "../backend_common/residue_layout.hpp"
#include "../backend_common/residue_layout_gemm.hpp"
#include "../backend_common/direct_crt_constants.hpp"
#include "../backend_common/quarter_crt.hpp"
#include "../backend_common/balanced_crt.hpp"
#include "../backend_common/group_crt.hpp"
#include "../backend_common/strassen_program.hpp"
#include "../backend_common/rank48_program.hpp"
#include "../backend_common/rank48_slp.hpp"
#include "../backend_common/gram_five_product.hpp"
#include "../backend_common/rank23_program.hpp"
#include "../backend_common/rank23_slp.hpp"
#include "../backend_common/rank23_october55.hpp"
#include "../backend_common/int4_encoding.hpp"
#include "../backend_common/native_tile_bound.hpp"
#include "../backend_common/native_wmma_queue.hpp"
#include "../backend_common/native_wmma_schedule.hpp"
#include "../backend_common/native_conversion_layout.hpp"
#include "../backend_common/fragment_layout.hpp"
#include "../backend_common/ragged_residues.hpp"
#include "../backend_common/mixed_radix_lift.hpp"
#include "../backend_common/distributed_device.hpp"

namespace {



constexpr int kRns8DefaultModulusCount = 28;

constexpr int rns8_default_moduli_host[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

#include "hip_direct_device_common.cuh"

#include "hip_direct_pack_kernels.cuh"

#include "hip_direct_gemm_kernels.cuh"
#include "hip_direct_wide_gemm_kernels.cuh"

#include "hip_direct_export_kernels.cuh"
#include "hip_direct_lift_kernels.cuh"
#include "hip_direct_split_k_kernels.cuh"
#include "hip_direct_epilogue_kernels.cuh"
#include "hip_direct_complex_kernels.cuh"
#include "hip_direct_narrow_kernels.cuh"
#include "hip_direct_native_byte_kernels.cuh"
#include "hip_direct_layout_kernels.cuh"
#include "hip_direct_direct_crt_kernels.cuh"
#include "hip_direct_balanced_crt_kernels.cuh"
#include "hip_direct_group_crt_kernels.cuh"
#include "hip_direct_strassen_kernels.cuh"
#include "hip_direct_rank23_slp_kernels.cuh"
#include "hip_direct_rank48_slp_kernels.cuh"
#include "hip_direct_integrity_kernels.cuh"
#include "hip_direct_int4_kernels.cuh"
#include "hip_direct_native_wmma_kernels.cuh"
#include "hip_direct_native_tiled_kernels.cuh"
#include "hip_direct_adaptive_native_kernels.cuh"
#include "hip_direct_bound_kernels.cuh"
#include "hip_direct_ragged_kernels.cuh"
#include "hip_direct_block_sparse_kernels.cuh"
#include "hip_direct_distributed.cuh"

}  // namespace

#include "hip_direct_complex_wrappers.inc"

extern "C" int rns8_hip_distributed_pack(const uint64_t* native_a, const uint64_t* native_b,
    int8_t* a, int8_t* b, uint64_t row, uint64_t col, uint64_t rows, uint64_t cols,
    uint64_t k, uint64_t lda, uint64_t ldb, bool signed_value, uint32_t modulus,uint64_t k_begin, void* stream) {
  const uint64_t blocks = ((rows + cols) * k + 255) / 256;
  hipLaunchKernelGGL(rns8_distributed_pack_kernel, dim3(unsigned(blocks > 4096 ? 4096 : blocks)),
      dim3(256), 0, static_cast<hipStream_t>(stream), native_a, native_b, a, b,
      row, col, rows, cols, k, lda, ldb, signed_value,
      rns8::detail::finite_u8::native64_reducer(modulus),k_begin);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_distributed_export(const int8_t* residues, uint64_t* output,
    int elements, int cols, uint32_t tile_rows, uint32_t tile_cols,
    const rns8::detail::distributed::DeviceTile* tiles, bool signed_value, bool bounded,
    int* status, void* stream) {
  hipLaunchKernelGGL(rns8_distributed_export_kernel, dim3((uint64_t(elements) + 255) / 256),
      dim3(256), 0, static_cast<hipStream_t>(stream), residues, output, elements, cols,
      tile_rows, tile_cols, tiles, signed_value, bounded, status);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_distributed_reduce_partials(const int8_t* partials,int8_t* output,
    const rns8::detail::distributed::DeviceTile* tiles,uint64_t rows,uint64_t cols,uint32_t tile_count,uint32_t prefix,void* stream) {
  if(!partials || !output || !tiles || !rows || !cols || !tile_count || !prefix || prefix>20)return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(rns8_distributed_reduce_partials_kernel,dim3(tile_count,prefix),dim3(256),0,
      static_cast<hipStream_t>(stream),partials,output,tiles,rows,cols);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_distributed_materialize(const uint64_t* limbs, int8_t* residues,
    int elements, uint32_t prefix, bool signed_value, void* stream) {
  hipLaunchKernelGGL(rns8_distributed_materialize_kernel,
      dim3((uint64_t(elements) * prefix + 255) / 256), dim3(256), 0,
      static_cast<hipStream_t>(stream), limbs, residues, elements, prefix, signed_value);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_distributed_integrity(const uint64_t* a,const uint64_t* b,const uint64_t* candidate,
    uint64_t m,uint64_t n,uint64_t k,uint64_t lda,uint64_t ldb,bool signed_value,int* status,void* stream) {
  hipLaunchKernelGGL(rns8_distributed_integrity_kernel,dim3((m*n+255)/256),dim3(256),0,
      static_cast<hipStream_t>(stream),a,b,candidate,m,n,k,lda,ldb,signed_value,status);
  return int(hipGetLastError());
}

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
        static_cast<const uint64_t*>(a), static_cast<const uint64_t*>(b), static_cast<uint64_t*>(c), m, n, k, lda, ldb, ldc,nullptr,0);
  } else {
    hipLaunchKernelGGL((rns8_native_byte_wmma_kernel<int64_t>), grid, dim3(32), 0, nullptr,
        static_cast<const int64_t*>(a), static_cast<const int64_t*>(b), static_cast<int64_t*>(c), m, n, k, lda, ldb, ldc,nullptr,0);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_integrity_check(const void* a,const void* b,const int8_t* candidate,
    int m,int n,int k,int lda,int ldb,int prefix,bool unsigned_inputs,const uint64_t* bound,uint32_t* failures,void* stream) {
  const dim3 grid((static_cast<uint64_t>(m)*n+255)/256);
  const rns8_u192_device wide_bound{bound[0],bound[1],bound[2]};
  if(unsigned_inputs) {
    hipLaunchKernelGGL((rns8_integrity_check_kernel<uint64_t>),grid,dim3(256),0,static_cast<hipStream_t>(stream),
        static_cast<const uint64_t*>(a),static_cast<const uint64_t*>(b),candidate,m,n,k,lda,ldb,prefix,false,wide_bound,failures);
  } else {
    hipLaunchKernelGGL((rns8_integrity_check_kernel<int64_t>),grid,dim3(256),0,static_cast<hipStream_t>(stream),
        static_cast<const int64_t*>(a),static_cast<const int64_t*>(b),candidate,m,n,k,lda,ldb,prefix,true,wide_bound,failures);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_split_k_canonicalize(uint8_t* values, int64_t cells, int prefix, int fragments) {
  hipLaunchKernelGGL(rns8_split_k_canonicalize_kernel, dim3((cells + 255) / 256, prefix, fragments),
                    dim3(256), 0, nullptr, values, cells, prefix, fragments);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_rank23_slp_transform(const int8_t* source,int8_t* output,
    int rows,int cols,int ld,int hr,int hc,int prefix,bool lhs) {
  const dim3 grid((uint64_t(hr)*hc+255)/256,prefix);
  if(lhs) {
    hipLaunchKernelGGL((rns8_rank23_slp_transform_kernel<true>),grid,dim3(256),0,nullptr,
        source,output,rows,cols,ld,hr,hc,prefix);
  } else {
    hipLaunchKernelGGL((rns8_rank23_slp_transform_kernel<false>),grid,dim3(256),0,nullptr,
        source,output,rows,cols,ld,hr,hc,prefix);
  }
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_rank23_slp_combine(const int8_t* products,int8_t* stage,
    int m,int n,int hm,int hn,int prefix) {
  hipLaunchKernelGGL(rns8_rank23_slp_combine_kernel,
      dim3((uint64_t(hm)*hn+255)/256,prefix),dim3(256),0,nullptr,
      products,stage,m,n,hm,hn,prefix);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_rank23_oct55_transform(const int8_t* source,int8_t* output,
    int rows,int cols,int ld,int hr,int hc,int prefix,bool lhs,int selected_product) {
  const dim3 grid((uint64_t(hr)*hc+255)/256,prefix);
  if(lhs)hipLaunchKernelGGL((rns8_rank23_oct55_transform_kernel<true>),grid,dim3(256),0,nullptr,
      source,output,rows,cols,ld,hr,hc,prefix,selected_product);
  else hipLaunchKernelGGL((rns8_rank23_oct55_transform_kernel<false>),grid,dim3(256),0,nullptr,
      source,output,rows,cols,ld,hr,hc,prefix,selected_product);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_rank23_oct55_combine(const int8_t* products,int8_t* stage,
    int m,int n,int hm,int hn,int prefix) {
  hipLaunchKernelGGL(rns8_rank23_oct55_combine_kernel,dim3((uint64_t(hm)*hn+255)/256,prefix),dim3(256),0,nullptr,
      products,stage,m,n,hm,hn,prefix);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_rank48_slp_transform(const int8_t* source,int8_t* output,
    int rows,int cols,int ld,int hr,int hc,int prefix,bool lhs,int selected_product) {
  const dim3 grid((uint64_t(hr)*hc+255)/256,prefix);
  if(lhs)hipLaunchKernelGGL((rns8_rank48_slp_transform_kernel<true>),grid,dim3(256),0,nullptr,
      source,output,rows,cols,ld,hr,hc,prefix,selected_product);
  else hipLaunchKernelGGL((rns8_rank48_slp_transform_kernel<false>),grid,dim3(256),0,nullptr,
      source,output,rows,cols,ld,hr,hc,prefix,selected_product);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_rank48_slp_combine(const int8_t* products,int8_t* stage,
    int m,int n,int hm,int hn,int prefix) {
  hipLaunchKernelGGL(rns8_rank48_slp_combine_kernel,dim3((uint64_t(hm)*hn+255)/256,prefix),dim3(256),0,nullptr,
      products,stage,m,n,hm,hn,prefix);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_strassen_transform(const int8_t* source,int8_t* output,
    int rows,int cols,int ld,int hr,int hc,int prefix,int product,bool lhs,int blocks) {
  const dim3 grid((static_cast<uint64_t>(hr)*hc+255)/256,prefix);
  if(blocks==4) {
    hipLaunchKernelGGL((rns8_bilinear_transform_kernel<4>),grid,dim3(256),0,nullptr,
        source,output,rows,cols,ld,hr,hc,product,lhs);
  } else if (blocks==3) {
    hipLaunchKernelGGL((rns8_bilinear_transform_kernel<3>),grid,dim3(256),0,nullptr,
        source,output,rows,cols,ld,hr,hc,product,lhs);
  } else {
    hipLaunchKernelGGL((rns8_bilinear_transform_kernel<2>),grid,dim3(256),0,nullptr,
        source,output,rows,cols,ld,hr,hc,product,lhs);
  }
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_strassen_combine(const int8_t* product,int8_t* stage,int m,int n,int hm,int hn,int prefix,int index,int blocks) {
  const dim3 grid((static_cast<uint64_t>(m)*n+255)/256,prefix);
  if(blocks==4) {
    hipLaunchKernelGGL((rns8_bilinear_combine_kernel<4>),grid,dim3(256),0,nullptr,
        product,stage,m,n,hm,hn,index);
  } else if (blocks==3) {
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
    const int8_t* residues,uint64_t* limbs,int cells,const rns8::detail::balanced_crt::Plan* plan,
    rns8::detail::ResidueLayout source) {
  if (source.kind == rns8::detail::ResidueLayoutKind::modulus_major && source.ld == source.cols) {
    hipLaunchKernelGGL((rns8_export_canonical_balanced_crt_kernel<false>),dim3((cells+255)/256),dim3(256),0,nullptr,
        residues,limbs,cells,*plan,source);
  } else {
    hipLaunchKernelGGL((rns8_export_canonical_balanced_crt_kernel<true>),dim3((cells+255)/256),dim3(256),0,nullptr,
        residues,limbs,cells,*plan,source);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_export_canonical_group_crt(const int8_t* residues,uint64_t* output,
    uint64_t* partials,int cells,const rns8::detail::group_crt::Plan* plan,uint32_t order,uint32_t stage) {
  const dim3 grid((cells+255)/256);
  if(stage==0) {
    hipLaunchKernelGGL(rns8_export_group_crt_local_kernel,grid,dim3(256),0,nullptr,residues,output,cells,*plan);
  } else if(stage==1) {
    hipLaunchKernelGGL(rns8_produce_group_crt_kernel,dim3(grid.x,plan->group_count),dim3(256),0,nullptr,
        residues,partials,cells,*plan,order);
  } else if(stage==2) {
    hipLaunchKernelGGL(rns8_merge_group_crt_kernel,grid,dim3(256),0,nullptr,partials,output,cells,*plan);
  } else return static_cast<int>(hipErrorInvalidValue);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_block_sparse_gemm(
    const int8_t* a,const uint32_t* ao,const uint32_t* ai,
    const int8_t* b,const uint32_t* bo,const uint32_t* bi,
    int8_t* c,int m,int n,int k,int ldc,int prefix,int method,uint32_t scalar_limit) {
  const uint64_t tile_cols=(static_cast<uint64_t>(n)+15)/16;
  const uint64_t tiles=((static_cast<uint64_t>(m)+15)/16)*tile_cols;
  if(method==0) hipLaunchKernelGGL((rns8_block_sparse_gemm_kernel<0>),dim3(tiles,prefix),dim3(256),0,nullptr,
      a,ao,ai,b,bo,bi,c,m,n,k,ldc,prefix,static_cast<int>(tile_cols),scalar_limit);
  else if(method==1) hipLaunchKernelGGL((rns8_block_sparse_gemm_kernel<1>),dim3(tiles,prefix),dim3(32),0,nullptr,
      a,ao,ai,b,bo,bi,c,m,n,k,ldc,prefix,static_cast<int>(tile_cols),scalar_limit);
  else if(method==2) hipLaunchKernelGGL((rns8_block_sparse_gemm_kernel<2>),dim3(tiles,prefix),dim3(256),0,nullptr,
      a,ao,ai,b,bo,bi,c,m,n,k,ldc,prefix,static_cast<int>(tile_cols),scalar_limit);
  else return static_cast<int>(hipErrorInvalidValue);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_discover_ragged_bounds(const int8_t* source, int rows, int cols,
    int prefix, int tile_rows, int tile_cols, int grid_cols, bool signed_value,
    uint64_t bound0, uint64_t bound1, uint64_t bound2,
    rns8::detail::BasisPacket basis, rns8::detail::RaggedResidueTile* directory, int* status) {
  const uint64_t tiles = ((uint64_t(rows) + tile_rows - 1) / tile_rows) * grid_cols;
  if(rns8::detail::ordered_basis_identity(basis)==rns8::detail::OrderedBasisIdentity{}) {
  hipLaunchKernelGGL((rns8_discover_ragged_bounds_kernel<true>), dim3(tiles), dim3(128), 0, nullptr,
      source, rows, cols, prefix, tile_rows, tile_cols, grid_cols, signed_value,
      (rns8_u192_device{bound0, bound1, bound2}), basis, directory, status);
  } else {
  hipLaunchKernelGGL((rns8_discover_ragged_bounds_kernel<false>), dim3(tiles), dim3(128), 0, nullptr,
      source, rows, cols, prefix, tile_rows, tile_cols, grid_cols, signed_value,
      (rns8_u192_device{bound0, bound1, bound2}), basis, directory, status);
  }
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_compact_ragged(const int8_t* source, int rows, int cols,
    const rns8::detail::RaggedResidueTile* directory, int tile_rows, int tile_cols,
    int grid_cols, int8_t* destination) {
  const uint64_t tiles = ((uint64_t(rows) + tile_rows - 1) / tile_rows) * grid_cols;
  hipLaunchKernelGGL(rns8_compact_ragged_kernel, dim3(tiles), dim3(128), 0, nullptr,
      source, rows, cols, directory, tile_rows, tile_cols, grid_cols, destination);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_materialize_ragged(
    const int8_t* source, const rns8::detail::RaggedResidueTile* tiles,
    int tile_rows, int tile_cols, int grid_cols, bool signed_value,
    int8_t* destination, rns8::detail::ResidueLayout target, rns8::detail::BasisPacket basis, int* status) {
  if(rns8::detail::ordered_basis_identity(basis)==rns8::detail::OrderedBasisIdentity{}) {
  hipLaunchKernelGGL((rns8_materialize_ragged_kernel<true>), dim3((target.rows * target.cols + 255) / 256),
      dim3(256), 0, nullptr, source, tiles, tile_rows, tile_cols, grid_cols, signed_value, destination, target, basis, status);
  } else {
  hipLaunchKernelGGL((rns8_materialize_ragged_kernel<false>), dim3((target.rows * target.cols + 255) / 256),
      dim3(256), 0, nullptr, source, tiles, tile_rows, tile_cols, grid_cols, signed_value, destination, target, basis, status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_int4_capability(int* supported) {
  hipLaunchKernelGGL(rns8_int4_capability_kernel, dim3(1), dim3(1), 0, nullptr, supported);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_finite_int4_gemm(
    const int8_t* a, const int8_t* b, int8_t* c, int m, int n, int k,
    int lda, int ldb, int ldc, int modulus) {
  hipLaunchKernelGGL((rns8_finite_int4_wmma_kernel<false>),
      dim3((static_cast<uint64_t>(n) + 15) / 16, (static_cast<uint64_t>(m) + 15) / 16),
      dim3(32), 0, nullptr, a, b, nullptr, c, m, n, k, lda, ldb, ldc, modulus);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_finite_int4_gemm_packed(const int8_t* a,const uint32_t* b,int8_t* c,
    int m,int n,int k,int lda,int ldc,int modulus) {
  hipLaunchKernelGGL((rns8_finite_int4_wmma_kernel<true>),
      dim3((uint64_t(n)+15)/16,(uint64_t(m)+15)/16),dim3(32),0,nullptr,a,nullptr,b,c,m,n,k,lda,n,ldc,modulus);
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_pack_finite_int4_b(const int8_t* b,uint32_t* packed,int k,int n,int ldb,int modulus) {
  const uint64_t words=((uint64_t(k)+15)/16)*uint64_t(n)*2;
  hipLaunchKernelGGL(rns8_pack_finite_int4_b_kernel,dim3((words+255)/256),dim3(256),0,nullptr,b,packed,k,n,ldb,modulus,words);
  return static_cast<int>(hipGetLastError());
}

#include "hip_direct_pack_wrappers.inc"
#include "hip_direct_native_wmma_wrappers.inc"
#include "hip_direct_native_tiled_wrappers.inc"
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
  // Enqueue-only, just like the generic and fixed-prefix pack wrappers.
  return code;
}

extern "C" int rns8_hip_direct_lift_suffix_device(
    const int8_t* residues, int8_t* suffix, int cells, int source_prefix, int target_prefix,
    bool signed_value, uint64_t bound0, uint64_t bound1, uint64_t bound2, int* status, bool mixed_radix, void* stream) {
  if (!residues || !suffix || !status || cells <= 0 || cells > kRns8MaxInt - 256 ||
      source_prefix <= 0 || target_prefix > kRns8MaxSupportedPrefix || source_prefix >= target_prefix)
    return static_cast<int>(hipErrorInvalidValue);
  if (mixed_radix) {
    hipLaunchKernelGGL((rns8_exact_lift_suffix_kernel<true>), dim3((cells + 255) / 256), dim3(256), 0, static_cast<hipStream_t>(stream),
                      residues, suffix, cells, source_prefix, target_prefix, signed_value,
                      (rns8_u192_device{bound0, bound1, bound2}), status);
  } else {
    hipLaunchKernelGGL((rns8_exact_lift_suffix_kernel<false>), dim3((cells + 255) / 256), dim3(256), 0, static_cast<hipStream_t>(stream),
                      residues, suffix, cells, source_prefix, target_prefix, signed_value,
                      (rns8_u192_device{bound0, bound1, bound2}), status);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_axpby_residues(
    const int8_t* x, const int8_t* d, int8_t* c,
    int m, int n, int ldx, int ldd, int ldc, int prefix, int64_t alpha, int64_t beta) {
  rns8::detail::ResidueAxpbyTable coefficients;
  if(!rns8::detail::lower_residue_axpby_table(alpha,beta,rns8_default_moduli_host,prefix,coefficients))return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(rns8_axpby_residues_kernel, dim3((int64_t{m} * n + 255) / 256, prefix), dim3(256), 0, nullptr,
                    x, d, c, m, n, ldx, ldd, ldc, coefficients);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_gemm_rns_axpby(
    const int8_t* a, const int8_t* b, const int8_t* d, int8_t* c,
    int m, int n, int k, int lda, int ldb, int ldd, int ldc, int prefix, int64_t alpha, int64_t beta, void* stream) {
  rns8::detail::ResidueAxpbyTable coefficients;
  if(!rns8::detail::lower_residue_axpby_table(alpha,beta,rns8_default_moduli_host,prefix,coefficients))return int(hipErrorInvalidValue);
  const auto blocks = static_cast<unsigned>((static_cast<int64_t>(m) * n + 255) / 256);
  hipLaunchKernelGGL(rns8_gemm_rns_axpby_kernel, dim3(blocks, prefix), dim3(256), 0, static_cast<hipStream_t>(stream),
                    a, b, d, c, m, n, k, lda, ldb, ldd, ldc, coefficients);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_native_abs_max_on_stream(const void* source, int rows, int cols, int ld, void* maximum, bool signed_values,void* stream) {
  const auto blocks=(uint64_t(rows)*cols+255)/256;
  hipLaunchKernelGGL(rns8_native_i64_abs_max_kernel,dim3(blocks),dim3(256),0,static_cast<hipStream_t>(stream),
      static_cast<const uint64_t*>(source),rows,cols,ld,static_cast<unsigned long long*>(maximum),signed_values);
  return int(hipGetLastError());
}
extern "C" int rns8_hip_native_abs_max(const void* source,int rows,int cols,int ld,void* maximum,bool signed_values) {
  return rns8_hip_native_abs_max_on_stream(source,rows,cols,ld,maximum,signed_values,nullptr);
}
extern "C" int rns8_hip_native_i64_abs_max(const int64_t* source, int rows, int cols, int ld, void* maximum) {
  return rns8_hip_native_abs_max(source,rows,cols,ld,maximum,true);
}

extern "C" int rns8_hip_native_fused_narrow_rns(const void* a,const void* b,int8_t* residues,int m,int n,int k,int lda,int ldb,int prefix,bool byte_wmma,bool unsigned_inputs) {
  if(byte_wmma){const dim3 grid((uint64_t(n)+15)/16,(uint64_t(m)+15)/16);
    if(unsigned_inputs)hipLaunchKernelGGL((rns8_native_byte_wmma_kernel<uint64_t,true>),grid,dim3(32),0,nullptr,static_cast<const uint64_t*>(a),static_cast<const uint64_t*>(b),nullptr,m,n,k,lda,ldb,n,residues,prefix);
    else hipLaunchKernelGGL((rns8_native_byte_wmma_kernel<int64_t,true>),grid,dim3(32),0,nullptr,static_cast<const int64_t*>(a),static_cast<const int64_t*>(b),nullptr,m,n,k,lda,ldb,n,residues,prefix);
  } else hipLaunchKernelGGL((rns8_native_i64_narrow_gemm_kernel<true>),dim3((uint64_t(m)*n+255)/256),dim3(256),0,nullptr,static_cast<const int64_t*>(a),static_cast<const int64_t*>(b),nullptr,m,n,k,lda,ldb,n,residues,prefix);
  return int(hipGetLastError());
}

extern "C" int rns8_hip_native_i64_narrow_gemm(
    const int64_t* a, const int64_t* b, int64_t* c, int m, int n, int k, int lda, int ldb, int ldc) {
  const auto blocks = static_cast<unsigned>((static_cast<int64_t>(m) * n + 255) / 256);
  hipLaunchKernelGGL((rns8_native_i64_narrow_gemm_kernel<false>), dim3(blocks), dim3(256), 0, nullptr,
                    a, b, c, m, n, k, lda, ldb, ldc,nullptr,0);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_convert_residue_planes(
    const int8_t* src,rns8::detail::ResidueLayout source,int8_t* dst,rns8::detail::ResidueLayout destination,
    uint32_t first,uint32_t end) {
  const auto blocks = static_cast<unsigned>((source.rows * source.cols * (end-first) + 255) / 256);
  hipLaunchKernelGGL(rns8_convert_residue_layout_kernel, dim3(blocks), dim3(256), 0, nullptr,
                    src,source,dst,destination,first,end);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_gemm_residue_layout(
    const int8_t* a,rns8::detail::ResidueLayout al,const int8_t* b,rns8::detail::ResidueLayout bl,
    int8_t* c,rns8::detail::ResidueLayout cl,uint32_t prefix,rns8::detail::ResidueLayoutModuli moduli) {
  hipLaunchKernelGGL(rns8_gemm_residue_layout_kernel,
      dim3((bl.cols+15)/16,(al.rows+15)/16,prefix),dim3(16,16),0,nullptr,a,al,b,bl,c,cl,prefix,moduli);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_export_canonical_direct_crt(const int8_t* residues, uint64_t* limbs, int cells, int prefix) {
  hipLaunchKernelGGL(rns8_export_canonical_direct_crt_kernel, dim3((cells + 255) / 256), dim3(256), 0, nullptr,
                    residues, limbs, cells, prefix);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_wide_gemm(const int8_t* a,const int8_t* b,int8_t* c,
    uint64_t m,uint64_t n,uint64_t k,uint64_t lda,uint64_t ldb,uint64_t ldc,uint64_t count) {
  const unsigned blocks=static_cast<unsigned>(count<262144 ? (count+255)/256 : 1024);
  hipLaunchKernelGGL(rns8_wide_gemm_kernel,dim3(blocks),dim3(256),0,nullptr,a,b,c,m,n,k,lda,ldb,ldc,count);
  return static_cast<int>(hipGetLastError());
}

extern "C" int rns8_hip_finite_indexed_window(const int8_t* a,const int8_t* b,int8_t* c,
    uint64_t n,uint64_t k,uint64_t lda,uint64_t ldb,uint64_t ldc,uint64_t first_row,uint64_t count,int modulus,int width) {
  if(!n || !k || !count || modulus<2 || modulus>256 || (width!=0 && width!=1))return int(hipErrorInvalidValue);
  const unsigned blocks=static_cast<unsigned>(count<262144?(count+255)/256:1024);
  if(width==0)
    hipLaunchKernelGGL(rns8_finite_indexed_window_kernel<int32_t>,dim3(blocks),dim3(256),0,nullptr,a,b,c,n,k,lda,ldb,ldc,first_row,count,modulus);
  else
    hipLaunchKernelGGL(rns8_finite_indexed_window_kernel<uint64_t>,dim3(blocks),dim3(256),0,nullptr,a,b,c,n,k,lda,ldb,ldc,first_row,count,modulus);
  return static_cast<int>(hipGetLastError());
}

#include "hip_direct_finite_verify.cuh"

#include "hip_direct_wrap_radix.cuh"
#include "hip_direct_wrap_mfma.cuh"

#include "hip_direct_gram.cuh"
#include "hip_direct_gram_five.cuh"

#include "hip_direct_fixed_conversion.cuh"

#include "hip_direct_word_mod.cuh"
#include "hip_direct_word_mod_mfma.cuh"

#include "hip_direct_instruction_roof.cuh"
#include "hip_direct_fp8_modular.cuh"
#include "hip_direct_native_tdm.cuh"

extern "C" int rns8_hip_export_quarter_crt(const int8_t* residues,uint64_t* output,int cells,int prefix,int digits,int limbs,int mode) {
  hipLaunchKernelGGL(rns8_export_quarter_crt_kernel,dim3((cells+255)/256),dim3(256),0,nullptr,residues,output,cells,prefix,digits,limbs,mode);
  return static_cast<int>(hipGetLastError());
}

// Generic status-gated compact/padded byte commit. The status-producing kernel
// finishes earlier on this same stream; all destination writes are conditional.
__global__ void rns8_guarded_copy_bytes_kernel(const uint8_t* source,uint8_t* destination,
    uint64_t rows,uint64_t row_bytes,uint64_t source_pitch,uint64_t destination_pitch,const int* status) {
  if(*status)return;
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(cell>=rows*row_bytes)return;
  const uint64_t row=cell/row_bytes,col=cell%row_bytes;
  destination[row*destination_pitch+col]=source[row*source_pitch+col];
}
extern "C" int rns8_hip_guarded_copy_bytes(const void* source,void* destination,uint64_t rows,
    uint64_t row_bytes,uint64_t source_pitch,uint64_t destination_pitch,const int* status,void* stream) {
  if(!source || !destination || !status || !rows || !row_bytes || row_bytes>source_pitch || row_bytes>destination_pitch ||
      row_bytes>uint64_t(PTRDIFF_MAX) || rows>(UINT64_MAX-255)/row_bytes || (rows*row_bytes+255)/256>uint64_t(INT32_MAX) ||
      rows-1>(uint64_t(PTRDIFF_MAX)-row_bytes)/source_pitch || rows-1>(uint64_t(PTRDIFF_MAX)-row_bytes)/destination_pitch)return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(rns8_guarded_copy_bytes_kernel,dim3((rows*row_bytes+255)/256),dim3(256),0,static_cast<hipStream_t>(stream),
      static_cast<const uint8_t*>(source),static_cast<uint8_t*>(destination),rows,row_bytes,source_pitch,destination_pitch,status);
  return int(hipGetLastError());
}

__global__ void rns8_native_bound_status_kernel(const uint64_t* maximum,uint64_t limit,int* status) {
  if(*maximum>limit)*status=static_cast<int>(RNS8_RANGE_ERROR);
}
extern "C" int rns8_hip_native_bound_status(const void* maximum,uint64_t limit,int* status,void* stream) {
  hipLaunchKernelGGL(rns8_native_bound_status_kernel,dim3(1),dim3(1),0,static_cast<hipStream_t>(stream),
      static_cast<const uint64_t*>(maximum),limit,status);
  return int(hipGetLastError());
}

#include "hip_direct_cache_displace.cuh"
#include "hip_direct_checked_crt.cuh"

#include "hip_direct_experimental_basis.cuh"
