#include "rns8/generated/metadata_registry.hpp"
#include "backend_common/address_domain.hpp"
#include "backend_amdgpu_builtins/amdgpu_builtins_backend.hpp"

#include "backend_common/finite_u8_reducer.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "core/backend_common.hpp"
#include "core/internal.hpp"

#include <cstring>
#include <limits>
#include <string_view>

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
#  include <hip/hip_runtime_api.h>


extern "C" int rns8_amdgpu_builtin_enqueue_dense(int,const void*,const void*,void*,int,int,int,int,int,int,unsigned,bool,void*);
extern "C" int rns8_amdgpu_builtin_gemm_rns_batch_device(int,const void*,const void*,void*,int,int,int,int,int,int,int,int,int);
extern "C" int rns8_amdgpu_builtin_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int prefix,
    int dense_kernel_variant);

extern "C" int rns8_amdgpu_builtin_gemm_finite_u8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    int dense_kernel_variant);

extern "C" int rns8_amdgpu_builtin_gemm_rns_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int m,
    int n,
    int k,
    int ldb,
    int ldc,
    int prefix);

extern "C" int rns8_amdgpu_builtin_gemm_finite_u8_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int m,
    int n,
    int k,
    int ldb,
    int ldc,
    int modulus);
#endif

namespace rns8::detail {

rns8_status amdgpu_builtins_sparse_a_contract(
    const char* target, int64_t m, int64_t n, int64_t k, int64_t ldb, int64_t ldc,
    uint32_t prefix_or_modulus, bool finite) {
  if (m <= 0 || n <= 0 || k <= 0 || (k % 4) != 0 || ldb < n || ldc < n ||
      m > INT32_MAX || n > INT32_MAX || k > INT32_MAX || ldb > INT32_MAX || ldc > INT32_MAX ||
      (finite ? (prefix_or_modulus<2 || prefix_or_modulus>256)
              : prefix_or_modulus == 0 || prefix_or_modulus > RNS8_MAX_SUPPORTED_PREFIX))
    return RNS8_INVALID_ARGUMENT;
  if (k > RNS8_SAFE_INT32_K_BLOCK) return RNS8_RANGE_ERROR;
  const uint32_t planes = finite ? 1 : prefix_or_modulus;
  if (!qualified_i32_matrix_address(m, k / 2, k / 2, planes) ||
      !qualified_i32_matrix_address(m, k / 4, k / 4, planes) ||
      !qualified_i32_matrix_address(k, n, ldb, planes) ||
      !qualified_i32_matrix_address(m, n, ldc, planes))
    return RNS8_UNSUPPORTED_BACKEND;
  const std::string_view name = target ? target : "";
  const auto arch = name.substr(0, name.find(':'));
  return arch == "gfx942" || arch == "gfx1200" || arch == "gfx1201"
      ? RNS8_SUCCESS : RNS8_UNSUPPORTED_BACKEND;
}

namespace {

thread_local amdgpu_builtins_dense_kernel_variant g_dense_kernel_variant_override =
    amdgpu_builtins_dense_kernel_variant::auto_select;

bool fits_int_argument(int64_t value) {
  return value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

bool valid_dense_shape(
    const void* a,
    const void* b,
    const void* c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  return a && b && c && m > 0 && n > 0 && k > 0 && lda >= k && ldb >= n && ldc >= n &&
         fits_int_argument(m) && fits_int_argument(n) && fits_int_argument(k) && fits_int_argument(lda) &&
         fits_int_argument(ldb) && fits_int_argument(ldc);
}

bool valid_sparse_shape(
    const void* values,
    const void* indices,
    const void* b,
    const void* c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    int64_t ldc) {
  return values && indices && b && c && m > 0 && n > 0 && k > 0 && (k % 4) == 0 && ldb >= n && ldc >= n &&
         fits_int_argument(m) && fits_int_argument(n) && fits_int_argument(k) && fits_int_argument(ldb) &&
         fits_int_argument(ldc);
}

bool amdgpu_target_is_cdna3(const char* target) {
  return target && std::strncmp(target, "gfx942", 6) == 0;
}

bool amdgpu_target_is_rdna3(const char* target) {
  return target && std::strncmp(target, "gfx110", 6) == 0;
}

bool amdgpu_target_is_rdna4(const char* target) {
  return target && (std::strncmp(target, "gfx1200", 7) == 0 || std::strncmp(target, "gfx1201", 7) == 0);
}

bool use_cdna3_mfma_32x32x16_for_override(int64_t m, int64_t n, int64_t k) {
  switch (g_dense_kernel_variant_override) {
    case amdgpu_builtins_dense_kernel_variant::cdna3_mfma_16x16x32:
      return false;
    case amdgpu_builtins_dense_kernel_variant::cdna3_mfma_32x32x16:
      return true;
    case amdgpu_builtins_dense_kernel_variant::auto_select:
      return amdgpu_builtins_use_cdna3_mfma_32x32x16(m, n, k);
  }
  return amdgpu_builtins_use_cdna3_mfma_32x32x16(m, n, k);
}

bool use_cdna3_smfmac_32x32x32(int64_t m, int64_t n, int64_t k) {
  return m >= 128 && n >= 128 && k >= 128;
}

const char* amdgpu_builtins_dense_event_label(int device_id, bool finite, int64_t m, int64_t n, int64_t k) {
  if(!hip_direct_timing_enabled())return "amdgpu_builtin_uninstrumented";
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  if (hip_direct_probe(device_id, info) == RNS8_SUCCESS) {
    if (amdgpu_target_is_rdna3(info.gcn_arch)) {
      return rns8::generated_metadata::builtin_event_label("rdna3_wmma_i32_16x16x16_iu8",finite);
    }
    if (amdgpu_target_is_rdna4(info.gcn_arch)) {
      return rns8::generated_metadata::builtin_event_label("rdna4_wmma_i32_16x16x16_iu8",finite);
    }
    if (amdgpu_target_is_cdna3(info.gcn_arch)) {
      if (use_cdna3_mfma_32x32x16_for_override(m, n, k)) {
        return rns8::generated_metadata::builtin_event_label("cdna3_mfma_i32_32x32x16_i8",finite);
      }
      return rns8::generated_metadata::builtin_event_label("cdna3_mfma_i32_16x16x32_i8",finite);
    }
  }
#else
  (void)device_id;
#endif
  (void)m;
  (void)n;
  (void)k;
  return finite ? "amdgpu_builtin_finite_matrix_core_kernel" : "amdgpu_builtin_rns_matrix_core_kernel";
}

const char* amdgpu_builtins_sparse_event_label(int device_id, bool finite, int64_t m, int64_t n, int64_t k) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  if (hip_direct_probe(device_id, info) == RNS8_SUCCESS) {
    if (amdgpu_target_is_rdna4(info.gcn_arch)) {
      return rns8::generated_metadata::builtin_event_label("rdna4_swmmac_i32_16x16x32_iu8",finite);
    }
    if (amdgpu_target_is_cdna3(info.gcn_arch)) {
      if (use_cdna3_smfmac_32x32x32(m, n, k)) {
        return rns8::generated_metadata::builtin_event_label("cdna3_smfmac_i32_32x32x32_i8",finite);
      }
      return rns8::generated_metadata::builtin_event_label("cdna3_smfmac_i32_16x16x64_i8",finite);
    }
  }
#else
  (void)device_id;
#endif
  (void)m;
  (void)n;
  (void)k;
  return finite ? "amdgpu_builtin_sparse_a_finite_matrix_core_kernel"
                : "amdgpu_builtin_sparse_a_rns_matrix_core_kernel";
}

}  // namespace

 rns8_status amdgpu_builtins_enqueue_dense(int device,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
     int64_t lda,int64_t ldb,int64_t ldc,uint32_t value,bool finite,void* stream) {
   if(!stream || !a || !b || !c || value<1 || (!finite && value>20) ||
       (finite && (value<2 || value>256)))return RNS8_INVALID_ARGUMENT;
   if(k>65536 || !qualified_i32_gemm_address(m,n,k,lda,ldb,ldc,finite?1:value))return RNS8_UNSUPPORTED_BACKEND;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
   return status_from_device_code(rns8_amdgpu_builtin_enqueue_dense(device,a,b,c,int(m),int(n),int(k),int(lda),int(ldb),int(ldc),value,finite,stream));
#else
   (void)device;return RNS8_UNSUPPORTED_BACKEND;
#endif
 }

bool amdgpu_builtins_compiled() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  return true;
#else
  return false;
#endif
}

rns8_status amdgpu_builtins_probe(int device_id, rns8_device_info& out) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  const rns8_status status = hip_direct_probe(device_id, out);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const bool supported = std::strncmp(out.gcn_arch, "gfx942", 6) == 0 ||
                         std::strncmp(out.gcn_arch, "gfx110", 6) == 0 ||
                         std::strncmp(out.gcn_arch, "gfx1200", 7) == 0 ||
                         std::strncmp(out.gcn_arch, "gfx1201", 7) == 0;
  if (!supported) {
    copy_c_string(out.detail, sizeof(out.detail), "AMDGPU builtin kernels require gfx942, gfx110x, gfx1200, or gfx1201");
    return RNS8_UNSUPPORTED_BACKEND;
  }
  out.backend = RNS8_BACKEND_AMDGPU_BUILTINS;
  copy_c_string(out.detail, sizeof(out.detail), "AMDGPU builtin matrix-core kernels detected for this target");
  return RNS8_SUCCESS;
#else
  (void)device_id;
  copy_c_string(out.detail, sizeof(out.detail), "RNS8 was built without AMDGPU builtin kernels");
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

bool amdgpu_builtins_workspace_requirements(int64_t max_m, int64_t max_n, int64_t k, std::size_t& total_bytes) {
  total_bytes = 0;
  return max_m > 0 && max_n > 0 && k > 0 && k <= RNS8_SAFE_INT32_K_BLOCK;
}

bool amdgpu_builtins_use_cdna3_mfma_32x32x16(int64_t m, int64_t n, int64_t k) {
  return m >= 128 && n >= 128 && k >= 128;
}

void amdgpu_builtins_set_dense_kernel_variant_override(amdgpu_builtins_dense_kernel_variant variant) {
  g_dense_kernel_variant_override = variant;
}

rns8_status amdgpu_builtins_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix) {
  if (m > 0 && n > 0 && k > 0 && lda >= k && ldb >= n && ldc >= n &&
      !rns8::detail::qualified_i32_gemm_address(m, n, k, lda, ldb, ldc, prefix > 0 && prefix <= RNS8_MAX_SUPPORTED_PREFIX ? prefix : 1)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  if (!valid_dense_shape(device_a_residues, device_b_residues, device_c_residues, m, n, k, lda, ldb, ldc) ||
      prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (k > RNS8_SAFE_INT32_K_BLOCK) {
    return RNS8_RANGE_ERROR;
  }
  const int code = run_timed_device_code(amdgpu_builtins_dense_event_label(device_id, false, m, n, k), [&]() {
    return rns8_amdgpu_builtin_gemm_rns_device(
        device_id,
        device_a_residues,
        device_b_residues,
        device_c_residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(prefix),
        static_cast<int>(g_dense_kernel_variant_override));
  });
  return status_from_device_code(code);
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status amdgpu_builtins_gemm_rns_batch_device(int device_id,const void* a,const void* b,void* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint32_t batches) {
  if(!a || !b || !c || !prefix || prefix>20 || !batches || batches>64)return RNS8_INVALID_ARGUMENT;
  if(k>RNS8_SAFE_INT32_K_BLOCK)return RNS8_RANGE_ERROR;
  if(!qualified_i32_gemm_address(m,n,k,lda,ldb,ldc,prefix))return RNS8_UNSUPPORTED_BACKEND;
  AddressDomain ad{},bd{},cd{};
  if(!checked_host_plane_address(m,k,lda,prefix*batches,1,ad) || !checked_host_plane_address(k,n,ldb,prefix*batches,1,bd) ||
      !checked_host_plane_address(m,n,ldc,prefix*batches,1,cd))return RNS8_RANGE_ERROR;
  if(!nonoverlapping_byte_ranges(c,cd.byte_extent,a,ad.byte_extent) || !nonoverlapping_byte_ranges(c,cd.byte_extent,b,bd.byte_extent))return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  const std::string label=std::string(amdgpu_builtins_dense_event_label(device_id,false,m,n,k))+"_batched";
  const int code=run_timed_device_code(label.c_str(),[&](){return rns8_amdgpu_builtin_gemm_rns_batch_device(
      device_id,a,b,c,int(m),int(n),int(k),int(lda),int(ldb),int(ldc),int(prefix),int(batches),int(g_dense_kernel_variant_override));});
  return status_from_device_code(code);
#else
  (void)device_id;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status amdgpu_builtins_gemm_finite_u8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
  if (m > 0 && n > 0 && k > 0 && lda >= k && ldb >= n && ldc >= n &&
      !rns8::detail::qualified_i32_gemm_address(m, n, k, lda, ldb, ldc)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  if (!valid_dense_shape(device_a_residues, device_b_residues, device_c_residues, m, n, k, lda, ldb, ldc) ||
      (modulus<2 || modulus>256)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (k > RNS8_SAFE_INT32_K_BLOCK) {
    return RNS8_RANGE_ERROR;
  }
  const int code = run_timed_device_code(amdgpu_builtins_dense_event_label(device_id, true, m, n, k), [&]() {
    return rns8_amdgpu_builtin_gemm_finite_u8_device(
        device_id,
        device_a_residues,
        device_b_residues,
        device_c_residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        static_cast<int>(g_dense_kernel_variant_override));
  });
  return status_from_device_code(code);
#else
  (void)device_id;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)lda;
  (void)ldb;
  (void)ldc;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status amdgpu_builtins_gemm_rns_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  if (!valid_sparse_shape(device_a_packed_values, device_a_packed_indices, device_b_residues, device_c_residues, m, n, k, ldb, ldc) ||
      prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (k > RNS8_SAFE_INT32_K_BLOCK) {
    return RNS8_RANGE_ERROR;
  }
  rns8_device_info sparse_device{};
  const auto probe = amdgpu_builtins_probe(device_id, sparse_device);
  if (probe != RNS8_SUCCESS) return probe;
  const auto contract = amdgpu_builtins_sparse_a_contract(sparse_device.gcn_arch, m, n, k, ldb, ldc, prefix, false);
  if (contract != RNS8_SUCCESS) return contract;
  const int code = run_timed_device_code(amdgpu_builtins_sparse_event_label(device_id, false, m, n, k), [&]() {
    return rns8_amdgpu_builtin_gemm_rns_sparse_a_device(
        device_id,
        device_a_packed_values,
        device_a_packed_indices,
        device_b_residues,
        device_c_residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(prefix));
  });
  return status_from_device_code(code);
#else
  (void)device_id;
  (void)device_a_packed_values;
  (void)device_a_packed_indices;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)ldb;
  (void)ldc;
  (void)prefix;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status amdgpu_builtins_gemm_finite_u8_sparse_a_device(
    int device_id,
    const void* device_a_packed_values,
    const void* device_a_packed_indices,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP && defined(RNS8_ENABLE_AMDGPU_BUILTINS) && \
    RNS8_ENABLE_AMDGPU_BUILTINS && defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && \
    RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
  if (!valid_sparse_shape(device_a_packed_values, device_a_packed_indices, device_b_residues, device_c_residues, m, n, k, ldb, ldc) ||
      (modulus<2 || modulus>256)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (k > RNS8_SAFE_INT32_K_BLOCK) {
    return RNS8_RANGE_ERROR;
  }
  rns8_device_info sparse_device{};
  const auto probe = amdgpu_builtins_probe(device_id, sparse_device);
  if (probe != RNS8_SUCCESS) return probe;
  const auto contract = amdgpu_builtins_sparse_a_contract(sparse_device.gcn_arch, m, n, k, ldb, ldc, modulus, true);
  if (contract != RNS8_SUCCESS) return contract;
  const int code = run_timed_device_code(amdgpu_builtins_sparse_event_label(device_id, true, m, n, k), [&]() {
    return rns8_amdgpu_builtin_gemm_finite_u8_sparse_a_device(
        device_id,
        device_a_packed_values,
        device_a_packed_indices,
        device_b_residues,
        device_c_residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus));
  });
  return status_from_device_code(code);
#else
  (void)device_id;
  (void)device_a_packed_values;
  (void)device_a_packed_indices;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)m;
  (void)n;
  (void)k;
  (void)ldb;
  (void)ldc;
  (void)modulus;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

}  // namespace rns8::detail
