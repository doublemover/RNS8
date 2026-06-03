#include "backend_hipblaslt/hipblaslt_backend.hpp"

#include "backend_hip_direct/hip_backend.hpp"
#include "core/internal.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
#  include <hip/hip_runtime_api.h>
#  include <hipblaslt/hipblaslt.h>

extern "C" int rns8_hipblaslt_reduce_i32_to_centered_device(
    const int32_t* scratch,
    int8_t* residues,
    int rows,
    int cols,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int accumulate);

extern "C" int rns8_hipblaslt_pack_transpose_centered_device(
    const int8_t* src,
    int8_t* dst,
    int src_rows,
    int src_cols,
    int src_ld,
    int dst_rows,
    int dst_cols,
    int dst_ld);

extern "C" int rns8_hipblaslt_reduce_i32_to_centered_strided_device(
    const int32_t* scratch,
    int8_t* residues,
    int rows,
    int cols,
    int scratch_ld,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int accumulate);
#endif

namespace rns8::detail {

namespace {

constexpr uint16_t kHipblasLtDefaultModuli[RNS8_DEFAULT_MODULUS_COUNT] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

constexpr uint64_t kReciprocalScale = 1ull << 32u;

uint32_t modulus_reciprocal_u32(uint16_t modulus) {
  return static_cast<uint32_t>(kReciprocalScale / static_cast<uint32_t>(modulus));
}

bool checked_common_shape(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  if (m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n) {
    return false;
  }
  return m <= std::numeric_limits<int>::max() && n <= std::numeric_limits<int>::max() &&
         k <= std::numeric_limits<int>::max() && lda <= std::numeric_limits<int>::max() &&
         ldb <= std::numeric_limits<int>::max() && ldc <= std::numeric_limits<int>::max();
}

bool checked_workspace(int64_t m, int64_t n, int64_t k, std::size_t scratch_bytes, std::size_t workspace_bytes) {
  std::size_t required_scratch = 0;
  std::size_t required_workspace = 0;
  if (!hipblaslt_baseline_workspace_requirements(m, n, k, required_scratch, required_workspace)) {
    return false;
  }
  return scratch_bytes >= required_scratch && workspace_bytes >= required_workspace;
}

#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
struct padded_block_shape {
  int m = 0;
  int n = 0;
  int k = 0;
};

bool padded_block_shape_for(int64_t m, int64_t n, int64_t k_block, padded_block_shape& out) {
  uint64_t padded_m = 0;
  uint64_t padded_n = 0;
  uint64_t padded_k = 0;
  if (!hipblaslt_round_up_aligned(static_cast<uint64_t>(m), padded_m) ||
      !hipblaslt_round_up_aligned(static_cast<uint64_t>(n), padded_n) ||
      !hipblaslt_round_up_aligned(static_cast<uint64_t>(k_block), padded_k)) {
    return false;
  }
  out.m = static_cast<int>(padded_m);
  out.n = static_cast<int>(padded_n);
  out.k = static_cast<int>(padded_k);
  return true;
}

rns8_status set_hip_device(int device_id) {
  if (device_id < 0) {
    device_id = 0;
  }
  const hipError_t err = hipSetDevice(device_id);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status status_from_hipblas(hipblasStatus_t status) {
  switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
      return RNS8_SUCCESS;
    case HIPBLAS_STATUS_NOT_SUPPORTED:
    case HIPBLAS_STATUS_ARCH_MISMATCH:
      return RNS8_UNSUPPORTED_BACKEND;
    case HIPBLAS_STATUS_INVALID_VALUE:
    case HIPBLAS_STATUS_INVALID_ENUM:
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
      return RNS8_INVALID_ARGUMENT;
    case HIPBLAS_STATUS_ALLOC_FAILED:
      return RNS8_INTERNAL_ERROR;
    case HIPBLAS_STATUS_NOT_INITIALIZED:
    case HIPBLAS_STATUS_MAPPING_ERROR:
    case HIPBLAS_STATUS_EXECUTION_FAILED:
    case HIPBLAS_STATUS_INTERNAL_ERROR:
    case HIPBLAS_STATUS_UNKNOWN:
      return RNS8_BACKEND_FAILURE;
  }
  return RNS8_BACKEND_FAILURE;
}

template <typename Fn>
hipblasStatus_t timed_hipblaslt_operation(const char* label, Fn&& fn) {
  if (!hip_direct_timing_enabled()) {
    return fn();
  }
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  hipError_t err = hipEventCreate(&start);
  if (err != hipSuccess) {
    return fn();
  }
  err = hipEventCreate(&stop);
  if (err != hipSuccess) {
    (void)hipEventDestroy(start);
    return fn();
  }
  err = hipEventRecord(start, nullptr);
  if (err != hipSuccess) {
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    return fn();
  }
  const hipblasStatus_t status = fn();
  if (status == HIPBLAS_STATUS_SUCCESS) {
    err = hipEventRecord(stop, nullptr);
    if (err == hipSuccess) {
      err = hipEventSynchronize(stop);
    }
    if (err == hipSuccess) {
      float milliseconds = 0.0f;
      err = hipEventElapsedTime(&milliseconds, start, stop);
      if (err == hipSuccess && milliseconds >= 0.0f) {
        hip_direct_timing_record_sample(label, static_cast<double>(milliseconds) * 1000.0);
      }
    }
  }
  (void)hipEventDestroy(stop);
  (void)hipEventDestroy(start);
  return status;
}

template <typename Fn>
hipError_t timed_hip_operation(const char* label, Fn&& fn) {
  if (!hip_direct_timing_enabled()) {
    return fn();
  }
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  hipError_t err = hipEventCreate(&start);
  if (err != hipSuccess) {
    return fn();
  }
  err = hipEventCreate(&stop);
  if (err != hipSuccess) {
    (void)hipEventDestroy(start);
    return fn();
  }
  err = hipEventRecord(start, nullptr);
  if (err != hipSuccess) {
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    return fn();
  }
  const hipError_t status = fn();
  if (status == hipSuccess) {
    err = hipEventRecord(stop, nullptr);
    if (err == hipSuccess) {
      err = hipEventSynchronize(stop);
    }
    if (err == hipSuccess) {
      float milliseconds = 0.0f;
      err = hipEventElapsedTime(&milliseconds, start, stop);
      if (err == hipSuccess && milliseconds >= 0.0f) {
        hip_direct_timing_record_sample(label, static_cast<double>(milliseconds) * 1000.0);
      }
    }
  }
  (void)hipEventDestroy(stop);
  (void)hipEventDestroy(start);
  return status;
}

struct matmul_descriptors {
  hipblasLtMatmulDesc_t op = nullptr;
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;

  matmul_descriptors() = default;
  matmul_descriptors(const matmul_descriptors&) = delete;
  matmul_descriptors& operator=(const matmul_descriptors&) = delete;

  ~matmul_descriptors() {
    if (preference) {
      (void)hipblasLtMatmulPreferenceDestroy(preference);
    }
    if (d) {
      (void)hipblasLtMatrixLayoutDestroy(d);
    }
    if (c) {
      (void)hipblasLtMatrixLayoutDestroy(c);
    }
    if (b) {
      (void)hipblasLtMatrixLayoutDestroy(b);
    }
    if (a) {
      (void)hipblasLtMatrixLayoutDestroy(a);
    }
    if (op) {
      (void)hipblasLtMatmulDescDestroy(op);
    }
  }
};

struct matmul_algorithm_cache_key {
  int device_id = -1;
  int m = 0;
  int n = 0;
  int k = 0;
  int scratch_ld = 0;
  std::size_t workspace_bytes = 0;

  bool operator==(const matmul_algorithm_cache_key& other) const {
    return device_id == other.device_id && m == other.m && n == other.n && k == other.k &&
           scratch_ld == other.scratch_ld && workspace_bytes == other.workspace_bytes;
  }
};

struct matmul_algorithm_cache_entry {
  matmul_algorithm_cache_key key{};
  hipblasLtMatmulAlgo_t algo{};
};

std::mutex& matmul_algorithm_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<matmul_algorithm_cache_entry>& matmul_algorithm_cache_entries() {
  static std::vector<matmul_algorithm_cache_entry> entries;
  return entries;
}

bool find_cached_matmul_algorithm(const matmul_algorithm_cache_key& key, hipblasLtMatmulAlgo_t& algo) {
  std::lock_guard<std::mutex> lock(matmul_algorithm_cache_mutex());
  const auto& entries = matmul_algorithm_cache_entries();
  const auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) { return entry.key == key; });
  if (it == entries.end()) {
    return false;
  }
  algo = it->algo;
  return true;
}

void remember_matmul_algorithm(const matmul_algorithm_cache_key& key, const hipblasLtMatmulAlgo_t& algo) {
  std::lock_guard<std::mutex> lock(matmul_algorithm_cache_mutex());
  auto& entries = matmul_algorithm_cache_entries();
  const auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) { return entry.key == key; });
  if (it != entries.end()) {
    it->algo = algo;
    return;
  }
  entries.push_back(matmul_algorithm_cache_entry{key, algo});
}

hipblasStatus_t create_matmul_descriptors(
    const padded_block_shape& shape,
    int64_t scratch_ld,
    std::size_t workspace_bytes,
    matmul_descriptors& descriptors) {
  hipblasStatus_t status = hipblasLtMatmulDescCreate(&descriptors.op, HIPBLAS_COMPUTE_32I, HIP_R_32I);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  const hipblasOperation_t op_n = HIPBLAS_OP_N;
  status = hipblasLtMatmulDescSetAttribute(descriptors.op, HIPBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = hipblasLtMatmulDescSetAttribute(descriptors.op, HIPBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = hipblasLtMatrixLayoutCreate(
      &descriptors.a, HIP_R_8I, static_cast<uint64_t>(shape.n), static_cast<uint64_t>(shape.k), shape.n);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = hipblasLtMatrixLayoutCreate(
      &descriptors.b, HIP_R_8I, static_cast<uint64_t>(shape.k), static_cast<uint64_t>(shape.m), shape.k);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = hipblasLtMatrixLayoutCreate(
      &descriptors.c, HIP_R_32I, static_cast<uint64_t>(shape.n), static_cast<uint64_t>(shape.m), scratch_ld);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = hipblasLtMatrixLayoutCreate(
      &descriptors.d, HIP_R_32I, static_cast<uint64_t>(shape.n), static_cast<uint64_t>(shape.m), scratch_ld);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  status = hipblasLtMatmulPreferenceCreate(&descriptors.preference);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  const uint64_t max_workspace = static_cast<uint64_t>(workspace_bytes);
  return hipblasLtMatmulPreferenceSetAttribute(
      descriptors.preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace));
}

hipblasStatus_t run_matmul_block(
    int device_id,
    hipblasLtHandle_t handle,
    const int8_t* packed_a_t,
    const int8_t* packed_b_t,
    int32_t* scratch,
    void* workspace,
    std::size_t workspace_bytes,
    const padded_block_shape& shape) {
  matmul_descriptors descriptors;
  const int64_t scratch_ld = shape.n;
  hipblasStatus_t status = create_matmul_descriptors(shape, scratch_ld, workspace_bytes, descriptors);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }

  const matmul_algorithm_cache_key cache_key{
      device_id,
      shape.m,
      shape.n,
      shape.k,
      static_cast<int>(scratch_ld),
      workspace_bytes};
  hipblasLtMatmulAlgo_t algo{};
  if (!find_cached_matmul_algorithm(cache_key, algo)) {
    hipblasLtMatmulHeuristicResult_t heuristic{};
    int heuristic_count = 0;
    status = hipblasLtMatmulAlgoGetHeuristic(
        handle,
        descriptors.op,
        descriptors.a,
        descriptors.b,
        descriptors.c,
        descriptors.d,
        descriptors.preference,
        1,
        &heuristic,
        &heuristic_count);
    if (status == HIPBLAS_STATUS_SUCCESS && heuristic_count > 0 && heuristic.state == HIPBLAS_STATUS_SUCCESS) {
      algo = heuristic.algo;
      remember_matmul_algorithm(cache_key, algo);
    } else if (status == HIPBLAS_STATUS_SUCCESS || status == HIPBLAS_STATUS_NOT_SUPPORTED) {
      return HIPBLAS_STATUS_NOT_SUPPORTED;
    } else {
      return status;
    }
  }

  const int32_t alpha = 1;
  const int32_t beta = 0;
  return timed_hipblaslt_operation("hipblaslt_int8_i32_matmul", [&]() {
    return hipblasLtMatmul(
        handle,
        descriptors.op,
        &alpha,
        packed_b_t,
        descriptors.a,
        packed_a_t,
        descriptors.b,
        &beta,
        scratch,
        descriptors.c,
        scratch,
        descriptors.d,
        &algo,
        workspace,
        workspace_bytes,
        nullptr);
  });
}

rns8_status pack_transpose_centered(
    const int8_t* src,
    int8_t* dst,
    int src_rows,
    int src_cols,
    int src_ld,
    int dst_rows,
    int dst_cols,
    int dst_ld) {
  const hipError_t err = timed_hip_operation("hipblaslt_pack_transpose_centered", [&]() {
    const int code = rns8_hipblaslt_pack_transpose_centered_device(
        src, dst, src_rows, src_cols, src_ld, dst_rows, dst_cols, dst_ld);
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status run_reduce_block(
    int32_t* scratch,
    int8_t* residues,
    int64_t m,
    int64_t n,
    int64_t scratch_ld,
    int64_t ldc,
    uint16_t modulus,
    bool accumulate) {
  const hipError_t err = timed_hip_operation("hipblaslt_i32_to_residue_reduce", [&]() {
    const int code = rns8_hipblaslt_reduce_i32_to_centered_strided_device(
        scratch,
        residues,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(scratch_ld),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        accumulate ? 1 : 0);
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status gemm_one_plane(
    int device_id,
    hipblasLtHandle_t handle,
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int32_t* scratch,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
  int64_t k_offset = 0;
  bool accumulate = false;
  while (k_offset < k) {
    const int64_t k_remaining = k - k_offset;
    const int64_t k_block =
        std::min<int64_t>(k_remaining, static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK));
    padded_block_shape shape;
    if (!padded_block_shape_for(m, n, k_block, shape)) {
      return RNS8_RANGE_ERROR;
    }
    const std::size_t packed_a_bytes = static_cast<std::size_t>(shape.m) * static_cast<std::size_t>(shape.k);
    const std::size_t packed_b_bytes = static_cast<std::size_t>(shape.n) * static_cast<std::size_t>(shape.k);
    if (workspace_bytes < packed_a_bytes ||
        workspace_bytes - packed_a_bytes < packed_b_bytes ||
        workspace_bytes - packed_a_bytes - packed_b_bytes < kHipblasLtBaselineWorkspaceBytes) {
      return RNS8_INVALID_ARGUMENT;
    }
    auto* workspace_bytes_base = static_cast<std::byte*>(workspace);
    auto* packed_a_t = reinterpret_cast<int8_t*>(workspace_bytes_base);
    auto* packed_b_t = reinterpret_cast<int8_t*>(workspace_bytes_base + packed_a_bytes);
    void* library_workspace = workspace_bytes_base + packed_a_bytes + packed_b_bytes;
    const std::size_t library_workspace_bytes = workspace_bytes - packed_a_bytes - packed_b_bytes;
    rns8_status pack_status = pack_transpose_centered(
        a + k_offset,
        packed_a_t,
        static_cast<int>(m),
        static_cast<int>(k_block),
        static_cast<int>(lda),
        shape.k,
        shape.m,
        shape.k);
    if (pack_status != RNS8_SUCCESS) {
      return pack_status;
    }
    pack_status = pack_transpose_centered(
        b + k_offset * ldb,
        packed_b_t,
        static_cast<int>(k_block),
        static_cast<int>(n),
        static_cast<int>(ldb),
        shape.n,
        shape.k,
        shape.n);
    if (pack_status != RNS8_SUCCESS) {
      return pack_status;
    }
    const hipblasStatus_t matmul_status = run_matmul_block(
        device_id,
        handle,
        packed_a_t,
        packed_b_t,
        scratch,
        library_workspace,
        library_workspace_bytes,
        shape);
    const rns8_status mapped_status = status_from_hipblas(matmul_status);
    if (mapped_status != RNS8_SUCCESS) {
      return mapped_status;
    }
    const rns8_status reduce_status = run_reduce_block(scratch, c, m, n, shape.n, ldc, modulus, accumulate);
    if (reduce_status != RNS8_SUCCESS) {
      return reduce_status;
    }
    accumulate = true;
    k_offset += k_block;
  }
  return RNS8_SUCCESS;
}
#endif

}  // namespace

bool hipblaslt_compiled() {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  return true;
#else
  return false;
#endif
}

rns8_status hipblaslt_create_context(int device_id, rns8_device_info& out, void** handle, std::string& version) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if (!handle) {
    return RNS8_INVALID_ARGUMENT;
  }
  *handle = nullptr;
  version.clear();
  const rns8_status probe_status = hip_direct_probe(device_id, out);
  if (probe_status != RNS8_SUCCESS) {
    return probe_status;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  hipblasLtHandle_t hipblaslt_handle = nullptr;
  hipblasStatus_t status = hipblasLtCreate(&hipblaslt_handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status_from_hipblas(status);
  }
  int raw_version = 0;
  status = hipblasLtGetVersion(hipblaslt_handle, &raw_version);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    (void)hipblasLtDestroy(hipblaslt_handle);
    return status_from_hipblas(status);
  }
  out.backend = RNS8_BACKEND_HIPBLASLT;
  copy_c_string(out.detail, sizeof(out.detail), "hipBLASLt baseline accelerator detected");
  version = "hipBLASLt ";
  version += std::to_string(raw_version);
  *handle = hipblaslt_handle;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  (void)out;
  (void)handle;
  (void)version;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hipblaslt_destroy_context(int device_id, void* handle) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if (!handle) {
    return RNS8_SUCCESS;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const hipblasStatus_t status = hipblasLtDestroy(static_cast<hipblasLtHandle_t>(handle));
  return status_from_hipblas(status);
#else
  (void)device_id;
  (void)handle;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hipblaslt_gemm_rns_device(
    int device_id,
    void* handle,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* int32_scratch,
    std::size_t int32_scratch_bytes,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if (!handle || !device_a_residues || !device_b_residues || !device_c_residues || !int32_scratch ||
      !workspace || !checked_common_shape(m, n, k, lda, ldb, ldc) ||
      !checked_workspace(m, n, k, int32_scratch_bytes, workspace_bytes) ||
      prefix == 0 || prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const auto* a_base = static_cast<const int8_t*>(device_a_residues);
  const auto* b_base = static_cast<const int8_t*>(device_b_residues);
  auto* c_base = static_cast<int8_t*>(device_c_residues);
  auto* scratch = static_cast<int32_t*>(int32_scratch);
  const auto typed_handle = static_cast<hipblasLtHandle_t>(handle);
  for (uint32_t p = 0; p < prefix; ++p) {
    const std::size_t a_offset =
        static_cast<std::size_t>(p) * static_cast<std::size_t>(m) * static_cast<std::size_t>(lda);
    const std::size_t b_offset =
        static_cast<std::size_t>(p) * static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb);
    const std::size_t c_offset =
        static_cast<std::size_t>(p) * static_cast<std::size_t>(m) * static_cast<std::size_t>(ldc);
    const rns8_status status = gemm_one_plane(
        device_id,
        typed_handle,
        a_base + a_offset,
        b_base + b_offset,
        c_base + c_offset,
        scratch,
        workspace,
        workspace_bytes,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        kHipblasLtDefaultModuli[p]);
    if (status != RNS8_SUCCESS) {
      return status;
    }
  }
  const hipError_t sync_status = hipDeviceSynchronize();
  return sync_status == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)handle;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)int32_scratch;
  (void)int32_scratch_bytes;
  (void)workspace;
  (void)workspace_bytes;
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

rns8_status hipblaslt_gemm_finite_u8_device(
    int device_id,
    void* handle,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* int32_scratch,
    std::size_t int32_scratch_bytes,
    void* workspace,
    std::size_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if (!handle || !device_a_residues || !device_b_residues || !device_c_residues || !int32_scratch ||
      !workspace || !checked_common_shape(m, n, k, lda, ldb, ldc) ||
      !checked_workspace(m, n, k, int32_scratch_bytes, workspace_bytes) || modulus < 2 || modulus > 256) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_status device_status = set_hip_device(device_id);
  if (device_status != RNS8_SUCCESS) {
    return device_status;
  }
  const rns8_status status = gemm_one_plane(
      device_id,
      static_cast<hipblasLtHandle_t>(handle),
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      static_cast<int32_t*>(int32_scratch),
      workspace,
      workspace_bytes,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      modulus);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const hipError_t sync_status = hipDeviceSynchronize();
  return sync_status == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  (void)handle;
  (void)device_a_residues;
  (void)device_b_residues;
  (void)device_c_residues;
  (void)int32_scratch;
  (void)int32_scratch_bytes;
  (void)workspace;
  (void)workspace_bytes;
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

}  // namespace rns8::detail
