#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <exception>

#include <hip/hip_runtime.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

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

namespace rns8::detail {
bool hip_direct_timing_enabled();
void hip_direct_timing_flush_pending_events();
void hip_direct_timing_record_sample(const char* label, double microseconds);
void hip_direct_timing_record_pending_event_with_alias(
    const char* label,
    const char* alias,
    void* start_event,
    void* stop_event);
}  // namespace rns8::detail

namespace {

constexpr uint32_t kRns8AbiVersion = 1;
constexpr uint32_t kRns8DefaultModulusCount = 28;
constexpr uint32_t kRns8MaxSupportedPrefix = 28;
constexpr int64_t kRns8CanonicalSafeInt32KBlock = 32768;
constexpr int64_t kRns8CkShapeAlignment = 64;
constexpr uint16_t kDefaultModuli[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};
constexpr uint64_t kReciprocalScale = 1ull << 32u;

enum ck_status_code {
  ck_success = 0,
  ck_invalid_argument = 1,
  ck_unsupported_backend = 2,
  ck_backend_failure = 3,
  ck_range_error = 4,
};

void prefix_label(char* buffer, std::size_t size, int prefix_index, const char* suffix) {
  std::snprintf(buffer, size, "ck_prefix_%02d_%s", prefix_index, suffix);
}

void destroy_timing_event_pair(hipEvent_t start, hipEvent_t stop) {
  if (stop) {
    (void)hipEventDestroy(stop);
  }
  if (start) {
    (void)hipEventDestroy(start);
  }
}

bool create_recorded_timing_event_pair(hipEvent_t* start, hipEvent_t* stop) {
  *start = nullptr;
  *stop = nullptr;
  for (int attempt = 0; attempt < 8; ++attempt) {
    hipEvent_t candidate_start = nullptr;
    hipEvent_t candidate_stop = nullptr;
    hipError_t status = hipEventCreate(&candidate_start);
    if (status == hipSuccess) {
      status = hipEventCreate(&candidate_stop);
    }
    if (status == hipSuccess) {
      status = hipEventRecord(candidate_start, nullptr);
    }
    if (status == hipSuccess) {
      *start = candidate_start;
      *stop = candidate_stop;
      return true;
    }
    destroy_timing_event_pair(candidate_start, candidate_stop);
    if (attempt == 0) {
      rns8::detail::hip_direct_timing_flush_pending_events();
    }
    if (attempt + 1 < 8) {
      (void)hipDeviceSynchronize();
    }
  }
  return false;
}

void record_synchronized_host_timing(
    const char* label,
    const char* alias,
    std::chrono::steady_clock::time_point start) {
  (void)hipDeviceSynchronize();
  const auto stop = std::chrono::steady_clock::now();
  const double microseconds = std::chrono::duration<double, std::micro>(stop - start).count();
  rns8::detail::hip_direct_timing_record_sample(label, microseconds);
  if (alias && alias[0] != '\0') {
    rns8::detail::hip_direct_timing_record_sample(alias, microseconds);
  }
}

template <typename Fn>
hipError_t timed_hip_launch(const char* label, const char* alias, Fn&& fn) {
  if (!rns8::detail::hip_direct_timing_enabled()) {
    return fn();
  }
  const auto host_start = std::chrono::steady_clock::now();
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (!create_recorded_timing_event_pair(&start, &stop)) {
    const hipError_t status = fn();
    if (status == hipSuccess) {
      record_synchronized_host_timing(label, alias, host_start);
    }
    return status;
  }
  hipError_t status = fn();
  if (status == hipSuccess) {
    hipError_t stop_status = hipEventRecord(stop, nullptr);
    if (stop_status != hipSuccess) {
      rns8::detail::hip_direct_timing_flush_pending_events();
      (void)hipDeviceSynchronize();
      stop_status = hipEventRecord(stop, nullptr);
    }
    if (stop_status == hipSuccess) {
      hipError_t sync_status = hipEventSynchronize(stop);
      if (sync_status != hipSuccess) {
        (void)hipDeviceSynchronize();
        sync_status = hipEventSynchronize(stop);
      }
      if (sync_status == hipSuccess) {
        float milliseconds = 0.0f;
        hipError_t elapsed_status = hipEventElapsedTime(&milliseconds, start, stop);
        if (elapsed_status != hipSuccess) {
          (void)hipDeviceSynchronize();
          elapsed_status = hipEventElapsedTime(&milliseconds, start, stop);
        }
        if (elapsed_status == hipSuccess && milliseconds >= 0.0f) {
          const double microseconds = static_cast<double>(milliseconds) * 1000.0;
          rns8::detail::hip_direct_timing_record_sample(label, microseconds);
          if (alias && alias[0] != '\0') {
            rns8::detail::hip_direct_timing_record_sample(alias, microseconds);
          }
          destroy_timing_event_pair(start, stop);
          return status;
        }
      }
    }
    record_synchronized_host_timing(label, alias, host_start);
  }
  destroy_timing_event_pair(start, stop);
  return status;
}

template <typename Fn>
int timed_ck_operation(const char* label, const char* alias, Fn&& fn) {
  if (!rns8::detail::hip_direct_timing_enabled()) {
    return fn();
  }
  const auto host_start = std::chrono::steady_clock::now();
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (!create_recorded_timing_event_pair(&start, &stop)) {
    const int status = fn();
    if (status == ck_success) {
      record_synchronized_host_timing(label, alias, host_start);
    }
    return status;
  }
  const int status = fn();
  if (status == ck_success) {
    hipError_t stop_status = hipEventRecord(stop, nullptr);
    if (stop_status != hipSuccess) {
      rns8::detail::hip_direct_timing_flush_pending_events();
      (void)hipDeviceSynchronize();
      stop_status = hipEventRecord(stop, nullptr);
    }
    if (stop_status == hipSuccess) {
      hipError_t sync_status = hipEventSynchronize(stop);
      if (sync_status != hipSuccess) {
        (void)hipDeviceSynchronize();
        sync_status = hipEventSynchronize(stop);
      }
      if (sync_status == hipSuccess) {
        float milliseconds = 0.0f;
        hipError_t elapsed_status = hipEventElapsedTime(&milliseconds, start, stop);
        if (elapsed_status != hipSuccess) {
          (void)hipDeviceSynchronize();
          elapsed_status = hipEventElapsedTime(&milliseconds, start, stop);
        }
        if (elapsed_status == hipSuccess && milliseconds >= 0.0f) {
          const double microseconds = static_cast<double>(milliseconds) * 1000.0;
          rns8::detail::hip_direct_timing_record_sample(label, microseconds);
          if (alias && alias[0] != '\0') {
            rns8::detail::hip_direct_timing_record_sample(alias, microseconds);
          }
          destroy_timing_event_pair(start, stop);
          return status;
        }
      }
    }
    record_synchronized_host_timing(label, alias, host_start);
  }
  destroy_timing_event_pair(start, stop);
  return status;
}

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

struct CenteredModulo {
  int32_t modulus;
  uint32_t reciprocal;

  __host__ __device__ CenteredModulo(int32_t m = 253, uint32_t r = 0) : modulus(m), reciprocal(r) {}

  template <typename Y, typename X>
  __host__ __device__ void operator()(Y& y, const X& x) const {
    int32_t value = static_cast<int32_t>(x);
    const bool negative = value < 0;
    uint32_t magnitude =
        negative ? static_cast<uint32_t>(-static_cast<int64_t>(value)) : static_cast<uint32_t>(value);
    uint32_t quotient = static_cast<uint32_t>((static_cast<uint64_t>(magnitude) * reciprocal) >> 32u);
    uint32_t residue = magnitude - quotient * static_cast<uint32_t>(modulus);
    while (residue >= static_cast<uint32_t>(modulus)) {
      residue -= static_cast<uint32_t>(modulus);
    }
    int32_t centered = static_cast<int32_t>(residue);
    if (negative && centered != 0) {
      centered = modulus - centered;
    }
    const int32_t threshold = (modulus + 1) / 2;
    if (centered >= threshold) {
      centered -= modulus;
    }
    y = static_cast<Y>(centered);
  }
};

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    Row,
    Col,
    Row,
    int8_t,
    int8_t,
    int8_t,
    int32_t,
    int32_t,
    PassThrough,
    PassThrough,
    CenteredModulo,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    1,
    128,
    64,
    128,
    64,
    2,
    16,
    16,
    2,
    4,
    S<4, 32, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    2,
    2,
    true,
    S<4, 32, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    2,
    2,
    true,
    1,
    1,
    S<1, 32, 1, 4>,
    8>;

uint32_t modulus_reciprocal_u32(uint16_t modulus) {
  return static_cast<uint32_t>(kReciprocalScale / static_cast<uint32_t>(modulus));
}

bool checked_i32_shape(int64_t value) {
  return value > 0 && value <= static_cast<int64_t>(INT32_MAX);
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out) {
  if (a != 0 && b > UINT64_MAX / a) {
    return false;
  }
  *out = a * b;
  return true;
}

bool round_up_aligned_i64(int64_t value, int64_t alignment, int64_t* out) {
  if (value <= 0 || alignment <= 0) {
    return false;
  }
  const int64_t remainder = value % alignment;
  if (remainder == 0) {
    *out = value;
    return true;
  }
  if (value > INT64_MAX - (alignment - remainder)) {
    return false;
  }
  *out = value + (alignment - remainder);
  return true;
}

bool ck_workspace_requirements_u64(
    int64_t max_m,
    int64_t max_n,
    int64_t k,
    uint64_t* a_bytes,
    uint64_t* b_bytes,
    uint64_t* temp_bytes) {
  if (!checked_i32_shape(max_m) || !checked_i32_shape(max_n) || !checked_i32_shape(k)) {
    return false;
  }
  const uint64_t k_block = static_cast<uint64_t>(k) < static_cast<uint64_t>(kRns8CanonicalSafeInt32KBlock)
                               ? static_cast<uint64_t>(k)
                               : static_cast<uint64_t>(kRns8CanonicalSafeInt32KBlock);
  const uint64_t padded_k_block = k_block < 64 ? 64 : k_block;
  int64_t padded_m = 0;
  int64_t padded_n = 0;
  if (!round_up_aligned_i64(max_m, kRns8CkShapeAlignment, &padded_m) ||
      !round_up_aligned_i64(max_n, kRns8CkShapeAlignment, &padded_n)) {
    return false;
  }
  if (!checked_mul_u64(static_cast<uint64_t>(padded_m), padded_k_block, a_bytes)) {
    return false;
  }
  if (!checked_mul_u64(padded_k_block, static_cast<uint64_t>(padded_n), b_bytes)) {
    return false;
  }
  if (!checked_mul_u64(static_cast<uint64_t>(padded_m), static_cast<uint64_t>(padded_n), temp_bytes)) {
    return false;
  }
  return *a_bytes <= UINT64_MAX - *b_bytes && *a_bytes + *b_bytes <= UINT64_MAX - *temp_bytes;
}

__device__ int8_t center_i32_device(int32_t value, int modulus, uint32_t reciprocal) {
  const bool negative = value < 0;
  uint32_t magnitude =
      negative ? static_cast<uint32_t>(-static_cast<int64_t>(value)) : static_cast<uint32_t>(value);
  uint32_t quotient = static_cast<uint32_t>((static_cast<uint64_t>(magnitude) * reciprocal) >> 32u);
  uint32_t residue = magnitude - quotient * static_cast<uint32_t>(modulus);
  while (residue >= static_cast<uint32_t>(modulus)) {
    residue -= static_cast<uint32_t>(modulus);
  }
  int32_t centered = static_cast<int32_t>(residue);
  if (negative && centered != 0) {
    centered = modulus - centered;
  }
  const int threshold = (modulus + 1) / 2;
  if (centered >= threshold) {
    centered -= modulus;
  }
  return static_cast<int8_t>(centered);
}

__device__ int8_t canonical_byte_from_centered_device(int8_t residue, int modulus) {
  int value = static_cast<int>(residue);
  if (value < 0) {
    value += modulus;
  }
  return static_cast<int8_t>(value);
}

__global__ void pack_a_canonical_rowmajor_kernel(
    const int8_t* a,
    int8_t* packed,
    int rows,
    int padded_rows,
    int valid_k_block,
    int gemm_k_block,
    int lda,
    int k_offset,
    int modulus) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = padded_rows * gemm_k_block;
  if (index >= total) {
    return;
  }
  const int row = index / gemm_k_block;
  const int kk = index - row * gemm_k_block;
  packed[row * gemm_k_block + kk] =
      row < rows && kk < valid_k_block ? canonical_byte_from_centered_device(a[row * lda + k_offset + kk], modulus)
                                       : 0;
}

__global__ void pack_b_colmajor_kernel(
    const int8_t* b,
    int8_t* packed,
    int valid_k_block,
    int gemm_k_block,
    int n,
    int padded_n,
    int ldb,
    int k_offset,
    int modulus) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = gemm_k_block * padded_n;
  if (index >= total) {
    return;
  }
  const int kk = index % gemm_k_block;
  const int col = index / gemm_k_block;
  packed[col * gemm_k_block + kk] =
      col < n && kk < valid_k_block ? canonical_byte_from_centered_device(b[(k_offset + kk) * ldb + col], modulus)
                                    : 0;
}

__global__ void copy_centered_residue_kernel(
    const int8_t* src,
    int8_t* dst,
    int rows,
    int cols,
    int src_ld,
    int dst_ld) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }
  const int row = index / cols;
  const int col = index - row * cols;
  dst[row * dst_ld + col] = src[row * src_ld + col];
}

__global__ void add_centered_residue_kernel(
    const int8_t* rhs,
    int8_t* dst,
    int rows,
    int cols,
    int rhs_ld,
    int dst_ld,
    int modulus,
    uint32_t reciprocal) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }
  const int row = index / cols;
  const int col = index - row * cols;
  const int32_t sum = static_cast<int32_t>(dst[row * dst_ld + col]) + static_cast<int32_t>(rhs[row * rhs_ld + col]);
  dst[row * dst_ld + col] = center_i32_device(sum, modulus, reciprocal);
}

hipError_t launch_pack_a(
    const int8_t* a,
    int8_t* packed,
    int rows,
    int padded_rows,
    int valid_k_block,
    int gemm_k_block,
    int lda,
    int k_offset,
    int modulus) {
  const int total = padded_rows * gemm_k_block;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  hipLaunchKernelGGL(
      pack_a_canonical_rowmajor_kernel,
      dim3(grid),
      dim3(block),
      0,
      nullptr,
      a,
      packed,
      rows,
      padded_rows,
      valid_k_block,
      gemm_k_block,
      lda,
      k_offset,
      modulus);
  return hipGetLastError();
}

hipError_t launch_pack_b(
    const int8_t* b,
    int8_t* packed,
    int valid_k_block,
    int gemm_k_block,
    int n,
    int padded_n,
    int ldb,
    int k_offset,
    int modulus) {
  const int total = gemm_k_block * padded_n;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  hipLaunchKernelGGL(
      pack_b_colmajor_kernel,
      dim3(grid),
      dim3(block),
      0,
      nullptr,
      b,
      packed,
      valid_k_block,
      gemm_k_block,
      n,
      padded_n,
      ldb,
      k_offset,
      modulus);
  return hipGetLastError();
}

hipError_t launch_copy_centered(
    const int8_t* src,
    int8_t* dst,
    int rows,
    int cols,
    int src_ld,
    int dst_ld) {
  const int total = rows * cols;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  hipLaunchKernelGGL(copy_centered_residue_kernel, dim3(grid), dim3(block), 0, nullptr, src, dst, rows, cols, src_ld,
                     dst_ld);
  return hipGetLastError();
}

hipError_t launch_add_centered(
    const int8_t* rhs,
    int8_t* dst,
    int rows,
    int cols,
    int rhs_ld,
    int dst_ld,
    int modulus,
    uint32_t reciprocal) {
  const int total = rows * cols;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  hipLaunchKernelGGL(
      add_centered_residue_kernel, dim3(grid), dim3(block), 0, nullptr, rhs, dst, rows, cols, rhs_ld, dst_ld, modulus,
      reciprocal);
  return hipGetLastError();
}

int run_ck_tile_modulus(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* a_workspace,
    int8_t* b_workspace,
    int8_t* temp_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t reciprocal,
    int prefix_index) {
  char pack_a_label[64];
  char pack_b_label[64];
  char matmul_label[64];
  char copy_label[64];
  char add_label[64];
  const bool have_prefix_label = prefix_index >= 0;
  if (have_prefix_label) {
    prefix_label(pack_a_label, sizeof(pack_a_label), prefix_index, "pack_a");
    prefix_label(pack_b_label, sizeof(pack_b_label), prefix_index, "pack_b");
    prefix_label(matmul_label, sizeof(matmul_label), prefix_index, "matmul");
    prefix_label(copy_label, sizeof(copy_label), prefix_index, "copy_centered");
    prefix_label(add_label, sizeof(add_label), prefix_index, "add_centered");
  }
  const char* pack_a_timing_label = have_prefix_label ? pack_a_label : "ck_pack_a_kernel";
  const char* pack_b_timing_label = have_prefix_label ? pack_b_label : "ck_pack_b_kernel";
  const char* matmul_timing_label = have_prefix_label ? matmul_label : "ck_wmma_cshuffle_matmul";
  const char* copy_timing_label = have_prefix_label ? copy_label : "ck_copy_centered_kernel";
  const char* add_timing_label = have_prefix_label ? add_label : "ck_add_centered_kernel";
  const char* pack_a_alias = have_prefix_label ? "ck_pack_a_kernel" : nullptr;
  const char* pack_b_alias = have_prefix_label ? "ck_pack_b_kernel" : nullptr;
  const char* matmul_alias = have_prefix_label ? "ck_wmma_cshuffle_matmul" : nullptr;
  const char* copy_alias = have_prefix_label ? "ck_copy_centered_kernel" : nullptr;
  const char* add_alias = have_prefix_label ? "ck_add_centered_kernel" : nullptr;
  bool have_accumulated = false;
  int64_t padded_m_i64 = 0;
  int64_t padded_n_i64 = 0;
  if (!round_up_aligned_i64(m, kRns8CkShapeAlignment, &padded_m_i64) ||
      !round_up_aligned_i64(n, kRns8CkShapeAlignment, &padded_n_i64) ||
      !checked_i32_shape(padded_m_i64) || !checked_i32_shape(padded_n_i64) || !temp_c) {
    return ck_invalid_argument;
  }
  const int gemm_m = static_cast<int>(padded_m_i64);
  const int gemm_n = static_cast<int>(padded_n_i64);
  const bool padded_output = gemm_m != m || gemm_n != n;
  for (int64_t k_offset = 0; k_offset < k; k_offset += kRns8CanonicalSafeInt32KBlock) {
    const int valid_k_block = static_cast<int>(
        (static_cast<int64_t>(k) - k_offset) < kRns8CanonicalSafeInt32KBlock
            ? (static_cast<int64_t>(k) - k_offset)
            : kRns8CanonicalSafeInt32KBlock);
    const int gemm_k_block = valid_k_block < 64 ? 64 : valid_k_block;
    hipError_t err = timed_hip_launch(pack_a_timing_label, pack_a_alias, [&]() {
      return launch_pack_a(
          a, a_workspace, m, gemm_m, valid_k_block, gemm_k_block, lda, static_cast<int>(k_offset), modulus);
    });
    if (err != hipSuccess) {
      return ck_backend_failure;
    }
    err = timed_hip_launch(pack_b_timing_label, pack_b_alias, [&]() {
      return launch_pack_b(
          b, b_workspace, valid_k_block, gemm_k_block, n, gemm_n, ldb, static_cast<int>(k_offset), modulus);
    });
    if (err != hipSuccess) {
      return ck_backend_failure;
    }
    const bool write_workspace_output = padded_output || have_accumulated;
    int8_t* output = write_workspace_output ? temp_c : c;
    const int output_ld = write_workspace_output ? gemm_n : ldc;
    auto arg = DeviceGemmInstance::MakeArgument(
        a_workspace,
        b_workspace,
        output,
        gemm_m,
        gemm_n,
        gemm_k_block,
        gemm_k_block,
        gemm_k_block,
        output_ld,
        PassThrough{},
        PassThrough{},
        CenteredModulo{modulus, reciprocal});
    if (!DeviceGemmInstance::IsSupportedArgument(arg)) {
      return ck_unsupported_backend;
    }
    const int matmul_status = timed_ck_operation(matmul_timing_label, matmul_alias, [&]() {
      try {
        auto invoker = DeviceGemmInstance::MakeInvoker();
        (void)invoker.Run(arg);
      } catch (const std::exception&) {
        return ck_backend_failure;
      } catch (...) {
        return ck_backend_failure;
      }
      return ck_success;
    });
    if (matmul_status != ck_success) {
      return ck_backend_failure;
    }
    if (write_workspace_output) {
      if (have_accumulated) {
        err = timed_hip_launch(add_timing_label, add_alias, [&]() {
          return launch_add_centered(temp_c, c, m, n, gemm_n, ldc, modulus, reciprocal);
        });
      } else {
        err = timed_hip_launch(copy_timing_label, copy_alias, [&]() {
          return launch_copy_centered(temp_c, c, m, n, gemm_n, ldc);
        });
      }
      if (err != hipSuccess) {
        return ck_backend_failure;
      }
    }
    have_accumulated = true;
  }
  return ck_success;
}

bool checked_tile_entry(const rns8_plan_tile_schedule_entry& entry, int64_t rows, int64_t cols) {
  if (entry.struct_size != sizeof(rns8_plan_tile_schedule_entry) || entry.abi_version != kRns8AbiVersion ||
      entry.flags != 0 || entry.row_offset < 0 || entry.col_offset < 0 || entry.row_extent <= 0 ||
      entry.col_extent <= 0 || entry.required_prefix == 0 || entry.selected_prefix == 0 ||
      entry.required_prefix > entry.selected_prefix || entry.selected_prefix > kRns8MaxSupportedPrefix ||
      entry.row_offset > rows || entry.col_offset > cols || entry.row_extent > rows - entry.row_offset ||
      entry.col_extent > cols - entry.col_offset) {
    return false;
  }
  return true;
}

int run_ck_entry(
    const int8_t* a_base,
    const int8_t* b_base,
    int8_t* c_base,
    int8_t* a_workspace,
    int8_t* b_workspace,
    int8_t* temp_c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t row_offset,
    int64_t col_offset,
    int64_t row_extent,
    int64_t col_extent,
    uint32_t selected_prefix) {
  if (!checked_i32_shape(row_extent) || !checked_i32_shape(col_extent) || !checked_i32_shape(k) ||
      !checked_i32_shape(lda) || !checked_i32_shape(ldb) || !checked_i32_shape(ldc) ||
      selected_prefix == 0 || selected_prefix > kRns8MaxSupportedPrefix) {
    return ck_invalid_argument;
  }
  for (uint32_t p = 0; p < selected_prefix; ++p) {
    const uint16_t modulus = kDefaultModuli[p];
    const uint32_t reciprocal = modulus_reciprocal_u32(modulus);
    const uint64_t a_plane_offset = static_cast<uint64_t>(p) * static_cast<uint64_t>(m) * static_cast<uint64_t>(lda);
    const uint64_t b_plane_offset = static_cast<uint64_t>(p) * static_cast<uint64_t>(k) * static_cast<uint64_t>(ldb);
    const uint64_t c_plane_offset = static_cast<uint64_t>(p) * static_cast<uint64_t>(m) * static_cast<uint64_t>(ldc);
    const int result = run_ck_tile_modulus(
        a_base + a_plane_offset + static_cast<uint64_t>(row_offset) * static_cast<uint64_t>(lda),
        b_base + b_plane_offset + static_cast<uint64_t>(col_offset),
        c_base + c_plane_offset + static_cast<uint64_t>(row_offset) * static_cast<uint64_t>(ldc) +
            static_cast<uint64_t>(col_offset),
        a_workspace,
        b_workspace,
        temp_c,
        static_cast<int>(row_extent),
        static_cast<int>(col_extent),
        static_cast<int>(k),
        static_cast<int>(lda),
        static_cast<int>(ldb),
        static_cast<int>(ldc),
        static_cast<int>(modulus),
        reciprocal,
        static_cast<int>(p));
    if (result != ck_success) {
      return result;
    }
  }
  return ck_success;
}

int run_ck_finite(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* a_workspace,
    int8_t* b_workspace,
    int8_t* temp_c,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus) {
  if (modulus < 2 || modulus > 256) {
    return ck_invalid_argument;
  }
  return run_ck_tile_modulus(
      a,
      b,
      c,
      a_workspace,
      b_workspace,
      temp_c,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      static_cast<int>(modulus),
      modulus_reciprocal_u32(modulus),
      -1);
}

int checked_common(
    int device_id,
    const void* a,
    const void* b,
    void* c,
    void* workspace,
    uint64_t workspace_bytes,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_m,
    int64_t max_n,
    int8_t** a_workspace,
    int8_t** b_workspace,
    int8_t** temp_c) {
  if (!a || !b || !c || !workspace || !checked_i32_shape(m) || !checked_i32_shape(n) || !checked_i32_shape(k) ||
      !checked_i32_shape(lda) || !checked_i32_shape(ldb) || !checked_i32_shape(ldc) || lda < k || ldb < n ||
      ldc < n) {
    return ck_invalid_argument;
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  uint64_t temp_bytes = 0;
  if (!ck_workspace_requirements_u64(max_m, max_n, k, &a_bytes, &b_bytes, &temp_bytes) ||
      a_bytes > UINT64_MAX - b_bytes || a_bytes + b_bytes > UINT64_MAX - temp_bytes ||
      workspace_bytes < a_bytes + b_bytes + temp_bytes) {
    return ck_invalid_argument;
  }
  hipError_t err = hipSetDevice(device_id < 0 ? 0 : device_id);
  if (err != hipSuccess) {
    return ck_backend_failure;
  }
  *a_workspace = static_cast<int8_t*>(workspace);
  *b_workspace = static_cast<int8_t*>(workspace) + a_bytes;
  *temp_c = temp_bytes == 0 ? nullptr : static_cast<int8_t*>(workspace) + a_bytes + b_bytes;
  return ck_success;
}

}  // namespace

extern "C" int rns8_ck_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    unsigned long long workspace_bytes,
    long long m,
    long long n,
    long long k,
    long long lda,
    long long ldb,
    long long ldc,
    unsigned int prefix) {
  if (prefix == 0 || prefix > kRns8MaxSupportedPrefix) {
    return ck_invalid_argument;
  }
  int8_t* a_workspace = nullptr;
  int8_t* b_workspace = nullptr;
  int8_t* temp_c = nullptr;
  int status = checked_common(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      workspace,
      workspace_bytes,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      m,
      n,
      &a_workspace,
      &b_workspace,
      &temp_c);
  if (status != ck_success) {
    return status;
  }
  status = run_ck_entry(
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      a_workspace,
      b_workspace,
      temp_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      0,
      0,
      m,
      n,
      prefix);
  if (status != ck_success) {
    return status;
  }
  return hipDeviceSynchronize() == hipSuccess ? ck_success : ck_backend_failure;
}

extern "C" int rns8_ck_gemm_rns_tiled_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    unsigned long long workspace_bytes,
    long long m,
    long long n,
    long long k,
    long long lda,
    long long ldb,
    long long ldc,
    const rns8_plan_tile_schedule_entry* entries,
    unsigned long long entry_count) {
  if (!entries || entry_count == 0 || entry_count > static_cast<unsigned long long>(INT32_MAX)) {
    return ck_invalid_argument;
  }
  int64_t max_m = 0;
  int64_t max_n = 0;
  for (uint64_t i = 0; i < entry_count; ++i) {
    if (!checked_tile_entry(entries[i], m, n)) {
      return ck_invalid_argument;
    }
    if (entries[i].row_extent > max_m) {
      max_m = entries[i].row_extent;
    }
    if (entries[i].col_extent > max_n) {
      max_n = entries[i].col_extent;
    }
  }
  int8_t* a_workspace = nullptr;
  int8_t* b_workspace = nullptr;
  int8_t* temp_c = nullptr;
  int status = checked_common(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      workspace,
      workspace_bytes,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      max_m,
      max_n,
      &a_workspace,
      &b_workspace,
      &temp_c);
  if (status != ck_success) {
    return status;
  }
  for (uint64_t i = 0; i < entry_count; ++i) {
    const auto& entry = entries[i];
    status = run_ck_entry(
        static_cast<const int8_t*>(device_a_residues),
        static_cast<const int8_t*>(device_b_residues),
        static_cast<int8_t*>(device_c_residues),
        a_workspace,
        b_workspace,
        temp_c,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        entry.row_offset,
        entry.col_offset,
        entry.row_extent,
        entry.col_extent,
        entry.selected_prefix);
    if (status != ck_success) {
      return status;
    }
  }
  return hipDeviceSynchronize() == hipSuccess ? ck_success : ck_backend_failure;
}

extern "C" int rns8_ck_gemm_finite_u8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    void* workspace,
    unsigned long long workspace_bytes,
    long long m,
    long long n,
    long long k,
    long long lda,
    long long ldb,
    long long ldc,
    unsigned int modulus) {
  int8_t* a_workspace = nullptr;
  int8_t* b_workspace = nullptr;
  int8_t* temp_c = nullptr;
  int status = checked_common(
      device_id,
      device_a_residues,
      device_b_residues,
      device_c_residues,
      workspace,
      workspace_bytes,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      m,
      n,
      &a_workspace,
      &b_workspace,
      &temp_c);
  if (status != ck_success) {
    return status;
  }
  status = run_ck_finite(
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      a_workspace,
      b_workspace,
      temp_c,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      static_cast<uint16_t>(modulus));
  if (status != ck_success) {
    return status;
  }
  return hipDeviceSynchronize() == hipSuccess ? ck_success : ck_backend_failure;
}
