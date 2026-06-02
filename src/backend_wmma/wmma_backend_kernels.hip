#include <cstdint>

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

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

namespace {

constexpr int kRns8DefaultModulusCount = 28;
constexpr int kRns8MaxSupportedPrefix = 28;
constexpr int kRns8WmmaTileM = 16;
constexpr int kRns8WmmaTileN = 16;
constexpr int kRns8WmmaTileK = 16;
constexpr int kRns8WmmaWaveThreads = 32;
constexpr int kRns8SignedSafeInt32KBlock = 65536;
constexpr int kRns8MaxInt = 2147483647;
constexpr uint64_t kReciprocalScale = 1ull << 32u;

constexpr uint16_t kDefaultModuli[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

enum wmma_status_code {
  wmma_success = 0,
  wmma_invalid_argument = 1,
  wmma_unsupported_backend = 2,
  wmma_backend_failure = 3,
  wmma_range_error = 4,
};

uint32_t modulus_reciprocal_u32(uint16_t modulus) {
  return static_cast<uint32_t>(kReciprocalScale / static_cast<uint32_t>(modulus));
}

bool checked_i32_shape(int64_t value) {
  return value > 0 && value <= static_cast<int64_t>(kRns8MaxInt);
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

bool wmma_workspace_requirements_u64(
    int64_t max_m,
    int64_t max_n,
    int64_t k,
    uint64_t* a_bytes,
    uint64_t* b_bytes) {
  if (!checked_i32_shape(max_m) || !checked_i32_shape(max_n) || !checked_i32_shape(k)) {
    return false;
  }
  int64_t padded_m = 0;
  int64_t padded_n = 0;
  if (!round_up_aligned_i64(max_m, kRns8WmmaTileM, &padded_m) ||
      !round_up_aligned_i64(max_n, kRns8WmmaTileN, &padded_n)) {
    return false;
  }
  int64_t k_block = k < kRns8SignedSafeInt32KBlock ? k : kRns8SignedSafeInt32KBlock;
  if (k_block < kRns8WmmaTileK) {
    k_block = kRns8WmmaTileK;
  }
  if (!round_up_aligned_i64(k_block, kRns8WmmaTileK, &k_block)) {
    return false;
  }
  if (!checked_mul_u64(static_cast<uint64_t>(padded_m), static_cast<uint64_t>(k_block), a_bytes)) {
    return false;
  }
  if (!checked_mul_u64(static_cast<uint64_t>(k_block), static_cast<uint64_t>(padded_n), b_bytes)) {
    return false;
  }
  return *a_bytes <= UINT64_MAX - *b_bytes;
}

__device__ uint32_t reduce_u32_small_modulus_device(uint32_t value, uint32_t modulus, uint32_t reciprocal) {
  const uint32_t quotient = static_cast<uint32_t>((static_cast<uint64_t>(value) * reciprocal) >> 32u);
  uint32_t residue = value - quotient * modulus;
  residue -= modulus & (0u - static_cast<uint32_t>(residue >= modulus));
  residue -= modulus & (0u - static_cast<uint32_t>(residue >= modulus));
  return residue;
}

__device__ int8_t reduce_to_centered_device(int32_t value, int modulus, uint32_t reciprocal) {
  const uint32_t unsigned_modulus = static_cast<uint32_t>(modulus);
  const uint32_t magnitude =
      value < 0 ? static_cast<uint32_t>(-static_cast<int64_t>(value)) : static_cast<uint32_t>(value);
  uint32_t residue = reduce_u32_small_modulus_device(magnitude, unsigned_modulus, reciprocal);
  const uint32_t negative_nonzero =
      static_cast<uint32_t>(value < 0) & static_cast<uint32_t>(residue != 0);
  const uint32_t negative_mask = 0u - negative_nonzero;
  residue = (residue & ~negative_mask) | ((unsigned_modulus - residue) & negative_mask);
  const uint32_t threshold = (unsigned_modulus + 1u) / 2u;
  residue -= unsigned_modulus & (0u - static_cast<uint32_t>(residue >= threshold));
  return static_cast<int8_t>(static_cast<int32_t>(residue));
}

__global__ void pack_a_rowmajor_kernel(
    const int8_t* a,
    int8_t* packed,
    int rows,
    int padded_rows,
    int valid_k_block,
    int gemm_k_block,
    int lda,
    int k_offset) {
  const int kk = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (row >= padded_rows || kk >= gemm_k_block) {
    return;
  }
  packed[row * gemm_k_block + kk] =
      row < rows && kk < valid_k_block ? a[row * lda + k_offset + kk] : static_cast<int8_t>(0);
}

__global__ void pack_b_colmajor_kernel(
    const int8_t* b,
    int8_t* packed,
    int valid_k_block,
    int gemm_k_block,
    int n,
    int padded_n,
    int ldb,
    int k_offset) {
  const int kk = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int col = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (col >= padded_n || kk >= gemm_k_block) {
    return;
  }
  packed[col * gemm_k_block + kk] =
      col < n && kk < valid_k_block ? b[(k_offset + kk) * ldb + col] : static_cast<int8_t>(0);
}

__global__ void __launch_bounds__(kRns8WmmaWaveThreads) rocwmma_i8_residue_gemm_kernel(
    const int8_t* a_pack,
    const int8_t* b_pack,
    int8_t* c,
    int rows,
    int cols,
    int padded_rows,
    int padded_cols,
    int gemm_k_block,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int accumulate) {
  using namespace rocwmma;
  using FragA = fragment<matrix_a, kRns8WmmaTileM, kRns8WmmaTileN, kRns8WmmaTileK, int8_t, row_major>;
  using FragB = fragment<matrix_b, kRns8WmmaTileM, kRns8WmmaTileN, kRns8WmmaTileK, int8_t, col_major>;
  using FragAcc = fragment<accumulator, kRns8WmmaTileM, kRns8WmmaTileN, kRns8WmmaTileK, int32_t>;

  __shared__ int32_t tile_acc[kRns8WmmaTileM * kRns8WmmaTileN];

  const int tile_col = static_cast<int>(blockIdx.x) * kRns8WmmaTileN;
  const int tile_row = static_cast<int>(blockIdx.y) * kRns8WmmaTileM;
  if (tile_row >= padded_rows || tile_col >= padded_cols) {
    return;
  }

  FragA frag_a;
  FragB frag_b;
  FragAcc acc;
  fill_fragment(acc, 0);

  for (int kk = 0; kk < gemm_k_block; kk += kRns8WmmaTileK) {
    load_matrix_sync(frag_a, a_pack + tile_row * gemm_k_block + kk, gemm_k_block);
    load_matrix_sync(frag_b, b_pack + tile_col * gemm_k_block + kk, gemm_k_block);
    mma_sync(acc, frag_a, frag_b, acc);
  }

  store_matrix_sync(tile_acc, acc, kRns8WmmaTileN, mem_row_major);
  __syncthreads();

  const int lane = static_cast<int>(threadIdx.x);
  for (int index = lane; index < kRns8WmmaTileM * kRns8WmmaTileN; index += kRns8WmmaWaveThreads) {
    const int local_row = index / kRns8WmmaTileN;
    const int local_col = index - local_row * kRns8WmmaTileN;
    const int row = tile_row + local_row;
    const int col = tile_col + local_col;
    if (row < rows && col < cols) {
      int32_t value = tile_acc[index];
      if (accumulate) {
        value += static_cast<int32_t>(c[row * ldc + col]);
      }
      c[row * ldc + col] = reduce_to_centered_device(value, modulus, modulus_reciprocal);
    }
  }
}

int launch_pack_and_gemm(
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* a_pack,
    int8_t* b_pack,
    int rows,
    int cols,
    int k,
    int lda,
    int ldb,
    int ldc,
    int modulus,
    uint32_t reciprocal,
    int safe_k_block) {
  int64_t padded_rows64 = 0;
  int64_t padded_cols64 = 0;
  if (!round_up_aligned_i64(rows, kRns8WmmaTileM, &padded_rows64) ||
      !round_up_aligned_i64(cols, kRns8WmmaTileN, &padded_cols64)) {
    return wmma_range_error;
  }
  const int padded_rows = static_cast<int>(padded_rows64);
  const int padded_cols = static_cast<int>(padded_cols64);
  int accumulate = 0;
  for (int k_offset = 0; k_offset < k; k_offset += safe_k_block) {
    int valid_k_block = k - k_offset < safe_k_block ? k - k_offset : safe_k_block;
    int64_t gemm_k_block64 = valid_k_block;
    if (gemm_k_block64 < kRns8WmmaTileK) {
      gemm_k_block64 = kRns8WmmaTileK;
    }
    if (!round_up_aligned_i64(gemm_k_block64, kRns8WmmaTileK, &gemm_k_block64)) {
      return wmma_range_error;
    }
    const int gemm_k_block = static_cast<int>(gemm_k_block64);
    const dim3 pack_block(16, 16, 1);
    const dim3 pack_a_grid(
        static_cast<unsigned int>((gemm_k_block + pack_block.x - 1) / pack_block.x),
        static_cast<unsigned int>((padded_rows + pack_block.y - 1) / pack_block.y),
        1);
    const dim3 pack_b_grid(
        static_cast<unsigned int>((gemm_k_block + pack_block.x - 1) / pack_block.x),
        static_cast<unsigned int>((padded_cols + pack_block.y - 1) / pack_block.y),
        1);
    pack_a_rowmajor_kernel<<<pack_a_grid, pack_block>>>(
        a, a_pack, rows, padded_rows, valid_k_block, gemm_k_block, lda, k_offset);
    if (hipGetLastError() != hipSuccess) {
      return wmma_backend_failure;
    }
    pack_b_colmajor_kernel<<<pack_b_grid, pack_block>>>(
        b, b_pack, valid_k_block, gemm_k_block, cols, padded_cols, ldb, k_offset);
    if (hipGetLastError() != hipSuccess) {
      return wmma_backend_failure;
    }
    const dim3 grid(
        static_cast<unsigned int>(padded_cols / kRns8WmmaTileN),
        static_cast<unsigned int>(padded_rows / kRns8WmmaTileM),
        1);
    rocwmma_i8_residue_gemm_kernel<<<grid, kRns8WmmaWaveThreads>>>(
        a_pack,
        b_pack,
        c,
        rows,
        cols,
        padded_rows,
        padded_cols,
        gemm_k_block,
        ldc,
        modulus,
        reciprocal,
        accumulate);
    if (hipGetLastError() != hipSuccess) {
      return wmma_backend_failure;
    }
    accumulate = 1;
  }
  return wmma_success;
}

int run_rns_prefix(
    int device_id,
    const int8_t* a,
    const int8_t* b,
    int8_t* c,
    int8_t* workspace,
    uint64_t workspace_bytes,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    uint32_t prefix) {
  if (prefix == 0 || prefix > kRns8MaxSupportedPrefix) {
    return wmma_invalid_argument;
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  if (!wmma_workspace_requirements_u64(m, n, k, &a_bytes, &b_bytes)) {
    return wmma_range_error;
  }
  if (!workspace || workspace_bytes < a_bytes + b_bytes) {
    return wmma_invalid_argument;
  }
  auto* a_pack = workspace;
  auto* b_pack = workspace + a_bytes;
  const int safe_k_block = k < kRns8SignedSafeInt32KBlock ? k : kRns8SignedSafeInt32KBlock;
  for (uint32_t modulus_index = 0; modulus_index < prefix; ++modulus_index) {
    const uint16_t modulus = kDefaultModuli[modulus_index];
    const auto plane_a = a + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * lda;
    const auto plane_b = b + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k) * ldb;
    auto plane_c = c + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * ldc;
    const int status = launch_pack_and_gemm(
        plane_a,
        plane_b,
        plane_c,
        a_pack,
        b_pack,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        static_cast<int>(modulus),
        modulus_reciprocal_u32(modulus),
        safe_k_block);
    if (status != wmma_success) {
      return status;
    }
  }
  return hipSetDevice(device_id) == hipSuccess && hipDeviceSynchronize() == hipSuccess ? wmma_success
                                                                                       : wmma_backend_failure;
}

}  // namespace

extern "C" int rns8_wmma_gemm_rns_device(
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
  if (hipSetDevice(device_id) != hipSuccess) {
    return wmma_backend_failure;
  }
  if (!device_a_residues || !device_b_residues || !device_c_residues || m <= 0 || n <= 0 || k <= 0 ||
      lda < k || ldb < n || ldc < n || !checked_i32_shape(m) || !checked_i32_shape(n) || !checked_i32_shape(k) ||
      !checked_i32_shape(lda) || !checked_i32_shape(ldb) || !checked_i32_shape(ldc)) {
    return wmma_invalid_argument;
  }
  return run_rns_prefix(
      device_id,
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      static_cast<int8_t*>(workspace),
      workspace_bytes,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      prefix);
}

extern "C" int rns8_wmma_gemm_rns_tiled_device(
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
  if (hipSetDevice(device_id) != hipSuccess) {
    return wmma_backend_failure;
  }
  if (!device_a_residues || !device_b_residues || !device_c_residues || !entries || entry_count == 0 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || !checked_i32_shape(m) ||
      !checked_i32_shape(n) || !checked_i32_shape(k) || !checked_i32_shape(lda) || !checked_i32_shape(ldb) ||
      !checked_i32_shape(ldc)) {
    return wmma_invalid_argument;
  }
  auto* a_pack = static_cast<int8_t*>(workspace);
  const auto* a = static_cast<const int8_t*>(device_a_residues);
  const auto* b = static_cast<const int8_t*>(device_b_residues);
  auto* c = static_cast<int8_t*>(device_c_residues);
  int max_rows = 0;
  int max_cols = 0;
  for (unsigned long long entry_index = 0; entry_index < entry_count; ++entry_index) {
    const auto& entry = entries[entry_index];
    if (entry.row_extent <= 0 || entry.col_extent <= 0 || entry.row_offset < 0 || entry.col_offset < 0 ||
        entry.row_offset + entry.row_extent > m || entry.col_offset + entry.col_extent > n ||
        entry.selected_prefix == 0 || entry.selected_prefix > kRns8MaxSupportedPrefix) {
      return wmma_invalid_argument;
    }
    if (entry.row_extent > max_rows) {
      max_rows = static_cast<int>(entry.row_extent);
    }
    if (entry.col_extent > max_cols) {
      max_cols = static_cast<int>(entry.col_extent);
    }
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  if (!wmma_workspace_requirements_u64(max_rows, max_cols, k, &a_bytes, &b_bytes)) {
    return wmma_range_error;
  }
  if (!workspace || workspace_bytes < a_bytes + b_bytes) {
    return wmma_invalid_argument;
  }
  auto* b_pack = a_pack + a_bytes;
  const int safe_k_block = k < kRns8SignedSafeInt32KBlock ? static_cast<int>(k) : kRns8SignedSafeInt32KBlock;
  for (unsigned long long entry_index = 0; entry_index < entry_count; ++entry_index) {
    const auto& entry = entries[entry_index];
    for (uint32_t modulus_index = 0; modulus_index < entry.selected_prefix; ++modulus_index) {
      const uint16_t modulus = kDefaultModuli[modulus_index];
      const auto plane_a = a + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * lda +
                           static_cast<int64_t>(entry.row_offset) * lda;
      const auto plane_b = b + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(k) * ldb +
                           static_cast<int64_t>(entry.col_offset);
      auto plane_c = c + static_cast<int64_t>(modulus_index) * static_cast<int64_t>(m) * ldc +
                     static_cast<int64_t>(entry.row_offset) * ldc + static_cast<int64_t>(entry.col_offset);
      const int status = launch_pack_and_gemm(
          plane_a,
          plane_b,
          plane_c,
          a_pack,
          b_pack,
          static_cast<int>(entry.row_extent),
          static_cast<int>(entry.col_extent),
          static_cast<int>(k),
          static_cast<int>(lda),
          static_cast<int>(ldb),
          static_cast<int>(ldc),
          static_cast<int>(modulus),
          modulus_reciprocal_u32(modulus),
          safe_k_block);
      if (status != wmma_success) {
        return status;
      }
    }
  }
  return hipDeviceSynchronize() == hipSuccess ? wmma_success : wmma_backend_failure;
}

extern "C" int rns8_wmma_gemm_finite_u8_device(
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
  if (hipSetDevice(device_id) != hipSuccess) {
    return wmma_backend_failure;
  }
  if (!device_a_residues || !device_b_residues || !device_c_residues || modulus < 2 || modulus > 256 ||
      m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n || !checked_i32_shape(m) ||
      !checked_i32_shape(n) || !checked_i32_shape(k) || !checked_i32_shape(lda) || !checked_i32_shape(ldb) ||
      !checked_i32_shape(ldc)) {
    return wmma_invalid_argument;
  }
  uint64_t a_bytes = 0;
  uint64_t b_bytes = 0;
  if (!wmma_workspace_requirements_u64(m, n, k, &a_bytes, &b_bytes)) {
    return wmma_range_error;
  }
  if (!workspace || workspace_bytes < a_bytes + b_bytes) {
    return wmma_invalid_argument;
  }
  auto* a_pack = static_cast<int8_t*>(workspace);
  auto* b_pack = a_pack + a_bytes;
  const int safe_k_block = k < kRns8SignedSafeInt32KBlock ? static_cast<int>(k) : kRns8SignedSafeInt32KBlock;
  const int status = launch_pack_and_gemm(
      static_cast<const int8_t*>(device_a_residues),
      static_cast<const int8_t*>(device_b_residues),
      static_cast<int8_t*>(device_c_residues),
      a_pack,
      b_pack,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(ldc),
      static_cast<int>(modulus),
      modulus_reciprocal_u32(static_cast<uint16_t>(modulus)),
      safe_k_block);
  if (status != wmma_success) {
    return status;
  }
  return hipDeviceSynchronize() == hipSuccess ? wmma_success : wmma_backend_failure;
}
