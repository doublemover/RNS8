namespace {

constexpr int kRns8DefaultModulusCount = 28;
constexpr int kRns8MaxSupportedPrefix = 28;
constexpr int kRns8WmmaTileM = 16;
constexpr int kRns8WmmaTileN = 16;
constexpr int kRns8WmmaTileK = 16;
#ifndef RNS8_ROCWMMA_WAVE_THREADS
#define RNS8_ROCWMMA_WAVE_THREADS 32
#endif
static_assert(RNS8_ROCWMMA_WAVE_THREADS == 32 || RNS8_ROCWMMA_WAVE_THREADS == 64);
constexpr int kRns8WmmaWaveThreads = RNS8_ROCWMMA_WAVE_THREADS;
constexpr int kRns8SignedSafeInt32KBlock = 65536;
constexpr int kRns8MaxInt = 2147483647;
constexpr uint32_t kRns8TileScheduleZeroOutput = 0x00000001u;
constexpr uint32_t kRns8TileScheduleZeroRowColProduct = 0x00000002u;
constexpr uint32_t kRns8SupportedTileScheduleFlags =
    kRns8TileScheduleZeroOutput | kRns8TileScheduleZeroRowColProduct;
constexpr uint64_t kReciprocalScale = 1ull << 32u;

constexpr uint16_t kDefaultModuli[kRns8DefaultModulusCount] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

enum rocwmma_status_code {
  rocwmma_success = 0,
  rocwmma_invalid_argument = 1,
  rocwmma_unsupported_backend = 2,
  rocwmma_backend_failure = 3,
  rocwmma_range_error = 4,
};

void prefix_label(char* buffer, std::size_t size, int prefix_index, const char* suffix) {
  std::snprintf(buffer, size, "rocwmma_prefix_%02d_%s", prefix_index, suffix);
}

using rns8::detail::timed_hip_backend_launch;

uint32_t modulus_reciprocal_u32(uint16_t modulus) {
  return static_cast<uint32_t>(kReciprocalScale / static_cast<uint32_t>(modulus));
}

bool checked_i32_shape(int64_t value) {
  return value > 0 && value <= static_cast<int64_t>(kRns8MaxInt);
}

bool rocwmma_workspace_requirements_u64(int64_t max_m, int64_t max_n, int64_t k, uint64_t* a_bytes,
                                        uint64_t* b_bytes) {
  rns8::detail::PackedGemmLayout layout;
  if (!a_bytes || !b_bytes || max_m <= 0 || max_n <= 0 || k <= 0 ||
      !rns8::detail::rocwmma_packed_gemm_layout(max_m, max_n, k, layout))
    return false;
  *a_bytes = layout.a_bytes;
  *b_bytes = layout.b_bytes;
  return true;
}

int zero_tile_residues(int8_t* c_base, int64_t m, int64_t ldc, int64_t row_offset,
                       int64_t col_offset, int64_t row_extent, int64_t col_extent,
                       uint32_t selected_prefix) {
  for (uint32_t p = 0; p < selected_prefix; ++p) {
    auto* tile_c =
        c_base + static_cast<uint64_t>(p) * static_cast<uint64_t>(m) * static_cast<uint64_t>(ldc) +
        static_cast<uint64_t>(row_offset) * static_cast<uint64_t>(ldc) +
        static_cast<uint64_t>(col_offset);
    const hipError_t status =
        timed_hip_backend_launch("rocwmma_zero_output_tile_memset", nullptr, [&]() {
          return hipMemset2D(tile_c, static_cast<std::size_t>(ldc) * sizeof(int8_t), 0,
                             static_cast<std::size_t>(col_extent) * sizeof(int8_t),
                             static_cast<std::size_t>(row_extent));
        });
    if (status != hipSuccess) {
      return rocwmma_backend_failure;
    }
  }
  return rocwmma_success;
}

__device__ uint32_t reduce_u32_small_modulus_device(uint32_t value, uint32_t modulus,
                                                    uint32_t reciprocal) {
  return rns8::detail::finite_u8::reduce_u32_small_modulus(value, modulus, reciprocal);
}

__host__ __device__ __forceinline__ bool rocwmma_static_modulus_supported(uint32_t modulus) {
  return modulus >= 2 && modulus <= 256;
}

__device__ int8_t reduce_to_centered_device(int32_t value, int modulus, uint32_t reciprocal) {
  return rns8::detail::finite_u8::reduce_to_centered_i32(value, static_cast<uint32_t>(modulus),
                                                         reciprocal);
}
