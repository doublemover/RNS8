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

class timing_event_pair {
 public:
  timing_event_pair() = default;
  timing_event_pair(const timing_event_pair&) = delete;
  timing_event_pair& operator=(const timing_event_pair&) = delete;
  ~timing_event_pair() {
    reset();
  }

  bool create_and_record_start() {
    reset();
    for (int attempt = 0; attempt < 8; ++attempt) {
      hipError_t status = hipEventCreate(&start_);
      if (status == hipSuccess) {
        status = hipEventCreate(&stop_);
      }
      if (status == hipSuccess) {
        status = hipEventRecord(start_, nullptr);
      }
      if (status == hipSuccess) {
        return true;
      }
      reset();
      if (attempt == 0) {
        rns8::detail::hip_direct_timing_flush_pending_events();
      }
      if (attempt + 1 < 8) {
        (void)hipDeviceSynchronize();
      }
    }
    return false;
  }

  hipEvent_t start() const {
    return start_;
  }

  hipEvent_t stop() const {
    return stop_;
  }

 private:
  void reset() {
    if (stop_) {
      (void)hipEventDestroy(stop_);
      stop_ = nullptr;
    }
    if (start_) {
      (void)hipEventDestroy(start_);
      start_ = nullptr;
    }
  }

  hipEvent_t start_ = nullptr;
  hipEvent_t stop_ = nullptr;
};

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
  timing_event_pair events;
  if (!events.create_and_record_start()) {
    const hipError_t status = fn();
    if (status == hipSuccess) {
      record_synchronized_host_timing(label, alias, host_start);
    }
    return status;
  }
  hipError_t status = fn();
  if (status == hipSuccess) {
    hipError_t stop_status = hipEventRecord(events.stop(), nullptr);
    if (stop_status != hipSuccess) {
      rns8::detail::hip_direct_timing_flush_pending_events();
      (void)hipDeviceSynchronize();
      stop_status = hipEventRecord(events.stop(), nullptr);
    }
    if (stop_status == hipSuccess) {
      hipError_t sync_status = hipEventSynchronize(events.stop());
      if (sync_status != hipSuccess) {
        (void)hipDeviceSynchronize();
        sync_status = hipEventSynchronize(events.stop());
      }
      if (sync_status == hipSuccess) {
        float milliseconds = 0.0f;
        hipError_t elapsed_status = hipEventElapsedTime(&milliseconds, events.start(), events.stop());
        if (elapsed_status != hipSuccess) {
          (void)hipDeviceSynchronize();
          elapsed_status = hipEventElapsedTime(&milliseconds, events.start(), events.stop());
        }
        if (elapsed_status == hipSuccess && milliseconds >= 0.0f) {
          const double microseconds = static_cast<double>(milliseconds) * 1000.0;
          rns8::detail::hip_direct_timing_record_sample(label, microseconds);
          if (alias && alias[0] != '\0') {
            rns8::detail::hip_direct_timing_record_sample(alias, microseconds);
          }
          return status;
        }
      }
    }
    record_synchronized_host_timing(label, alias, host_start);
  }
  return status;
}

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

bool rocwmma_workspace_requirements_u64(
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

int zero_tile_residues(
    int8_t* c_base,
    int64_t m,
    int64_t ldc,
    int64_t row_offset,
    int64_t col_offset,
    int64_t row_extent,
    int64_t col_extent,
    uint32_t selected_prefix) {
  for (uint32_t p = 0; p < selected_prefix; ++p) {
    auto* tile_c = c_base + static_cast<uint64_t>(p) * static_cast<uint64_t>(m) * static_cast<uint64_t>(ldc) +
                   static_cast<uint64_t>(row_offset) * static_cast<uint64_t>(ldc) +
                   static_cast<uint64_t>(col_offset);
    const hipError_t status = timed_hip_launch("rocwmma_zero_output_tile_memset", nullptr, [&]() {
      return hipMemset2D(
          tile_c,
          static_cast<std::size_t>(ldc) * sizeof(int8_t),
          0,
          static_cast<std::size_t>(col_extent) * sizeof(int8_t),
          static_cast<std::size_t>(row_extent));
    });
    if (status != hipSuccess) {
      return rocwmma_backend_failure;
    }
  }
  return rocwmma_success;
}

__device__ uint32_t reduce_u32_small_modulus_device(uint32_t value, uint32_t modulus, uint32_t reciprocal) {
  return rns8::detail::finite_u8::reduce_u32_small_modulus(value, modulus, reciprocal);
}

__host__ __device__ __forceinline__ bool rocwmma_static_modulus_supported(uint32_t modulus) {
  return rns8::detail::finite_u8::static_byte_modulus_supported(modulus);
}

__device__ int8_t reduce_to_centered_device(int32_t value, int modulus, uint32_t reciprocal) {
  (void)reciprocal;
  return rns8::detail::finite_u8::reduce_to_centered_static_byte_modulus_i32(
      value, static_cast<uint32_t>(modulus));
}

