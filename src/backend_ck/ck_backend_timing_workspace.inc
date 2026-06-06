namespace {

constexpr uint32_t kRns8AbiVersion = 1;
constexpr uint32_t kRns8DefaultModulusCount = 28;
constexpr uint32_t kRns8MaxSupportedPrefix = 28;
constexpr uint32_t kRns8TileScheduleZeroOutput = 0x00000001u;
constexpr uint32_t kRns8TileScheduleZeroRowColProduct = 0x00000002u;
constexpr uint32_t kRns8SupportedTileScheduleFlags =
    kRns8TileScheduleZeroOutput | kRns8TileScheduleZeroRowColProduct;
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

template <typename Fn>
int timed_ck_operation(const char* label, const char* alias, Fn&& fn) {
  if (!rns8::detail::hip_direct_timing_enabled()) {
    return fn();
  }
  const auto host_start = std::chrono::steady_clock::now();
  timing_event_pair events;
  if (!events.create_and_record_start()) {
    const int status = fn();
    if (status == ck_success) {
      record_synchronized_host_timing(label, alias, host_start);
    }
    return status;
  }
  const int status = fn();
  if (status == ck_success) {
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
    y = static_cast<Y>(rns8::detail::finite_u8::reduce_to_centered_ck_i32(
        static_cast<int32_t>(x), static_cast<uint32_t>(modulus), reciprocal));
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
  return rns8::detail::finite_u8::reduce_to_centered_ck_i32(value, static_cast<uint32_t>(modulus), reciprocal);
}

__device__ int8_t canonical_byte_from_centered_device(int8_t residue, int modulus) {
  return static_cast<int8_t>(
      rns8::detail::finite_u8::canonical_from_centered(residue, static_cast<uint32_t>(modulus)));
}

