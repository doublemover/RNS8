namespace {

#ifndef RNS8_CK_SHAPE_ALIGNMENT
#define RNS8_CK_SHAPE_ALIGNMENT 64
#endif
#ifndef RNS8_CK_USE_XDL
#define RNS8_CK_USE_XDL 0
#endif

constexpr uint32_t kRns8AbiVersion = 1;
constexpr uint32_t kRns8DefaultModulusCount = 28;
constexpr uint32_t kRns8MaxSupportedPrefix = 28;
constexpr uint32_t kRns8TileScheduleZeroOutput = 0x00000001u;
constexpr uint32_t kRns8TileScheduleZeroRowColProduct = 0x00000002u;
constexpr uint32_t kRns8SupportedTileScheduleFlags =
    kRns8TileScheduleZeroOutput | kRns8TileScheduleZeroRowColProduct;
constexpr auto kCkImplementation = rns8::detail::compiled_ck_implementation();
static_assert(kCkImplementation.valid(), "CK requires a detected physical implementation contract");
constexpr int64_t kRns8CanonicalSafeInt32KBlock = kCkImplementation.slab_k;
constexpr int64_t kRns8CkShapeAlignment = kCkImplementation.shape_alignment;
static_assert(kRns8CkShapeAlignment == 64 || kRns8CkShapeAlignment == 128,
              "unsupported CK shape alignment");
static_assert(RNS8_CK_USE_XDL == 0 || RNS8_CK_USE_XDL == 1,
              "unsupported CK kernel family selector");
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

using rns8::detail::timed_hip_backend_launch;

template <typename Fn>
int timed_ck_operation(const char* label, const char* alias, Fn&& fn,
                       hipStream_t stream = nullptr) {
  return rns8::detail::timed_hip_backend_operation(
      label, alias, std::forward<Fn>(fn), int(ck_success),
      [](hipError_t) { return int(ck_backend_failure); }, stream);
}

template <ck::index_t... Is> using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

__host__ __device__ __forceinline__ bool ck_static_modulus_supported(uint32_t modulus) {
  return modulus >= 2 && modulus <= 256;
}

struct CenteredModulo {
  int32_t modulus;
  uint32_t reciprocal;

  __host__ __device__ CenteredModulo(int32_t m = 253, uint32_t r = 0) : modulus(m), reciprocal(r) {}

  template <typename Y, typename X> __host__ __device__ void operator()(Y& y, const X& x) const {
    y = static_cast<Y>(rns8::detail::finite_u8::reduce_to_centered_i32(
        static_cast<int32_t>(x), static_cast<uint32_t>(modulus), reciprocal));
  }
};

template <uint32_t Modulus> struct CenteredModuloFixed {
  template <typename Y, typename X> __host__ __device__ void operator()(Y& y, const X& x) const {
    y = static_cast<Y>(
        rns8::detail::finite_u8::reduce_to_centered_fixed_i32<Modulus>(static_cast<int32_t>(x)));
  }
};

#if RNS8_CK_USE_XDL
template <typename CElementwiseOperation>
using DeviceGemmInstanceFor = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
    Row, Col, Row, int8_t, int8_t, int8_t, int32_t, int32_t, PassThrough, PassThrough,
    CElementwiseOperation, ck::tensor_operation::device::GemmSpecialization::MNKPadding, 1, 256,
    128, 128, 64, 16, 16, 32, 32, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, true,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, 1, 1, S<1, 32, 1, 4>, 16>;
#else
template <typename CElementwiseOperation>
using DeviceGemmInstanceFor = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    Row, Col, Row, int8_t, int8_t, int8_t, int32_t, int32_t, PassThrough, PassThrough,
    CElementwiseOperation, ck::tensor_operation::device::GemmSpecialization::MNKPadding, 1, 128, 64,
    128, 64, 2, 16, 16, 2, 4, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, true, S<4, 32, 1>,
    S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, true, 1, 1, S<1, 32, 1, 4>, 8>;
#endif

using DeviceGemmInstance = DeviceGemmInstanceFor<CenteredModulo>;

uint32_t modulus_reciprocal_u32(uint16_t modulus) {
  return static_cast<uint32_t>(kReciprocalScale / static_cast<uint32_t>(modulus));
}

bool checked_i32_shape(int64_t value) {
  return value > 0 && value <= static_cast<int64_t>(INT32_MAX);
}

bool ck_workspace_requirements_u64(int64_t max_m, int64_t max_n, int64_t k, uint64_t* a_bytes,
                                   uint64_t* b_bytes, uint64_t* temp_bytes) {
  rns8::detail::PackedGemmLayout layout;
  if (!a_bytes || !b_bytes || !temp_bytes || max_m <= 0 || max_n <= 0 || k <= 0 ||
      !rns8::detail::ck_packed_gemm_layout(max_m, max_n, k, kCkImplementation, layout))
    return false;
  *a_bytes = layout.a_bytes;
  *b_bytes = layout.b_bytes;
  *temp_bytes = layout.c_bytes;
  return true;
}

__device__ int8_t center_i32_device(int32_t value, int modulus, uint32_t reciprocal) {
  return rns8::detail::finite_u8::reduce_to_centered_i32(value, static_cast<uint32_t>(modulus),
                                                         reciprocal);
}

__device__ int8_t ck_input_byte_from_centered_device(int8_t residue, int modulus) {
  return rns8::detail::ck_encode_matrix_input(residue, uint32_t(modulus),
                                              kCkImplementation.signed_operands);
}
