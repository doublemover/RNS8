
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
int default_visible_hip_test_device() {
  return 0;
}

class HipBuffer {
 public:
  explicit HipBuffer(std::size_t bytes) {
    REQUIRE(bytes > 0);
    REQUIRE(hipSetDevice(default_visible_hip_test_device()) == hipSuccess);
    REQUIRE(hipMalloc(&ptr_, bytes) == hipSuccess);
  }

  ~HipBuffer() {
    if (ptr_) {
      (void)hipSetDevice(default_visible_hip_test_device());
      (void)hipFree(ptr_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;

  void* get() const {
    return ptr_;
  }

 private:
  void* ptr_ = nullptr;
};

class HipScratchBuffer {
 public:
  ~HipScratchBuffer() {
    if (ptr_) {
      (void)hipSetDevice(default_visible_hip_test_device());
      (void)hipFree(ptr_);
    }
  }

  HipScratchBuffer(const HipScratchBuffer&) = delete;
  HipScratchBuffer& operator=(const HipScratchBuffer&) = delete;

  HipScratchBuffer() = default;

  void** ptr_address() {
    return &ptr_;
  }

  std::size_t* bytes_address() {
    return &bytes_;
  }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

rns8_context* create_backend_context(rns8_backend_kind backend) {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = backend;
  rns8_context* ctx = nullptr;
  const int device_id = backend == RNS8_BACKEND_CPU_REFERENCE ? -1 : default_visible_hip_test_device();
  REQUIRE(rns8_create_context(device_id, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

bool rocwmma_available() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_ROCWMMA;
  rns8_context* ctx = nullptr;
  const rns8_status status = rns8_create_context(default_visible_hip_test_device(), &options, &ctx);
  rns8_destroy_context(ctx);
  return status == RNS8_SUCCESS;
}

rns8_gemm_desc bounded_i64_desc(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc bounded_u64_desc(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc per_tile_u64_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<uint64_t>& bounds,
    rns8_backend_kind backend) {
  rns8_gemm_desc desc = bounded_u64_desc(m, n, k, 0, backend);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.tile_m = 64;
  desc.tile_n = 64;
  desc.tile_bounds = bounds.data();
  desc.tile_bounds_count = bounds.size();
  return desc;
}

rns8_gemm_desc finite_desc(
    int64_t m,
    int64_t n,
    int64_t k,
    rns8_semantics semantics,
    rns8_backend_kind backend,
    uint16_t modulus = 0) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.finite_modulus =
      modulus != 0 ? modulus : (semantics == RNS8_FINITE_FIELD_U8 ? uint16_t{251} : uint16_t{255});
  return desc;
}

rns8_gemm_desc exact_signed_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_EXACT_WIDE_SIGNED;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_gemm_desc exact_unsigned_desc(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_EXACT_WIDE_UNSIGNED;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_matrix_desc matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix = 0) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  desc.max_prefix = prefix;
  return desc;
}

void require_same_i64(const std::vector<int64_t>& expected, const std::vector<int64_t>& actual) {
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(actual[i] == expected[i]);
  }
}

void require_same_u64(const std::vector<uint64_t>& expected, const std::vector<uint64_t>& actual) {
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(actual[i] == expected[i]);
  }
}

void require_wrap64_output_matches_oracle(
    const std::vector<uint64_t>& A,
    int64_t lda,
    const std::vector<uint64_t>& B,
    int64_t ldb,
    const std::vector<uint64_t>& C,
    int64_t ldc,
    int64_t m,
    int64_t n,
    int64_t k) {
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      CHECK(C[static_cast<std::size_t>(row * ldc + col)] ==
            rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
    }
  }
}

void require_wrap64_sampled_output_matches_oracle(
    const std::vector<uint64_t>& A,
    int64_t lda,
    const std::vector<uint64_t>& B,
    int64_t ldb,
    const std::vector<uint64_t>& C,
    int64_t ldc,
    int64_t m,
    int64_t n,
    int64_t k) {
  std::vector<uint64_t> sample_offsets;
  const auto add_sample = [&](int64_t row, int64_t col) {
    REQUIRE(row >= 0);
    REQUIRE(row < m);
    REQUIRE(col >= 0);
    REQUIRE(col < n);
    const auto offset = static_cast<uint64_t>(row * n + col);
    if (std::find(sample_offsets.begin(), sample_offsets.end(), offset) == sample_offsets.end()) {
      sample_offsets.push_back(offset);
    }
  };

  add_sample(0, 0);
  add_sample(0, n - 1);
  add_sample(m - 1, 0);
  add_sample(m - 1, n - 1);
  add_sample(m / 2, n / 2);
  add_sample(m / 3, n / 5);
  add_sample((2 * m) / 3, (3 * n) / 5);
  add_sample((5 * m) / 7, (2 * n) / 7);

  for (const uint64_t offset : sample_offsets) {
    const int64_t row = static_cast<int64_t>(offset / static_cast<uint64_t>(n));
    const int64_t col = static_cast<int64_t>(offset % static_cast<uint64_t>(n));
    CHECK(C[static_cast<std::size_t>(row * ldc + col)] ==
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), lda, B.data(), ldb, row, col, k));
  }
}

struct Wrap64CandidateShape {
  const char* name;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t lda_padding;
  int64_t ldb_padding;
  int64_t ldc_padding;
  uint64_t seed;
};

uint64_t wrap64_candidate_a_value(int64_t row, int64_t col, std::mt19937_64& rng) {
  const auto selector = static_cast<uint32_t>((row * 13 + col * 17) % 11);
  if (selector == 0) {
    return std::numeric_limits<uint64_t>::max();
  }
  if (selector == 1) {
    return 0x8080808080808080ull;
  }
  if (selector == 2) {
    return 0x7f807f807f807f80ull;
  }
  if (selector == 3) {
    return 0x0000000000000001ull << static_cast<uint32_t>((row + col) % 63);
  }
  return rng();
}

uint64_t wrap64_candidate_b_value(int64_t row, int64_t col, std::mt19937_64& rng) {
  const auto selector = static_cast<uint32_t>((row * 19 + col * 23) % 11);
  if (selector == 0) {
    return std::numeric_limits<uint64_t>::max() - 1;
  }
  if (selector == 1) {
    return 0xfefdfcfbfaf9f8f7ull;
  }
  if (selector == 2) {
    return 0x0102030405060708ull;
  }
  if (selector == 3) {
    return 0x8000000000000000ull >> static_cast<uint32_t>((row + col) % 63);
  }
  return rng();
}

void require_wrap64_rocwmma_candidate_matches_direct_hip_and_oracle(
    const Wrap64CandidateShape& shape,
    bool exhaustive_oracle = true) {
  INFO(shape.name);

  const int device_id = default_visible_hip_test_device();
  constexpr uint64_t sentinel = 0xfeedfacecafebeefull;
  const int64_t lda = shape.k + shape.lda_padding;
  const int64_t ldb = shape.n + shape.ldb_padding;
  const int64_t ldc = shape.n + shape.ldc_padding;
  REQUIRE(lda >= shape.k);
  REQUIRE(ldb >= shape.n);
  REQUIRE(ldc >= shape.n);

  std::vector<uint64_t> A(static_cast<std::size_t>(shape.m * lda), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> B(static_cast<std::size_t>(shape.k * ldb), 0xbbbbbbbbbbbbbbbbull);
  std::vector<uint64_t> direct_out(static_cast<std::size_t>(shape.m * ldc), sentinel);
  std::vector<uint64_t> rocwmma_out(static_cast<std::size_t>(shape.m * ldc), sentinel);
  std::mt19937_64 rng(shape.seed);

  for (int64_t row = 0; row < shape.m; ++row) {
    for (int64_t col = 0; col < shape.k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] = wrap64_candidate_a_value(row, col, rng);
    }
  }
  for (int64_t row = 0; row < shape.k; ++row) {
    for (int64_t col = 0; col < shape.n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] = wrap64_candidate_b_value(row, col, rng);
    }
  }

  const std::size_t a_limb_bytes = static_cast<std::size_t>(shape.m * shape.k * 8);
  const std::size_t b_limb_bytes = static_cast<std::size_t>(shape.k * shape.n * 8);
  const std::size_t c_limb_bytes = static_cast<std::size_t>(shape.m * shape.n * 8);
  HipBuffer a_limbs(a_limb_bytes);
  HipBuffer b_limbs(b_limb_bytes);
  HipBuffer direct_limbs(c_limb_bytes);
  HipBuffer rocwmma_limbs(c_limb_bytes);
  HipScratchBuffer upload_buffer;
  HipScratchBuffer export_buffer;

  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              device_id,
              A.data(),
              upload_buffer.ptr_address(),
              upload_buffer.bytes_address(),
              a_limbs.get(),
              shape.m,
              shape.k,
              lda) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              device_id,
              B.data(),
              upload_buffer.ptr_address(),
              upload_buffer.bytes_address(),
              b_limbs.get(),
              shape.k,
              shape.n,
              ldb) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
              device_id, a_limbs.get(), b_limbs.get(), direct_limbs.get(), shape.m, shape.n, shape.k) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::rocwmma_wrap64_gemm_byte_limbs_candidate_device(
              device_id, a_limbs.get(), b_limbs.get(), rocwmma_limbs.get(), shape.m, shape.n, shape.k) ==
          RNS8_SUCCESS);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              device_id,
              direct_limbs.get(),
              export_buffer.ptr_address(),
              export_buffer.bytes_address(),
              shape.m,
              shape.n,
              direct_out.data(),
              ldc) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              device_id,
              rocwmma_limbs.get(),
              export_buffer.ptr_address(),
              export_buffer.bytes_address(),
              shape.m,
              shape.n,
              rocwmma_out.data(),
              ldc) == RNS8_SUCCESS);

  for (int64_t row = 0; row < shape.m; ++row) {
    for (int64_t col = shape.n; col < ldc; ++col) {
      CHECK(direct_out[static_cast<std::size_t>(row * ldc + col)] == sentinel);
      CHECK(rocwmma_out[static_cast<std::size_t>(row * ldc + col)] == sentinel);
    }
  }
  require_same_u64(direct_out, rocwmma_out);
  if (exhaustive_oracle) {
    require_wrap64_output_matches_oracle(A, lda, B, ldb, rocwmma_out, ldc, shape.m, shape.n, shape.k);
  } else {
    require_wrap64_sampled_output_matches_oracle(A, lda, B, ldb, rocwmma_out, ldc, shape.m, shape.n, shape.k);
  }
}
#endif

