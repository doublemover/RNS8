
rns8_context* create_cpu() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_context* create_wrap64() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

template <typename Enum>
Enum invalid_public_enum_value() {
  volatile unsigned value = 0x7fffu;
  return static_cast<Enum>(value);
}

rns8_gemm_desc gemm_desc(rns8_semantics semantics, rns8_bound_kind bound_kind) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = bound_kind;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = 1;
  desc.n = 1;
  desc.k = 1;
  desc.bound = 1;
  return desc;
}

rns8_gemm_desc finite_desc(rns8_semantics semantics, int64_t m, int64_t n, int64_t k, uint16_t modulus = 0) {
  auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = 0;
  desc.max_prefix = 0;
  desc.finite_modulus =
      modulus != 0 ? modulus : (semantics == RNS8_FINITE_FIELD_U8 ? uint16_t{251} : uint16_t{255});
  return desc;
}

uint8_t exact_finite_cell(
    const std::vector<uint8_t>& A,
    const std::vector<uint8_t>& B,
    int64_t lda,
    int64_t ldb,
    int64_t k,
    int64_t row,
    int64_t col,
    uint16_t modulus) {
  uint64_t acc = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    acc += static_cast<uint64_t>(A[static_cast<std::size_t>(row * lda + kk)] % modulus) *
           static_cast<uint64_t>(B[static_cast<std::size_t>(kk * ldb + col)] % modulus);
    acc %= modulus;
  }
  return static_cast<uint8_t>(acc % modulus);
}

rns8_gemm_desc bounded_looking_desc(
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    rns8_backend_kind requested_backend = RNS8_BACKEND_CPU_REFERENCE) {
  auto desc = gemm_desc(semantics, bound_kind);
  desc.requested_backend = requested_backend;
  desc.bound = UINT64_MAX;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

rns8_matrix_desc matrix_desc(rns8_semantics semantics, rns8_bound_kind bound_kind) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = 1;
  desc.cols = 1;
  desc.logical_ld = 1;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  return desc;
}

rns8_matrix_desc matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t max_prefix = 0) {
  auto desc = matrix_desc(semantics, bound_kind);
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.max_prefix = max_prefix;
  return desc;
}

rns8_matrix_desc bounded_looking_matrix_desc(rns8_semantics semantics, rns8_bound_kind bound_kind) {
  auto desc = matrix_desc(semantics, bound_kind);
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

