
uint64_t low64(boost::multiprecision::cpp_int value) {
  const boost::multiprecision::cpp_int modulus = boost::multiprecision::cpp_int(1) << 64;
  value %= modulus;
  if (value < 0) {
    value += modulus;
  }
  return static_cast<uint64_t>(value);
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

rns8_context* create_cpu() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_gemm_desc wrap_desc(int64_t m, int64_t n, int64_t k) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  return desc;
}

rns8_matrix_desc wrap_matrix_desc(int64_t rows, int64_t cols) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  return desc;
}

uint64_t corrected_signed_i8_comba_product(uint64_t a, uint64_t b) {
  uint64_t out = 0;
  uint64_t carry = 0;
  for (uint32_t diagonal = 0; diagonal < 8; ++diagonal) {
    uint64_t column = carry;
    for (uint32_t i = 0; i <= diagonal; ++i) {
      const uint32_t j = diagonal - i;
      const auto a_byte = static_cast<uint8_t>((a >> (8u * i)) & 0xffu);
      const auto b_byte = static_cast<uint8_t>((b >> (8u * j)) & 0xffu);
      column += rns8::detail::wrap64_unsigned_byte_product_from_signed_i8(a_byte, b_byte);
    }
    out |= (column & 0xffu) << (8u * diagonal);
    carry = column >> 8u;
  }
  return out;
}

uint64_t expected_wrap_cell(
    const std::vector<uint64_t>& A,
    int64_t lda,
    const std::vector<uint64_t>& B,
    int64_t ldb,
    int64_t row,
    int64_t col,
    int64_t k) {
  boost::multiprecision::cpp_int exact = 0;
  for (int64_t kk = 0; kk < k; ++kk) {
    exact += boost::multiprecision::cpp_int(A[static_cast<std::size_t>(row * lda + kk)]) *
             boost::multiprecision::cpp_int(B[static_cast<std::size_t>(kk * ldb + col)]);
  }
  return low64(exact);
}

uint64_t load_compact_u64_limbs(const std::vector<uint8_t>& limbs, int64_t cell) {
  uint64_t value = 0;
  for (uint32_t limb = 0; limb < 8; ++limb) {
    value |= static_cast<uint64_t>(limbs[static_cast<std::size_t>(cell * 8 + limb)]) << (8u * limb);
  }
  return value;
}

