
rns8_context* create_cpu() {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}

rns8_gemm_desc exact_desc(rns8_semantics semantics, int64_t m, int64_t n, int64_t k) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

rns8_matrix_desc exact_matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  return desc;
}

void check_signed_residues(
    const rns8_matrix& matrix,
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t m,
    int64_t n,
    int64_t k) {
  for (uint32_t p = 0; p < matrix.prefix; ++p) {
    const uint16_t modulus = rns8::detail::kDefaultModuli[p];
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto exact = rns8::detail::exact_i64_gemm_cell(A, lda, B, ldb, row, col, k);
        CHECK(matrix.residues[rns8::detail::residue_index(matrix, p, row, col)] ==
              rns8::detail::centered_residue(exact, modulus));
      }
    }
  }
}

void check_unsigned_residues(
    const rns8_matrix& matrix,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    int64_t m,
    int64_t n,
    int64_t k) {
  for (uint32_t p = 0; p < matrix.prefix; ++p) {
    const uint16_t modulus = rns8::detail::kDefaultModuli[p];
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const auto exact = rns8::detail::exact_u64_gemm_cell(A, lda, B, ldb, row, col, k);
        CHECK(matrix.residues[rns8::detail::residue_index(matrix, p, row, col)] ==
              rns8::detail::centered_residue(exact, modulus));
      }
    }
  }
}

std::vector<uint64_t> unsigned_limbs(boost::multiprecision::cpp_int value, uint32_t limb_count) {
  std::vector<uint64_t> limbs(limb_count);
  for (uint32_t limb = 0; limb < limb_count; ++limb) {
    limbs[limb] = static_cast<uint64_t>(value & boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()));
    value >>= 64u;
  }
  return limbs;
}

std::vector<uint64_t> signed_twos_complement_limbs(boost::multiprecision::cpp_int value, uint32_t limb_count) {
  if (value < 0) {
    value += boost::multiprecision::cpp_int(1) << (64u * limb_count);
  }
  return unsigned_limbs(value, limb_count);
}

std::vector<int8_t> residues_for(boost::multiprecision::cpp_int value, uint32_t prefix) {
  std::vector<int8_t> residues(prefix);
  for (uint32_t p = 0; p < prefix; ++p) {
    residues[p] = rns8::detail::centered_residue(value, rns8::detail::kDefaultModuli[p]);
  }
  return residues;
}

void fill_exact_residue_matrix(rns8_matrix* matrix, const std::vector<boost::multiprecision::cpp_int>& values) {
  REQUIRE(matrix != nullptr);
  REQUIRE(values.size() == static_cast<std::size_t>(matrix->desc.rows * matrix->desc.cols));
  std::vector<std::vector<int8_t>> residues;
  residues.reserve(values.size());
  for (const auto& value : values) {
    residues.push_back(residues_for(value, matrix->prefix));
  }
  for (uint32_t p = 0; p < matrix->prefix; ++p) {
    for (int64_t row = 0; row < matrix->desc.rows; ++row) {
      for (int64_t col = 0; col < matrix->desc.cols; ++col) {
        const std::size_t value_index = static_cast<std::size_t>(row * matrix->desc.cols + col);
        const std::size_t residue_index = rns8::detail::residue_index(*matrix, p, row, col);
        matrix->residues[residue_index] = residues[value_index][p];
      }
    }
  }
  rns8::test::set_host_residues_current(*matrix, true);
  rns8::test::set_device_residues_current(*matrix, false);
  rns8::test::set_host_byte_limbs_current(*matrix, false);
  rns8::test::set_device_byte_limbs_current(*matrix, false);
}

}  // namespace
