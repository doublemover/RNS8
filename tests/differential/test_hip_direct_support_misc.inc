bool has_timing_label(const std::vector<rns8::detail::hip_direct_timing_sample>& samples, const std::string& label) {
  for (const auto& sample : samples) {
    if (sample.label == label) {
      return true;
    }
  }
  return false;
}

void fill_wrap64_carry_heavy_inputs(
    std::vector<uint64_t>& A,
    int64_t m,
    int64_t k,
    int64_t lda,
    std::vector<uint64_t>& B,
    int64_t n,
    int64_t ldb) {
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      A[static_cast<std::size_t>(row * lda + col)] =
          row % 3 == 0 ? std::numeric_limits<uint64_t>::max()
                       : row % 3 == 1 ? 0x8080808080808080ull : 0xfefdfcfbfaf9f8f7ull;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      B[static_cast<std::size_t>(row * ldb + col)] =
          col % 3 == 0 ? std::numeric_limits<uint64_t>::max()
                       : col % 3 == 1 ? 0x7f807f807f807f80ull : 0x0102030405060708ull;
    }
  }
}

