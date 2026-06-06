void fill_bounded_i64_inputs(
    const Args& args,
    std::vector<int64_t>& A,
    std::vector<int64_t>& B,
    std::mt19937_64& rng) {
  if (args.input_profile == InputProfile::UniformSmall) {
    std::uniform_int_distribution<int64_t> dist(-16, 16);
    for (auto& value : A) value = dist(rng);
    for (auto& value : B) value = dist(rng);
    return;
  }
  for (int64_t row = 0; row < args.m; ++row) {
    const uint64_t row_band = profile_band_for_coordinate(row, args.tile_m, true);
    for (int64_t kk = 0; kk < args.k; ++kk) {
      A[row_major_index(row, kk, args.k, "A")] = sample_signed_band(rng, row_band);
    }
  }
  for (int64_t kk = 0; kk < args.k; ++kk) {
    for (int64_t col = 0; col < args.n; ++col) {
      const uint64_t col_band = profile_band_for_coordinate(col, args.tile_n, false);
      B[row_major_index(kk, col, args.n, "B")] = sample_signed_band(rng, col_band);
    }
  }
}

void fill_bounded_u64_inputs(
    const Args& args,
    std::vector<uint64_t>& A,
    std::vector<uint64_t>& B,
    std::mt19937_64& rng) {
  if (args.input_profile == InputProfile::UniformSmall) {
    std::uniform_int_distribution<uint64_t> dist(0, 16);
    for (auto& value : A) value = dist(rng);
    for (auto& value : B) value = dist(rng);
    return;
  }
  for (int64_t row = 0; row < args.m; ++row) {
    const uint64_t row_band = profile_band_for_coordinate(row, args.tile_m, true);
    for (int64_t kk = 0; kk < args.k; ++kk) {
      A[row_major_index(row, kk, args.k, "A")] = sample_unsigned_band(rng, row_band);
    }
  }
  for (int64_t kk = 0; kk < args.k; ++kk) {
    for (int64_t col = 0; col < args.n; ++col) {
      const uint64_t col_band = profile_band_for_coordinate(col, args.tile_n, false);
      B[row_major_index(kk, col, args.n, "B")] = sample_unsigned_band(rng, col_band);
    }
  }
}

int8_t uniform_small_i64_to_i8(int64_t value) {
  if (value < -16 || value > 16) {
    usage_error("uniform-small i64 fast path received an input outside [-16, 16]");
  }
  return static_cast<int8_t>(value);
}

int8_t uniform_small_u64_to_i8(uint64_t value) {
  if (value > 16) {
    usage_error("uniform-small u64 fast path received an input outside [0, 16]");
  }
  return static_cast<int8_t>(value);
}

std::vector<int8_t> make_uniform_small_i8_inputs(const std::vector<int64_t>& values) {
  std::vector<int8_t> packed(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    packed[index] = uniform_small_i64_to_i8(values[index]);
  }
  return packed;
}

std::vector<int8_t> make_uniform_small_i8_inputs(const std::vector<uint64_t>& values) {
  std::vector<int8_t> packed(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    packed[index] = uniform_small_u64_to_i8(values[index]);
  }
  return packed;
}

uint64_t benchmark_bound(const Args& args) {
  if (exact_wide_benchmark_semantics(args.semantics) || args.semantics == BenchSemantics::WrapU64Mod2_64 ||
      finite_benchmark_semantics(args.semantics) || args.bound_mode == BoundMode::PerTile) {
    return 0;
  }
  const uint64_t max_term = 16u * 16u;
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > std::numeric_limits<uint64_t>::max() / max_term) {
    usage_error("k is too large for the benchmark bound");
  }
  uint64_t bound = u_k * max_term;
  if (vector_to_rns_chain_requested(args)) {
    const auto u_n = static_cast<uint64_t>(args.n);
    if (u_n > std::numeric_limits<uint64_t>::max() / 16u) {
      usage_error("n is too large for the vector-to-RNS chain benchmark bound");
    }
    const uint64_t chain_multiplier = u_n * 16u;
    if (chain_multiplier != 0 && bound > std::numeric_limits<uint64_t>::max() / chain_multiplier) {
      usage_error("vector-to-RNS chain benchmark bound exceeds uint64_t");
    }
    bound *= chain_multiplier;
  }
  if (args.residue_chain_length > 1) {
    if (u_k > std::numeric_limits<uint64_t>::max() / 16u) {
      usage_error("k is too large for the bounded residue chain benchmark bound");
    }
    const uint64_t chain_multiplier = u_k * 16u;
    for (uint32_t chain_index = 1; chain_index < args.residue_chain_length; ++chain_index) {
      if (chain_multiplier != 0 && bound > std::numeric_limits<uint64_t>::max() / chain_multiplier) {
        usage_error("bounded residue chain benchmark bound exceeds uint64_t");
      }
      bound *= chain_multiplier;
    }
  }
  if (args.semantics == BenchSemantics::BoundedI64 &&
      bound > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    usage_error("bounded-i64 benchmark bound exceeds int64 output range");
  }
  return bound;
}

uint64_t vector_to_rns_chain_producer_bound(const Args& args) {
  const uint64_t max_term = 16u * 16u;
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > std::numeric_limits<uint64_t>::max() / max_term) {
    usage_error("k is too large for the vector-to-RNS producer benchmark bound");
  }
  const uint64_t bound = u_k * max_term;
  if (args.semantics == BenchSemantics::BoundedI64 &&
      bound > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    usage_error("bounded-i64 vector-to-RNS producer bound exceeds int64 output range");
  }
  return bound;
}

uint64_t checked_add_bound(uint64_t a, uint64_t b, const char* label) {
  if (a > std::numeric_limits<uint64_t>::max() - b) {
    usage_error(std::string(label) + " bound scan overflowed uint64_t");
  }
  return a + b;
}

uint64_t checked_mul_bound(uint64_t a, uint64_t b, const char* label) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    usage_error(std::string(label) + " bound scan overflowed uint64_t");
  }
  return a * b;
}

uint64_t magnitude_for_bound_scan(int64_t value) {
  if (value == std::numeric_limits<int64_t>::min()) {
    usage_error("bounded-i64 input scan cannot represent abs(INT64_MIN) in int64_t");
  }
  return value < 0 ? static_cast<uint64_t>(-value) : static_cast<uint64_t>(value);
}

uint64_t magnitude_for_bound_scan(uint64_t value) {
  return value;
}

struct GlobalBoundScanStats {
  uint64_t discovered_bound = 0;
  uint64_t candidate_row_sum_col_max = 0;
  uint64_t candidate_row_max_col_sum = 0;
  uint64_t row_abs_sum_max = 0;
  uint64_t row_abs_max = 0;
  uint64_t col_abs_sum_max = 0;
  uint64_t col_abs_max = 0;
  uint64_t zero_row_count = 0;
  uint64_t zero_col_count = 0;
};

template <typename T>
GlobalBoundScanStats compute_global_bound_scan(
    const Args& args,
    const std::vector<T>& A,
    const std::vector<T>& B,
    uint64_t static_bound) {
  GlobalBoundScanStats stats{};

  for (int64_t row = 0; row < args.m; ++row) {
    uint64_t sum = 0;
    uint64_t maximum = 0;
    for (int64_t kk = 0; kk < args.k; ++kk) {
      const uint64_t value = magnitude_for_bound_scan(A[row_major_index(row, kk, args.k, "A")]);
      sum = checked_add_bound(sum, value, "row absolute sum");
      maximum = std::max(maximum, value);
    }
    stats.row_abs_sum_max = std::max(stats.row_abs_sum_max, sum);
    stats.row_abs_max = std::max(stats.row_abs_max, maximum);
    if (sum == 0) {
      ++stats.zero_row_count;
    }
  }

  for (int64_t col = 0; col < args.n; ++col) {
    uint64_t sum = 0;
    uint64_t maximum = 0;
    for (int64_t kk = 0; kk < args.k; ++kk) {
      const uint64_t value = magnitude_for_bound_scan(B[row_major_index(kk, col, args.n, "B")]);
      sum = checked_add_bound(sum, value, "column absolute sum");
      maximum = std::max(maximum, value);
    }
    stats.col_abs_sum_max = std::max(stats.col_abs_sum_max, sum);
    stats.col_abs_max = std::max(stats.col_abs_max, maximum);
    if (sum == 0) {
      ++stats.zero_col_count;
    }
  }

  stats.candidate_row_sum_col_max =
      checked_mul_bound(stats.row_abs_sum_max, stats.col_abs_max, "row-sum/column-max");
  stats.candidate_row_max_col_sum =
      checked_mul_bound(stats.row_abs_max, stats.col_abs_sum_max, "row-max/column-sum");
  stats.discovered_bound = std::min(stats.candidate_row_sum_col_max, stats.candidate_row_max_col_sum);
  if (static_bound != 0) {
    stats.discovered_bound = std::min(stats.discovered_bound, static_bound);
  }
  return stats;
}

void record_global_bound_scan(BenchmarkResult& result, const GlobalBoundScanStats& stats, uint64_t static_bound) {
  result.global_bound_scan_available = true;
  result.static_bound = static_bound;
  result.discovered_global_bound = stats.discovered_bound;
  result.bound_candidate_row_sum_col_max = stats.candidate_row_sum_col_max;
  result.bound_candidate_row_max_col_sum = stats.candidate_row_max_col_sum;
  result.row_abs_sum_max = stats.row_abs_sum_max;
  result.row_abs_max = stats.row_abs_max;
  result.col_abs_sum_max = stats.col_abs_sum_max;
  result.col_abs_max = stats.col_abs_max;
  result.zero_row_count = stats.zero_row_count;
  result.zero_col_count = stats.zero_col_count;
}

template <typename T>
uint64_t resolve_bounded_global_bound(
    const Args& args,
    const std::vector<T>& A,
    const std::vector<T>& B,
    uint64_t static_bound,
    BenchmarkResult& result) {
  result.static_bound = static_bound;
  result.effective_bound = static_bound;
  result.effective_bound_available = true;
  if (args.bound_source != BoundSource::InputScan || args.bound_mode != BoundMode::Global) {
    return static_bound;
  }
  const auto scan_start = std::chrono::steady_clock::now();
  const GlobalBoundScanStats stats = compute_global_bound_scan(args, A, B, static_bound);
  const auto scan_end = std::chrono::steady_clock::now();
  result.global_bound_scan_us = elapsed_us(scan_start, scan_end);
  record_global_bound_scan(result, stats, static_bound);
  result.effective_bound = stats.discovered_bound;
  return result.effective_bound;
}

uint64_t checked_tile_count(const Args& args) {
  const uint64_t tile_rows = ceil_div_i64_u32(args.m, args.tile_m);
  const uint64_t tile_cols = ceil_div_i64_u32(args.n, args.tile_n);
  if (tile_cols != 0 && tile_rows > std::numeric_limits<uint64_t>::max() / tile_cols) {
    usage_error("tile grid overflows uint64_t");
  }
  const uint64_t tile_count = tile_rows * tile_cols;
  if (tile_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    usage_error("tile grid overflows size_t");
  }
  return tile_count;
}

struct TileBoundScanResult {
  std::vector<uint64_t> bounds{};
  std::vector<uint8_t> zero_a_rows{};
  std::vector<uint8_t> zero_b_cols{};
};

template <typename T>
std::vector<uint8_t> compute_nonzero_a_rows(const Args& args, const std::vector<T>& A) {
  std::vector<uint8_t> rows(static_cast<std::size_t>(args.m), 0);
  for (int64_t row = 0; row < args.m; ++row) {
    for (int64_t kk = 0; kk < args.k; ++kk) {
      if (A[row_major_index(row, kk, args.k, "A")] != T{}) {
        rows[static_cast<std::size_t>(row)] = 1;
        break;
      }
    }
  }
  return rows;
}

template <typename T>
std::vector<uint8_t> compute_nonzero_b_cols(const Args& args, const std::vector<T>& B) {
  std::vector<uint8_t> cols(static_cast<std::size_t>(args.n), 0);
  for (int64_t kk = 0; kk < args.k; ++kk) {
    for (int64_t col = 0; col < args.n; ++col) {
      if (B[row_major_index(kk, col, args.n, "B")] != T{}) {
        cols[static_cast<std::size_t>(col)] = 1;
      }
    }
  }
  return cols;
}

bool range_has_nonzero_flag(const std::vector<uint8_t>& flags, int64_t begin, int64_t end) {
  for (int64_t index = begin; index < end; ++index) {
    if (flags[static_cast<std::size_t>(index)] != 0) {
      return true;
    }
  }
  return false;
}

std::vector<uint8_t> invert_nonzero_flags(const std::vector<uint8_t>& nonzero_flags) {
  std::vector<uint8_t> zero_flags(nonzero_flags.size(), 0);
  for (std::size_t index = 0; index < nonzero_flags.size(); ++index) {
    zero_flags[index] = nonzero_flags[index] == 0 ? 1u : 0u;
  }
  return zero_flags;
}

uint64_t count_set_flags(const std::vector<uint8_t>& flags) {
  uint64_t count = 0;
  for (const uint8_t value : flags) {
    count += value != 0 ? 1u : 0u;
  }
  return count;
}

uint64_t zero_row_col_product_count(int64_t m, int64_t n, uint64_t zero_a_rows, uint64_t zero_b_cols) {
  const uint64_t rows = static_cast<uint64_t>(m);
  const uint64_t cols = static_cast<uint64_t>(n);
  const uint64_t nonzero_a_rows = rows - zero_a_rows;
  return zero_a_rows * cols + nonzero_a_rows * zero_b_cols;
}

TileBoundScanResult compute_i64_tile_bounds(
    const Args& args,
    const std::vector<int64_t>& A,
    const std::vector<int64_t>& B) {
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 256u) {
    usage_error("bounded-i64 per-tile benchmark k is too large for exact int64 tile-bound prepass");
  }
  const uint64_t tile_rows = ceil_div_i64_u32(args.m, args.tile_m);
  const uint64_t tile_cols = ceil_div_i64_u32(args.n, args.tile_n);
  TileBoundScanResult scan{};
  scan.bounds.assign(static_cast<std::size_t>(checked_tile_count(args)), 0);
  const std::vector<uint8_t> nonzero_a_rows = compute_nonzero_a_rows(args, A);
  const std::vector<uint8_t> nonzero_b_cols = compute_nonzero_b_cols(args, B);
  scan.zero_a_rows = invert_nonzero_flags(nonzero_a_rows);
  scan.zero_b_cols = invert_nonzero_flags(nonzero_b_cols);
  for (uint64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
    const int64_t row_begin = static_cast<int64_t>(tile_row * static_cast<uint64_t>(args.tile_m));
    const int64_t row_end = std::min<int64_t>(args.m, row_begin + static_cast<int64_t>(args.tile_m));
    for (uint64_t tile_col = 0; tile_col < tile_cols; ++tile_col) {
      const int64_t col_begin = static_cast<int64_t>(tile_col * static_cast<uint64_t>(args.tile_n));
      const int64_t col_end = std::min<int64_t>(args.n, col_begin + static_cast<int64_t>(args.tile_n));
      if (!range_has_nonzero_flag(nonzero_a_rows, row_begin, row_end) ||
          !range_has_nonzero_flag(nonzero_b_cols, col_begin, col_end)) {
        continue;
      }
      uint64_t& tile_max = scan.bounds[static_cast<std::size_t>(tile_row * tile_cols + tile_col)];
      for (int64_t row = row_begin; row < row_end; ++row) {
        if (nonzero_a_rows[static_cast<std::size_t>(row)] == 0) {
          continue;
        }
        for (int64_t col = col_begin; col < col_end; ++col) {
          if (nonzero_b_cols[static_cast<std::size_t>(col)] == 0) {
            continue;
          }
          int64_t acc = 0;
          for (int64_t kk = 0; kk < args.k; ++kk) {
            acc += A[row_major_index(row, kk, args.k, "A")] * B[row_major_index(kk, col, args.n, "B")];
          }
          const uint64_t abs_value = acc < 0 ? static_cast<uint64_t>(-acc) : static_cast<uint64_t>(acc);
          tile_max = std::max(tile_max, abs_value);
        }
      }
    }
  }
  return scan;
}

TileBoundScanResult compute_u64_tile_bounds(
    const Args& args,
    const std::vector<uint64_t>& A,
    const std::vector<uint64_t>& B) {
  const auto u_k = static_cast<uint64_t>(args.k);
  if (u_k > std::numeric_limits<uint64_t>::max() / 256u) {
    usage_error("bounded-u64 per-tile benchmark k is too large for exact uint64 tile-bound prepass");
  }
  const uint64_t tile_rows = ceil_div_i64_u32(args.m, args.tile_m);
  const uint64_t tile_cols = ceil_div_i64_u32(args.n, args.tile_n);
  TileBoundScanResult scan{};
  scan.bounds.assign(static_cast<std::size_t>(checked_tile_count(args)), 0);
  const std::vector<uint8_t> nonzero_a_rows = compute_nonzero_a_rows(args, A);
  const std::vector<uint8_t> nonzero_b_cols = compute_nonzero_b_cols(args, B);
  scan.zero_a_rows = invert_nonzero_flags(nonzero_a_rows);
  scan.zero_b_cols = invert_nonzero_flags(nonzero_b_cols);
  for (uint64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
    const int64_t row_begin = static_cast<int64_t>(tile_row * static_cast<uint64_t>(args.tile_m));
    const int64_t row_end = std::min<int64_t>(args.m, row_begin + static_cast<int64_t>(args.tile_m));
    for (uint64_t tile_col = 0; tile_col < tile_cols; ++tile_col) {
      const int64_t col_begin = static_cast<int64_t>(tile_col * static_cast<uint64_t>(args.tile_n));
      const int64_t col_end = std::min<int64_t>(args.n, col_begin + static_cast<int64_t>(args.tile_n));
      if (!range_has_nonzero_flag(nonzero_a_rows, row_begin, row_end) ||
          !range_has_nonzero_flag(nonzero_b_cols, col_begin, col_end)) {
        continue;
      }
      uint64_t& tile_max = scan.bounds[static_cast<std::size_t>(tile_row * tile_cols + tile_col)];
      for (int64_t row = row_begin; row < row_end; ++row) {
        if (nonzero_a_rows[static_cast<std::size_t>(row)] == 0) {
          continue;
        }
        for (int64_t col = col_begin; col < col_end; ++col) {
          if (nonzero_b_cols[static_cast<std::size_t>(col)] == 0) {
            continue;
          }
          uint64_t acc = 0;
          for (int64_t kk = 0; kk < args.k; ++kk) {
            acc += A[row_major_index(row, kk, args.k, "A")] * B[row_major_index(kk, col, args.n, "B")];
          }
          tile_max = std::max(tile_max, acc);
        }
      }
    }
  }
  return scan;
}

void record_tile_bounds(BenchmarkResult& result, std::vector<uint64_t> bounds) {
  result.tile_bounds = std::move(bounds);
  if (result.tile_bounds.empty()) {
    result.tile_bound_min = 0;
    result.tile_bound_max = 0;
    result.tile_bound_hash = 0;
    return;
  }
  result.tile_bound_min = *std::min_element(result.tile_bounds.begin(), result.tile_bounds.end());
  result.tile_bound_max = *std::max_element(result.tile_bounds.begin(), result.tile_bounds.end());
  uint64_t hash = 1469598103934665603ull;
  mix_checksum(hash, static_cast<uint64_t>(result.tile_bounds.size()));
  for (const uint64_t bound : result.tile_bounds) {
    mix_checksum(hash, bound);
  }
  result.tile_bound_hash = hash;
}

template <typename Fn>
void record_timed_tile_bounds(BenchmarkResult& result, Fn&& compute_bounds) {
  const auto scan_start = std::chrono::steady_clock::now();
  auto scan = compute_bounds();
  const auto scan_end = std::chrono::steady_clock::now();
  result.tile_bound_scan_us = elapsed_us(scan_start, scan_end);
  result.tile_bound_scan_available = true;
  result.zero_a_rows = std::move(scan.zero_a_rows);
  result.zero_b_cols = std::move(scan.zero_b_cols);
  result.zero_a_row_proof_count = count_set_flags(result.zero_a_rows);
  result.zero_b_col_proof_count = count_set_flags(result.zero_b_cols);
  result.zero_row_col_product_count =
      zero_row_col_product_count(
          result.zero_a_rows.empty() ? 0 : static_cast<int64_t>(result.zero_a_rows.size()),
          result.zero_b_cols.empty() ? 0 : static_cast<int64_t>(result.zero_b_cols.size()),
          result.zero_a_row_proof_count,
          result.zero_b_col_proof_count);
  record_tile_bounds(result, std::move(scan.bounds));
}

uint32_t benchmark_prefix(const Args& args) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 || finite_benchmark_semantics(args.semantics)) {
    return 0;
  }
  if (args.max_prefix_override != 0) {
    return args.max_prefix_override;
  }
  if (exact_wide_benchmark_semantics(args.semantics)) {
    return RNS8_MAX_SUPPORTED_PREFIX;
  }
  return RNS8_DEFAULT_BOUNDED_PREFIX;
}

bool fixed_requested_prefix_policy(const Args& args) {
  return args.prefix_policy == PrefixPolicy::FixedRequested || args.residue_chain_length > 1;
}

const char* prefix_policy_name(const Args& args, const BenchmarkResult& result) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 || finite_benchmark_semantics(args.semantics)) {
    return "semantic_specific_no_rns_prefix";
  }
  if (args.bound_mode == BoundMode::PerTile && !fixed_requested_prefix_policy(args)) {
    return "per_tile_minimum";
  }
  if (args.residue_chain_length > 1 && args.prefix_policy != PrefixPolicy::FixedRequested) {
    return "fixed_requested_residue_chain";
  }
  if (fixed_requested_prefix_policy(args)) {
    return "fixed_requested";
  }
  if (result.schedule_info_available && result.schedule_info.max_selected_prefix < benchmark_prefix(args)) {
    return "minimum_proven";
  }
  return "minimum_proven";
}

uint32_t selected_execution_prefix(const Args& args, const BenchmarkResult& result) {
  if (args.semantics == BenchSemantics::WrapU64Mod2_64 || finite_benchmark_semantics(args.semantics)) {
    return 0;
  }
  if (result.schedule_info_available && result.schedule_info.max_selected_prefix > 0) {
    return result.schedule_info.max_selected_prefix;
  }
  return benchmark_prefix(args);
}

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols, const Args& args, uint32_t max_prefix) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = c_semantics(args.semantics);
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind(args);
  desc.tile_m = args.tile_m;
  desc.tile_n = args.tile_n;
  desc.max_prefix = max_prefix;
  return desc;
}

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols, const Args& args) {
  return matrix_desc(rows, cols, args, benchmark_prefix(args));
}

rns8_backend_kind requested_backend_for_plan_desc(const Args& args) {
  return native_to_rns_bridge_requested(args) ? RNS8_BACKEND_HIP_DIRECT : args.backend;
}

rns8_gemm_desc gemm_desc(
    const Args& args,
    uint64_t bound,
    const std::vector<uint64_t>* tile_bounds = nullptr,
    const std::vector<uint8_t>* zero_a_rows = nullptr,
    const std::vector<uint8_t>* zero_b_cols = nullptr) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = c_semantics(args.semantics);
  desc.bound_kind = bound_kind(args);
  desc.requested_backend = requested_backend_for_plan_desc(args);
  desc.m = args.m;
  desc.n = args.n;
  desc.k = args.k;
  desc.bound = bound;
  desc.max_prefix = benchmark_prefix(args);
  desc.tile_m = args.tile_m;
  desc.tile_n = args.tile_n;
  if (fixed_requested_prefix_policy(args) &&
      (bounded_benchmark_semantics(args.semantics) || exact_wide_benchmark_semantics(args.semantics))) {
    desc.flags |= RNS8_PLAN_FORCE_FIXED_PREFIX;
  }
  if (finite_benchmark_semantics(args.semantics)) {
    desc.finite_modulus = args.finite_modulus;
  }
  if (args.bound_mode == BoundMode::PerTile) {
    if (!tile_bounds) {
      usage_error("per-tile bound mode requires generated tile bounds");
    }
    desc.bound = 0;
    desc.tile_bounds = tile_bounds->data();
    desc.tile_bounds_count = static_cast<uint64_t>(tile_bounds->size());
    if (bounded_benchmark_semantics(args.semantics) && !tile_bounds->empty()) {
      desc.flags |= RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS;
    }
    if (bounded_benchmark_semantics(args.semantics) && zero_a_rows && zero_b_cols &&
        zero_a_rows->size() == static_cast<std::size_t>(args.m) &&
        zero_b_cols->size() == static_cast<std::size_t>(args.n) &&
        zero_row_col_product_count(
            args.m,
            args.n,
            count_set_flags(*zero_a_rows),
            count_set_flags(*zero_b_cols)) != 0) {
      desc.flags |= RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS;
      desc.zero_a_rows = zero_a_rows->data();
      desc.zero_a_rows_count = static_cast<uint64_t>(zero_a_rows->size());
      desc.zero_b_cols = zero_b_cols->data();
      desc.zero_b_cols_count = static_cast<uint64_t>(zero_b_cols->size());
    }
  }
  return desc;
}

