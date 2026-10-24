#ifndef RNS8_WORKLOAD_COMPLEX_HPP
#define RNS8_WORKLOAD_COMPLEX_HPP
#include "workload_baselines.hpp"

namespace rns8_workload {
struct ComplexData {
  std::array<std::vector<int64_t>, 4> values; // Ar, Ai, Br, Bi
  double preparation_ns = 0;
  std::string recipe;
};
inline ComplexData make_complex_data(const Options& options) {
  const auto start = Clock::now();
  ComplexData data;
  data.values[0].resize(std::size_t(options.m) * options.k);
  data.values[1].resize(data.values[0].size());
  data.values[2].resize(std::size_t(options.k) * options.n);
  data.values[3].resize(data.values[2].size());
  data.recipe = "splitmix64_split_complex_signed_v1";
  uint64_t state = options.seed;
  const uint64_t max = (uint64_t{1} << options.bits) - 1, span = max * 2 + 1;
  for (auto& operand : data.values) for (std::size_t index = 0; index < operand.size(); ++index) {
    const uint64_t value = next_word(state) % span;
    operand[index] = options.pattern == "zero" || (options.pattern == "sparse" && index % 8)
        ? 0 : value <= max ? static_cast<int64_t>(value) : -static_cast<int64_t>(value - max);
  }
  if (options.size == "tiny" && options.m == 2 && options.n == 2 && options.k == 3 && options.pattern == "seeded"
      && !options.supplied.count("--seed") && !options.supplied.count("--bits")) {
    data.values = {{{2, -3, 4, -5, 6, 7}, {1, 0, -1, 2, -2, 1},
                    {1, 8, -2, 9, 3, -4}, {0, 1, 2, 0, -1, 3}}};
    data.recipe = "tiny_split_complex_v1";
  }
  data.preparation_ns = since(start);
  return data;
}
inline Words complex_input_words(const ComplexData& data) {
  Words result;
  for (const auto& operand : data.values) for (const auto value : operand) result.push_back(bits_of(value));
  return result;
}
inline void mutate_complex_input(ComplexData& data) {
  auto& value = data.values[0][0];
  value = value == INT64_MAX ? value - 1 : value + 1;
}
inline cpp_int complex_final_bound(const ComplexData& data, const Options& options) {
  std::array<cpp_int, 4> bounds;
  for (std::size_t index = 0; index < bounds.size(); ++index) bounds[index] = maximum(data.values[index]);
  const int64_t value = data.values[0][0];
  const int64_t changed = value == INT64_MAX ? value - 1 : value + 1;
  bounds[0] = std::max(bounds[0], cpp_int(magnitude(changed)));
  const cpp_int real = cpp_int(options.k) * (bounds[0] * bounds[2] + bounds[1] * bounds[3]);
  const cpp_int imag = cpp_int(options.k) * (bounds[0] * bounds[3] + bounds[1] * bounds[2]);
  return std::max(real, imag);
}
template<class Integer> void complex_four_product(const ComplexData& data, const Options& options,
    bool blocked, std::vector<Integer>& real, std::vector<Integer>& imag) {
  std::fill(real.begin(), real.end(), Integer(0));
  std::fill(imag.begin(), imag.end(), Integer(0));
  const auto& ar = data.values[0]; const auto& ai = data.values[1];
  const auto& br = data.values[2]; const auto& bi = data.values[3];
  const auto accumulate = [&](uint32_t row, uint32_t col, uint32_t q) {
    const std::size_t a = std::size_t(row) * options.k + q, b = std::size_t(q) * options.n + col;
    const std::size_t c = std::size_t(row) * options.n + col;
    real[c] += Integer(ar[a]) * Integer(br[b]) - Integer(ai[a]) * Integer(bi[b]);
    imag[c] += Integer(ar[a]) * Integer(bi[b]) + Integer(ai[a]) * Integer(br[b]);
  };
  if (!blocked) {
    for (uint32_t row = 0; row < options.m; ++row) for (uint32_t col = 0; col < options.n; ++col)
      for (uint32_t q = 0; q < options.k; ++q) accumulate(row, col, q);
    return;
  }
  constexpr uint32_t tile = 32;
  for (uint32_t ii = 0; ii < options.m; ii += tile) for (uint32_t qq = 0; qq < options.k; qq += tile)
    for (uint32_t jj = 0; jj < options.n; jj += tile)
      for (uint32_t row = ii; row < std::min(options.m, ii + tile); ++row)
        for (uint32_t q = qq; q < std::min(options.k, qq + tile); ++q)
          for (uint32_t col = jj; col < std::min(options.n, jj + tile); ++col) accumulate(row, col, q);
}
inline Words complex_reference(const ComplexData& data, const Options& options) {
  std::vector<cpp_int> real(std::size_t(options.m) * options.n), imag(real.size());
  complex_four_product(data, options, false, real, imag);
  auto words = integer_words(real, 3); const auto imaginary = integer_words(imag, 3);
  words.insert(words.end(), imaginary.begin(), imaginary.end());
  return words;
}
template<class Integer> struct ComplexBaseline {
  const ComplexData& data;
  const Options& options;
  bool blocked;
  std::vector<Integer> real, imag;
  Words real_words, imag_words, words;
  ComplexBaseline(const ComplexData& input, const Options& config, bool tiled)
      : data(input), options(config), blocked(tiled), real(std::size_t(config.m) * config.n), imag(real.size()),
        real_words(real.size() * 3), imag_words(real_words.size()), words(real_words.size() * 2) {}
  void execute(Phases& phases, uint64_t) {
    phases.measure("four_product_compute_and_pair_serialization_ns", [&] {
      complex_four_product(data, options, blocked, real, imag);
      encode_into(real, 3, real_words); encode_into(imag, 3, imag_words);
      std::copy(real_words.begin(), real_words.end(), words.begin());
      std::copy(imag_words.begin(), imag_words.end(), words.begin() + real_words.size());
    });
  }
};
inline void add_complex_baselines(std::vector<Variant>& variants, Json& skipped, const ComplexData& data,
                                  const Options& options, const cpp_int& bound) {
  if (4 * uint64_t(options.m) * options.n * options.k <= options.baseline_max_products)
    variants.push_back(state_variant<ComplexBaseline<cpp_int>>("naive_cpp_int_complex_4m", data, options, false));
  else skipped.push_back({{"name", "naive_cpp_int_complex_4m"}, {"status", "not_run"},
      {"reason", "four-product naive comparator exceeds --baseline-max-products; full independent reference still computed"}});
  variants.push_back(state_variant<ComplexBaseline<cpp_int>>("blocked_cpp_int_complex_4m", data, options, true));
  if (bound <= INT64_MAX)
    variants.push_back(state_variant<ComplexBaseline<int64_t>>("blocked_native_i64_complex_4m", data, options, true));
  else skipped.push_back({{"name", "blocked_native_i64_complex_4m"}, {"status", "insufficient_range"},
                         {"reason", "complete component absolute partial-sum bound exceeds signed64"}});
#if defined(__SIZEOF_INT128__)
  if (bound < (cpp_int(1) << 127))
    variants.push_back(state_variant<ComplexBaseline<Native128>>("blocked_native_i128_complex_4m", data, options, true));
  else skipped.push_back({{"name", "blocked_native_i128_complex_4m"}, {"status", "insufficient_range"},
                         {"reason", "complete component absolute partial-sum bound exceeds signed128"}});
#else
  skipped.push_back({{"name", "blocked_native_i128_complex_4m"}, {"status", "unavailable"},
                    {"reason", "compiler has no genuine __int128; Boost fixed types are not native comparators"}});
#endif
}
} // namespace rns8_workload
#endif
