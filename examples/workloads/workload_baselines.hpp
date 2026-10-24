#ifndef RNS8_WORKLOAD_BASELINES_HPP
#define RNS8_WORKLOAD_BASELINES_HPP
#include "workload_common.hpp"

namespace rns8_workload {
inline bool checked_product(int64_t a, int64_t b, int64_t& product) {
  if ((a > 0 && b > 0 && a > INT64_MAX / b) ||
      (a > 0 && b < 0 && b < INT64_MIN / a) ||
      (a < 0 && b > 0 && a < INT64_MIN / b) ||
      (a < 0 && b < 0 && a < INT64_MAX / b)) return false;
  product = a * b; return true;
}
inline bool checked_sum(int64_t a, int64_t b, int64_t& sum) {
  if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return false;
  sum = a + b; return true;
}
template<class Integer, class Left> void naive_product(const std::vector<Left>& a, const std::vector<int64_t>& b,
    uint32_t m, uint32_t n, uint32_t k, std::vector<Integer>& c) {
  std::fill(c.begin(), c.end(), Integer(0));
  for (uint32_t row = 0; row < m; ++row) for (uint32_t col = 0; col < n; ++col)
    for (uint32_t q = 0; q < k; ++q)
      c[std::size_t(row) * n + col] += Integer(a[std::size_t(row) * k + q]) * Integer(b[std::size_t(q) * n + col]);
}
template<class Integer, class Left> void blocked_product(const std::vector<Left>& a, const std::vector<int64_t>& b,
    uint32_t m, uint32_t n, uint32_t k, std::vector<Integer>& c) {
  std::fill(c.begin(), c.end(), Integer(0));
  constexpr uint32_t tile = 32;
  for (uint32_t ii = 0; ii < m; ii += tile) for (uint32_t qq = 0; qq < k; qq += tile)
    for (uint32_t jj = 0; jj < n; jj += tile)
      for (uint32_t row = ii; row < std::min(m, ii + tile); ++row)
        for (uint32_t q = qq; q < std::min(k, qq + tile); ++q) {
          const Integer av = Integer(a[std::size_t(row) * k + q]);
          for (uint32_t col = jj; col < std::min(n, jj + tile); ++col)
            c[std::size_t(row) * n + col] += av * Integer(b[std::size_t(q) * n + col]);
        }
}
inline void encode_into(const std::vector<cpp_int>& values, uint32_t limbs, Words& result) {
  const cpp_int limit = cpp_int(1) << (64 * limbs), half = limit >> 1, mask = (cpp_int(1) << 64) - 1;
  require(result.size() == values.size() * limbs, "output serialization shape mismatch");
  for (std::size_t cell = 0; cell < values.size(); ++cell) {
    cpp_int value = values[cell]; require(value >= -half && value < half, "baseline export width insufficient");
    if (value < 0) value += limit;
    for (uint32_t limb = 0; limb < limbs; ++limb) {
      result[cell * limbs + limb] = (value & mask).convert_to<uint64_t>(); value >>= 64;
    }
  }
}
inline void encode_into(const std::vector<int64_t>& values, uint32_t limbs, Words& result) {
  for (std::size_t cell = 0; cell < values.size(); ++cell) {
    result[cell * limbs] = bits_of(values[cell]);
    for (uint32_t limb = 1; limb < limbs; ++limb) result[cell * limbs + limb] = values[cell] < 0 ? UINT64_MAX : 0;
  }
}
#if defined(__SIZEOF_INT128__)
__extension__ typedef __int128 Native128;
__extension__ typedef unsigned __int128 NativeU128;
inline void encode_into(const std::vector<Native128>& values, uint32_t limbs, Words& result) {
  for (std::size_t cell = 0; cell < values.size(); ++cell) {
    NativeU128 bits = static_cast<NativeU128>(values[cell]);
    for (uint32_t limb = 0; limb < limbs; ++limb) {
      result[cell * limbs + limb] = limb < 2 ? static_cast<uint64_t>(bits) : values[cell] < 0 ? UINT64_MAX : 0;
      if (limb < 2) bits >>= 64;
    }
  }
}
#endif
inline std::vector<cpp_int> exact_product(const Dataset& data, const Options& options, bool chain = false) {
  std::vector<cpp_int> first(std::size_t(options.m) * options.n);
  naive_product(data.a, data.b, options.m, options.n, options.k, first);
  if (!chain) return first;
  std::vector<cpp_int> result(first.size());
  // Independent straightforward full-expression loop, distinct from the blocked comparator.
  for (uint32_t row = 0; row < options.m; ++row) for (uint32_t col = 0; col < options.n; ++col)
    for (uint32_t q = 0; q < options.n; ++q)
      result[std::size_t(row) * options.n + col] += first[std::size_t(row) * options.n + q] * data.t[std::size_t(q) * options.n + col];
  return result;
}
template<class Integer> struct IntegerBaseline {
  const Dataset& data;
  const Options& options;
  bool blocked, chain;
  uint32_t limbs;
  std::vector<Integer> first, final;
  Words words;
  IntegerBaseline(const Dataset& input, const Options& config, bool tiled, bool chained, uint32_t width)
      : data(input), options(config), blocked(tiled), chain(chained), limbs(width),
        first(std::size_t(config.m) * config.n), final(chained ? first.size() : 0), words(first.size() * width) {}
  void execute(Phases& phases, uint64_t) {
    phases.measure("compute_and_output_serialization_ns", [&] {
      if (blocked) blocked_product(data.a, data.b, options.m, options.n, options.k, first);
      else naive_product(data.a, data.b, options.m, options.n, options.k, first);
      if (chain) {
        if (blocked) blocked_product(first, data.t, options.m, options.n, options.n, final);
        else naive_product(first, data.t, options.m, options.n, options.n, final);
        encode_into(final, limbs, words);
      } else encode_into(first, limbs, words);
    });
  }
};
template<class State, class... Arguments> Variant state_variant(const std::string& name, Arguments&&... arguments) {
  const auto start = Clock::now();
  auto state = std::make_shared<State>(std::forward<Arguments>(arguments)...);
  const double setup = since(start);
  Variant result;
  result.name = name; result.setup_ns = setup;
  result.execute = [state](Phases& phases, uint64_t version) { state->execute(phases, version); };
  result.output = [state]() -> const Words& { return state->words; };
  return result;
}
inline void add_integer_baselines(std::vector<Variant>& variants, Json& skipped, const Dataset& data,
    const Options& options, const cpp_int& bound, uint32_t limbs, bool chain = false) {
  const uint64_t products = (uint64_t(options.m) * options.n * options.k +
      (chain ? uint64_t(options.m) * options.n * options.n : 0))*std::max(options.gram_reuse,1u);
  if (products <= options.baseline_max_products)
    variants.push_back(state_variant<IntegerBaseline<cpp_int>>(chain ? "naive_cpp_int_chain" : "naive_cpp_int", data, options, false, chain, limbs));
  else skipped.push_back({{"name", "naive_cpp_int"}, {"status", "not_run"},
      {"reason", "timed naive comparator exceeds declared --baseline-max-products; independent full reference still computed once"}});
  if (bound <= INT64_MAX)
    variants.push_back(state_variant<IntegerBaseline<int64_t>>("blocked_native_i64", data, options, true, chain, limbs));
  else skipped.push_back({{"name", "blocked_native_i64"}, {"status", "unsupported"}, {"reason", "proven absolute partial-sum envelope exceeds INT64_MAX"}});
#if defined(__SIZEOF_INT128__)
  if (bound < (cpp_int(1) << 127))
    variants.push_back(state_variant<IntegerBaseline<Native128>>("blocked_native_i128", data, options, true, chain, limbs));
  else skipped.push_back({{"name", "blocked_native_i128"}, {"status", "unsupported"}, {"reason", "proven absolute envelope exceeds native signed128 range"}});
#else
  skipped.push_back({{"name", "blocked_native_i128"}, {"status", "unavailable"},
      {"reason", "compiler has no genuine __int128 type; Boost int128 is not substituted"}});
#endif
  variants.push_back(state_variant<IntegerBaseline<cpp_int>>("blocked_cpp_int", data, options, true, chain, limbs));
}
} // namespace rns8_workload
#endif
