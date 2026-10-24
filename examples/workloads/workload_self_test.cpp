#include "workload_baselines.hpp"
#include "workload_engine.hpp"
#include <cstdlib>
using namespace rns8_workload;
int main() {
  try {
    int64_t value = 0;
    require(!checked_product(INT64_MIN, -1, value), "min*-1 was not rejected");
    require(!checked_product(INT64_MAX, 2, value), "max*2 was not rejected");
    require(checked_product(INT64_MIN, 1, value) && value == INT64_MIN, "valid signed minimum rejected");
    require(checked_product(-3, -5, value) && value == 15, "negative product failed");
    require(!checked_sum(INT64_MAX, 1, value) && !checked_sum(INT64_MIN, -1, value), "checked addition failed");
    require(checked_sum(INT64_MIN, INT64_MAX, value) && value == -1, "cancellation failed");
    MemoryBudget budget; budget.reserve(60, 100, "first retained state");
    budget.reserve(30, 100, "second retained state");
    bool exceeded = false;
    try { budget.reserve(11, 100, "third retained state"); } catch (const std::invalid_argument&) { exceeded = true; }
    require(exceeded && budget.reserved == 90, "aggregate memory reservation was not checked transactionally");
    uint64_t state = 0;
    require(next_word(state) == UINT64_C(0xe220a8397b1dcdaf), "splitmix64 version changed");
    const Words encoded = integer_words({cpp_int(-1), cpp_int(0), (cpp_int(1) << 129)}, 3);
    require(decode(encoded, 0, 3) == -1 && decode(encoded, 2, 3) == (cpp_int(1) << 129), "signed limbs lost information");
    for (const auto shape : {std::array<uint32_t, 3>{1, 1, 1}, {2, 3, 17}, {17, 5, 33}}) {
      Options options; options.m = shape[0]; options.n = shape[1]; options.k = shape[2]; options.bits = 8;
      Dataset data = make_data(options);
      const auto expected = exact_product(data, options);
      std::vector<int64_t> native(expected.size());
      blocked_product(data.a, data.b, options.m, options.n, options.k, native);
      Words words(native.size() * 3); encode_into(native, 3, words);
      require(words == integer_words(expected, 3), "blocked native baseline differs from independent Boost");
      std::vector<cpp_int> wide(expected.size());
      blocked_product(data.a, data.b, options.m, options.n, options.k, wide);
      require(wide == expected, "blocked cpp_int baseline differs");
#if defined(__SIZEOF_INT128__)
      std::vector<Native128> native128(expected.size());
      blocked_product(data.a, data.b, options.m, options.n, options.k, native128);
      encode_into(native128, 3, words);
      require(words == integer_words(expected, 3), "genuine native128 baseline differs");
#endif
    }
    Options tiny; tiny.m = tiny.n = 2; tiny.k = 3; tiny.size = "tiny";
    Dataset data = make_data(tiny, true);
    require(exact_product(data, tiny) == std::vector<cpp_int>({20, -27, 4, -14}), "hand product changed");
    require(exact_product(data, tiny, true) == std::vector<cpp_int>({67, -34, 22, -24}), "hand chain changed");
    tiny.gram = true; data = make_data(tiny);
    require(exact_product(data, tiny) == std::vector<cpp_int>({14, -22, -22, 161}), "Gram transpose changed");
    Dataset full; full.a = {INT64_MIN, INT64_MAX}; full.b = {1, 1};
    Options edge; edge.m = edge.n = 1; edge.k = 2;
    require(exact_product(full, edge)[0] == -1, "full-width reference failed");
    bool rejected = false;
    try { integer_words({cpp_int(1) << 70}, 1); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "narrow serialization silently truncated");
    std::cout << "workload independent baselines: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return EXIT_FAILURE; }
}
