#include "workload_baselines.hpp"
#include "workload_engine.hpp"
#include "workload_report.hpp"
#include "workload_complex.hpp"
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
    auto complex = make_complex_data(tiny);
    const auto pair = complex_reference(complex, tiny);
    const std::vector<cpp_int> known_pair{19, -25, 9, -19, -12, 26, 14, 10};
    for (std::size_t cell = 0; cell < known_pair.size(); ++cell)
      require(decode(pair, cell, 3) == known_pair[cell], "hand split-complex product changed");
    std::vector<cpp_int> complex_real(4), complex_imag(4);
    complex_four_product(complex, tiny, true, complex_real, complex_imag);
    require(complex_real == std::vector<cpp_int>({19, -25, 9, -19}) &&
            complex_imag == std::vector<cpp_int>({-12, 26, 14, 10}), "blocked complex baseline differs");
    require(complex_final_bound(complex, tiny) == 207, "complex complete-expression bound changed");
    mutate_complex_input(complex);
    const auto changed_pair = complex_reference(complex, tiny);
    const std::vector<cpp_int> known_changed{20, -17, 9, -19, -12, 27, 14, 10};
    for (std::size_t cell = 0; cell < known_changed.size(); ++cell)
      require(decode(changed_pair, cell, 3) == known_changed[cell], "complex generation2 differs");
    Options complex_edge; complex_edge.m = complex_edge.n = complex_edge.k = 1;
    ComplexData full_complex;
    full_complex.values = {{{INT64_MIN}, {INT64_MAX}, {INT64_MAX}, {INT64_MIN}}};
    const auto full_pair = complex_reference(full_complex, complex_edge);
    require(decode(full_pair, 0, 3) == 0 &&
            decode(full_pair, 1, 3) == (cpp_int(1) << 127) - (cpp_int(1) << 64) + 1,
            "full-width complex exact reference lost signed extrema");
    tiny.gram = true; data = make_data(tiny);
    require(exact_product(data, tiny) == std::vector<cpp_int>({14, -22, -22, 161}), "Gram transpose changed");
    for(uint32_t reuse:{1u,3u}) for(const std::string policy:{"each","once"}) {
      Options cohort=tiny;cohort.backend=RNS8_BACKEND_CPU_REFERENCE;cohort.device=-1;
      cohort.gram_reuse=reuse;cohort.gram_export=policy;
      cohort.memory=std::make_shared<MemoryBudget>();
      Dataset input=make_data(cohort);
      const cpp_int bound=cpp_int(cohort.k)*maximum_after_mutation(input)*maximum_after_change(input.b);
      auto dense=dense_variant(input,cohort,bound);
      auto single=gram_variant(input,cohort,bound,RNS8_GRAM_SCALAR);
      const auto original=integer_words(exact_product(input,cohort),1);
      for(auto* variant:{&dense,&single}) {
        const auto first=verified_sample(*variant,1,original);
        require(first["gram_products"]==reuse && first["rns_exports"]==(policy=="each"?reuse:1u),"Gram batch lost product/export counts");
        Phases repeated;variant->execute(repeated,1);
        require(variant->output()==original && !repeated.values.contains("pack_and_upload_ns"),"Gram reuse repacked or changed output");
      }
      mutate_input(input);input.b[0]=input.a[0];
      const auto changed=integer_words(exact_product(input,cohort),1);
      for(auto* variant:{&dense,&single}) {
        Phases refreshed;variant->execute(refreshed,2);
        require(variant->output()==changed && refreshed.values.contains("changed_input_pack_and_prepared_refresh_ns"),"Gram generation2 did not refresh");
      }
      require(dense.metadata["gram_batch"]["implementation"]=="public_prepared_dense" &&
          single.metadata["gram_batch"]["prepared_gram_api"]==false,"Gram preparation boundary was mislabeled");
      require(single.metadata["independent_integer_products_per_call"]==9*reuse &&
          single.metadata["dense_equivalent_integer_products_per_call"]==12*reuse,"Gram mathematical work omitted batch reuse");
    }
    Dataset full; full.a = {INT64_MIN, INT64_MAX}; full.b = {1, 1};
    Options edge; edge.m = edge.n = 1; edge.k = 2;
    require(exact_product(full, edge)[0] == -1, "full-width reference failed");
    bool rejected = false;
    try { integer_words({cpp_int(1) << 70}, 1); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "narrow serialization silently truncated");
    Words produced{0};
    uint32_t executions = 0;
    Variant measured;
    measured.name = "sustained_fixture";
    measured.execute = [&](Phases& phases, uint64_t version) {
      require(version == 1 && !phases.enabled, "sustained call changed source identity or enabled phase clocks");
      phases.measure("disabled", [&] { produced[0] = 42; ++executions; });
      require(phases.values.empty(), "disabled phase timing still recorded a sample");
    };
    measured.output = [&]() -> const Words& { return produced; };
    const auto sustained = sustained_sample(measured, 1, Words{42});
    require(sustained["calls"] == executions && executions > 0 && executions <= 4096,
            "sustained report lost application calls");
    rejected = false;
    try { sustained_sample(measured, 1, Words{43}); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "sustained measurement accepted an incorrect final output");
    std::cout << "workload independent baselines: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return EXIT_FAILURE; }
}
