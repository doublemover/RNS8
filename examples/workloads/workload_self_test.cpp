#include "workload_baselines.hpp"
#include "workload_engine.hpp"
#include "workload_report.hpp"
#include "workload_complex.hpp"
#include "workload_optional_references.hpp"
#include <cstdlib>
using namespace rns8_workload;
int main() {
  try {
    // Admission only: exercise large-index/work bounds without allocating data.
    for (uint32_t order : {1u, 15u, 16u, 17u, 32u, 33u, 128u, 512u, 1024u, 2048u})
      for (uint32_t k : {32u, 256u, 4096u, 65535u, 65536u, 65537u})
        for (uint32_t reuse : {1u, 2u, 8u, 32u}) {
          Options candidate;
          candidate.m = candidate.n = order;
          candidate.k = k;
          candidate.gram = true;
          candidate.gram_reuse = reuse;
          candidate.size = "large";
          candidate.memory_mib = 16384;
          bool admitted = true;
          try {
            validate_workload_extent(candidate);
          } catch (const std::invalid_argument&) {
            admitted = false;
          }
          require(admitted == (uint64_t(order) * order * k <= 134217728) &&
                      candidate.memory->reserved == 0,
                  "Gram extent admission changed the work bound or allocated storage");
          candidate.reference_max_products = uint64_t(order) * order * k;
          candidate.supplied.insert("--reference-max-products");
          validate_workload_extent(candidate);
          require(gram_reference_budget(candidate)["total_reference_products"] ==
                          2 * candidate.reference_max_products &&
                      candidate.memory->reserved == 0,
                  "Gram explicit oracle budget counted repeated products or allocated storage");
          --candidate.reference_max_products;
          bool too_small_rejected = false;
          try {
            validate_workload_extent(candidate);
          } catch (const std::invalid_argument&) {
            too_small_rejected = true;
          }
          require(too_small_rejected, "Gram admitted a reference larger than its explicit budget");
        }
    {
      // Admission arithmetic only: no dataset, comparator or GPU allocation.
      Options large;
      large.m = large.n = 2048;
      large.k = 4096;
      large.gram = true;
      large.gram_reuse = 32;
      large.size = "large";
      large.memory_mib = 131072;
      large.reference_max_products = uint64_t(large.m) * large.n * large.k;
      memory_admission(large);
      for (unsigned comparator = 0; comparator < 4; ++comparator)
        optional_reference_memory(large, false);
      require(large.memory->reserved < large.memory_mib * 1024 * 1024 &&
                  large.memory_mib * 1024 * 1024 - large.memory->reserved > (uint64_t{24} << 30),
              "large Gram conservative host/comparator models exhaust the explicit allowance");
      for (const bool extended : {false, true}) {
        Options invalid;
        invalid.gram = extended;
        invalid.gram_reuse = extended ? 1 : 0;
        invalid.size = extended ? "large" : "modest";
        invalid.memory_mib = extended ? 131073 : 4097;
        bool rejected = false;
        try {
          validate_workload_extent(invalid);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        require(rejected && invalid.memory->reserved == 0,
                "memory allowance escaped its explicit cohort or allocated storage");
      }
    }
    int64_t value = 0;
    require(!checked_product(INT64_MIN, -1, value), "min*-1 was not rejected");
    require(!checked_product(INT64_MAX, 2, value), "max*2 was not rejected");
    require(checked_product(INT64_MIN, 1, value) && value == INT64_MIN,
            "valid signed minimum rejected");
    require(checked_product(-3, -5, value) && value == 15, "negative product failed");
    require(!checked_sum(INT64_MAX, 1, value) && !checked_sum(INT64_MIN, -1, value),
            "checked addition failed");
    require(checked_sum(INT64_MIN, INT64_MAX, value) && value == -1, "cancellation failed");
    MemoryBudget budget;
    budget.reserve(60, 100, "first retained state");
    budget.reserve(30, 100, "second retained state");
    bool exceeded = false;
    try {
      budget.reserve(11, 100, "third retained state");
    } catch (const std::invalid_argument&) {
      exceeded = true;
    }
    require(exceeded && budget.reserved == 90,
            "aggregate memory reservation was not checked transactionally");
    {
      Options malformed;
      malformed.m = UINT32_MAX;
      const auto reserved = malformed.memory->reserved;
      bool rejected = false;
      try {
        optional_reference_memory(malformed, true);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected && malformed.memory->reserved == reserved,
              "optional memory arithmetic accepted an invalid shape");
    }
#if defined(RNS8_WORKLOAD_HAVE_GMP) || defined(RNS8_WORKLOAD_HAVE_FLINT)
    {
      bool rejected = false;
      try {
        OptionalWordEncoder invalid(0);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "optional zero-width encoder was accepted");
      OptionalWordEncoder encoder(3);
      WorkloadMpz one;
      mpz_set_ui(one.value, 1);
      Words short_output{77};
      rejected = false;
      try {
        encoder.store(one.value, short_output, 0);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected && short_output == Words{77},
              "optional encoder wrote past a malformed capacity");
      Words words(3, 77);
      rejected = false;
      try {
        encoder.store(one.value, words, 1);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected && words == Words(3, 77), "optional encoder accepted an out-of-range cell");
    }
#endif
    {
      Options config;
      config.m = 1;
      config.n = 1;
      config.k = 2;
      Dataset input = make_data(config, true);
      CPUComparatorLifecycle cache;
      std::array<std::vector<int64_t>, 3> converted;
      auto import = [&](unsigned role, const std::vector<int64_t>& values) {
        converted[role] = values;
      };
      cache.refresh(input, true, false, import);
      cache.refresh(input, true, false, import);
      require(cache.conversion_passes == std::array<uint64_t, 3>{1, 1, 1},
              "stable input generations were reconverted");
      ++input.source_versions[0];
      ++input.source_versions[1];
      input.a[0] += 1;
      input.b[0] += 1;
      bool failed = false;
      try {
        cache.refresh(input, true, false, [&](unsigned role, const std::vector<int64_t>& values) {
          if (role == 1) {
            converted[role][0] = values[0];
            throw std::invalid_argument("injected partial conversion");
          }
          import(role, values);
        });
      } catch (const std::invalid_argument&) {
        failed = true;
      }
      require(failed && !cache.current(input, true) && !cache.converted[1].valid,
              "failed refresh published a usable converted generation");
      cache.refresh(input, true, false, import);
      require(cache.current(input, true) && converted[0] == input.a && converted[1] == input.b,
              "failed converted-input refresh did not recover");
      require(cache.conversion_passes == std::array<uint64_t, 3>{2, 2, 1},
              "retry reconverted unrelated completed inputs");
      auto replacement = input.t;
      input.t.swap(replacement);
      cache.refresh(input, true, false, import);
      require(cache.conversion_passes[2] == 2, "changed input storage identity was reused");
    }
    for (const std::string policy : {"each", "once"}) {
      Options config;
      config.m = 1;
      config.n = 1;
      config.k = 2;
      config.gram_reuse = 3;
      config.gram_export = policy;
      Dataset input = make_data(config);
      input.a = {INT64_MIN, INT64_MAX};
      input.b = {1, 1};
      IntegerBaseline<int64_t> baseline(input, config, true, false, 1, true);
      Phases phases;
      phases.enabled = false;
      baseline.execute(phases, 1);
      require(baseline.words == integer_words(exact_product(input, config), 1),
              "matched native64 extrema/cancellation differ");
      require(baseline.lifecycle.computed_expressions == 3 &&
                  baseline.lifecycle.gemm_products == 3 &&
                  baseline.lifecycle.materialized_outputs == (policy == "each" ? 3u : 1u),
              "matched CPU batch skipped products or over-materialized");
      mutate_input(input);
      baseline.execute(phases, 2);
      require(baseline.words == integer_words(exact_product(input, config), 1),
              "matched native64 changed input differs");
      auto native_view =
          matched_integer_variant<int64_t>("blocked_native_i64", input, config, true, false, 1);
      native_view.execute(phases, 2);
      const auto words = integer_words(exact_product(input, config), 1);
      require(native_view.output() == words && checksum(native_view.output()) == checksum(words),
              "native64 retained output view changed public signed word bits");
      require(native_view.metadata["cpu_lifecycle"]["output_materialization"] == "native_i64_view",
              "native64 comparator did not expose its existing output representation");
      IntegerBaseline<int64_t> direct_view(input, config, true, false, 1, true, true);
      direct_view.execute(phases, 2);
      require(direct_view.words.empty() && WordView(direct_view.first) == words,
              "native64 view allocated or populated a redundant serialization buffer");
    }
    {
      Options config;
      config.m = 1;
      config.n = 1;
      config.k = 2;
      config.gram_reuse = 3;
      config.gram_export = "once";
      Dataset input = make_data(config, true);
      input.a = {INT64_MAX, INT64_MAX};
      input.b = {INT64_MAX, INT64_MAX};
      input.t = {INT64_MAX};
      const auto expected = exact_product(input, config, true);
      require(expected[0] > (cpp_int(1) << 128), "wide CPU fixture did not exceed128 bits");
      IntegerBaseline<cpp_int> boost(input, config, true, true, 3, true);
      Phases phases;
      phases.enabled = false;
      boost.execute(phases, 1);
      require(boost.words == integer_words(expected, 3) &&
                  boost.lifecycle.computed_expressions == 3 && boost.lifecycle.gemm_products == 6 &&
                  boost.lifecycle.materialized_outputs == 1,
              "matched Boost chain did not recompute all products before final serialization");
#if defined(RNS8_WORKLOAD_HAVE_GMP) || defined(RNS8_WORKLOAD_HAVE_FLINT)
      auto check_retained = [&](auto& baseline) {
        baseline.execute(phases, 1);
        baseline.execute(phases, 1);
        require(baseline.words == integer_words(exact_product(input, config, true), 3),
                "retained optional wide output differs");
        require(baseline.lifecycle.conversion_passes == std::array<uint64_t, 3>{1, 1, 1},
                "optional steady input generations were reconverted");
        input.a[0] = -input.a[0];
        ++input.source_versions[0];
        baseline.execute(phases, 2);
        input.b[0] = -input.b[0];
        ++input.source_versions[1];
        baseline.execute(phases, 2);
        input.t[0] = -input.t[0];
        ++input.source_versions[2];
        baseline.execute(phases, 2);
        require(baseline.words == integer_words(exact_product(input, config, true), 3),
                "retained optional A/B/T refresh differs from independent Boost");
        require(baseline.lifecycle.conversion_passes == std::array<uint64_t, 3>{2, 2, 2} &&
                    baseline.lifecycle.computed_expressions == 15 &&
                    baseline.lifecycle.gemm_products == 30 &&
                    baseline.lifecycle.materialized_outputs == 5,
                "optional retained lifecycle operation counts differ");
      };
#endif
#if defined(RNS8_WORKLOAD_HAVE_GMP)
      {
        GmpBaseline gmp(input, config, true, 3, true);
        check_retained(gmp);
      }
#endif
#if defined(RNS8_WORKLOAD_HAVE_FLINT)
      {
        FlintBaseline flint(input, config, true, 3, true);
        check_retained(flint);
      }
#endif
    }
    uint64_t state = 0;
    require(next_word(state) == UINT64_C(0xe220a8397b1dcdaf), "splitmix64 version changed");
    const Words encoded = integer_words({cpp_int(-1), cpp_int(0), (cpp_int(1) << 129)}, 3);
    require(decode(encoded, 0, 3) == -1 && decode(encoded, 2, 3) == (cpp_int(1) << 129),
            "signed limbs lost information");
    for (const auto shape : {std::array<uint32_t, 3>{1, 1, 1}, {2, 3, 17}, {17, 5, 33}}) {
      Options options;
      options.m = shape[0];
      options.n = shape[1];
      options.k = shape[2];
      options.bits = 8;
      Dataset data = make_data(options);
      const auto expected = exact_product(data, options);
      std::vector<int64_t> native(expected.size());
      blocked_product(data.a, data.b, options.m, options.n, options.k, native);
      Words words(native.size() * 3);
      encode_into(native, 3, words);
      require(words == integer_words(expected, 3),
              "blocked native baseline differs from independent Boost");
      std::vector<cpp_int> wide(expected.size());
      blocked_product(data.a, data.b, options.m, options.n, options.k, wide);
      require(wide == expected, "blocked cpp_int baseline differs");
      std::vector<Variant> optional;
      Json omitted = Json::array();
      add_optional_integer_baselines(optional, omitted, data, options, 3);
      for (auto& variant : optional) {
        Phases phases;
        variant.execute(phases, 1);
        require(variant.output() == integer_words(expected, 3),
                "optional exact matrix comparator differs from Boost");
        mutate_input(data);
        variant.execute(phases, 2);
        require(variant.output() == integer_words(exact_product(data, options), 3),
                "optional changed native inputs were not converted");
        data = make_data(options);
      }
#if defined(__SIZEOF_INT128__)
      std::vector<Native128> native128(expected.size());
      blocked_product(data.a, data.b, options.m, options.n, options.k, native128);
      encode_into(native128, 3, words);
      require(words == integer_words(expected, 3), "genuine native128 baseline differs");
#endif
    }
    Options tiny;
    tiny.m = tiny.n = 2;
    tiny.k = 3;
    tiny.size = "tiny";
    Dataset data = make_data(tiny, true);
    require(exact_product(data, tiny) == std::vector<cpp_int>({20, -27, 4, -14}),
            "hand product changed");
    require(exact_product(data, tiny, true) == std::vector<cpp_int>({67, -34, 22, -24}),
            "hand chain changed");
    auto complex = make_complex_data(tiny);
    const auto pair = complex_reference(complex, tiny);
    const std::vector<cpp_int> known_pair{19, -25, 9, -19, -12, 26, 14, 10};
    for (std::size_t cell = 0; cell < known_pair.size(); ++cell)
      require(decode(pair, cell, 3) == known_pair[cell], "hand split-complex product changed");
    std::vector<cpp_int> complex_real(4), complex_imag(4);
    complex_four_product(complex, tiny, true, complex_real, complex_imag);
    require(complex_real == std::vector<cpp_int>({19, -25, 9, -19}) &&
                complex_imag == std::vector<cpp_int>({-12, 26, 14, 10}),
            "blocked complex baseline differs");
    require(complex_final_bound(complex, tiny) == 207, "complex complete-expression bound changed");
    mutate_complex_input(complex);
    const auto changed_pair = complex_reference(complex, tiny);
    const std::vector<cpp_int> known_changed{20, -17, 9, -19, -12, 27, 14, 10};
    for (std::size_t cell = 0; cell < known_changed.size(); ++cell)
      require(decode(changed_pair, cell, 3) == known_changed[cell], "complex generation2 differs");
    Options complex_edge;
    complex_edge.m = complex_edge.n = complex_edge.k = 1;
    ComplexData full_complex;
    full_complex.values = {{{INT64_MIN}, {INT64_MAX}, {INT64_MAX}, {INT64_MIN}}};
    const auto full_pair = complex_reference(full_complex, complex_edge);
    require(decode(full_pair, 0, 3) == 0 &&
                decode(full_pair, 1, 3) == (cpp_int(1) << 127) - (cpp_int(1) << 64) + 1,
            "full-width complex exact reference lost signed extrema");
    tiny.gram = true;
    data = make_data(tiny);
    require(exact_product(data, tiny) == std::vector<cpp_int>({14, -22, -22, 161}),
            "Gram transpose changed");
    for (uint32_t reuse : {1u, 3u})
      for (const std::string policy : {"each", "once"}) {
        Options cohort = tiny;
        cohort.backend = RNS8_BACKEND_CPU_REFERENCE;
        cohort.device = -1;
        cohort.gram_reuse = reuse;
        cohort.gram_export = policy;
        cohort.memory = std::make_shared<MemoryBudget>();
        Dataset input = make_data(cohort);
        const cpp_int bound =
            cpp_int(cohort.k) * maximum_after_mutation(input) * maximum_after_change(input.b);
        auto dense = dense_variant(input, cohort, bound);
        auto single = gram_variant(input, cohort, bound, RNS8_GRAM_SCALAR);
        auto prepared = gram_variant(input, cohort, bound, RNS8_GRAM_SCALAR, true);
        const auto original = integer_words(exact_product(input, cohort), 1);
        for (auto* variant : {&dense, &single, &prepared}) {
          const auto first = verified_sample(*variant, 1, original);
          require(first["gram_products"] == reuse &&
                      first["rns_exports"] == (policy == "each" ? reuse : 1u),
                  "Gram batch lost product/export counts");
          Phases repeated;
          variant->execute(repeated, 1);
          require(variant->output() == original && !repeated.values.contains("pack_and_upload_ns"),
                  "Gram reuse repacked or changed output");
        }
        mutate_input(input);
        input.b[0] = input.a[0];
        const auto changed = integer_words(exact_product(input, cohort), 1);
        for (auto* variant : {&dense, &single, &prepared}) {
          Phases refreshed;
          variant->execute(refreshed, 2);
          require(variant->output() == changed &&
                      refreshed.values.contains("changed_input_pack_and_prepared_refresh_ns"),
                  "Gram generation2 did not refresh");
        }
        require(dense.metadata["gram_batch"]["implementation"] == "public_prepared_dense" &&
                    single.metadata["gram_batch"]["prepared_gram_api"] == false,
                "Gram preparation boundary was mislabeled");
        require(single.metadata["independent_integer_products_per_call"] == 9 * reuse &&
                    single.metadata["dense_equivalent_integer_products_per_call"] == 12 * reuse,
                "Gram mathematical work omitted batch reuse");
        const auto refreshed = prepared.execution_evidence();
        require(refreshed["generation_action"] == "refreshed_owner" &&
                    refreshed["owner_builds"] == 1 && refreshed["owner_refreshes"] == 1,
                "prepared Gram changed-input refresh was mislabeled");
      }
    {
      Options zero = tiny;
      zero.pattern = "zero";
      zero.gram_reuse = 3;
      zero.backend = RNS8_BACKEND_CPU_REFERENCE;
      zero.device = -1;
      zero.memory = std::make_shared<MemoryBudget>();
      Dataset input = make_data(zero);
      auto prepared = gram_variant(input, zero, cpp_int(3), RNS8_GRAM_SCALAR, true);
      const auto first = verified_sample(prepared, 1, integer_words(exact_product(input, zero), 1));
      require(first["execution_evidence"]["owner_first_execution"] == true,
              "prepared Gram first-use setup was hidden");
      mutate_input(input);
      input.b[0] = input.a[0];
      const auto changed =
          verified_sample(prepared, 2, integer_words(exact_product(input, zero), 1));
      require(changed["execution_evidence"]["generation_action"] ==
                      "recreated_owner_for_larger_bound" &&
                  changed["execution_evidence"]["owner_builds"] == 2 &&
                  changed["execution_evidence"]["owner_refreshes"] == 0 &&
                  changed["execution_evidence"]["owner_first_execution"] == true,
              "expanded Gram proof did not recreate explicitly");
    }
    Dataset full;
    full.a = {INT64_MIN, INT64_MAX};
    full.b = {1, 1};
    Options edge;
    edge.m = edge.n = 1;
    edge.k = 2;
    require(exact_product(full, edge)[0] == -1, "full-width reference failed");
    {
      Options wide;
      wide.m = wide.n = wide.k = 2;
      Dataset input;
      input.a = {INT64_MIN, INT64_MAX, INT64_MAX, INT64_MIN};
      input.b = {INT64_MIN, INT64_MAX, INT64_MAX, INT64_MIN};
      std::vector<Variant> optional;
      Json omitted = Json::array();
      add_optional_integer_baselines(optional, omitted, input, wide, 3);
      const auto exact = integer_words(exact_product(input, wide), 3);
      for (auto& variant : optional) {
        Phases phases;
        variant.execute(phases, 1);
        require(variant.output() == exact, "optional full64 product lost information");
      }
    }
    {
      Options chain;
      chain.m = 1;
      chain.n = 8;
      chain.k = 1;
      Dataset input;
      input.a = {INT64_C(1) << 31};
      input.b.assign(8, INT64_C(1) << 31);
      input.t.assign(64, INT64_MAX);
      const auto exact = integer_words(exact_product(input, chain, true), 3);
      require(decode(exact, 0, 3) > (cpp_int(1) << 127),
              "optional chain witness does not exceed native128");
      std::vector<Variant> optional;
      Json omitted = Json::array();
      add_optional_integer_baselines(optional, omitted, input, chain, 3, true);
      for (auto& variant : optional) {
        Phases phases;
        variant.execute(phases, 1);
        require(variant.output() == exact, "optional complete chain differs from Boost");
      }
    }
    bool rejected = false;
    try {
      integer_words({cpp_int(1) << 70}, 1);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "narrow serialization silently truncated");
    Words produced{0};
    uint32_t executions = 0;
    Variant measured;
    measured.name = "sustained_fixture";
    measured.execute = [&](Phases& phases, uint64_t version) {
      require(version == 1 && !phases.enabled,
              "sustained call changed source identity or enabled phase clocks");
      phases.measure("disabled", [&] {
        produced[0] = 42;
        ++executions;
      });
      require(phases.values.empty(), "disabled phase timing still recorded a sample");
    };
    measured.output = [&]() -> const Words& { return produced; };
    const auto sustained = sustained_sample(measured, 1, Words{42});
    require(sustained["calls"] == executions && executions > 0 && executions <= 4096,
            "sustained report lost application calls");
    rejected = false;
    try {
      sustained_sample(measured, 1, Words{43});
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "sustained measurement accepted an incorrect final output");
    std::cout << "workload independent baselines: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
