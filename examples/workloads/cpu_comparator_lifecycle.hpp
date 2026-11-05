#ifndef RNS8_CPU_COMPARATOR_LIFECYCLE_HPP
#define RNS8_CPU_COMPARATOR_LIFECYCLE_HPP
#include "workload_common.hpp"
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace rns8_workload {
// Keep every requested expression real even when only the final identical
// output is observed. This is a compiler boundary, not an extra data copy.
inline void preserve_cpu_product(const void* output) {
#if defined(_MSC_VER)
  (void)output;
  _ReadWriteBarrier();
#elif defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("" : : "g"(output) : "memory");
#else
  (void)output;
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

struct ConvertedSource {
  uint64_t identity = 0, generation = 0;
  const int64_t* address = nullptr;
  std::size_t cells = 0;
  bool valid = false;
  bool matches(const Dataset& data, unsigned role, const std::vector<int64_t>& input) const {
    return valid && identity == data.identity && generation == data.source_versions[role]
        && address == input.data() && cells == input.size();
  }
};

struct CPUComparatorLifecycle {
  std::array<ConvertedSource, 3> converted{};
  std::array<uint64_t, 3> conversion_passes{}, converted_values{};
  uint64_t computed_expressions = 0, gemm_products = 0, materialized_outputs = 0;

  template<class Convert>
  void refresh(const Dataset& data, bool chain, bool force, Convert&& convert) {
    const std::array<const std::vector<int64_t>*, 3> sources{&data.a, &data.b, &data.t};
    for (unsigned role = 0; role < (chain ? 3u : 2u); ++role) {
      const auto& input = *sources[role];
      require(data.source_versions[role] != 0, "CPU input generation must be nonzero");
      auto& bound = converted[role];
      if (!force && bound.matches(data, role, input)) continue;
      // A failed partial import cannot leave a reusable generation receipt.
      bound.valid = false;
      require(conversion_passes[role] != UINT64_MAX && input.size() <= UINT64_MAX - converted_values[role],
          "CPU conversion counters exhausted");
      convert(role, input);
      bound = {data.identity, data.source_versions[role], input.data(), input.size(), true};
      ++conversion_passes[role];
      converted_values[role] += input.size();
    }
  }

  void computed(bool chain, const void* output) {
    preserve_cpu_product(output);
    const uint64_t products = chain ? 2 : 1;
    require(computed_expressions != UINT64_MAX && gemm_products <= UINT64_MAX - products,
        "CPU product counters exhausted");
    ++computed_expressions; gemm_products += products;
  }
  void materialized() {
    require(materialized_outputs != UINT64_MAX, "CPU materialization counters exhausted");
    ++materialized_outputs;
  }

  bool current(const Dataset& data, bool chain) const {
    return converted[0].matches(data, 0, data.a) && converted[1].matches(data, 1, data.b)
        && (!chain || converted[2].matches(data, 2, data.t));
  }

  Json evidence(const Dataset& data, uint64_t retained_bytes, bool has_conversion) const {
    Json generations = Json::array();
    for (const auto& source : converted) generations.push_back(source.valid ? Json(source.generation) : Json(nullptr));
    return {{"schema", "rns8_cpu_comparator_lifecycle_v2"}, {"source_identity", data.identity},
        {"source_generations", data.source_versions}, {"converted_generations", generations},
        {"conversion_passes", conversion_passes}, {"converted_values", converted_values},
        {"computed_expressions", computed_expressions}, {"gemm_products", gemm_products},
        {"materialized_outputs", materialized_outputs}, {"retained_static_storage_bytes", retained_bytes},
        {"converted_inputs", has_conversion}, {"dynamic_library_peak_measured", false}};
  }
};

inline uint32_t cpu_batch_products(const Options& options) { return std::max(options.gram_reuse, 1u); }
inline bool cpu_materialize_each(const Options& options) {
  return !options.gram_reuse || options.gram_export == "each";
}

template<class State>
void execute_matched_cpu_batch(State& state, Phases& phases, uint64_t version, const Options& options) {
  state.prepare(phases, version);
  const auto products = cpu_batch_products(options);
  for (uint32_t product = 0; product < products; ++product) {
    state.compute(phases);
    if (cpu_materialize_each(options) || product + 1 == products) state.materialize(phases);
  }
}

inline Json cpu_lifecycle_contract(const Options& options, bool converted, bool native_view = false) {
  return {{"version", 2}, {"input_conversion_policy", converted ? "generation_bound_converted_inputs" : "native_inputs"},
      {"materialization_policy", cpu_materialize_each(options) ? "each" : "once"},
      {"computed_expressions_per_batch", cpu_batch_products(options)},
      {"materialized_outputs_per_batch", cpu_materialize_each(options) ? cpu_batch_products(options) : 1},
      {"output_cached", false}, {"recomputation_policy", "compiler_memory_barrier_after_each_expression_v1"}, {"output_materialization", native_view ? "native_i64_view" : "checked_signed_word_serialization"}, {"initial_conversion_charged", converted ? "variant_setup" : "not_applicable_native_inputs"},
      {"source_identity_policy", "dataset_identity_address_length_and_explicit_operand_generation"},
      {"complete_caller_output_per_batch", true}};
}
} // namespace rns8_workload
#endif
