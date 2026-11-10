#ifndef RNS8_WORKLOAD_ENGINE_HPP
#define RNS8_WORKLOAD_ENGINE_HPP
#include "workload_common.hpp"
namespace rns8_workload {
inline bool gram_matrix_method(int method) {
  return method == RNS8_GRAM_WMMA || method == RNS8_GRAM_WMMA_SEPARATE_MIRROR ||
         method == RNS8_GRAM_MFMA_GFX942 || method == RNS8_GRAM_MFMA_GFX942_SEPARATE_MIRROR;
}
inline bool gram_mirror_method(int method) {
  return method == RNS8_GRAM_SCALAR_SEPARATE_MIRROR || method == RNS8_GRAM_WMMA_SEPARATE_MIRROR ||
         method == RNS8_GRAM_MFMA_GFX942_SEPARATE_MIRROR;
}
inline const char* gram_method_name(int method) {
  return method == RNS8_GRAM_MFMA_GFX942 || method == RNS8_GRAM_MFMA_GFX942_SEPARATE_MIRROR
             ? "mfma_gfx942"
         : gram_matrix_method(method) ? "wmma"
                                      : "scalar";
}
inline cpp_int maximum_after_change(const std::vector<int64_t>& values) {
  const int64_t changed = values[0] == INT64_MAX ? values[0] - 1 : values[0] + 1;
  return std::max(maximum(values), cpp_int(magnitude(changed)));
}
inline cpp_int maximum_after_mutation(const Dataset& data) {
  return maximum_after_change(data.a);
}
struct WorkloadPreparedDeleter {
  void operator()(rns8_prepared_execution* value) const noexcept {
    (void)rns8_destroy_prepared_execution(value);
  }
};
struct WorkloadPreparedGramDeleter {
  void operator()(rns8_prepared_gram* value) const noexcept {
    (void)rns8_destroy_prepared_gram(value);
  }
};
struct DenseEngine {
  const Dataset& data;
  const Options& options;
  rns8::Context context;
  std::unique_ptr<rns8::Plan> plan;
  std::unique_ptr<rns8::Workspace> workspace;
  std::unique_ptr<rns8::Matrix> a, b, c;
  std::unique_ptr<rns8_prepared_execution, WorkloadPreparedDeleter> prepared;
  std::unique_ptr<rns8_prepared_gram, WorkloadPreparedGramDeleter> prepared_gram;
  uint64_t packed_version = 0;
  uint32_t gram_owner_builds = 0, gram_owner_refreshes = 0;
  bool prepared_gram_route = false, gram_owner_used = false, last_owner_first_execution = false;
  const char* gram_generation_action = "retained_owner";
  Words words;
  std::vector<int64_t> native;
  Json metadata;
  uint32_t prefix = 0;
  int gram_method = -1;
  rns8_gram_input_policy gram_input = RNS8_GRAM_RESIDENT_RESIDUES;
  bool gram_once = false;
  bool gram_five = false;
  uint64_t gram_payload_reservation = 0;
  DenseEngine(const Dataset& input, const Options& config, const cpp_int& bound, int gram = -1,
              bool gram_owner = false, rns8_gram_input_policy policy = RNS8_GRAM_RESIDENT_RESIDUES,
              bool once = false, bool five = false)
      : data(input), options(config), context(config.device, config.backend),
        prepared_gram_route(gram_owner),
        words(std::size_t(config.m) * config.n * (config.wide ? 3 : 1)),
        native(std::size_t(config.m) * config.n), gram_method(gram), gram_input(policy),
        gram_once(once), gram_five(five) {
    require(!prepared_gram_route || (options.gram_reuse && gram_method >= 0),
            "prepared Gram route requires an explicit Gram batch");
    require(!gram_five || (prepared_gram_route && policy == RNS8_GRAM_RESIDENT_RESIDUES && !once &&
                           gram <= int(RNS8_GRAM_WMMA)),
            "five-product comparison requires a resident prepared scalar/matrix recipe");
    metadata["device"] = device_info(context, options);
    const auto desc = gemm_desc(options, options.wide ? RNS8_EXACT_WIDE_SIGNED : RNS8_BOUNDED_I64,
                                options.m, options.n, options.k, bound);
    plan = std::make_unique<rns8::Plan>(context, desc);
    require(plan->backend_info().backend == options.backend,
            "plan changed the explicitly requested backend");
    prefix = plan->schedule_info().max_selected_prefix;
    const auto matrix = [&](uint32_t rows, uint32_t cols, bool retained_native = false) {
      auto descriptor = matrix_desc(rows, cols, desc.semantics, desc.bound_kind, prefix);
      if (retained_native)
        descriptor.flags |= RNS8_MATRIX_RETAIN_NATIVE_INPUT;
      return std::make_unique<rns8::Matrix>(context, descriptor);
    };
    const uint64_t modeled_bytes =
        (uint64_t(options.m) * options.k + uint64_t(options.k) * options.n +
         uint64_t(options.m) * options.n) *
        128;
    const uint64_t workspace_bytes = plan->backend_info().workspace_required_bytes;
    const uint64_t total_memory = context.device_info().global_mem_bytes;
    require(
        !total_memory || modeled_bytes + workspace_bytes <= total_memory,
        "estimated device storage/workspace exceeds total device memory; free-memory availability remains runtime checked");
    options.memory->reserve(modeled_bytes + workspace_bytes, options.memory_mib * 1024 * 1024,
                            "retained dense candidate storage/workspace");
    workspace = std::make_unique<rns8::Workspace>(context, *plan);
    if (gram_method < 0)
      a = matrix(options.m, options.k);
    b = matrix(options.k, options.n, gram_input == RNS8_GRAM_RETAINED_NATIVE_I64);
    c = matrix(options.m, options.n);
    metadata["plan"] = plan_info(*plan);
    metadata["storage"] = {{"b", storage_info(*b)}, {"c", storage_info(*c)}};
    if (a)
      metadata["storage"]["a"] = storage_info(*a);
    metadata["single_source_gram"] = gram_method >= 0;
    if (gram_method >= 0) {
      metadata["independent_integer_products_per_call"] =
          uint64_t(options.n) * (options.n + 1) / 2 * options.k * std::max(options.gram_reuse, 1u);
      metadata["dense_equivalent_integer_products_per_call"] =
          uint64_t(options.n) * options.n * options.k * std::max(options.gram_reuse, 1u);
      metadata["gram_method"] = (gram_method & 1) ? "wmma" : "scalar";
    }
    metadata["modeled_device_storage_envelope_bytes"] = modeled_bytes;
    if (options.gram_reuse) {
      options.memory->reserve(4096, options.memory_mib * UINT64_C(1024) * 1024,
                              "Gram retained preparation metadata allowance");
      pack(1);
      if (gram_method < 0) {
        rns8_prepared_execution* handle = nullptr;
        rns8::check(rns8_prepare_gemm(context.get(), plan->get(), a->get(), b->get(), c->get(),
                                      workspace->get(), &handle));
        prepared.reset(handle);
      } else if (prepared_gram_route) {
        const uint64_t tiles = (uint64_t(options.n) + 15) / 16;
        const uint64_t jobs = tiles * (tiles + 1) / 2;
        const uint64_t job_envelope = jobs * 16 + 4;
        options.memory->reserve(job_envelope, options.memory_mib * UINT64_C(1024) * 1024,
                                "modeled retained Gram host/device job payload");
        metadata["modeled_retained_gram_job_payload_bytes"] = job_envelope;
        // Admit all planes before preparation; the queried compact recipe can
        // retain fewer. This is a payload bound, not a process allocation peak.
        const uint64_t half_n = (uint64_t(options.n) + 1) / 2;
        const uint64_t half_k = ((uint64_t(options.k) + 1) / 2 + 1) & ~UINT64_C(1);
        const uint64_t extra =
            gram_five
                ? prefix * (2 * half_n * half_k + 5 * half_n * half_n +
                            uint64_t(options.n) * options.n) +
                      3
                : (gram_input == RNS8_GRAM_RETAINED_TRANSPOSE
                       ? uint64_t(options.k) * options.n * prefix
                       : 0) +
                      (gram_once ? uint64_t(options.n) * options.n * (options.wide ? 3 : 1) * 8 *
                                       (options.backend == RNS8_BACKEND_HIP_DIRECT ? 2 : 1)
                                 : 0) +
                      prefix + 16;
        options.memory->reserve(extra, options.memory_mib * UINT64_C(1024) * 1024,
                                "checked Gram adapter/export payload envelope");
        gram_payload_reservation = job_envelope + extra;
        prepare_gram_owner();
      }
      metadata["gram_batch"] = gram_batch_contract(options);
      metadata["gram_batch"]["implementation"] = prepared_gram_route
                                                     ? "public_prepared_single_source"
                                                 : gram_method < 0 ? "public_prepared_dense"
                                                                   : "retained_single_source";
      metadata["gram_batch"]["public_api"] = prepared_gram_route ? "rns8_execute_prepared_gram"
                                             : gram_method < 0   ? "rns8_execute_prepared"
                                                                 : "rns8_gram_residues";
      metadata["gram_batch"]["rns_exports_per_batch"] =
          options.gram_export == "each" ? options.gram_reuse : 1;
      metadata["gram_batch"]["native_serializations_per_batch"] = 0;
      metadata["gram_batch"]["prepared_gram_api"] = prepared_gram_route;
      metadata["gram_batch"]["job_list_allocated_per_call"] =
          gram_method >= 0 && !prepared_gram_route;
      metadata["gram_batch"]["capability_probed_per_call"] =
          gram_matrix_method(gram_method) && !prepared_gram_route;
      if (!gram_five && gram_method >= 6)
        metadata["gram_method"] = "gfx942_mfma_i8_wave64";
      if (prepared_gram_route) {
        metadata["gram_batch"]["generation_policy"] =
            "refresh_or_explicit_recreate_on_expanded_bound";
        metadata["gram_batch"]["host_job_preparation"] =
            "once_per_owner_during_setup_or_changed_input";
        metadata["gram_batch"]["device_job_allocation"] =
            options.backend == RNS8_BACKEND_HIP_DIRECT
                ? "once_per_owner_during_setup_or_changed_input"
                : "not_applicable";
        metadata["gram_batch"]["device_job_initialization"] =
            options.backend == RNS8_BACKEND_HIP_DIRECT ? "first_execution_per_owner"
                                                       : "not_applicable";
        metadata["gram_batch"]["wmma_capability_probe"] =
            !gram_five && gram_matrix_method(gram_method) && gram_method < 6
                ? "first_execution_per_owner"
                : "not_applicable";
        if (!gram_five && gram_method >= 6)
          metadata["gram_batch"]["mfma_capability_probe"] = "first_execution_per_owner";
      }
    }
  }
  void prepare_gram_owner() {
    rns8_prepared_gram* handle = nullptr;
    if (gram_five)
      rns8::check(rns8_prepare_gram_five_product(
          context.get(), b->get(), c->get(), RNS8_GRAM_TRANSPOSE_TIMES_SELF, uint32_t(gram_method),
          gram_payload_reservation, &handle));
    else
      rns8::check(rns8_prepare_gram_ex(context.get(), b->get(), c->get(),
                                       RNS8_GRAM_TRANSPOSE_TIMES_SELF,
                                       static_cast<rns8_gram_method>(gram_method), gram_input,
                                       gram_once ? (options.wide ? 3 : 1) : 0,
                                       options.memory_mib * UINT64_C(1024) * 1024, &handle));
    prepared_gram.reset(handle);
    ++gram_owner_builds;
    gram_owner_used = false;
    rns8_prepared_gram_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    rns8::check(rns8_get_prepared_gram_info(handle, &info));
    require(info.owned_payload_bytes <= gram_payload_reservation,
            "actual Gram payload exceeds pre-admitted allocation envelope");
    metadata["gram_ablations"] = {
        {"version", 1},
        {"input_policy", uint32_t(gram_input)},
        {"reconstruction_policy", gram_once ? "independent_once_then_mirror" : "dense_each_cell"},
        {"owned_payload_bytes", info.owned_payload_bytes},
        {"adapter_bytes", info.orientation_adapter_bytes},
        {"host_export_bytes", info.host_export_bytes},
        {"device_export_bytes", info.device_export_bytes},
        {"independent_cells", info.independent_cells},
        {"dense_cells", info.dense_cells},
        {"native_storage_retained", gram_input == RNS8_GRAM_RETAINED_NATIVE_I64},
        {"adapter_generation_policy", "first_execute_after_prepare_or_refresh"},
        {"mirror_policy", gram_mirror_method(gram_method) ? "separate_tiled" : "producer_stores"},
        {"output_validity", "complete_dense_residues"}};
    if (gram_five) {
      rns8_gram_classical_info classical{};
      classical.struct_size = sizeof(classical);
      classical.abi_version = RNS8_ABI_VERSION;
      rns8::check(rns8_get_gram_classical_info(prepared_gram.get(), &classical));
      metadata["gram_method"] = gram_method ? "builtin_matrix" : "scalar";
      require(info.method == 4 + uint32_t(gram_method) && info.algebraic_product_count == 5 &&
                  info.algebraic_prime_planes > 0 &&
                  info.algebraic_prime_planes + info.classical_planes == prefix,
              "five-product owner query disagrees with the requested recipe");
      metadata["gram_five_product"] = {
          {"version", 2},
          {"matrix_subproducts", bool(gram_method)},
          {"classical_method", classical.method},
          {"algebraic_product_count", info.algebraic_product_count},
          {"algebraic_prime_planes", info.algebraic_prime_planes},
          {"classical_planes", info.classical_planes},
          {"algebraic_operand_bytes", info.algebraic_operand_bytes},
          {"algebraic_product_bytes", info.algebraic_product_bytes},
          {"algebraic_staging_bytes", info.algebraic_staging_bytes},
          {"host_job_bytes", info.host_job_bytes},
          {"device_job_bytes", info.device_job_bytes},
          {"owned_payload_bytes", info.owned_payload_bytes},
          {"modeled_payload_envelope_bytes", gram_payload_reservation},
          {"count_scope",
           "queried_retained_requests_and_algebraic_products_not_observed_instructions"}};
      metadata.erase("gram_ablations");
    }
  }
  Json gram_execution_evidence() const {
    Json evidence = {{"generation_action", gram_generation_action},
                     {"packed_source_version", packed_version},
                     {"owner_first_execution", last_owner_first_execution},
                     {"owner_builds", gram_owner_builds},
                     {"owner_refreshes", gram_owner_refreshes}};
    if (gram_five) {
      rns8_gram_classical_info classical{};
      classical.struct_size = sizeof(classical);
      classical.abi_version = RNS8_ABI_VERSION;
      rns8::check(rns8_get_gram_classical_info(prepared_gram.get(), &classical));
      evidence["classical_provider"] = {
          {"method", classical.method},
          {"classical_planes", classical.classical_planes},
          {"matrix_probe_observed", classical.matrix_probe_observed},
          {"matrix_probe_supported", classical.matrix_probe_supported}};
    }
    return evidence;
  }
  void pack(uint64_t version) {
    if (a)
      rns8::check(rns8_pack_i64(context.get(), a->get(), data.a.data(), options.k, version));
    rns8::check(rns8_pack_i64(context.get(), b->get(), data.b.data(), options.n,
                              options.gram ? version : 1));
    packed_version = version;
  }
  void compute(Phases& phases) {
    phases.measure("gemm_and_required_sync_ns", [&] {
      if (prepared_gram)
        rns8::check(rns8_execute_prepared_gram(prepared_gram.get()));
      else if (gram_method >= 0)
        rns8::check(rns8_gram_residues(context.get(), b->get(), c->get(),
                                       RNS8_GRAM_TRANSPOSE_TIMES_SELF,
                                       static_cast<rns8_gram_method>(gram_method),
                                       options.memory_mib * UINT64_C(1024) * 1024));
      else if (prepared) {
        rns8_completion_token token{};
        rns8::check(rns8_execute_prepared(prepared.get(), &token));
      } else
        rns8::check(rns8_gemm_rns(context.get(), plan->get(), a->get(), b->get(), c->get(),
                                  workspace->get()));
    });
  }
  void reconstruct(Phases& phases) {
    phases.measure("reconstruction_and_download_ns", [&] {
      if (gram_once)
        rns8::check(rns8_export_prepared_gram_limbs(prepared_gram.get(), words.data(), options.n));
      else if (options.wide)
        rns8::check(rns8_export_exact_wide_signed_limbs(context.get(), plan->get(), c->get(),
                                                        words.data(), options.n, 3));
      else {
        rns8::check(
            rns8_export_i64(context.get(), plan->get(), c->get(), native.data(), options.n));
        for (size_t index = 0; index < native.size(); ++index)
          words[index] = bits_of(native[index]);
      }
    });
  }
  void execute(Phases& phases, uint64_t version) {
    gram_generation_action = "retained_owner";
    if (options.gram_reuse) {
      if (version != packed_version)
        phases.measure("changed_input_pack_and_prepared_refresh_ns", [&] {
          pack(version);
          if (prepared)
            rns8::check(rns8_refresh_prepared_inputs(prepared.get()));
          if (prepared_gram) {
            const auto refreshed = rns8_refresh_prepared_gram(prepared_gram.get());
            if (refreshed == RNS8_RANGE_ERROR) {
              rns8::check(rns8_destroy_prepared_gram(prepared_gram.get()));
              (void)prepared_gram.release();
              prepare_gram_owner();
              gram_generation_action = "recreated_owner_for_larger_bound";
            } else {
              rns8::check(refreshed);
              ++gram_owner_refreshes;
              gram_generation_action = "refreshed_owner";
            }
          }
        });
      last_owner_first_execution = prepared_gram_route && !gram_owner_used;
      for (uint32_t product = 0; product < options.gram_reuse; ++product) {
        compute(phases);
        if (prepared_gram_route)
          gram_owner_used = true;
        if (options.gram_export == "each")
          reconstruct(phases);
      }
      if (options.gram_export == "once")
        reconstruct(phases);
    } else {
      phases.measure("pack_and_upload_ns", [&] { pack(version); });
      compute(phases);
      reconstruct(phases);
    }
  }
};
inline void bind_dense_variant(Variant& result, const std::shared_ptr<DenseEngine>& state) {
  result.metadata = state->metadata;
  result.execute = [state](Phases& phases, uint64_t version) { state->execute(phases, version); };
  result.output = [state]() -> const Words& { return state->words; };
  result.separate_observation = [state](const Variant::NativeBatchObserver& observe) {
    Phases phases;
    phases.enabled = false;
    return observe(state->context.get(), [&] { state->execute(phases, 1); });
  };
}
inline Variant dense_variant(const Dataset& data, const Options& options, const cpp_int& bound) {
  const auto start = Clock::now();
  auto state = std::make_shared<DenseEngine>(data, options, bound);
  Variant result;
  result.name = options.gram_reuse ? "rns8_prepared_dense_gram_control"
                                   : "rns8_" + std::string(backend_name(options.backend));
  result.setup_ns = since(start);
  bind_dense_variant(result, state);
  return result;
}
inline Variant gram_variant(const Dataset& data, const Options& options, const cpp_int& bound,
                            rns8_gram_method method, bool prepared = false,
                            rns8_gram_input_policy policy = RNS8_GRAM_RESIDENT_RESIDUES,
                            bool once = false, bool five = false) {
  const auto start = Clock::now();
  auto state = std::make_shared<DenseEngine>(data, options, bound, int(method), prepared, policy,
                                             once, five);
  Variant result;
  result.name =
      std::string(prepared ? "rns8_prepared_single_source_gram_" : "rns8_single_source_gram_") +
      gram_method_name(method);
  if (policy || once || gram_mirror_method(method))
    result.name = std::string("rns8_prepared_single_source_gram_") +
                  (policy == RNS8_GRAM_RETAINED_NATIVE_I64  ? "native_"
                   : policy == RNS8_GRAM_RETAINED_TRANSPOSE ? "transpose_"
                                                            : "") +
                  (gram_mirror_method(method) ? "separate_mirror_" : "") +
                  (once ? "reconstruct_once_" : "") + gram_method_name(method);
  if (five)
    result.name = std::string("rns8_prepared_single_source_gram_five_") +
                  (method == RNS8_GRAM_WMMA ? "matrix" : "scalar");
  result.setup_ns = since(start);
  bind_dense_variant(result, state);
  if (prepared)
    result.execution_evidence = [state] { return state->gram_execution_evidence(); };
  return result;
}
inline void add_five_gram_variants(std::vector<Variant>& variants, Json& skipped,
                                   const Dataset& data, const Options& options,
                                   const cpp_int& bound) {
  for (const auto method : {RNS8_GRAM_SCALAR, RNS8_GRAM_WMMA}) {
    const std::string name = std::string("rns8_prepared_single_source_gram_five_") +
                             (method == RNS8_GRAM_WMMA ? "matrix" : "scalar");
    if (method == RNS8_GRAM_WMMA && options.backend == RNS8_BACKEND_CPU_REFERENCE) {
      skipped.push_back({{"name", name},
                         {"status", "unsupported"},
                         {"reason", "matrix subproducts require a compiled GPU matrix provider"}});
      continue;
    }
    const auto reserved = options.memory->reserved;
    try {
      variants.push_back(gram_variant(data, options, bound, method, true,
                                      RNS8_GRAM_RESIDENT_RESIDUES, false, true));
    } catch (const rns8::Error& error) {
      if (error.status() != RNS8_UNSUPPORTED_BACKEND && error.status() != RNS8_UNSUPPORTED_ARCH)
        throw;
      // Construction unwound the candidate; rejected recipes retain no memory.
      options.memory->reserved = reserved;
      skipped.push_back({{"name", name}, {"status", "unsupported"}, {"reason", error.what()}});
    }
  }
}
} // namespace rns8_workload
#endif
