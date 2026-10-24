#ifndef RNS8_WORKLOAD_ENGINE_HPP
#define RNS8_WORKLOAD_ENGINE_HPP
#include "workload_common.hpp"
namespace rns8_workload {
inline cpp_int maximum_after_change(const std::vector<int64_t>& values) {
  const int64_t changed = values[0] == INT64_MAX ? values[0] - 1 : values[0] + 1;
  return std::max(maximum(values), cpp_int(magnitude(changed)));
}
inline cpp_int maximum_after_mutation(const Dataset& data) { return maximum_after_change(data.a); }
inline Json storage_info(const rns8::Matrix& matrix) {
  const auto info = matrix.storage_info();
  return {{"host_residue_bytes", info.host_residue_bytes}, {"device_residue_bytes", info.device_residue_bytes},
          {"host_native_bytes", info.host_native_bytes}, {"device_native_bytes", info.device_native_bytes},
          {"layout", info.layout_version}, {"source_version", info.source_version}};
}
struct DenseEngine {
  const Dataset& data;
  const Options& options;
  rns8::Context context;
  std::unique_ptr<rns8::Plan> plan;
  std::unique_ptr<rns8::Workspace> workspace;
  std::unique_ptr<rns8::Matrix> a, b, c;
  Words words;
  std::vector<int64_t> native;
  Json metadata;
  uint32_t prefix = 0;
  DenseEngine(const Dataset& input, const Options& config, const cpp_int& bound)
      : data(input), options(config), context(config.device, config.backend),
        words(std::size_t(config.m) * config.n * (config.wide ? 3 : 1)), native(std::size_t(config.m) * config.n) {
    metadata["device"] = device_info(context, options);
    const auto desc = gemm_desc(options, options.wide ? RNS8_EXACT_WIDE_SIGNED : RNS8_BOUNDED_I64,
                                options.m, options.n, options.k, bound);
    plan = std::make_unique<rns8::Plan>(context, desc);
    require(plan->backend_info().backend == options.backend, "plan changed the explicitly requested backend");
    prefix = plan->schedule_info().max_selected_prefix;
    const auto matrix = [&](uint32_t rows, uint32_t cols) {
      return std::make_unique<rns8::Matrix>(context, matrix_desc(rows, cols, desc.semantics, desc.bound_kind, prefix));
    };
    const uint64_t modeled_bytes = (uint64_t(options.m) * options.k + uint64_t(options.k) * options.n + uint64_t(options.m) * options.n) * 128;
    const uint64_t workspace_bytes = plan->backend_info().workspace_required_bytes;
    const uint64_t total_memory = context.device_info().global_mem_bytes;
    require(!total_memory || modeled_bytes + workspace_bytes <= total_memory,
            "estimated device storage/workspace exceeds total device memory; free-memory availability remains runtime checked");
    options.memory->reserve(modeled_bytes + workspace_bytes, options.memory_mib * 1024 * 1024, "retained dense candidate storage/workspace");
    workspace = std::make_unique<rns8::Workspace>(context, *plan);
    a = matrix(options.m, options.k); b = matrix(options.k, options.n); c = matrix(options.m, options.n);
    metadata["plan"] = plan_info(*plan);
    metadata["storage"] = {{"a", storage_info(*a)}, {"b", storage_info(*b)}, {"c", storage_info(*c)}};
    metadata["modeled_device_storage_envelope_bytes"] = modeled_bytes;
  }
  void execute(Phases& phases, uint64_t version) {
    phases.measure("pack_and_upload_ns", [&] {
      rns8::check(rns8_pack_i64(context.get(), a->get(), data.a.data(), options.k, version));
      rns8::check(rns8_pack_i64(context.get(), b->get(), data.b.data(), options.n, options.gram ? version : 1));
    });
    phases.measure("gemm_and_required_sync_ns", [&] {
      rns8::check(rns8_gemm_rns(context.get(), plan->get(), a->get(), b->get(), c->get(), workspace->get()));
    });
    phases.measure("reconstruction_and_download_ns", [&] {
      if (options.wide) rns8::check(rns8_export_exact_wide_signed_limbs(context.get(), plan->get(), c->get(), words.data(), options.n, 3));
      else {
        rns8::check(rns8_export_i64(context.get(), plan->get(), c->get(), native.data(), options.n));
        for (std::size_t index = 0; index < native.size(); ++index) words[index] = bits_of(native[index]);
      }
    });
  }
};
inline Variant dense_variant(const Dataset& data, const Options& options, const cpp_int& bound) {
  const auto start = Clock::now(); auto state = std::make_shared<DenseEngine>(data, options, bound);
  Variant result;
  result.name = "rns8_" + std::string(backend_name(options.backend)); result.setup_ns = since(start);
  result.metadata = state->metadata;
  result.execute = [state](Phases& phases, uint64_t version) { state->execute(phases, version); };
  result.output = [state]() -> const Words& { return state->words; };
  return result;
}
} // namespace rns8_workload
#endif
