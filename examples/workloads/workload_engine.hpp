#ifndef RNS8_WORKLOAD_ENGINE_HPP
#define RNS8_WORKLOAD_ENGINE_HPP
#include "workload_common.hpp"
namespace rns8_workload {
inline cpp_int maximum_after_change(const std::vector<int64_t>& values) {
  const int64_t changed = values[0] == INT64_MAX ? values[0] - 1 : values[0] + 1;
  return std::max(maximum(values), cpp_int(magnitude(changed)));
}
inline cpp_int maximum_after_mutation(const Dataset& data) { return maximum_after_change(data.a); }
struct WorkloadPreparedDeleter {
  void operator()(rns8_prepared_execution* value) const noexcept { (void)rns8_destroy_prepared_execution(value); }
};
struct WorkloadPreparedGramDeleter {
  void operator()(rns8_prepared_gram* value) const noexcept { (void)rns8_destroy_prepared_gram(value); }
};
struct DenseEngine {
  const Dataset& data;
  const Options& options;
  rns8::Context context;
  std::unique_ptr<rns8::Plan> plan;
  std::unique_ptr<rns8::Workspace> workspace;
  std::unique_ptr<rns8::Matrix> a, b, c;
  std::unique_ptr<rns8_prepared_execution,WorkloadPreparedDeleter> prepared;
  std::unique_ptr<rns8_prepared_gram,WorkloadPreparedGramDeleter> prepared_gram;
  uint64_t packed_version=0;
  uint32_t gram_owner_builds=0,gram_owner_refreshes=0;
  bool prepared_gram_route=false,gram_owner_used=false,last_owner_first_execution=false;
  const char* gram_generation_action="retained_owner";
  Words words;
  std::vector<int64_t> native;
  Json metadata;
  uint32_t prefix = 0;
  int gram_method = -1;
  DenseEngine(const Dataset& input, const Options& config, const cpp_int& bound, int gram = -1,bool gram_owner=false)
      : data(input), options(config), context(config.device, config.backend),
        prepared_gram_route(gram_owner),words(std::size_t(config.m) * config.n * (config.wide ? 3 : 1)), native(std::size_t(config.m) * config.n), gram_method(gram) {
    require(!prepared_gram_route || (options.gram_reuse && gram_method>=0),"prepared Gram route requires an explicit Gram batch");
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
    if (gram_method < 0) a = matrix(options.m, options.k);
    b = matrix(options.k, options.n); c = matrix(options.m, options.n);
    metadata["plan"] = plan_info(*plan);
    metadata["storage"] = {{"b", storage_info(*b)}, {"c", storage_info(*c)}};
    if (a) metadata["storage"]["a"] = storage_info(*a);
    metadata["single_source_gram"] = gram_method >= 0;
    if (gram_method >= 0) {
      metadata["independent_integer_products_per_call"] = uint64_t(options.n)*(options.n+1)/2*options.k*std::max(options.gram_reuse,1u);
      metadata["dense_equivalent_integer_products_per_call"] = uint64_t(options.n)*options.n*options.k*std::max(options.gram_reuse,1u);
      metadata["gram_method"] = gram_method == RNS8_GRAM_WMMA ? "wmma" : "scalar";
    }
    metadata["modeled_device_storage_envelope_bytes"] = modeled_bytes;
    if (options.gram_reuse) {
      options.memory->reserve(4096,options.memory_mib*UINT64_C(1024)*1024,"Gram retained preparation metadata allowance");
      pack(1);
      if (gram_method < 0) {
        rns8_prepared_execution* handle=nullptr;
        rns8::check(rns8_prepare_gemm(context.get(),plan->get(),a->get(),b->get(),c->get(),workspace->get(),&handle));
        prepared.reset(handle);
      } else if(prepared_gram_route) {
        const uint64_t tiles=(uint64_t(options.n)+15)/16;
        const uint64_t jobs=tiles*(tiles+1)/2;
        const uint64_t job_envelope=jobs*16+4;
        options.memory->reserve(job_envelope,options.memory_mib*UINT64_C(1024)*1024,"modeled retained Gram host/device job payload");
        metadata["modeled_retained_gram_job_payload_bytes"]=job_envelope;
        prepare_gram_owner();
      }
      metadata["gram_batch"]=gram_batch_contract(options);
      metadata["gram_batch"]["implementation"]=prepared_gram_route?"public_prepared_single_source":gram_method < 0 ? "public_prepared_dense" : "retained_single_source";
      metadata["gram_batch"]["public_api"]=prepared_gram_route?"rns8_execute_prepared_gram":gram_method < 0 ? "rns8_execute_prepared" : "rns8_gram_residues";
      metadata["gram_batch"]["rns_exports_per_batch"]=options.gram_export=="each"?options.gram_reuse:1;
      metadata["gram_batch"]["native_serializations_per_batch"]=0;
      metadata["gram_batch"]["prepared_gram_api"]=prepared_gram_route;
      metadata["gram_batch"]["job_list_allocated_per_call"]=gram_method>=0 && !prepared_gram_route;
      metadata["gram_batch"]["capability_probed_per_call"]=gram_method==RNS8_GRAM_WMMA && !prepared_gram_route;
      if(prepared_gram_route) {
        metadata["gram_batch"]["generation_policy"]="refresh_or_explicit_recreate_on_expanded_bound";
        metadata["gram_batch"]["host_job_preparation"]="once_per_owner_during_setup_or_changed_input";
        metadata["gram_batch"]["device_job_allocation"]=options.backend==RNS8_BACKEND_HIP_DIRECT?"once_per_owner_during_setup_or_changed_input":"not_applicable";
        metadata["gram_batch"]["device_job_initialization"]=options.backend==RNS8_BACKEND_HIP_DIRECT?"first_execution_per_owner":"not_applicable";
        metadata["gram_batch"]["wmma_capability_probe"]=gram_method==RNS8_GRAM_WMMA?"first_execution_per_owner":"not_applicable";
      }
    }
  }
  void prepare_gram_owner() {
    rns8_prepared_gram* handle=nullptr;
    rns8::check(rns8_prepare_gram(context.get(),b->get(),c->get(),RNS8_GRAM_TRANSPOSE_TIMES_SELF,
        static_cast<rns8_gram_method>(gram_method),options.memory_mib*UINT64_C(1024)*1024,&handle));
    prepared_gram.reset(handle);++gram_owner_builds;gram_owner_used=false;
  }
  Json gram_execution_evidence() const {
    return {{"generation_action",gram_generation_action},{"packed_source_version",packed_version},
      {"owner_first_execution",last_owner_first_execution},{"owner_builds",gram_owner_builds},{"owner_refreshes",gram_owner_refreshes}};
  }
  void pack(uint64_t version) {
    if (a) rns8::check(rns8_pack_i64(context.get(),a->get(),data.a.data(),options.k,version));
    rns8::check(rns8_pack_i64(context.get(),b->get(),data.b.data(),options.n,options.gram?version:1));
    packed_version=version;
  }
  void compute(Phases& phases) {
    phases.measure("gemm_and_required_sync_ns",[&] {
      if(prepared_gram) rns8::check(rns8_execute_prepared_gram(prepared_gram.get()));
      else if(gram_method>=0)
        rns8::check(rns8_gram_residues(context.get(),b->get(),c->get(),RNS8_GRAM_TRANSPOSE_TIMES_SELF,
            static_cast<rns8_gram_method>(gram_method),options.memory_mib*UINT64_C(1024)*1024));
      else if(prepared) {
        rns8_completion_token token{};rns8::check(rns8_execute_prepared(prepared.get(),&token));
      } else rns8::check(rns8_gemm_rns(context.get(),plan->get(),a->get(),b->get(),c->get(),workspace->get()));
    });
  }
  void reconstruct(Phases& phases) {
    phases.measure("reconstruction_and_download_ns",[&] {
      if(options.wide) rns8::check(rns8_export_exact_wide_signed_limbs(context.get(),plan->get(),c->get(),words.data(),options.n,3));
      else {
        rns8::check(rns8_export_i64(context.get(),plan->get(),c->get(),native.data(),options.n));
        for(size_t index=0;index<native.size();++index) words[index]=bits_of(native[index]);
      }
    });
  }
  void execute(Phases& phases,uint64_t version) {
    gram_generation_action="retained_owner";
    if(options.gram_reuse) {
      if(version!=packed_version) phases.measure("changed_input_pack_and_prepared_refresh_ns",[&] {
        pack(version);if(prepared) rns8::check(rns8_refresh_prepared_inputs(prepared.get()));
        if(prepared_gram) {
          const auto refreshed=rns8_refresh_prepared_gram(prepared_gram.get());
          if(refreshed==RNS8_RANGE_ERROR) {
            rns8::check(rns8_destroy_prepared_gram(prepared_gram.get()));(void)prepared_gram.release();
            prepare_gram_owner();gram_generation_action="recreated_owner_for_larger_bound";
          } else {
            rns8::check(refreshed);++gram_owner_refreshes;gram_generation_action="refreshed_owner";
          }
        }
      });
      last_owner_first_execution=prepared_gram_route && !gram_owner_used;
      for(uint32_t product=0;product<options.gram_reuse;++product) {
        compute(phases);
        if(prepared_gram_route) gram_owner_used=true;
        if(options.gram_export=="each") reconstruct(phases);
      }
      if(options.gram_export=="once") reconstruct(phases);
    } else {
      phases.measure("pack_and_upload_ns",[&] {pack(version);});compute(phases);reconstruct(phases);
    }
  }

};
inline Variant dense_variant(const Dataset& data, const Options& options, const cpp_int& bound) {
  const auto start = Clock::now(); auto state = std::make_shared<DenseEngine>(data, options, bound);
  Variant result;
  result.name = options.gram_reuse ? "rns8_prepared_dense_gram_control" : "rns8_" + std::string(backend_name(options.backend)); result.setup_ns = since(start);
  result.metadata = state->metadata;
  result.execute = [state](Phases& phases, uint64_t version) { state->execute(phases, version); };
  result.output = [state]() -> const Words& { return state->words; };
  return result;
}
inline Variant gram_variant(const Dataset& data, const Options& options, const cpp_int& bound,
                            rns8_gram_method method,bool prepared=false) {
  const auto start=Clock::now();auto state=std::make_shared<DenseEngine>(data,options,bound,int(method),prepared);
  Variant result;
  result.name=std::string(prepared?"rns8_prepared_single_source_gram_":"rns8_single_source_gram_")+(method==RNS8_GRAM_WMMA?"wmma":"scalar");
  result.setup_ns=since(start);result.metadata=state->metadata;
  result.execute=[state](Phases& phases,uint64_t version){state->execute(phases,version);};
  result.output=[state]() -> const Words& {return state->words;};
  if(prepared) result.execution_evidence=[state] {return state->gram_execution_evidence();};
  return result;
}
} // namespace rns8_workload
#endif
