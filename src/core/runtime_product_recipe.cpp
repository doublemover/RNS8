#include "core/runtime_product_recipe.hpp"
#include "core/api_internal.hpp"
#include "core/hip_resources.hpp"
#include <mutex>
#include <unordered_set>
namespace rns8::detail {
using namespace api;
struct RuntimeProductRecipe::Impl {
  rns8_context* ctx=nullptr;
  struct Input { const rns8_matrix* matrix; uint64_t instance,generation; };
  std::vector<Input> inputs;
  std::vector<ProductNode> nodes;
  std::vector<uint32_t> outputs;
  ProductArenaPlan allocation;
  std::vector<cpp_int> bounds;
  std::vector<uint32_t> compute_prefixes,capacities;
  uint64_t lift_scratch=0;
  ProductPrefixSelection selection;
  uint32_t prefix=0;
  rns8_semantics semantics=RNS8_EXACT_WIDE_SIGNED;
  std::vector<int8_t> host;
  void* device=nullptr;
  bool quarantined=false,closed=false,output_touched=false;
  std::vector<const rns8_matrix*> active_inputs;
  std::vector<rns8_matrix*> active_outputs;
  std::vector<uint64_t*> holds;
  std::mutex mutex;
  void release_active() noexcept {
    if(active_outputs.empty()) return;
    for(const auto* input:active_inputs) --input->in_flight_readers;
    for(auto* output:active_outputs) {output->in_flight_writer=false;--output->prepared_references;}
    active_outputs.clear();active_inputs.clear();output_touched=false;
  }
  void invalidate_touched() noexcept {
    if(output_touched) for(auto* output:active_outputs) {
      clear_residue_current(*output);clear_byte_limb_current(*output);clear_native_current(*output);
    }
  }
  ~Impl() {
    // An executing/quarantined owner is never destroyed: close preserves it.
    if(device) (void)hip_direct_free(ctx->device_id,device);
    for(auto* hold:holds) --*hold;
  }
};
RuntimeProductRecipe::RuntimeProductRecipe(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
RuntimeProductRecipe::~RuntimeProductRecipe() {
  // Private RAII users cannot return a destruction error. Preserve the complete
  // owner (including matrix/context holds) rather than free in-flight storage.
  if(close()!=RNS8_SUCCESS) (void)impl_.release();
}
rns8_status RuntimeProductRecipe::close() {
  return guard_api([&]() -> rns8_status {
    auto& state=*impl_;std::unique_lock<std::mutex> lock(state.mutex,std::try_to_lock);
    if(!lock.owns_lock()) return RNS8_BUSY;
    if(state.closed) return RNS8_SUCCESS;
    if(state.quarantined || !state.active_outputs.empty()) {
      if(state.ctx->backend==RNS8_BACKEND_HIP_DIRECT && hip_direct_synchronize(state.ctx->device_id)!=RNS8_SUCCESS)
        return RNS8_BACKEND_FAILURE;
      state.invalidate_touched();state.release_active();state.quarantined=false;
    }
    if(state.device) {
      const auto status=hip_direct_free(state.ctx->device_id,state.device);
      if(status!=RNS8_SUCCESS) return status;state.device=nullptr;
    }
    state.closed=true;return RNS8_SUCCESS;
  });
}
const ProductPrefixSelection& RuntimeProductRecipe::cost_selection() const { return impl_->selection; }
const ProductArenaPlan& RuntimeProductRecipe::allocation() const { return impl_->allocation; }
uint64_t RuntimeProductRecipe::lift_scratch_bytes() const { return impl_->lift_scratch; }
const std::vector<uint32_t>& RuntimeProductRecipe::compute_prefixes() const { return impl_->compute_prefixes; }
rns8_status RuntimeProductRecipe::create(rns8_context* ctx,
    const std::vector<const rns8_matrix*>& inputs,const std::vector<ProductNode>& nodes,
    uint64_t budget,std::unique_ptr<RuntimeProductRecipe>& out,bool minimize_compute_prefix,const ProductCostProfile* costs,const std::vector<uint32_t>& outputs) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || inputs.empty() || inputs.size()>4096 || nodes.empty() || nodes.size()>4096 || out)
      return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if(ctx->backend!=RNS8_BACKEND_CPU_REFERENCE && ctx->backend!=RNS8_BACKEND_HIP_DIRECT)
      return RNS8_UNSUPPORTED_BACKEND;
    auto state=std::make_unique<Impl>(); state->ctx=ctx;
    std::vector<ProductShape> shapes;
    std::unordered_set<uint64_t*> holds{&ctx->prepared_references};
    for(const auto* input:inputs) {
      if(!input) return RNS8_INVALID_ARGUMENT;
      if(input->in_flight_writer) return RNS8_BUSY;
      if(!exact_wide_semantics(input->desc.semantics) || input->backend!=ctx->backend ||
          input->generation_exhausted || !rns_residue_state_current_for_backend(*input,ctx->backend) ||
          !rns_matrix_storage_matches(*input,ctx->backend,input->desc.rows,input->desc.cols,input->prefix) ||
          (ctx->backend==RNS8_BACKEND_HIP_DIRECT && input->hip_device_id!=ctx->device_id))
        return RNS8_INVALID_ARGUMENT;
      if(state->inputs.empty()) { state->prefix=input->prefix; state->semantics=input->desc.semantics; }
      if(input->prefix!=state->prefix || input->desc.semantics!=state->semantics ||
          input->exact_range_prefix<state->prefix ||
          !exact_range_fits(state->semantics,input->exact_max_magnitude,modulus_product(state->prefix)))
        return RNS8_RANGE_ERROR;
      state->inputs.push_back({input,input->matrix_instance_id,input->content_generation});
      shapes.push_back({input->desc.rows,input->desc.cols});
      state->bounds.push_back(input->exact_max_magnitude);
      holds.insert(&input->prepared_references);
    }
    state->outputs=outputs;
    if(state->outputs.empty()) state->outputs.push_back(static_cast<uint32_t>(inputs.size()+nodes.size()-1));
    // Existing cost selection models one final publication; do not misreport a
    // multi-output choice as its optimum.
    if(costs && (state->outputs.size()!=1 || state->outputs[0]!=inputs.size()+nodes.size()-1))
      return RNS8_UNSUPPORTED_BACKEND;
    state->allocation=plan_product_arena(shapes,nodes,state->prefix,UINT64_MAX,{},state->outputs);
    if(state->allocation.status!=RNS8_SUCCESS) return state->allocation.status;
    state->nodes=nodes;
    for(size_t i=0;i<nodes.size();++i) {
      const auto node=nodes[i];
      const auto a=state->allocation.values[node.a].shape,b=state->allocation.values[node.b].shape;
      if(ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
        const auto support=query_residue_gemm_support(a.rows,b.cols,a.cols,a.cols,b.cols,b.cols,state->prefix);
        if(support.status!=RNS8_SUCCESS) return support.status;
      }
      cpp_int bound=cpp_int(a.cols)*state->bounds[node.a]*state->bounds[node.b];
      if(!exact_range_fits(state->semantics,bound,modulus_product(state->prefix))) return RNS8_RANGE_ERROR;
      uint32_t selected=state->prefix;
      if(minimize_compute_prefix || costs) {
        selected=1;
        while(selected<state->prefix && !exact_range_fits(state->semantics,bound,modulus_product(selected))) ++selected;
      }
      state->compute_prefixes.push_back(selected);
      state->bounds.push_back(std::move(bound));
    }
    if(costs) {
      const std::string target=ctx->backend==RNS8_BACKEND_HIP_DIRECT?ctx->device_info.gcn_arch:"cpu";
      if(costs->backend!=ctx->backend || costs->semantics!=state->semantics || costs->target!=target)
        return RNS8_INVALID_ARGUMENT;
      state->selection=select_product_prefixes(shapes,nodes,state->prefix,state->compute_prefixes,budget,
          ctx->backend==RNS8_BACKEND_HIP_DIRECT,*costs);
      if(state->selection.status!=RNS8_SUCCESS) return state->selection.status;
      state->compute_prefixes=state->selection.prefixes;
    }
    state->capacities=state->compute_prefixes;
    for(size_t i=0;i<nodes.size();++i) for(auto id:{nodes[i].a,nodes[i].b})
      if(id>=inputs.size()) state->capacities[id-inputs.size()]=
          std::max(state->capacities[id-inputs.size()],state->compute_prefixes[i]);
    for(auto id:state->outputs) state->capacities[id-inputs.size()]=state->prefix; // complete observed values
    for(size_t i=0;i<nodes.size();++i) if(state->capacities[i]>state->compute_prefixes[i]) {
      const auto shape=state->allocation.values[inputs.size()+i].shape;
      if(ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
        const auto lift=query_exact_lift_support(shape.rows*shape.cols,state->compute_prefixes[i],
            state->capacities[i],UINT64_MAX);
        if(lift.status!=RNS8_SUCCESS) return lift.status;
        state->lift_scratch=std::max(state->lift_scratch,lift.scratch_bytes);
      }
    }
    if(state->lift_scratch>budget) return RNS8_WORKSPACE_TOO_SMALL;
    state->allocation=plan_product_arena(shapes,nodes,state->prefix,budget-state->lift_scratch,state->capacities,state->outputs);
    if(state->allocation.status!=RNS8_SUCCESS) return state->allocation.status;
    for(auto* hold:holds) if(*hold==UINT64_MAX) return RNS8_RANGE_ERROR;
    if(ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
      const auto status=hip_direct_allocate(ctx->device_id,static_cast<size_t>(state->allocation.arena_bytes),&state->device);
      if(status!=RNS8_SUCCESS) return status;
    } else state->host.resize(static_cast<size_t>(state->allocation.arena_bytes));
    // Allocate the outer owner and reference list before taking any references.
    auto result=std::unique_ptr<RuntimeProductRecipe>(new RuntimeProductRecipe(std::move(state)));
    result->impl_->holds.assign(holds.begin(),holds.end());
    for(auto* hold:result->impl_->holds) ++*hold;
    out=std::move(result);
    return RNS8_SUCCESS;
  });
}
rns8_status RuntimeProductRecipe::rebind(const std::vector<const rns8_matrix*>& inputs) {
  return guard_api([&]() -> rns8_status {
    auto& state=*impl_;std::unique_lock<std::mutex> lock(state.mutex,std::try_to_lock);
    if(!lock.owns_lock()) return RNS8_BUSY;
    if(state.closed || inputs.size()!=state.inputs.size()) return RNS8_INVALID_ARGUMENT;
    if(state.ctx->hip_failure_poisoned || state.quarantined) return RNS8_BACKEND_FAILURE;
    std::vector<Impl::Input> replacement;replacement.reserve(inputs.size());
    std::unordered_set<uint64_t*> new_holds{&state.ctx->prepared_references};
    const std::unordered_set<uint64_t*> old_holds(state.holds.begin(),state.holds.end());
    for(size_t i=0;i<inputs.size();++i) {
      const auto* input=inputs[i];if(!input) return RNS8_INVALID_ARGUMENT;
      if(input->in_flight_writer) return RNS8_BUSY;
      const auto shape=state.allocation.values[i].shape;
      if(input->backend!=state.ctx->backend || input->desc.semantics!=state.semantics || input->prefix!=state.prefix ||
         input->generation_exhausted || input->desc.rows!=shape.rows || input->desc.cols!=shape.cols ||
         !rns_residue_state_current_for_backend(*input,state.ctx->backend) ||
         !rns_matrix_storage_matches(*input,state.ctx->backend,shape.rows,shape.cols,state.prefix) ||
         (state.ctx->backend==RNS8_BACKEND_HIP_DIRECT && input->hip_device_id!=state.ctx->device_id)) return RNS8_INVALID_ARGUMENT;
      if(input->exact_range_prefix<state.prefix || input->exact_max_magnitude<0 ||
         input->exact_max_magnitude>state.bounds[i] ||
         !exact_range_fits(state.semantics,input->exact_max_magnitude,modulus_product(state.prefix))) return RNS8_RANGE_ERROR;
      replacement.push_back({input,input->matrix_instance_id,input->content_generation});
      new_holds.insert(&input->prepared_references);
    }
    std::vector<uint64_t*> replacement_holds(new_holds.begin(),new_holds.end());
    for(auto* hold:replacement_holds) if(!old_holds.count(hold) && *hold==UINT64_MAX) return RNS8_RANGE_ERROR;
    // Nothing below allocates or validates. Take new unique owners before
    // dropping old owners, then atomically publish all generation snapshots.
    for(auto* hold:replacement_holds) if(!old_holds.count(hold)) ++*hold;
    for(auto* hold:state.holds) if(!new_holds.count(hold)) --*hold;
    state.inputs.swap(replacement);state.holds.swap(replacement_holds);
    return RNS8_SUCCESS;
  });
}
rns8_status RuntimeProductRecipe::execute(rns8_matrix* output) {
  return guard_api([&]() {return execute(std::vector<rns8_matrix*>{output});});
}
rns8_status RuntimeProductRecipe::execute(const std::vector<rns8_matrix*>& outputs) {
  return guard_api([&]() -> rns8_status {
    auto& state=*impl_;
    std::unique_lock<std::mutex> lock(state.mutex,std::try_to_lock);
    if(!lock.owns_lock()) return RNS8_BUSY;
    if(state.closed || outputs.size()!=state.outputs.size()) return RNS8_INVALID_ARGUMENT;
    if(state.ctx->hip_failure_poisoned || state.quarantined) return RNS8_BACKEND_FAILURE;
    std::unordered_set<rns8_matrix*> unique_outputs;
    for(size_t i=0;i<outputs.size();++i) {
      auto* output=outputs[i];
      if(!output || !unique_outputs.insert(output).second) return RNS8_INVALID_ARGUMENT;
      if(output->in_flight_readers || output->in_flight_writer) return RNS8_BUSY;
      const auto shape=state.allocation.values[state.outputs[i]].shape;
      if(output->backend!=state.ctx->backend || output->desc.semantics!=state.semantics ||
          output->desc.rows!=shape.rows || output->desc.cols!=shape.cols || output->prefix!=state.prefix ||
          !rns_matrix_storage_matches(*output,state.ctx->backend,shape.rows,shape.cols,state.prefix) ||
          (state.ctx->backend==RNS8_BACKEND_HIP_DIRECT && output->hip_device_id!=state.ctx->device_id))
        return RNS8_INVALID_ARGUMENT;
      if(output->generation_exhausted || output->prepared_references==UINT64_MAX) return RNS8_RANGE_ERROR;
    }
    for(const auto& input:state.inputs) {
      const auto& matrix=*input.matrix;
      if(unique_outputs.count(const_cast<rns8_matrix*>(input.matrix))) return RNS8_INVALID_ARGUMENT;
      if(matrix.in_flight_writer) return RNS8_BUSY;
      if(matrix.generation_exhausted || matrix.matrix_instance_id!=input.instance ||
          matrix.content_generation!=input.generation ||
          !rns_residue_state_current_for_backend(matrix,state.ctx->backend)) return RNS8_INVALID_ARGUMENT;
    }
    // Copy every proof before work; final publication swaps without allocation.
    std::vector<cpp_int> bounds;bounds.reserve(outputs.size());
    for(auto id:state.outputs) bounds.push_back(state.bounds[id]);
    const bool hip=state.ctx->backend==RNS8_BACKEND_HIP_DIRECT;
    auto* arena=hip?static_cast<int8_t*>(state.device):state.host.data();
    const auto pointer=[&](uint32_t id) -> const int8_t* {
      if(id<state.inputs.size()) {
        const auto& matrix=*state.inputs[id].matrix;
        return hip?static_cast<const int8_t*>(matrix.hip_residues):matrix.residues.data();
      }
      return arena+state.allocation.values[id].offset;
    };
    std::vector<uint32_t> current(state.allocation.values.size(),0);
    std::fill_n(current.begin(),state.inputs.size(),state.prefix);
    // Allocate and validate the complete reservation list before taking any
    // counters. Duplicate external input handles get one reader reservation.
    std::unordered_set<const rns8_matrix*> unique_inputs;
    for(const auto& input:state.inputs) unique_inputs.insert(input.matrix);
    state.active_inputs.assign(unique_inputs.begin(),unique_inputs.end());
    for(const auto* input:state.active_inputs) if(input->in_flight_readers==UINT64_MAX) return RNS8_RANGE_ERROR;
    // Allocate output reservation storage before taking any counters.
    state.active_outputs=outputs;
    for(const auto* input:state.active_inputs) ++input->in_flight_readers;
    for(auto* output:state.active_outputs) {++output->prepared_references;output->in_flight_writer=true;}
    state.output_touched=false;
    struct RunGuard {
      Impl& state;bool completed=false;
      ~RunGuard() noexcept {
        if(completed) return;
        state.invalidate_touched();
        if(state.ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
          try {
            if(hip_direct_synchronize(state.ctx->device_id)!=RNS8_SUCCESS) {
              state.quarantined=true;state.ctx->hip_failure_poisoned=true;return;
            }
          } catch(...) {state.quarantined=true;state.ctx->hip_failure_poisoned=true;return;}
        }
        state.release_active();
      }
      void finish() noexcept {state.release_active();completed=true;}
    } run{state};
    const auto materialize=[&](uint32_t id,uint32_t target) -> rns8_status {
      if(current[id]>=target) return RNS8_SUCCESS;
      if(id<state.inputs.size() || !current[id]) return RNS8_INVALID_ARGUMENT;
      auto* residues=arena+state.allocation.values[id].offset;
      const auto shape=state.allocation.values[id].shape;
      const int64_t cells=shape.rows*shape.cols;
      if(hip) {
        uint64_t words[3]{};
        cpp_int magnitude=state.bounds[id];
        const cpp_int mask=(cpp_int(1)<<64)-1;
        for(auto& word:words) { word=(magnitude&mask).convert_to<uint64_t>(); magnitude>>=64; }
        if(magnitude!=0) return RNS8_RANGE_ERROR;
        bool touched=false;
        const auto status=hip_direct_lift_exact_suffix(state.ctx->device_id,residues,cells,current[id],target,
            state.semantics==RNS8_EXACT_WIDE_SIGNED,words,touched);
        if(status!=RNS8_SUCCESS) return status;
      } else {
        std::vector<int8_t> cell_residues(current[id]);
        const auto product=modulus_product(current[id]);
        const cpp_int threshold=(product+1)/2;
        for(int64_t cell=0;cell<cells;++cell) {
          for(uint32_t p=0;p<current[id];++p) {
            const int value=residues[int64_t(p)*cells+cell],modulus=kDefaultModuli[p];
            if(value<-(modulus/2) || value>(modulus-1)/2) return RNS8_INVALID_ARGUMENT;
            cell_residues[p]=static_cast<int8_t>(value);
          }
          auto value=reconstruct_canonical(cell_residues,current[id]);
          if(state.semantics==RNS8_EXACT_WIDE_SIGNED && value>=threshold) value-=product;
          if(boost::multiprecision::abs(value)>state.bounds[id]) return RNS8_RANGE_ERROR;
          for(uint32_t p=current[id];p<target;++p)
            residues[int64_t(p)*cells+cell]=centered_residue(value,kDefaultModuli[p]);
        }
      }
      current[id]=target;
      return RNS8_SUCCESS;
    };
    for(size_t i=0;i<state.nodes.size();++i) {
      const auto node=state.nodes[i];
      const auto prefix=state.compute_prefixes[i];
      for(auto id:{node.a,node.b}) {
        const auto status=materialize(id,prefix);
        if(status!=RNS8_SUCCESS) return status;
      }
      const auto a=state.allocation.values[node.a].shape,b=state.allocation.values[node.b].shape;
      auto* c=arena+state.allocation.values[state.inputs.size()+i].offset;
      const auto* av=pointer(node.a); const auto* bv=pointer(node.b);
      if(hip) {
        const auto status=hip_direct_gemm_rns_device(state.ctx->device_id,av,bv,c,
            a.rows,b.cols,a.cols,a.cols,b.cols,b.cols,prefix);
        if(status!=RNS8_SUCCESS) return status;
      } else {
        for(uint32_t p=0;p<prefix;++p) {
          const int64_t modulus=rns8_default_modulus(p);
          for(int64_t row=0;row<a.rows;++row) for(int64_t col=0;col<b.cols;++col) {
            int64_t sum=0;
            for(int64_t q=0;q<a.cols;++q)
              sum=(sum+int64_t(av[(int64_t(p)*a.rows+row)*a.cols+q])*
                  bv[(int64_t(p)*b.rows+q)*b.cols+col])%modulus;
            if(sum<0) sum+=modulus;
            if(sum>=(modulus+1)/2) sum-=modulus;
            c[(int64_t(p)*a.rows+row)*b.cols+col]=static_cast<int8_t>(sum);
          }
        }
      }
      current[state.inputs.size()+i]=prefix;
    }
    // Finish all private materializations before starting any public write.
    for(auto id:state.outputs) {
      const auto completed=materialize(id,state.prefix);
      if(completed!=RNS8_SUCCESS) return completed;
    }
    state.output_touched=true;
    state.invalidate_touched();
    for(size_t i=0;i<outputs.size();++i) {
      auto* output=outputs[i];const auto& value=state.allocation.values[state.outputs[i]];
      const auto* result=arena+value.offset;
      if(hip) {
        const auto status=hip_direct_copy_device_to_device(state.ctx->device_id,output->hip_residues,
            result,static_cast<size_t>(value.bytes));
        if(status!=RNS8_SUCCESS) return status;
      } else std::copy_n(result,static_cast<size_t>(value.bytes),output->residues.data());
    }
    // Publish only after every copy succeeds. Any failed publication invalidates
    // the entire output batch, including copies which had already completed.
    for(size_t i=0;i<outputs.size();++i) {
      auto* output=outputs[i];
      if(hip) mark_output_device_residues_current(*output);
      else mark_output_host_residues_current(*output);
      output->exact_max_magnitude.swap(bounds[i]);
      output->exact_axis_ranges.rows.clear();output->exact_axis_ranges.cols.clear();
      output->exact_range_prefix=state.prefix;
    }
    run.finish();
    return RNS8_SUCCESS;
  });
}
} // namespace rns8::detail

struct rns8_product_recipe {
  std::unique_ptr<rns8::detail::RuntimeProductRecipe> owner;
};
static rns8_status create_product_recipe(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const std::vector<uint32_t>& outputs,rns8_product_recipe** out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!ctx || !desc || !out || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) ||
        !desc->inputs || !desc->nodes || !desc->input_count || desc->input_count>4096 ||
        !desc->node_count || desc->node_count>4096 ||
        (desc->prefix_policy!=RNS8_PRODUCT_FULL_PREFIX && desc->prefix_policy!=RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX))
      return RNS8_INVALID_ARGUMENT;
    std::vector<const rns8_matrix*> inputs(desc->inputs,desc->inputs+desc->input_count);
    std::vector<ProductNode> nodes; nodes.reserve(desc->node_count);
    for(uint32_t i=0;i<desc->node_count;++i) nodes.push_back({desc->nodes[i].a,desc->nodes[i].b});
    auto result=std::make_unique<rns8_product_recipe>();
    const auto status=RuntimeProductRecipe::create(ctx,inputs,nodes,desc->max_arena_and_lift_bytes,result->owner,
        desc->prefix_policy==RNS8_PRODUCT_MINIMUM_COMPUTE_PREFIX,nullptr,outputs);
    if(status!=RNS8_SUCCESS) return status;
    *out=result.release(); return RNS8_SUCCESS;
  });
}
rns8_status rns8_create_product_recipe(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    rns8_product_recipe** out) {
  return create_product_recipe(ctx,desc,{},out);
}
rns8_status rns8_create_product_recipe_outputs(rns8_context* ctx,const rns8_product_recipe_desc* desc,
    const uint32_t* ids,uint32_t count,rns8_product_recipe** out) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if(!ids || !count || count>4096) return RNS8_INVALID_ARGUMENT;
    return create_product_recipe(ctx,desc,std::vector<uint32_t>(ids,ids+count),out);
  });
}
rns8_status rns8_execute_product_recipe_outputs(rns8_product_recipe* recipe,
    rns8_matrix* const* outputs,uint32_t count) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if(!recipe || !outputs || !count || count>4096) return RNS8_INVALID_ARGUMENT;
    return recipe->owner->execute(std::vector<rns8_matrix*>(outputs,outputs+count));
  });
}
rns8_status rns8_execute_product_recipe(rns8_product_recipe* recipe,rns8_matrix* output) {
  return recipe?recipe->owner->execute(output):RNS8_INVALID_ARGUMENT;
}
rns8_status rns8_get_product_recipe_resources(const rns8_product_recipe* recipe,rns8_product_recipe_resources* out) {
  if(!recipe || !out || !rns8::detail::valid_abi(out->struct_size,out->abi_version,sizeof(*out)))
    return RNS8_INVALID_ARGUMENT;
  const auto& allocation=recipe->owner->allocation();
  out->node_count=static_cast<uint32_t>(recipe->owner->compute_prefixes().size());
  out->arena_bytes=allocation.arena_bytes;
  out->lift_scratch_bytes=recipe->owner->lift_scratch_bytes();
  out->unreused_payload_bytes=allocation.unreused_bytes;
  return RNS8_SUCCESS;
}
rns8_status rns8_get_product_recipe_prefixes(const rns8_product_recipe* recipe,uint32_t* prefixes,uint32_t capacity) {
  if(!recipe || !prefixes) return RNS8_INVALID_ARGUMENT;
  const auto& selected=recipe->owner->compute_prefixes();
  if(capacity<selected.size()) return RNS8_WORKSPACE_TOO_SMALL;
  std::copy(selected.begin(),selected.end(),prefixes);
  return RNS8_SUCCESS;
}
rns8_status rns8_destroy_product_recipe(rns8_product_recipe* recipe) {
  if(!recipe) return RNS8_SUCCESS;
  const auto status=recipe->owner->close();if(status!=RNS8_SUCCESS) return status;
  delete recipe;return RNS8_SUCCESS;
}

rns8_status rns8_rebind_product_recipe_inputs(rns8_product_recipe* recipe,
    const rns8_matrix* const* inputs,uint32_t count) {
  return rns8::detail::api::guard_api([&]() -> rns8_status {
    if(!recipe || !inputs || !count || count>4096) return RNS8_INVALID_ARGUMENT;
    return recipe->owner->rebind(std::vector<const rns8_matrix*>(inputs,inputs+count));
  });
}
