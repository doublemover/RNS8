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
  ProductArenaPlan allocation;
  std::vector<cpp_int> bounds;
  uint32_t prefix=0;
  rns8_semantics semantics=RNS8_EXACT_WIDE_SIGNED;
  std::vector<int8_t> host;
  hip_direct_device_buffer device;
  std::vector<uint64_t*> holds;
  std::mutex mutex;
  ~Impl() { for(auto* hold:holds) --*hold; }
};
RuntimeProductRecipe::RuntimeProductRecipe(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
RuntimeProductRecipe::~RuntimeProductRecipe()=default;
const ProductArenaPlan& RuntimeProductRecipe::allocation() const { return impl_->allocation; }
rns8_status RuntimeProductRecipe::create(rns8_context* ctx,
    const std::vector<const rns8_matrix*>& inputs,const std::vector<ProductNode>& nodes,
    uint64_t budget,std::unique_ptr<RuntimeProductRecipe>& out) {
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
    state->allocation=plan_product_arena(shapes,nodes,state->prefix,budget);
    if(state->allocation.status!=RNS8_SUCCESS) return state->allocation.status;
    state->nodes=nodes;
    for(size_t i=0;i<nodes.size();++i) {
      const auto node=nodes[i];
      const auto a=state->allocation.values[node.a].shape,b=state->allocation.values[node.b].shape;
      if(ctx->backend==RNS8_BACKEND_HIP_DIRECT &&
          !qualified_i32_gemm_address(a.rows,b.cols,a.cols,a.cols,b.cols,b.cols,state->prefix))
        return RNS8_UNSUPPORTED_BACKEND;
      cpp_int bound=cpp_int(a.cols)*state->bounds[node.a]*state->bounds[node.b];
      if(!exact_range_fits(state->semantics,bound,modulus_product(state->prefix))) return RNS8_RANGE_ERROR;
      state->bounds.push_back(std::move(bound));
    }
    for(auto* hold:holds) if(*hold==UINT64_MAX) return RNS8_RANGE_ERROR;
    if(ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
      const auto status=state->device.allocate(ctx->device_id,static_cast<size_t>(state->allocation.arena_bytes));
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
rns8_status RuntimeProductRecipe::execute(rns8_matrix* output) {
  return guard_api([&]() -> rns8_status {
    auto& state=*impl_;
    std::unique_lock<std::mutex> lock(state.mutex,std::try_to_lock);
    if(!lock.owns_lock()) return RNS8_BUSY;
    if(!output) return RNS8_INVALID_ARGUMENT;
    if(state.ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if(output->in_flight_readers || output->in_flight_writer) return RNS8_BUSY;
    const auto shape=state.allocation.values.back().shape;
    if(output->backend!=state.ctx->backend || output->desc.semantics!=state.semantics ||
        output->desc.rows!=shape.rows || output->desc.cols!=shape.cols || output->prefix!=state.prefix ||
        !rns_matrix_storage_matches(*output,state.ctx->backend,shape.rows,shape.cols,state.prefix) ||
        (state.ctx->backend==RNS8_BACKEND_HIP_DIRECT && output->hip_device_id!=state.ctx->device_id))
      return RNS8_INVALID_ARGUMENT;
    if(output->generation_exhausted) return RNS8_RANGE_ERROR;
    for(const auto& input:state.inputs) {
      const auto& matrix=*input.matrix;
      if(input.matrix==output) return RNS8_INVALID_ARGUMENT;
      if(matrix.in_flight_writer) return RNS8_BUSY;
      if(matrix.generation_exhausted || matrix.matrix_instance_id!=input.instance ||
          matrix.content_generation!=input.generation ||
          !rns_residue_state_current_for_backend(matrix,state.ctx->backend)) return RNS8_INVALID_ARGUMENT;
    }
    // Copy the proof before work; final publication swaps it without allocation.
    cpp_int bound=state.bounds.back();
    const bool hip=state.ctx->backend==RNS8_BACKEND_HIP_DIRECT;
    auto* arena=hip?static_cast<int8_t*>(state.device.get()):state.host.data();
    const auto pointer=[&](uint32_t id) -> const int8_t* {
      if(id<state.inputs.size()) {
        const auto& matrix=*state.inputs[id].matrix;
        return hip?static_cast<const int8_t*>(matrix.hip_residues):matrix.residues.data();
      }
      return arena+state.allocation.values[id].offset;
    };
    for(size_t i=0;i<state.nodes.size();++i) {
      const auto node=state.nodes[i];
      const auto a=state.allocation.values[node.a].shape,b=state.allocation.values[node.b].shape;
      auto* c=arena+state.allocation.values[state.inputs.size()+i].offset;
      const auto* av=pointer(node.a); const auto* bv=pointer(node.b);
      if(hip) {
        const auto status=hip_direct_gemm_rns_device(state.ctx->device_id,av,bv,c,
            a.rows,b.cols,a.cols,a.cols,b.cols,b.cols,state.prefix);
        if(status!=RNS8_SUCCESS) return status;
      } else {
        for(uint32_t p=0;p<state.prefix;++p) {
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
    }
    MatrixWriteGuard write(*output);
    const auto* result=arena+state.allocation.values.back().offset;
    if(hip) {
      const auto status=hip_direct_copy_device_to_device(state.ctx->device_id,output->hip_residues,
          result,static_cast<size_t>(state.allocation.values.back().bytes));
      if(status!=RNS8_SUCCESS) return status;
      mark_output_device_residues_current(*output);
    } else {
      std::copy_n(result,static_cast<size_t>(state.allocation.values.back().bytes),output->residues.data());
      mark_output_host_residues_current(*output);
    }
    output->exact_max_magnitude.swap(bound);
    output->exact_axis_ranges.rows.clear(); output->exact_axis_ranges.cols.clear();
    output->exact_range_prefix=state.prefix;
    write.committed=true;
    return RNS8_SUCCESS;
  });
}
} // namespace rns8::detail
