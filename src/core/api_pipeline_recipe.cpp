#include "rns8/pipeline_recipe.h"
#include "core/api_internal.hpp"
#include "backend_common/checked_crt_export.hpp"
#include "backend_common/encoded_b_layout.hpp"
#include "backend_common/native_wmma_queue_plan.hpp"
#include "backend_common/native_locality.hpp"
#include "backend_common/pinned_host_contract.hpp"
#include <atomic>
#include <cmath>
#include <cstring>
using namespace rns8::detail;
using namespace rns8::detail::api;
namespace {
bool add_payload(uint64_t& value,uint64_t amount){if(value>uint64_t(PTRDIFF_MAX) || amount>uint64_t(PTRDIFF_MAX)-value)return false;value+=amount;return true;}
bool terminated(const char* text,size_t size){return std::memchr(text,0,size)!=nullptr;}
bool valid_recipe_options(const rns8_pipeline_recipe_options& o) {
  return valid_abi(o.struct_size,o.abi_version,sizeof(o)) && o.route_mask && !(o.route_mask&~31u) &&
      (o.execution_mode==RNS8_EXEC_SYNC || o.execution_mode==RNS8_EXEC_BORROWED_STREAM || o.execution_mode==RNS8_EXEC_OWNED_GRAPH) &&
      o.output_mode<=5 && (o.output_mode?(o.output_limbs>=1 && o.output_limbs<=(o.output_mode>=4?1u:32u) && o.export_algorithm<=(o.output_mode>=4?0u:4u) && o.output_ld>0):(!o.output_limbs && !o.export_algorithm && !o.output_ld)) &&
      (o.modulus_group==1 || o.modulus_group==2 || o.modulus_group==4) && o.workers && o.workers<=1024 &&
      o.fragments && o.fragments<=4096 && o.fragment_window && o.fragment_window<=o.fragments &&
      !(o.locality_order_mask&~15u) && o.reuse_count && o.max_extra_payload_bytes;
}
bool native_current(const rns8_matrix& value,bool sign) {
  return (value.desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT) && value.device_native_current && (sign?value.hip_native_i64:value.hip_native_u64)!=nullptr;
}
std::string recipe_identity(const rns8_plan& p,const rns8_matrix& a,const rns8_matrix& b,const rns8_matrix& c,
    const rns8_pipeline_recipe_options& o,unsigned route,unsigned method,const char* provider,bool binding=true) {
  std::string key="rns8-whole-pipeline-v1;";
  const auto text=[&](const std::string& x){key+=std::to_string(x.size())+":"+x;};
  const auto number=[&](uint64_t x){text(std::to_string(x));};
  text(p.backend_target_id);text(p.arithmetic_guard?p.arithmetic_guard->checked.sha256:"");text(provider);
  number(p.backend);number(p.desc.semantics);number(p.desc.finite_modulus);number(p.desc.m);number(p.desc.n);number(p.desc.k);number(p.prefix);number(p.desc.flags);
  number(a.prefix);number(b.prefix);number(c.prefix);number(a.exact_range_prefix);number(b.exact_range_prefix);
  text(a.exact_max_magnitude.str());text(b.exact_max_magnitude.str());number(a.device_native_current);number(b.device_native_current);
  if(binding){number(a.matrix_instance_id);number(a.content_generation);number(b.matrix_instance_id);number(b.content_generation);number(c.matrix_instance_id);}
  number(route);number(method);number(o.execution_mode);number(o.output_mode);number(o.output_limbs);
  number(o.export_algorithm);number(o.output_ld);number(o.modulus_group);number(o.workers);number(o.fragments);number(o.fragment_window);
  number(o.reuse_count);number(o.max_extra_payload_bytes);
  number(o.locality.target_id);number(o.locality.topology_version);number(o.locality.cache_domains);
  number(o.locality.concurrent_groups_per_domain);number(o.locality.cache_bytes_per_domain);
  char build[65]{};if(rns8_get_pipeline_build_identity(build,sizeof(build))==RNS8_SUCCESS)text(build);
  return sha256(key);
}
}
rns8_status rns8_query_pipeline_recipes(const rns8_context* ctx,const rns8_plan* plan,const rns8_matrix* a,
    const rns8_matrix* b,const rns8_matrix* c,const rns8_pipeline_recipe_options* options,
    const rns8_pipeline_recipe_prices* prices,uint32_t count,rns8_pipeline_recipe_candidate* output,uint32_t capacity,uint32_t* required) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || !plan || !a || !b || !c || !options || !required || (capacity && !output) ||
        (count && !prices) || count>256 || !valid_recipe_options(*options))return RNS8_INVALID_ARGUMENT;
    *required=0;const auto& o=*options;
    if(!plan->tile_schedule.empty())return RNS8_UNSUPPORTED_BACKEND;
    const bool wide=exact_wide_semantics(plan->desc.semantics),finite=uses_finite_storage(plan->desc.semantics);
    if((o.output_mode && o.output_mode<=3 && !wide) ||
        (o.output_mode==4 && plan->desc.semantics!=RNS8_BOUNDED_I64 && plan->desc.semantics!=RNS8_BOUNDED_U64 && plan->desc.semantics!=RNS8_WRAP_U64_MOD_2_64) ||
        (o.output_mode==5 && !finite))return RNS8_INVALID_ARGUMENT;
    AddressDomain host_output;if(o.output_mode && (o.output_ld<plan->desc.n ||
        !checked_host_plane_address(plan->desc.m,plan->desc.n,o.output_ld,1,uint64_t(o.output_limbs)*8,host_output)))return RNS8_RANGE_ERROR;
    if(!plan_schedule_contract_matches(*plan) || !plan->arithmetic_guard)return RNS8_INVALID_ARGUMENT;
    const auto admitted=finite?validate_finite_gemm_operands(*ctx,*plan,uint16_t(plan->desc.finite_modulus),*a,*b,*c):
        plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64?validate_wrap_gemm_operands(*ctx,*plan,*a,*b,*c):
        validate_rns_gemm_operands(*ctx,*plan,*a,*b,*c);
    if(admitted!=RNS8_SUCCESS)return admitted;
    char build[65]{};auto status=rns8_get_pipeline_build_identity(build,sizeof(build));if(status!=RNS8_SUCCESS)return status;
    for(uint32_t i=0;i<count;++i) {
      const auto& p=prices[i];if(!valid_abi(p.struct_size,p.abi_version,sizeof(p)) || p.known_mask>7 ||
          (p.origin!=1 && p.origin!=2) || !terminated(p.identity_sha256,sizeof(p.identity_sha256)) ||
          !terminated(p.build_identity_sha256,sizeof(p.build_identity_sha256)) || !terminated(p.target_id,sizeof(p.target_id)) ||
          !terminated(p.calibration_id,sizeof(p.calibration_id)) || !p.calibration_id[0])return RNS8_INVALID_ARGUMENT;
      for(uint32_t j=0;j<i;++j)if(std::strcmp(p.identity_sha256,prices[j].identity_sha256)==0)return RNS8_INVALID_ARGUMENT;
      for(unsigned stage=0;stage<3;++stage)if(p.known_mask&(1u<<stage))
        if(!std::isfinite(p.lower_ns[stage]) || !std::isfinite(p.upper_ns[stage]) || p.lower_ns[stage]<=0 || p.upper_ns[stage]<p.lower_ns[stage])return RNS8_INVALID_ARGUMENT;
    }
    std::vector<rns8_pipeline_recipe_candidate> candidates;
    const bool sign=plan->desc.semantics==RNS8_EXACT_WIDE_SIGNED;
    for(unsigned route=0;route<5;++route)if(o.route_mask&(1u<<route)) {
      const unsigned methods=route==RNS8_ROUTE_NARROW || route==RNS8_ROUTE_NATIVE_QUEUE?4:1;
      for(unsigned method=0;method<methods;++method) {
        if(route==RNS8_ROUTE_NATIVE_QUEUE && !((o.locality_order_mask?o.locality_order_mask:1u)&(1u<<method)))continue;
        rns8_pipeline_recipe_candidate x{};x.struct_size=sizeof(x);x.abi_version=RNS8_ABI_VERSION;x.route=route;x.method=method;x.execution_mode=o.execution_mode;x.resource_status=1;
        const auto reject=[&](rns8_status s,uint32_t reason,const char* text){x.status=s;x.reason_code=reason;if(reason==RNS8_SUPPORT_RESOURCE_OVERFLOW)x.resource_status=2;set_text(x.explanation,sizeof(x.explanation),text);};
        rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.problem=plan->desc;
        q.execution_modes=o.execution_mode;q.scratch_budget=UINT64_MAX;q.lda=a->desc.cols;q.ldb=b->desc.cols;q.ldc=c->desc.cols;
        q.operation=route==RNS8_ROUTE_NARROW?RNS8_OP_NARROW_NATIVE:
            route==RNS8_ROUTE_ENCODED_NATIVE_B || route==RNS8_ROUTE_NATIVE_QUEUE?RNS8_OP_NATIVE_GROUP:RNS8_OP_GEMM;
        q.algorithm=route==RNS8_ROUTE_NARROW?method:0;
        if(native_current(*a,sign) && native_current(*b,sign))q.proof_flags=RNS8_QUERY_NATIVE_RETAINED;
        if(route==RNS8_ROUTE_NARROW && a->exact_range_prefix && b->exact_range_prefix) {
          const boost::multiprecision::cpp_int product=a->exact_max_magnitude*b->exact_max_magnitude*plan->desc.k;
          const boost::multiprecision::cpp_int limit=sign?boost::multiprecision::cpp_int(INT64_MAX):boost::multiprecision::cpp_int(UINT64_MAX);
          const bool byte=method==1 || method==3;
          if(product<=limit && (!byte || (a->exact_max_magnitude<=(sign?127:255) && b->exact_max_magnitude<=(sign?127:255))))
            q.proof_flags|=RNS8_QUERY_NARROW_INPUT_PROOF;
        }
        const auto support=operation_support(*ctx,q,plan->backend);x.status=support.status;x.reason_code=support.reason_code;
        if(support.resource_status==2)x.resource_status=2;
        set_text(x.provider_identity_sha256,sizeof(x.provider_identity_sha256),support.provider_identity_sha256);
        set_text(x.explanation,sizeof(x.explanation),support.explanation);
        if(x.status==RNS8_SUCCESS && !wide && route!=RNS8_ROUTE_UNIFORM)
          reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"this semantic contract uses the uniform prepared producer");
        if(x.status==RNS8_SUCCESS && o.execution_mode!=RNS8_EXEC_SYNC && o.output_mode &&
            (plan->backend!=RNS8_BACKEND_HIP_DIRECT || o.export_algorithm ||
             (o.output_mode!=1 && o.output_mode!=2) || (o.output_mode==1)!=(plan->desc.semantics==RNS8_EXACT_WIDE_SIGNED)))
          reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_EXECUTION_MODE,"async host output requires direct-HIP signed/unsigned checked Garner limbs matching the exact-wide semantic");
        uint64_t extra=route==RNS8_ROUTE_SPLIT_K?0:plan->backend_workspace_required_bytes;
        if(x.status==RNS8_SUCCESS && route==RNS8_ROUTE_NARROW &&
            (!add_payload(extra,support.required_device_bytes) || !add_payload(extra,support.required_host_bytes)))
          reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_RESOURCE_OVERFLOW,"narrow producer payload overflow");
        if(x.status==RNS8_SUCCESS && route==RNS8_ROUTE_SPLIT_K) {
          if(plan->backend!=RNS8_BACKEND_HIP_DIRECT || o.execution_mode!=RNS8_EXEC_SYNC || a->prefix!=plan->prefix || b->prefix!=plan->prefix)
            reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"split-K recipe requires synchronous direct-HIP equal-prefix operands");
          else {rns8_split_k_resources r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
            const auto s=rns8_query_split_k_windowed(plan->desc.m,plan->desc.n,plan->desc.k,plan->prefix,o.fragments,o.fragment_window,UINT64_MAX,&r);
            if(s!=RNS8_SUCCESS)reject(s,RNS8_SUPPORT_RESOURCE_OVERFLOW,"split-K resource query rejected this schedule");else extra=r.scratch_bytes;}
        }
        if(x.status==RNS8_SUCCESS && (route==RNS8_ROUTE_ENCODED_NATIVE_B || route==RNS8_ROUTE_NATIVE_QUEUE)) {
          if(route==RNS8_ROUTE_ENCODED_NATIVE_B && b->prefix!=plan->prefix)
            reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"encoded-B recipe requires the cache and producer to have the same ordered prefix");
          if(route==RNS8_ROUTE_ENCODED_NATIVE_B) {
            const auto layout=encoded_b_layout(plan->desc.k,plan->desc.n,plan->desc.n,plan->prefix,UINT64_MAX);
            if(layout.status!=RNS8_SUCCESS || !add_payload(extra,layout.total_bytes))reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_RESOURCE_OVERFLOW,"encoded-B payload overflow");
          }
          if(x.status==RNS8_SUCCESS && extra>=o.max_extra_payload_bytes)
            reject(RNS8_WORKSPACE_TOO_SMALL,RNS8_SUPPORT_BUDGET,"representation leaves no payload budget for its native schedule");
          if(x.status==RNS8_SUCCESS) {
          NativeWmmaTask task;task.a=sign?a->hip_native_i64:a->hip_native_u64;task.b=sign?b->hip_native_i64:b->hip_native_u64;
          task.c=static_cast<int8_t*>(c->hip_residues);task.m=plan->desc.m;task.n=plan->desc.n;task.k=plan->desc.k;
          task.lda=task.k;task.ldb=task.n;task.ldc=task.n;task.prefix=plan->prefix;task.group=o.modulus_group;task.unsigned_inputs=!sign;
          NativeWmmaSchedulePlan schedule;
          const auto s=build_native_wmma_schedule({task},o.workers,o.max_extra_payload_bytes-extra,{1,NativeWmmaScheduleMode::explicit_round_robin,1},schedule);
          if(s!=RNS8_SUCCESS)reject(s,RNS8_SUPPORT_RESOURCE_OVERFLOW,"native queue descriptor admission failed");
          else {
            if(!add_payload(extra,schedule.device_bytes) || !add_payload(extra,schedule.peak_host_payload_bytes+sizeof(NativeWmmaTask)))
              reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_RESOURCE_OVERFLOW,"native queue peak payload overflow");
            if(x.status==RNS8_SUCCESS && route==RNS8_ROUTE_NATIVE_QUEUE && method) {
              const uint64_t intervals=uint64_t(task.m)+uint64_t(task.k);
              if(extra>o.max_extra_payload_bytes || intervals>(o.max_extra_payload_bytes-extra)/sizeof(std::pair<uintptr_t,uintptr_t>))
                reject(RNS8_WORKSPACE_TOO_SMALL,RNS8_SUPPORT_BUDGET,"locality source-interval preparation exceeds the extra-payload budget");
              else {
              auto locality=o.locality;locality.order=rns8_locality_order(method);rns8_locality_info info{};uint64_t transient=0;
              const auto lowered=lower_native_locality({task},schedule.explicit_work,locality,plan->backend_target_id,info,transient);
              if(lowered!=RNS8_SUCCESS || !info.applied)reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_TARGET,"requested locality has no matching explicit-work target lowering");
              else if(!add_payload(extra,transient))reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_RESOURCE_OVERFLOW,"locality preparation peak overflow");
              }
            }
          }
          }
        }
        x.retained_payload_bytes=extra;x.preparation_peak_bytes=extra;
        if(x.status==RNS8_SUCCESS && o.output_mode && o.execution_mode!=RNS8_EXEC_SYNC) {
          rns8_pinned_host_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
          d.element_type=o.output_mode==1?RNS8_NATIVE_DEVICE_SIGNED_LIMBS:RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS;
          d.access=RNS8_NATIVE_DEVICE_WRITE;d.rows=plan->desc.m;d.cols=plan->desc.n;d.ld=o.output_ld;d.limb_count=o.output_limbs;
          PinnedHostResources r;uint64_t device=0,pinned=0,total=0;
          const auto s=query_pinned_host_contract(d,r);
          if(s!=RNS8_SUCCESS || !pinned_host_transfer_bytes(r,plan->prefix,false,false,device,pinned,total))
            reject(s==RNS8_SUCCESS?RNS8_RANGE_ERROR:s,RNS8_SUPPORT_RESOURCE_OVERFLOW,"async checked host export payload query failed");
          else {x.export_payload_bytes=total;if(!add_payload(extra,total))reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_RESOURCE_OVERFLOW,"producer/export payload overflow");}
        } else if(x.status==RNS8_SUCCESS && o.output_mode>=4) {
          // Finite output is staged as bytes then widened to the documented
          // word representation. Native exporters own their checked staging.
          x.export_payload_bytes=plan->desc.m*plan->desc.n*uint64_t(o.output_mode==5?1:8);
          if(!add_payload(extra,x.export_payload_bytes))reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_RESOURCE_OVERFLOW,"native export payload overflow");
        } else if(x.status==RNS8_SUCCESS && o.output_mode) {
          const bool gpu=hip_device_backend_kind(plan->backend);
          if(!checked_crt_backend_supported(plan->backend))reject(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"selected producer cannot bind the requested checked CRT owner");
          else {
            const auto r=checked_crt::resources(plan->desc.m*plan->desc.n,plan->prefix,o.output_limbs,o.export_algorithm,gpu,UINT64_MAX);
            if(r.status!=RNS8_SUCCESS)reject(r.status,RNS8_SUPPORT_RESOURCE_OVERFLOW,"checked CRT payload query failed");
            else {x.export_payload_bytes=r.total_bytes;if(!add_payload(extra,r.total_bytes))reject(RNS8_RANGE_ERROR,RNS8_SUPPORT_RESOURCE_OVERFLOW,"producer/export payload overflow");}
          }
        }
        x.peak_extra_payload_bytes=extra;
        if(x.status==RNS8_SUCCESS && extra>o.max_extra_payload_bytes)reject(RNS8_WORKSPACE_TOO_SMALL,RNS8_SUPPORT_BUDGET,"known complete-route payload exceeds the explicit extra-payload budget");
        const auto identity=recipe_identity(*plan,*a,*b,*c,o,route,method,x.provider_identity_sha256);
        const auto cost_identity=recipe_identity(*plan,*a,*b,*c,o,route,method,x.provider_identity_sha256,false);
        set_text(x.identity_sha256,sizeof(x.identity_sha256),identity);
        set_text(x.cost_identity_sha256,sizeof(x.cost_identity_sha256),cost_identity);
        set_text(x.build_identity_sha256,sizeof(x.build_identity_sha256),build);set_text(x.target_id,sizeof(x.target_id),plan->backend_target_id);
        if(x.status==RNS8_SUCCESS) {
          const uint32_t needed=o.output_mode?7:3;
          for(uint32_t i=0;i<count;++i)if(cost_identity==prices[i].identity_sha256 && std::strcmp(build,prices[i].build_identity_sha256)==0 &&
              plan->backend_target_id==prices[i].target_id && (prices[i].known_mask&needed)==needed) {
            x.cost_known=1;x.cost_origin=prices[i].origin;x.lower_amortized_ns=prices[i].lower_ns[0]/o.reuse_count+prices[i].lower_ns[1]+(o.output_mode?prices[i].lower_ns[2]:0);
            x.upper_amortized_ns=prices[i].upper_ns[0]/o.reuse_count+prices[i].upper_ns[1]+(o.output_mode?prices[i].upper_ns[2]:0);
            if(!std::isfinite(x.upper_amortized_ns))return RNS8_RANGE_ERROR;
          }
          set_text(x.explanation,sizeof(x.explanation),x.cost_known?"complete bound route priced explicitly; no AUTO or promotion":"complete bound route; cost unknown until an exact route/build/target profile is supplied");
        }
        candidates.push_back(x);
      }
    }
    for(auto& x:candidates)if(x.status==RNS8_SUCCESS && x.cost_known) {
      x.pareto=1;for(const auto& y:candidates)if(y.status==RNS8_SUCCESS && y.cost_known && y.peak_extra_payload_bytes<=x.peak_extra_payload_bytes &&
          y.upper_amortized_ns<=x.upper_amortized_ns && (y.peak_extra_payload_bytes<x.peak_extra_payload_bytes || y.upper_amortized_ns<x.upper_amortized_ns))x.pareto=0;
    }
    *required=uint32_t(candidates.size());for(uint32_t i=0;i<std::min(capacity,*required);++i)output[i]=candidates[i];return RNS8_SUCCESS;
  });
}
rns8_status rns8_select_pipeline_recipe(const rns8_pipeline_recipe_candidate* candidates,uint32_t count,uint32_t* selected) {
  if(!candidates || !count || count>256 || !selected)return RNS8_INVALID_ARGUMENT;
  uint32_t best=UINT32_MAX;
  for(uint32_t i=0;i<count;++i) {
    const auto& x=candidates[i];if(!valid_abi(x.struct_size,x.abi_version,sizeof(x)))return RNS8_INVALID_ARGUMENT;
    if(x.status!=RNS8_SUCCESS || !x.cost_known)continue;
    if(!std::isfinite(x.lower_amortized_ns) || !std::isfinite(x.upper_amortized_ns) || x.lower_amortized_ns<=0 || x.upper_amortized_ns<x.lower_amortized_ns)return RNS8_INVALID_ARGUMENT;
    if(best==UINT32_MAX || x.upper_amortized_ns<candidates[best].upper_amortized_ns ||
        (x.upper_amortized_ns==candidates[best].upper_amortized_ns && x.peak_extra_payload_bytes<candidates[best].peak_extra_payload_bytes))best=i;
  }
  if(best==UINT32_MAX)return RNS8_UNSUPPORTED_BACKEND;*selected=best;return RNS8_SUCCESS;
}

struct rns8_pipeline_recipe {
  rns8_context* ctx=nullptr;const rns8_plan* plan=nullptr;
  const rns8_matrix *a=nullptr,*b=nullptr;rns8_matrix* c=nullptr;
  uint64_t a_instance=0,b_instance=0,c_instance=0,a_generation=0,b_generation=0;
  rns8_pipeline_recipe_options options{};rns8_pipeline_recipe_candidate candidate{};
  rns8_workspace* workspace=nullptr;rns8_prepared_execution* execution=nullptr;
  rns8_prepared_group* group=nullptr;rns8_encoded_b* encoded=nullptr;
  rns8_split_k* split=nullptr;rns8_crt_export* export_owner=nullptr;
  rns8_pinned_host_buffer* host_output=nullptr;rns8_prepared_execution* host_export=nullptr;
  rns8_prepared_execution* latest_owner=nullptr;
  rns8_completion_token latest{};bool async_issued=false,refs=false,usable=false;
  rns8_pipeline_recipe* cleanup_next=nullptr;
};
namespace {
std::atomic<rns8_pipeline_recipe*> recipe_cleanup{nullptr};
void quarantine_recipe(rns8_pipeline_recipe* p) noexcept {
  auto* head=recipe_cleanup.load(std::memory_order_relaxed);
  do{p->cleanup_next=head;}while(!recipe_cleanup.compare_exchange_weak(head,p,std::memory_order_release,std::memory_order_relaxed));
}
rns8_status close_recipe(rns8_pipeline_recipe& p) {
  p.usable=false;
  // A failed operation may nevertheless have drained. The actual child owners
  // decide whether resources can be destroyed; a failed wait never permits free.
  if(p.async_issued && p.latest_owner) {
    rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    (void)rns8_wait_prepared_hip(p.latest_owner,p.latest,&info);
  }
  if(p.host_export){const auto s=rns8_destroy_prepared_execution(p.host_export);if(s!=RNS8_SUCCESS)return s;
    if(p.latest_owner==p.host_export){p.latest_owner=nullptr;p.async_issued=false;}p.host_export=nullptr;}
  if(p.host_output){const auto s=rns8_destroy_pinned_host_buffer(p.host_output);if(s!=RNS8_SUCCESS)return s;p.host_output=nullptr;}
  if(p.export_owner){const auto s=rns8_destroy_crt_export(p.export_owner);if(s!=RNS8_SUCCESS)return s;p.export_owner=nullptr;}
  if(p.group){const auto s=rns8_destroy_prepared_group(p.group);if(s!=RNS8_SUCCESS)return s;p.group=nullptr;}
  if(p.execution){const auto s=rns8_destroy_prepared_execution(p.execution);if(s!=RNS8_SUCCESS)return s;
    if(p.latest_owner==p.execution){p.latest_owner=nullptr;p.async_issued=false;}p.execution=nullptr;}
  if(p.encoded){const auto s=rns8_destroy_encoded_b(p.encoded);if(s!=RNS8_SUCCESS)return s;p.encoded=nullptr;}
  if(p.split){const auto s=rns8_destroy_split_k(p.split);if(s!=RNS8_SUCCESS)return s;p.split=nullptr;}
  if(p.workspace){const auto s=rns8_destroy_workspace(p.workspace);if(s!=RNS8_SUCCESS)return s;p.workspace=nullptr;}
  if(p.refs) {
    --p.ctx->prepared_references;--p.plan->prepared_references;--p.a->prepared_references;
    if(p.b!=p.a)--p.b->prepared_references;
    if(p.c!=p.a && p.c!=p.b)--p.c->prepared_references;
    p.refs=false;
  }
  return RNS8_SUCCESS;
}
struct RecipeDeleter {
  void operator()(rns8_pipeline_recipe* p)const noexcept {
    if(!p)return;
    try{if(close_recipe(*p)==RNS8_SUCCESS){delete p;return;}}catch(...){}
    quarantine_recipe(p);
  }
};
rns8_status recipe_current(const rns8_pipeline_recipe* p) {
  if(!p || !p->usable)return RNS8_INVALID_ARGUMENT;
  if(p->a->matrix_instance_id!=p->a_instance || p->b->matrix_instance_id!=p->b_instance || p->c->matrix_instance_id!=p->c_instance ||
      p->a->content_generation!=p->a_generation || p->b->content_generation!=p->b_generation)return RNS8_INVALID_ARGUMENT;
  return RNS8_SUCCESS;
}
}
rns8_status rns8_prepare_pipeline_recipe(rns8_context* ctx,const rns8_plan* plan,const rns8_matrix* a,
    const rns8_matrix* b,rns8_matrix* c,const rns8_pipeline_recipe_options* options,
    const rns8_pipeline_recipe_candidate* selected,rns8_pipeline_recipe** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!selected || !valid_abi(selected->struct_size,selected->abi_version,sizeof(*selected)) ||
        !terminated(selected->identity_sha256,sizeof(selected->identity_sha256)))return RNS8_INVALID_ARGUMENT;
    rns8_pipeline_recipe_candidate fresh[16]{};uint32_t count=0;
    auto status=rns8_query_pipeline_recipes(ctx,plan,a,b,c,options,nullptr,0,fresh,16,&count);if(status!=RNS8_SUCCESS)return status;
    const rns8_pipeline_recipe_candidate* match=nullptr;
    for(uint32_t i=0;i<count;++i)if(fresh[i].route==selected->route && fresh[i].method==selected->method &&
        std::strcmp(fresh[i].identity_sha256,selected->identity_sha256)==0)match=&fresh[i];
    if(!match)return RNS8_INVALID_ARGUMENT;if(match->status!=RNS8_SUCCESS)return match->status;
    if(ctx->prepared_references==UINT64_MAX || plan->prepared_references==UINT64_MAX || a->prepared_references==UINT64_MAX ||
        b->prepared_references==UINT64_MAX || c->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
    std::unique_ptr<rns8_pipeline_recipe,RecipeDeleter> p(new rns8_pipeline_recipe);
    p->ctx=ctx;p->plan=plan;p->a=a;p->b=b;p->c=c;p->options=*options;p->candidate=*match;
    p->a_instance=a->matrix_instance_id;p->b_instance=b->matrix_instance_id;p->c_instance=c->matrix_instance_id;
    p->a_generation=a->content_generation;p->b_generation=b->content_generation;
    ++ctx->prepared_references;++plan->prepared_references;++a->prepared_references;
    if(b!=a)++b->prepared_references;if(c!=a && c!=b)++c->prepared_references;p->refs=true;
    const auto fail=[&](rns8_status failure){if(close_recipe(*p)!=RNS8_SUCCESS)*out=p.release();return failure;};
    const auto budget=p->options.max_extra_payload_bytes-p->candidate.export_payload_bytes;
    if(match->route==RNS8_ROUTE_SPLIT_K) {
      status=rns8_create_split_k_windowed(ctx,a,b,options->fragments,options->fragment_window,budget,&p->split);
      if(status!=RNS8_SUCCESS)return fail(status);
    }else {
      status=rns8_create_workspace(ctx,plan,&p->workspace);if(status!=RNS8_SUCCESS)return fail(status);
      if(match->route!=RNS8_ROUTE_NARROW) {
        status=rns8_prepare_gemm(ctx,plan,a,b,c,p->workspace,&p->execution);if(status!=RNS8_SUCCESS)return fail(status);
        if(options->execution_mode!=RNS8_EXEC_SYNC) {
          status=rns8_prepare_hip_submission(p->execution);if(status!=RNS8_SUCCESS)return fail(status);
        }
      }
      if(match->route==RNS8_ROUTE_ENCODED_NATIVE_B || match->route==RNS8_ROUTE_NATIVE_QUEUE) {
        uint64_t encoded_bytes=0;
        if(match->route==RNS8_ROUTE_ENCODED_NATIVE_B) {
          const auto layout=encoded_b_layout(plan->desc.k,plan->desc.n,b->desc.cols,plan->prefix,UINT64_MAX);encoded_bytes=layout.total_bytes;
          status=rns8_prepare_encoded_b(ctx,b,encoded_bytes,&p->encoded);if(status!=RNS8_SUCCESS)return fail(status);
        }
        status=rns8_create_prepared_group(&p->execution,1,&p->group);if(status!=RNS8_SUCCESS)return fail(status);
        rns8_native_group_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.claim_chunk=1;d.workers=options->workers;
        d.ordinary_grid_control=match->route==RNS8_ROUTE_ENCODED_NATIVE_B;
        d.max_metadata_bytes=budget-plan->backend_workspace_required_bytes-encoded_bytes;
        rns8_native_group_binding binding{};binding.struct_size=sizeof(binding);binding.abi_version=RNS8_ABI_VERSION;
        binding.modulus_group=options->modulus_group;binding.encoded_b=p->encoded;
        if(match->route==RNS8_ROUTE_NATIVE_QUEUE && match->method) {
          auto locality=options->locality;locality.order=rns8_locality_order(match->method);
          status=rns8_prepare_native_group_locality(p->group,&d,&binding,1,&locality);
        }else status=rns8_prepare_native_group(p->group,&d,&binding,1);
        if(status!=RNS8_SUCCESS)return fail(status);
      }
      if(options->execution_mode==RNS8_EXEC_OWNED_GRAPH) {
        status=p->group?rns8_capture_native_group_hip_graph(p->group):rns8_capture_prepared_hip_graph(p->execution);
        if(status!=RNS8_SUCCESS)return fail(status);
      }
    }
    if(options->output_mode && options->execution_mode!=RNS8_EXEC_SYNC) {
      rns8_pinned_host_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
      d.element_type=options->output_mode==1?RNS8_NATIVE_DEVICE_SIGNED_LIMBS:RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS;
      d.access=RNS8_NATIVE_DEVICE_WRITE;d.rows=plan->desc.m;d.cols=plan->desc.n;d.ld=options->output_ld;d.limb_count=options->output_limbs;
      boost::multiprecision::cpp_int envelope;status=exact_gemm_range(*plan,*a,*b,envelope);
      if(status!=RNS8_SUCCESS)return fail(status);
      status=rns8_create_pinned_host_buffer_hip(ctx,&d,p->candidate.export_payload_bytes,&p->host_output);
      if(status!=RNS8_SUCCESS)return fail(status);
      status=prepare_bound_host_export_hip(ctx,c,p->host_output,plan->prefix,envelope,p->candidate.export_payload_bytes,&p->host_export);
      if(status!=RNS8_SUCCESS)return fail(status);
    }
    p->usable=true;*out=p.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_pipeline_recipe(rns8_pipeline_recipe* p,uint64_t* dst,int64_t ld) {
  return guard_api([&]() -> rns8_status {
    auto status=recipe_current(p);if(status!=RNS8_SUCCESS)return status;
    if(p->options.execution_mode!=RNS8_EXEC_SYNC)return RNS8_UNSUPPORTED_BACKEND;
    if(p->options.output_mode?(!dst || ld!=p->options.output_ld):(dst || ld))return RNS8_INVALID_ARGUMENT;
    // Full external address admission before the producer can mutate C.
    AddressDomain span;if(dst && (!checked_host_plane_address(p->plan->desc.m,p->plan->desc.n,ld,1,uint64_t(p->options.output_limbs)*8,span) ||
        span.byte_extent>UINTPTR_MAX-reinterpret_cast<uintptr_t>(dst)))return RNS8_RANGE_ERROR;
    if(p->split)status=rns8_execute_split_k(p->split,p->c);
    else if(p->candidate.route==RNS8_ROUTE_NARROW)status=rns8_gemm_narrow_exact(p->ctx,p->plan,p->a,p->b,p->c,
        rns8_narrow_product_method(p->candidate.method),p->options.max_extra_payload_bytes-p->candidate.export_payload_bytes-p->plan->backend_workspace_required_bytes,p->workspace);
    else if(p->group){uint32_t done=0;status=rns8_execute_prepared_group(p->group,&p->latest,1,&done);}
    else status=rns8_execute_prepared(p->execution,&p->latest);
    if(status!=RNS8_SUCCESS || !dst)return status;
    if(p->options.output_mode==4)return p->plan->desc.semantics==RNS8_BOUNDED_I64?
        rns8_export_i64(p->ctx,p->plan,p->c,reinterpret_cast<int64_t*>(dst),ld):rns8_export_u64(p->ctx,p->plan,p->c,dst,ld);
    if(p->options.output_mode==5) {
      std::vector<uint8_t> values(size_t(p->plan->desc.m*p->plan->desc.n));
      status=rns8_export_finite_u8(p->ctx,p->plan,uint16_t(p->plan->desc.finite_modulus),p->c,values.data(),p->plan->desc.n);
      if(status!=RNS8_SUCCESS)return status;
      for(int64_t row=0;row<p->plan->desc.m;++row)for(int64_t col=0;col<p->plan->desc.n;++col)dst[row*ld+col]=values[size_t(row*p->plan->desc.n+col)];
      return RNS8_SUCCESS;
    }
    if(!p->export_owner) {
      rns8_crt_export_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
      d.algorithm=p->options.export_algorithm;d.mode=p->options.output_mode-1;d.limb_count=p->options.output_limbs;d.max_scratch_bytes=p->candidate.export_payload_bytes;
      status=rns8_prepare_crt_export(p->ctx,p->plan,p->c,&d,&p->export_owner);
    }else status=rns8_rebind_crt_export(p->export_owner,p->c);
    return status==RNS8_SUCCESS?rns8_export_crt_limbs(p->export_owner,dst,ld):status;
  });
}
rns8_status rns8_submit_pipeline_recipe_hip(rns8_pipeline_recipe* p,void* stream,rns8_completion_token* out) {
  return guard_api([&]() -> rns8_status {
    if(!out || !stream)return RNS8_INVALID_ARGUMENT;
    auto status=recipe_current(p);if(status!=RNS8_SUCCESS)return status;
    if(p->options.execution_mode==RNS8_EXEC_SYNC || !p->execution)return RNS8_UNSUPPORTED_BACKEND;
    if(p->host_output) {
      rns8_pinned_host_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      status=rns8_get_pinned_host_info(p->host_output,&info);if(status!=RNS8_SUCCESS)return status;
      if(info.device_writer || info.host_readers || info.write_window || info.device_readers)return RNS8_BUSY;
    }
    const bool graph=p->options.execution_mode==RNS8_EXEC_OWNED_GRAPH;rns8_completion_token token{};
    if(p->group){uint32_t issued=0;status=graph?rns8_submit_native_group_hip_graph(p->group,stream,&token,1,&issued):rns8_submit_native_group_hip(p->group,stream,&token,1,&issued);}
    else status=graph?rns8_submit_prepared_hip_graph(p->execution,stream,&token):rns8_submit_prepared_hip(p->execution,stream,&token);
    if(token.epoch){p->latest=token;p->latest_owner=p->execution;p->async_issued=true;*out=token;}
    if(status!=RNS8_SUCCESS || !p->host_export)return status;
    status=rns8_bind_prepared_dependency(p->host_export,p->execution,token);
    if(status!=RNS8_SUCCESS)return status;
    rns8_completion_token exported{};status=rns8_submit_prepared_hip_pipeline(p->host_export,stream,&exported);
    if(exported.epoch){p->latest=exported;p->latest_owner=p->host_export;*out=exported;}
    return status;
  });
}
rns8_status rns8_query_pipeline_recipe_hip(rns8_pipeline_recipe* p,rns8_completion_token token,rns8_completion_info* info) {
  if(!p || !p->latest_owner || !p->async_issued)return RNS8_INVALID_ARGUMENT;return rns8_query_prepared_hip(p->latest_owner,token,info);
}
rns8_status rns8_wait_pipeline_recipe_hip(rns8_pipeline_recipe* p,rns8_completion_token token,rns8_completion_info* info) {
  if(!p || !p->latest_owner || !p->async_issued)return RNS8_INVALID_ARGUMENT;return rns8_wait_prepared_hip(p->latest_owner,token,info);
}
rns8_status rns8_cancel_pipeline_recipe_hip(rns8_pipeline_recipe* p,rns8_completion_token token) {
  if(!p || !p->latest_owner || !p->async_issued)return RNS8_INVALID_ARGUMENT;return rns8_cancel_prepared_hip(p->latest_owner,token);
}
rns8_status rns8_begin_pipeline_recipe_host_read(rns8_pipeline_recipe* p,const uint64_t** data,uint64_t* bytes) {
  if(!p || !p->host_output || !data || !bytes)return RNS8_INVALID_ARGUMENT;
  const void* pointer=nullptr;const auto status=rns8_begin_pinned_host_read(p->host_output,&pointer,bytes);
  if(status==RNS8_SUCCESS)*data=static_cast<const uint64_t*>(pointer);return status;
}
rns8_status rns8_end_pipeline_recipe_host_read(rns8_pipeline_recipe* p) {
  return p && p->host_output?rns8_end_pinned_host_read(p->host_output):RNS8_INVALID_ARGUMENT;
}
rns8_status rns8_get_pipeline_recipe_candidate(const rns8_pipeline_recipe* p,rns8_pipeline_recipe_candidate* out) {
  if(!p || !out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out)))return RNS8_INVALID_ARGUMENT;*out=p->candidate;return RNS8_SUCCESS;
}
rns8_status rns8_destroy_pipeline_recipe(rns8_pipeline_recipe* p) {
  if(!p)return RNS8_SUCCESS;const auto status=close_recipe(*p);if(status!=RNS8_SUCCESS)return status;delete p;return RNS8_SUCCESS;
}
void rns8_reap_pipeline_recipe_cleanup(void) {
  auto* p=recipe_cleanup.exchange(nullptr,std::memory_order_acquire);
  while(p){auto* next=p->cleanup_next;p->cleanup_next=nullptr;RecipeDeleter{}(p);p=next;}
}
