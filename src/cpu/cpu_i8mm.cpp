#include "core/api_internal.hpp"
#include "backend_common/cpu_i8mm_layout.hpp"
#include "rns8/cpu_i8mm.h"
#include <algorithm>
#include <mutex>
#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
#include <omp.h>
#endif
#ifndef RNS8_CPU_I8MM_COMPILED
#define RNS8_CPU_I8MM_COMPILED 0
#endif
#if RNS8_CPU_I8MM_COMPILED
extern "C" void rns8_cpu_i8mm_tile(const int8_t*,const int8_t*,int8_t*,int64_t,int64_t,int64_t,int64_t,int64_t,uint32_t) noexcept;
#endif
using namespace rns8::detail;
using namespace rns8::detail::api;
using boost::multiprecision::cpp_int;
struct rns8_prepared_cpu_i8mm {
  rns8_context* ctx=nullptr;const rns8_plan* plan=nullptr;
  const rns8_matrix *a=nullptr,*b=nullptr;rns8_matrix* c=nullptr;
  rns8_workspace* workspace=nullptr;
  CpuI8mmLayout layout;uint32_t prefix=0,last_threads=0;
  uint64_t a_instance=0,b_instance=0,a_generation=0,b_generation=0,packs=0,reuses=0;
  cpp_int envelope;bool bound_frozen=false;std::vector<int8_t> packed,output;
  mutable std::mutex mutex;
};
namespace {
bool admitted(rns8_cpu_i8mm_info& info) noexcept {
  info.compiled=RNS8_CPU_I8MM_COMPILED;info.safe_k=65536;info.layout_version=1;
#if defined(__linux__) && defined(__aarch64__)
  info.linux_aarch64=1;
#if defined(HWCAP_ASIMD) && defined(HWCAP2_I8MM)
  info.os_asimd=(getauxval(AT_HWCAP)&HWCAP_ASIMD)!=0;
  info.os_i8mm=(getauxval(AT_HWCAP2)&HWCAP2_I8MM)!=0;
#endif
#endif
  info.admitted=info.compiled && info.linux_aarch64 && info.os_asimd && info.os_i8mm;
  return info.admitted!=0;
}
rns8_status validate(rns8_prepared_cpu_i8mm& o,bool refresh,cpp_int& bound,rns8_exact_matrix_ranges& ranges) {
  auto status=validate_plan_context_workspace(*o.ctx,*o.plan,*o.workspace);if(status!=RNS8_SUCCESS)return status;
  if(o.plan->backend!=RNS8_BACKEND_CPU_REFERENCE || o.ctx->backend!=RNS8_BACKEND_CPU_REFERENCE ||
      !o.plan->tile_schedule.empty() || o.a->experimental_basis || o.b->experimental_basis || o.c->experimental_basis)
    return RNS8_UNSUPPORTED_BACKEND;
  if(!refresh && (o.a->matrix_instance_id!=o.a_instance || o.b->matrix_instance_id!=o.b_instance ||
      o.a->content_generation!=o.a_generation || o.b->content_generation!=o.b_generation))return RNS8_INVALID_ARGUMENT;
  if(uses_finite_storage(o.plan->desc.semantics)) {
    status=validate_finite_gemm_operands(*o.ctx,*o.plan,uint16_t(o.plan->desc.finite_modulus),*o.a,*o.b,*o.c);
    bound=0;ranges={};
  } else {
    status=validate_rns_gemm_operands(*o.ctx,*o.plan,*o.a,*o.b,*o.c);
    if(status==RNS8_SUCCESS)status=exact_gemm_range(*o.plan,*o.a,*o.b,bound,&ranges);
  }
  if(status!=RNS8_SUCCESS)return status;
  if(o.a->residues.size()<uint64_t(o.plan->desc.m)*o.plan->desc.k*o.prefix ||
      o.b->residues.size()<uint64_t(o.plan->desc.k)*o.plan->desc.n*o.prefix ||
      o.c->residues.size()<o.layout.output)return RNS8_INVALID_ARGUMENT;
  return o.bound_frozen && bound>o.envelope?RNS8_RANGE_ERROR:RNS8_SUCCESS;
}
void bind(rns8_prepared_cpu_i8mm& o) {
  o.a_instance=o.a->matrix_instance_id;o.a_generation=o.a->content_generation;
  o.b_instance=o.b->matrix_instance_id;o.b_generation=o.b->content_generation;
}
void pack(rns8_prepared_cpu_i8mm& o) noexcept {
  const auto& d=o.plan->desc;
  for(uint32_t p=0;p<o.prefix;++p)for(uint64_t col=0;col<o.layout.nt;++col)for(uint64_t kb=0;kb<o.layout.kt;++kb)
    cpu_i8mm_pack_b(o.b->residues.data()+uint64_t(p)*d.k*d.n,d.k,d.n,d.n,int64_t(kb*8),int64_t(col*2),
        o.packed.data()+((uint64_t(p)*o.layout.nt+col)*o.layout.kt+kb)*16);
  if(o.packs!=UINT64_MAX)++o.packs;
}
}
rns8_status rns8_query_cpu_i8mm(rns8_cpu_i8mm_info* out) {
  if(!out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out)))return RNS8_INVALID_ARGUMENT;
  rns8_cpu_i8mm_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
  admitted(info);*out=info;return RNS8_SUCCESS;
}
rns8_status rns8_prepare_cpu_i8mm(rns8_context* ctx,const rns8_plan* plan,const rns8_matrix* a,const rns8_matrix* b,
    rns8_matrix* c,uint64_t budget,rns8_prepared_cpu_i8mm** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!ctx || !plan || !a || !b || !c || a==c || b==c)return RNS8_INVALID_ARGUMENT;
    rns8_cpu_i8mm_info support{};if(!admitted(support))return RNS8_UNSUPPORTED_BACKEND;
    if(plan->backend!=RNS8_BACKEND_CPU_REFERENCE || ctx->backend!=RNS8_BACKEND_CPU_REFERENCE ||
        plan->desc.semantics==RNS8_WRAP_U64_MOD_2_64 || !plan->tile_schedule.empty())return RNS8_UNSUPPORTED_BACKEND;
    auto owner=std::make_unique<rns8_prepared_cpu_i8mm>();owner->ctx=ctx;owner->plan=plan;owner->a=a;owner->b=b;owner->c=c;
    owner->prefix=uses_finite_storage(plan->desc.semantics)?1:rns_storage_prefix_for_plan(*plan);
    if(!cpu_i8mm_layout(plan->desc.m,plan->desc.n,plan->desc.k,owner->prefix,owner->layout))return RNS8_RANGE_ERROR;
    if(owner->layout.total>budget)return RNS8_WORKSPACE_TOO_SMALL;
    for(auto* slot:{&ctx->prepared_references,&plan->prepared_references,&a->prepared_references,&b->prepared_references,&c->prepared_references})
      if(*slot>UINT64_MAX-2)return RNS8_RANGE_ERROR;
    auto status=rns8_create_workspace(ctx,plan,&owner->workspace);if(status!=RNS8_SUCCESS)return status;
    struct Cleanup {rns8_prepared_cpu_i8mm& o;bool done=false;~Cleanup(){if(!done && o.workspace)(void)rns8_destroy_workspace(o.workspace);}} cleanup{*owner};
    cpp_int bound;rns8_exact_matrix_ranges ranges;
    status=validate(*owner,true,bound,ranges);if(status!=RNS8_SUCCESS)return status;
    owner->envelope=bound;owner->bound_frozen=true;owner->packed.resize(size_t(owner->layout.packed));owner->output.resize(size_t(owner->layout.output));
    if(owner->packed.capacity()>budget || owner->output.capacity()>budget-owner->packed.capacity())return RNS8_WORKSPACE_TOO_SMALL;
    pack(*owner);bind(*owner);
    ++ctx->prepared_references;++plan->prepared_references;++a->prepared_references;if(a!=b)++b->prepared_references;++c->prepared_references;
    cleanup.done=true;*out=owner.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_refresh_cpu_i8mm(rns8_prepared_cpu_i8mm* o) {
  return guard_api([&]() -> rns8_status {
    if(!o)return RNS8_INVALID_ARGUMENT;std::unique_lock<std::mutex> lock(o->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    cpp_int bound;rns8_exact_matrix_ranges ranges;const auto status=validate(*o,true,bound,ranges);if(status!=RNS8_SUCCESS)return status;
    if(o->b_instance!=o->b->matrix_instance_id || o->b_generation!=o->b->content_generation)pack(*o);
    bind(*o);return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_cpu_i8mm(rns8_prepared_cpu_i8mm* o) {
  return guard_api([&]() -> rns8_status {
    if(!o)return RNS8_INVALID_ARGUMENT;std::unique_lock<std::mutex> lock(o->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    rns8_cpu_i8mm_info support{};if(!admitted(support))return RNS8_UNSUPPORTED_BACKEND;
    cpp_int bound;rns8_exact_matrix_ranges ranges;auto status=validate(*o,false,bound,ranges);if(status!=RNS8_SUCCESS)return status;
    bool written=false;size_t token=0;status=begin_synchronous_lease(*o->ctx,*o->a,*o->b,*o->c,*o->workspace,token);if(status!=RNS8_SUCCESS)return status;
    struct Lease {rns8_prepared_cpu_i8mm& o;size_t token;bool& written;rns8_status operation=RNS8_INTERNAL_ERROR;
      ~Lease(){finish_synchronous_lease(*o.ctx,token,operation,written);}} lease{*o,token,written};
    const auto& d=o->plan->desc;uint32_t threads=1;
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
    if(!omp_in_parallel() && cpu_parallel_should_use(cpu_parallel_saturating_mul3(d.m,d.n,d.k)))
      threads=uint32_t(std::min<uint64_t>(o->layout.jobs,cpu_parallel_effective_threads()));
#endif
    o->last_threads=1;
#if RNS8_CPU_I8MM_COMPILED
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
#pragma omp parallel num_threads(threads) if(threads>1)
#endif
    {
      uint32_t worker=0,team=1;
#if defined(RNS8_CPU_PARALLEL_OPENMP) && RNS8_CPU_PARALLEL_OPENMP
      worker=uint32_t(omp_get_thread_num());team=uint32_t(omp_get_num_threads());
#pragma omp single
      o->last_threads=team;
#endif
      for(uint64_t job=worker;job<o->layout.jobs;job+=team) {
        const uint64_t col=job%o->layout.nt,row=(job/o->layout.nt)%o->layout.mt,p=job/(o->layout.nt*o->layout.mt);
        const auto modulus=uses_finite_storage(d.semantics)?d.finite_modulus:kDefaultModuli[p];
        rns8_cpu_i8mm_tile(o->a->residues.data()+p*d.m*d.k,o->packed.data()+(p*o->layout.nt+col)*o->layout.kt*16,
            o->output.data()+p*d.m*d.n,d.m,d.n,d.k,int64_t(row*2),int64_t(col*2),modulus);
      }
    }
#else
    (void)d;(void)threads;return RNS8_UNSUPPORTED_BACKEND;
#endif
    if(o->reuses!=UINT64_MAX)++o->reuses;
    written=true;invalidate_output_currentness(*o->c);std::copy(o->output.begin(),o->output.end(),o->c->residues.begin());
    mark_output_host_residues_current(*o->c,o->plan);
    if(uses_finite_storage(d.semantics))o->c->finite_modulus=uint16_t(d.finite_modulus);
    else commit_exact_range(*o->c,*o->plan,bound,&ranges);
    lease.operation=RNS8_SUCCESS;return RNS8_SUCCESS;
  });
}
rns8_status rns8_get_cpu_i8mm_info(const rns8_prepared_cpu_i8mm* o,rns8_cpu_i8mm_info* out) {
  if(!o || !out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out)))return RNS8_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lock(o->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
  const auto status=rns8_query_cpu_i8mm(out);if(status!=RNS8_SUCCESS)return status;
  out->packed_b_bytes=o->packed.capacity();out->output_staging_bytes=o->output.capacity();out->b_pack_count=o->packs;out->b_reuse_count=o->reuses;
  out->a_generation=o->a_generation;out->b_generation=o->b_generation;out->last_execution_threads=o->last_threads;return RNS8_SUCCESS;
}
rns8_status rns8_destroy_cpu_i8mm(rns8_prepared_cpu_i8mm* o) {
  if(!o)return RNS8_SUCCESS;std::unique_lock<std::mutex> lock(o->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
  const auto status=rns8_destroy_workspace(o->workspace);if(status!=RNS8_SUCCESS)return status;o->workspace=nullptr;
  --o->ctx->prepared_references;--o->plan->prepared_references;--o->a->prepared_references;if(o->a!=o->b)--o->b->prepared_references;--o->c->prepared_references;
  lock.unlock();delete o;return RNS8_SUCCESS;
}
