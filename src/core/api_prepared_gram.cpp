#include "core/gram_contract.hpp"
#include <mutex>
#include <cstring>
using namespace rns8::detail;
using namespace rns8::detail::api;
struct rns8_prepared_gram {
  rns8_context* ctx=nullptr;const rns8_matrix* x=nullptr;rns8_matrix* c=nullptr;
  GramProblem problem;cpp_int bound;std::vector<GramTile> jobs;
  std::vector<int8_t> host_adapter,export_residues;std::vector<uint64_t> host_export;
  void* device_jobs=nullptr;void* device_adapter=nullptr;void* device_export=nullptr;int* device_export_status=nullptr;
  uint64_t device_job_bytes=0,device_total_bytes=0,adapter_bytes=0,export_bytes=0,owned_bytes=0,output_generation=0;
  rns8_gram_input_policy input_policy=RNS8_GRAM_RESIDENT_RESIDUES;uint32_t export_limbs=0;
  int export_status=0;
  bool separate_mirror=false,wmma=false,jobs_ready=false,adapter_ready=false,active=false,commit_started=false,quarantined=false;
  mutable std::mutex mutex;
};
namespace {
void release_gram_reservation(rns8_prepared_gram& owner) noexcept {
  if(!owner.active)return;--owner.x->in_flight_readers;owner.c->in_flight_writer=false;owner.active=false;
}
void invalidate_gram_commit(rns8_prepared_gram& owner) noexcept {
  if(owner.active && owner.commit_started){clear_residue_current(*owner.c);clear_byte_limb_current(*owner.c);clear_native_current(*owner.c);owner.output_generation=0;}
}
struct GramReservation {
  rns8_prepared_gram& owner;bool completed=false;
  ~GramReservation() noexcept {
    if(completed)return;invalidate_gram_commit(owner);
    if(owner.x->backend==RNS8_BACKEND_HIP_DIRECT){try{if(hip_direct_synchronize(owner.ctx->device_id)!=RNS8_SUCCESS){owner.quarantined=true;owner.ctx->hip_failure_poisoned=true;return;}}
      catch(...){owner.quarantined=true;owner.ctx->hip_failure_poisoned=true;return;}}
    release_gram_reservation(owner);
  }
  void finish() noexcept{release_gram_reservation(owner);completed=true;}
};
bool gram_native_current(const rns8_matrix& x){
  if(!(x.desc.flags&RNS8_MATRIX_RETAIN_NATIVE_INPUT) || uses_finite_storage(x.desc.semantics))return false;
  return x.backend==RNS8_BACKEND_HIP_DIRECT ? x.device_native_current && x.hip_native_i64 && x.hip_native_i64_bytes==uint64_t(x.desc.rows)*x.desc.cols*8
      :x.host_native_current && x.native_i64.size()==uint64_t(x.desc.rows)*x.desc.cols;
}
rns8_status prepare_gram_impl(rns8_context* ctx,const rns8_matrix* x,rns8_matrix* c,rns8_gram_orientation orientation,
    rns8_gram_method method,rns8_gram_input_policy policy,uint32_t limbs,uint64_t budget,rns8_prepared_gram** out){
  if(!out || uint32_t(policy)>2 || limbs>32)return RNS8_INVALID_ARGUMENT;*out=nullptr;
  auto owner=std::make_unique<rns8_prepared_gram>();
  const auto status=validate_gram_contract(ctx,x,c,orientation,method,owner->problem,owner->bound);if(status!=RNS8_SUCCESS)return status;
  const bool hip=x->backend==RNS8_BACKEND_HIP_DIRECT,finite=uses_finite_storage(x->desc.semantics);
  if(policy==RNS8_GRAM_RETAINED_NATIVE_I64 && !gram_native_current(*x))return RNS8_INVALID_ARGUMENT;
  if(policy==RNS8_GRAM_RETAINED_NATIVE_I64 && hip && !qualified_i32_matrix_address(x->desc.rows,x->desc.cols,x->desc.cols,1,8))return RNS8_UNSUPPORTED_BACKEND;
  if(limbs && finite)return RNS8_UNSUPPORTED_BACKEND;
  if(limbs && owner->bound >= (cpp_int(1)<<(64*limbs-1)))return RNS8_RANGE_ERROR;
  const auto support=query_gram_support(owner->problem,budget,hip,bool(uint32_t(method)&1));if(support.status!=RNS8_SUCCESS)return support.status;
  if(limbs && (!valid_limb_export_access(support.order,support.order,support.order,limbs) ||
     (hip && !qualified_i32_matrix_address(support.order,support.order,support.order,1,uint64_t(limbs)*8))))return RNS8_RANGE_ERROR;
  const auto storage=query_gram_prepared_storage(owner->problem,hip,bool(uint32_t(method)&1),policy,limbs,budget);
  if(storage.status!=RNS8_SUCCESS)return storage.status;
  const uint64_t adapter=storage.adapter_bytes,exported=storage.export_bytes;
  if(ctx->prepared_references==UINT64_MAX || x->prepared_references==UINT64_MAX || c->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
  owner->ctx=ctx;owner->x=x;owner->c=c;owner->wmma=bool(uint32_t(method)&1);owner->separate_mirror=bool(uint32_t(method)&2);owner->input_policy=policy;owner->export_limbs=limbs;
  owner->jobs=gram_tiles(support);owner->adapter_bytes=adapter;owner->export_bytes=exported;
  if(!hip)owner->host_adapter.resize(static_cast<size_t>(adapter));
  owner->host_export.resize(static_cast<size_t>(exported/8));owner->export_residues.resize(static_cast<size_t>(storage.export_residue_bytes));
  const uint64_t host_capacity=owner->jobs.capacity()*sizeof(GramTile)+owner->host_adapter.capacity()+owner->host_export.capacity()*8+owner->export_residues.capacity();
  owner->device_job_bytes=storage.job_device_bytes;
  owner->device_total_bytes=storage.device_bytes;
  if(host_capacity>budget || owner->device_total_bytes>budget-host_capacity)return RNS8_WORKSPACE_TOO_SMALL;
  owner->owned_bytes=host_capacity+owner->device_total_bytes;
  if(hip){const auto allocated=hip_direct_allocate(ctx->device_id,static_cast<size_t>(owner->device_total_bytes),&owner->device_jobs);if(allocated!=RNS8_SUCCESS)return allocated;
    auto* base=static_cast<uint8_t*>(owner->device_jobs);owner->device_adapter=adapter?base+storage.adapter_offset:nullptr;
    // Align the limb staging after byte-oriented adapter padding.
    owner->device_export=exported?base+storage.export_offset:nullptr;
    owner->device_export_status=limbs?reinterpret_cast<int*>(base+storage.status_offset):nullptr;
  }
  ++ctx->prepared_references;++x->prepared_references;++c->prepared_references;*out=owner.release();return RNS8_SUCCESS;
}
rns8_status gram_ready(rns8_prepared_gram& owner){
  if(owner.quarantined || owner.ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
  const auto& x=*owner.x;const auto& c=*owner.c;
  if(x.in_flight_writer || c.in_flight_writer || c.in_flight_readers)return RNS8_BUSY;
  if(x.generation_exhausted || c.generation_exhausted || x.in_flight_readers==UINT64_MAX)return RNS8_RANGE_ERROR;
  if(x.matrix_instance_id!=owner.problem.source_instance || x.content_generation!=owner.problem.source_generation || !rns_residue_state_current_for_backend(x,x.backend) ||
      (owner.input_policy==RNS8_GRAM_RETAINED_NATIVE_I64 && !gram_native_current(x)))return RNS8_INVALID_ARGUMENT;
  return RNS8_SUCCESS;
}
}
rns8_status rns8_prepare_gram(rns8_context* ctx,const rns8_matrix* x,rns8_matrix* c,rns8_gram_orientation orientation,rns8_gram_method method,uint64_t budget,rns8_prepared_gram** out){
  return guard_api([&]{return prepare_gram_impl(ctx,x,c,orientation,method,RNS8_GRAM_RESIDENT_RESIDUES,0,budget,out);});
}
rns8_status rns8_prepare_gram_ex(rns8_context* ctx,const rns8_matrix* x,rns8_matrix* c,rns8_gram_orientation orientation,rns8_gram_method method,rns8_gram_input_policy policy,uint32_t limbs,uint64_t budget,rns8_prepared_gram** out){
  return guard_api([&]{return prepare_gram_impl(ctx,x,c,orientation,method,policy,limbs,budget,out);});
}
rns8_status rns8_refresh_prepared_gram(rns8_prepared_gram* owner){
  return guard_api([&]() -> rns8_status{
    if(!owner)return RNS8_INVALID_ARGUMENT;std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    if(owner->quarantined || owner->ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    GramProblem replacement;cpp_int bound;
    auto status=validate_gram_contract(owner->ctx,owner->x,owner->c,static_cast<rns8_gram_orientation>(owner->problem.orientation),static_cast<rns8_gram_method>(uint32_t(owner->wmma)+2*uint32_t(owner->separate_mirror)),replacement,bound);
    if(status!=RNS8_SUCCESS)return status;if(bound>owner->bound || replacement.modulus!=owner->problem.modulus)return RNS8_RANGE_ERROR;
    if(owner->input_policy==RNS8_GRAM_RETAINED_NATIVE_I64 && !gram_native_current(*owner->x))return RNS8_INVALID_ARGUMENT;
    owner->problem.source_instance=replacement.source_instance;owner->problem.source_generation=replacement.source_generation;
    owner->adapter_ready=false;owner->output_generation=0;return RNS8_SUCCESS;
  });
}
rns8_status rns8_execute_prepared_gram(rns8_prepared_gram* owner){
  return guard_api([&]() -> rns8_status{
    if(!owner)return RNS8_INVALID_ARGUMENT;std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    auto status=gram_ready(*owner);if(status!=RNS8_SUCCESS)return status;
    const auto* x=owner->x;auto* c=owner->c;const bool hip=x->backend==RNS8_BACKEND_HIP_DIRECT;cpp_int bound=owner->bound;
    ++x->in_flight_readers;c->in_flight_writer=true;owner->active=true;owner->commit_started=false;GramReservation reservation{*owner};
    GramProblem p=owner->problem;const void* source=hip?x->hip_residues:x->residues.data();
    if(owner->input_policy==RNS8_GRAM_RETAINED_TRANSPOSE){
      if(!owner->adapter_ready){if(hip){status=hip_direct_gram_transpose(owner->ctx->device_id,source,owner->device_adapter,p);if(status!=RNS8_SUCCESS)return status;}
        else for(uint32_t plane=0;plane<p.prefix;++plane)for(int64_t row=0;row<p.rows;++row)for(int64_t col=0;col<p.cols;++col)
          owner->host_adapter[(int64_t(plane)*p.cols+col)*p.rows+row]=x->residues[(int64_t(plane)*p.rows+row)*p.cols+col];
        owner->adapter_ready=true;}
      std::swap(p.rows,p.cols);p.source_ld=p.cols;p.orientation=p.orientation==GramOrientation::transpose_times_self?GramOrientation::self_times_transpose:GramOrientation::transpose_times_self;
      source=hip?owner->device_adapter:owner->host_adapter.data();
    }
    const bool native=owner->input_policy==RNS8_GRAM_RETAINED_NATIVE_I64;
    if(native)source=hip?x->hip_native_i64:x->native_i64.data();
    if(hip){status=hip_direct_gram_prepared_device(owner->ctx->device_id,source,c->hip_residues,p,owner->jobs.data(),owner->device_jobs,owner->device_job_bytes,owner->wmma,owner->jobs_ready,owner->commit_started,native,owner->separate_mirror);if(status!=RNS8_SUCCESS)return status;mark_output_device_residues_current(*c);}
    else {owner->commit_started=true;if(native){const int64_t n=c->desc.rows,k=p.orientation==GramOrientation::transpose_times_self?p.rows:p.cols;const auto* values=static_cast<const int64_t*>(source);
        for(auto tile:owner->jobs)for(unsigned r=0;r<16;++r)for(unsigned t=0;t<16;++t){const int64_t row=int64_t(tile.row)*16+r,col=int64_t(tile.col)*16+t;if(row>=n || col>=n || row>col)continue;
          cpp_int sum=0;for(int64_t q=0;q<k;++q){const auto at=[&](int64_t index){return p.orientation==GramOrientation::transpose_times_self?q*p.source_ld+index:index*p.source_ld+q;};sum+=cpp_int(values[at(row)])*values[at(col)];}
          for(uint32_t plane=0;plane<p.prefix;++plane){const int8_t value=centered_residue(sum,kDefaultModuli[plane]);c->residues[(int64_t(plane)*n+row)*n+col]=value;if(row!=col && !owner->separate_mirror)c->residues[(int64_t(plane)*n+col)*n+row]=value;}}}
      else gram_residues_cpu(static_cast<const int8_t*>(source),c->residues.data(),p,owner->jobs,kDefaultModuli,owner->separate_mirror);
      if(native && owner->separate_mirror)for(uint32_t plane=0;plane<p.prefix;++plane)for(int64_t row=0;row<c->desc.rows;++row)for(int64_t col=0;col<row;++col)
        c->residues[(int64_t(plane)*c->desc.rows+row)*c->desc.cols+col]=c->residues[(int64_t(plane)*c->desc.rows+col)*c->desc.cols+row];
      mark_output_host_residues_current(*c);}
    c->finite_modulus=owner->problem.modulus;c->exact_axis_ranges.rows.clear();c->exact_axis_ranges.cols.clear();
    if(!uses_finite_storage(c->desc.semantics)){c->materialized_rns_prefix=c->prefix;c->exact_max_magnitude.swap(bound);c->exact_range_prefix=c->prefix;}
    owner->output_generation=c->content_generation;reservation.finish();return RNS8_SUCCESS;
  });
}
rns8_status rns8_export_prepared_gram_limbs(rns8_prepared_gram* owner,uint64_t* dst,int64_t ld){
  return guard_api([&]() -> rns8_status{
    if(!owner || !dst || !owner->export_limbs)return RNS8_INVALID_ARGUMENT;std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    auto status=gram_ready(*owner);if(status!=RNS8_SUCCESS)return status;auto& c=*owner->c;const int64_t n=c.desc.rows;const uint32_t limbs=owner->export_limbs;
    if(!owner->output_generation || c.content_generation!=owner->output_generation || !rns_residue_state_current_for_backend(c,c.backend) || !valid_limb_export_access(n,n,ld,limbs))return RNS8_INVALID_ARGUMENT;
    const uint64_t destination_bytes=(uint64_t(n-1)*ld+n)*limbs*8;
    const void* source=c.backend==RNS8_BACKEND_HIP_DIRECT?c.hip_residues:c.residues.data();
    const auto disjoint=[&](const void* pointer,uint64_t bytes){return !bytes || nonoverlapping_byte_ranges(dst,destination_bytes,pointer,bytes);};
    if(!disjoint(source,uint64_t(n)*n*c.prefix) || !disjoint(owner->host_export.data(),owner->export_bytes) ||
       !disjoint(owner->host_adapter.data(),owner->host_adapter.size()) || !disjoint(owner->export_residues.data(),owner->export_residues.size()) ||
       !disjoint(owner->jobs.data(),owner->jobs.size()*sizeof(GramTile)) || !disjoint(owner->x->residues.data(),owner->x->residues.size()) ||
       !disjoint(owner->x->native_i64.data(),owner->x->native_i64.size()*8))return RNS8_INVALID_ARGUMENT;
    ++owner->x->in_flight_readers;c.in_flight_writer=true;owner->active=true;owner->commit_started=false;GramReservation reservation{*owner};
    if(c.backend==RNS8_BACKEND_HIP_DIRECT){status=hip_direct_gram_export_once(owner->ctx->device_id,source,owner->device_export,owner->device_export_status,owner->device_jobs,n,c.prefix,limbs,owner->jobs.size());
      if(status!=RNS8_SUCCESS)return status;
      status=hip_direct_copy_device_to_host(owner->ctx->device_id,&owner->export_status,owner->device_export_status,4);if(status!=RNS8_SUCCESS)return status;
      if(owner->export_status)return RNS8_RANGE_ERROR;
      status=hip_direct_copy_device_to_host(owner->ctx->device_id,owner->host_export.data(),owner->device_export,owner->export_bytes);if(status!=RNS8_SUCCESS)return status;
    }else for(auto tile:owner->jobs)for(unsigned r=0;r<16;++r)for(unsigned t=0;t<16;++t){const int64_t row=int64_t(tile.row)*16+r,col=int64_t(tile.col)*16+t;if(row>=n || col>=n || row>col)continue;
      for(uint32_t plane=0;plane<c.prefix;++plane)owner->export_residues[plane]=c.residues[(uint64_t(plane)*n+row)*n+col];auto* value=owner->host_export.data()+(row*n+col)*limbs;
      status=export_exact_wide_signed_limbs(owner->export_residues,c.prefix,value,limbs);if(status!=RNS8_SUCCESS)return status;
      if(row!=col)std::copy_n(value,limbs,owner->host_export.data()+(col*n+row)*limbs);}
    // Complete private staging precedes every user-visible store and padding stays untouched.
    for(int64_t row=0;row<n;++row)std::copy_n(owner->host_export.data()+row*n*limbs,size_t(n)*limbs,dst+row*ld*limbs);
    reservation.finish();return RNS8_SUCCESS;
  });
}
rns8_status rns8_get_prepared_gram_info(const rns8_prepared_gram* owner,rns8_prepared_gram_info* info){
  return guard_api([&]() -> rns8_status{
    if(!owner || !info || !valid_abi(info->struct_size,info->abi_version,sizeof(*info)))return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    *info={};info->struct_size=sizeof(*info);info->abi_version=RNS8_ABI_VERSION;info->input_policy=owner->input_policy;info->method=uint32_t(owner->wmma)+2*uint32_t(owner->separate_mirror);info->export_limbs=owner->export_limbs;
    info->source_instance=owner->problem.source_instance;info->source_generation=owner->problem.source_generation;info->output_generation=owner->output_generation;
    info->host_job_bytes=owner->jobs.capacity()*sizeof(GramTile);info->device_job_bytes=owner->device_job_bytes;info->orientation_adapter_bytes=owner->adapter_bytes;
    info->host_export_bytes=owner->host_export.capacity()*8;info->device_export_bytes=owner->device_export?owner->export_bytes+4:0;info->owned_payload_bytes=owner->owned_bytes;
    const uint64_t n=owner->c->desc.rows;info->dense_cells=n*n;info->independent_cells=n*(n+1)/2;return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_prepared_gram(rns8_prepared_gram* owner){
  return guard_api([&]() -> rns8_status{
    if(!owner)return RNS8_SUCCESS;std::unique_lock<std::mutex> lock(owner->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    if(owner->quarantined || owner->active){if(owner->x->backend==RNS8_BACKEND_HIP_DIRECT && hip_direct_synchronize(owner->ctx->device_id)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
      invalidate_gram_commit(*owner);release_gram_reservation(*owner);owner->quarantined=false;}
    if(owner->device_jobs){const auto status=hip_direct_free(owner->ctx->device_id,owner->device_jobs);if(status!=RNS8_SUCCESS)return status;owner->device_jobs=nullptr;}
    --owner->ctx->prepared_references;--owner->x->prepared_references;--owner->c->prepared_references;lock.unlock();delete owner;return RNS8_SUCCESS;
  });
}
