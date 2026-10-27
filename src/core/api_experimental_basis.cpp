#include "core/api_internal.hpp"
#include "core/experimental_basis.hpp"
#include "backend_common/ordered_basis_identity.hpp"
#include "backend_common/sha256.hpp"
#include "rns8/experimental_basis.h"
#include "backend_common/basis_word_arithmetic.hpp"
#include "core/prepared_transfer.hpp"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <numeric>
using namespace rns8::detail;
using namespace rns8::detail::api;
using boost::multiprecision::cpp_int;

struct rns8_basis {std::shared_ptr<const ExperimentalBasis> data;};
struct rns8_basis_plan {
  rns8_context* ctx=nullptr;
  rns8_basis_plan_desc desc{};
  std::shared_ptr<const ExperimentalBasis> basis,source_basis;
  std::mutex mutex;
  rns8_workspace lease_work;
  hip_direct_device_buffer device;
  std::vector<uint64_t> host_words;
  std::vector<int8_t> host_residues;
  uint64_t max_cells=0,packet_target=0,packet_source=0,native_upload=0,scratch=0;
  bool target_uploaded=false,retiring=false;
  int device_status=0;
};
namespace {
void basis_words(cpp_int value,uint64_t* out) {
  for(unsigned i=0;i<3;++i){out[i]=static_cast<uint64_t>(value&UINT64_MAX);value>>=64;}
}
std::shared_ptr<const ExperimentalBasis> make_experimental_basis(const uint16_t* moduli,uint32_t count) {
  auto data=std::make_shared<ExperimentalBasis>();data->packet.count=count;data->products[0]=1;
  for(uint32_t i=0;i<count;++i) {
    const auto modulus=moduli[i];data->packet.moduli[i]=modulus;
    const auto remainder=static_cast<uint32_t>(data->products[i]%modulus);
    uint32_t inverse=1;while((remainder*inverse)%modulus!=1)++inverse;
    data->packet.inverse_prefix[i]=uint16_t(inverse);
    data->products[i+1]=data->products[i]*modulus;
  }
  data->packet.identity=ordered_basis_fingerprint(data->packet.moduli,count);
  for(uint32_t i=0;i<=count;++i) {
    basis_words(data->products[i],data->packet.products[i]);
    basis_words((data->products[i]+1)/2,data->packet.negative_thresholds[i]);
  }
  return data;
}
std::shared_ptr<const ExperimentalBasis> basis_data(const rns8_basis* basis) {
  return basis?basis->data:default_experimental_basis();
}
std::shared_ptr<const ExperimentalBasis> matrix_basis(const rns8_matrix& matrix) {
  return matrix.experimental_basis?matrix.experimental_basis:default_experimental_basis();
}
bool equal_basis(const ExperimentalBasis& a,const ExperimentalBasis& b) {
  return a.packet.version==b.packet.version && a.packet.count==b.packet.count &&
      std::equal(a.packet.moduli,a.packet.moduli+a.packet.count,b.packet.moduli);
}
struct BasisStorage {
  uint64_t cells=0,host=0,device=0,target=0,source=0,native=0,scratch=0;
};
rns8_status basis_plan_storage(const ExperimentalBasis& basis,const rns8_basis_plan_desc& d,BasisStorage& out) {
  if(!valid_abi(d.struct_size,d.abi_version,sizeof(d)) || d.flags || !d.prefix || d.prefix>basis.packet.count ||
      d.m<=0 || d.n<=0 || d.k<=0)return RNS8_INVALID_ARGUMENT;
  if(!exact_wide_semantics(d.semantics) || (d.backend!=RNS8_BACKEND_CPU_REFERENCE && d.backend!=RNS8_BACKEND_HIP_DIRECT))return RNS8_UNSUPPORTED_BACKEND;
  for(const auto shape:{std::pair<int64_t,int64_t>{d.m,d.k},{d.k,d.n},{d.m,d.n}}) {
    AddressDomain domain;
    if(!checked_host_plane_address(shape.first,shape.second,shape.second,1,24,domain))return RNS8_RANGE_ERROR;
    out.cells=std::max(out.cells,domain.byte_extent/24);
  }
  const auto limit=uint64_t(PTRDIFF_MAX),count=uint64_t(basis.packet.count);
  if(out.cells>(limit-4)/(24+count))return RNS8_RANGE_ERROR;
  out.host=out.cells*24;
  if(d.backend==RNS8_BACKEND_CPU_REFERENCE)out.host+=out.cells*count;
  else {
    if(!qualified_i32_gemm_address(d.m,d.n,d.k,d.k,d.n,d.n,basis.packet.count))return RNS8_UNSUPPORTED_BACKEND;
    const auto add=[&](uint64_t bytes,uint64_t& offset) {
      if(out.device>limit-255 || bytes>limit-255)return false;
      offset=(out.device+255)&~UINT64_C(255);const auto padded=(bytes+255)&~UINT64_C(255);
      if(padded>limit-offset)return false;out.device=offset+padded;return true;
    };
    if(!add(sizeof(BasisPacket),out.target) || !add(sizeof(BasisPacket),out.source) ||
        !add(out.cells*8,out.native) || !add(std::max(out.cells*count,out.cells*24+4),out.scratch))return RNS8_RANGE_ERROR;
  }
  return out.device>d.max_device_bytes || out.host>d.max_host_bytes?RNS8_WORKSPACE_TOO_SMALL:RNS8_SUCCESS;
}
rns8_status describe_basis(const ExperimentalBasis& basis,uint32_t prefix,rns8_basis_info* out) {
  if(!out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out)) || !prefix || prefix>basis.packet.count)return RNS8_INVALID_ARGUMENT;
  rns8_basis_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;info.representation=1;
  info.identity=basis.packet.identity;info.count=basis.packet.count;info.selected_prefix=prefix;
  std::copy_n(basis.packet.moduli,20,info.moduli);std::copy_n(basis.packet.products[prefix],3,info.prefix_product);
  *out=info;return RNS8_SUCCESS;
}
}
namespace rns8::detail {
std::shared_ptr<const ExperimentalBasis> default_experimental_basis() {
  static const auto data=make_experimental_basis(kDefaultModuli,20);return data;
}
}
rns8_status rns8_create_basis(const rns8_basis_desc* desc,rns8_basis** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!desc || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) || desc->representation!=1 || desc->flags ||
        !desc->moduli || !desc->count || desc->count>20)return RNS8_INVALID_ARGUMENT;
    for(uint32_t i=0;i<desc->count;++i) {
      if(desc->moduli[i]<2 || desc->moduli[i]>256)return RNS8_INVALID_ARGUMENT;
      for(uint32_t j=0;j<i;++j)if(std::gcd(desc->moduli[i],desc->moduli[j])!=1)return RNS8_INVALID_ARGUMENT;
    }
    auto owner=std::make_unique<rns8_basis>();owner->data=make_experimental_basis(desc->moduli,desc->count);
    *out=owner.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_basis(rns8_basis* basis) {delete basis;return RNS8_SUCCESS;}
rns8_status rns8_get_basis_info(const rns8_basis* basis,uint32_t prefix,rns8_basis_info* out) {
  return guard_api([&](){return describe_basis(*basis_data(basis),prefix,out);});
}
rns8_status rns8_get_matrix_basis(const rns8_matrix* matrix,rns8_basis_info* out) {
  return guard_api([&](){return matrix?describe_basis(*matrix_basis(*matrix),matrix->prefix,out):RNS8_INVALID_ARGUMENT;});
}
rns8_status rns8_create_basis_matrix(rns8_context* ctx,const rns8_basis* basis,const rns8_matrix_desc* desc,rns8_matrix** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!ctx || !desc || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)))return RNS8_INVALID_ARGUMENT;
    if(!exact_wide_semantics(desc->semantics) || (ctx->backend!=RNS8_BACKEND_CPU_REFERENCE && ctx->backend!=RNS8_BACKEND_HIP_DIRECT))return RNS8_UNSUPPORTED_BACKEND;
    if(desc->flags || desc->bound_kind!=RNS8_BOUND_NONE)return RNS8_INVALID_ARGUMENT;
    auto data=basis_data(basis);auto normalized=*desc;
    if(!normalized.max_prefix)normalized.max_prefix=data->packet.count;
    if(normalized.max_prefix>data->packet.count)return RNS8_INVALID_ARGUMENT;
    const auto status=rns8_create_matrix(ctx,&normalized,out);if(status!=RNS8_SUCCESS)return status;
    if(basis)(*out)->experimental_basis=std::move(data);
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_query_basis_plan(const rns8_basis* basis,const rns8_basis_plan_desc* desc,rns8_basis_resources* out) {
  return guard_api([&]() -> rns8_status {
    if(!desc || !out || !valid_abi(out->struct_size,out->abi_version,sizeof(*out)))return RNS8_INVALID_ARGUMENT;
    const auto data=basis_data(basis);BasisStorage storage;
    const auto status=basis_plan_storage(*data,*desc,storage);
    if(status!=RNS8_SUCCESS && status!=RNS8_WORKSPACE_TOO_SMALL)return status;
    rns8_basis_resources info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    info.required_device_bytes=storage.device;info.required_host_bytes=storage.host;
    info.ordered_basis_identity=data->packet.identity;info.prefix=desc->prefix;info.representation=1;
    *out=info;return status;
  });
}
rns8_status rns8_query_basis_operation(const rns8_context* ctx,const rns8_basis* basis,
    const rns8_basis_plan_desc* desc,rns8_operation_kind operation,uint32_t modes,
    rns8_operation_support* support,rns8_basis_info* info) {
  return guard_api([&]()->rns8_status {
    if(!ctx || !desc || !support || !info || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) ||
        !valid_abi(support->struct_size,support->abi_version,sizeof(*support)) ||
        !valid_abi(info->struct_size,info->abi_version,sizeof(*info)))return RNS8_INVALID_ARGUMENT;
    const auto data=basis_data(basis);const auto described=describe_basis(*data,desc->prefix,info);
    if(described!=RNS8_SUCCESS)return described;
    rns8_operation_support out{};out.struct_size=sizeof(out);out.abi_version=RNS8_ABI_VERSION;
    out.backend=desc->backend;out.operation=operation;out.requires_binding=1;
    out.supported_execution_modes=RNS8_EXEC_SYNC;
    if(operation==RNS8_OP_GEMM && desc->backend==RNS8_BACKEND_HIP_DIRECT)
      out.supported_execution_modes|=RNS8_EXEC_BORROWED_STREAM|RNS8_EXEC_OWNED_GRAPH;
    out.safe_k=65536;
    out.alignment=desc->backend==RNS8_BACKEND_CPU_REFERENCE?alignof(uint64_t):256;
    const auto text=[](char* dst,size_t bytes,const char* src){std::strncpy(dst,src,bytes-1);};
    text(out.target_id,sizeof(out.target_id),desc->backend==RNS8_BACKEND_CPU_REFERENCE?"cpu":ctx->device_info.gcn_arch);
    const auto finish=[&](rns8_status status,uint32_t reason,const char* field,const char* explanation) {
      out.status=status;out.reason_code=reason;text(out.offending_field,sizeof(out.offending_field),field);
      text(out.explanation,sizeof(out.explanation),explanation);*support=out;return status;
    };
    if(operation>=RNS8_OP_COUNT || !modes || (modes&~127u))
      return finish(RNS8_INVALID_ARGUMENT,RNS8_SUPPORT_BAD_DESCRIPTOR,"operation/modes","invalid operation or execution mode");
    if(desc->backend!=RNS8_BACKEND_CPU_REFERENCE && desc->backend!=RNS8_BACKEND_HIP_DIRECT)
      return finish(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"backend","explicit ordered basis has only CPU/direct-HIP owners");
    out.source_implemented=1;out.compiled_inventory=desc->backend==RNS8_BACKEND_CPU_REFERENCE || hip_direct_compiled();
    if(!out.compiled_inventory)return finish(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_NOT_COMPILED,"backend","direct-HIP source is not compiled in this build");
    if(ctx->backend!=desc->backend)return finish(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_TARGET,"context","owner backend must match its context");
    out.runtime_observed=desc->backend==RNS8_BACKEND_CPU_REFERENCE || ctx->device_info.hip_available;
    if(operation!=RNS8_OP_GEMM && operation!=RNS8_OP_PACK && operation!=RNS8_OP_EXPORT &&
        operation!=RNS8_OP_BASE_EXTEND && operation!=RNS8_OP_EXPERIMENTAL_BASIS)
      return finish(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_OPERATION,"operation","operation has no explicit ordered-basis adapter");
    if(!exact_wide_semantics(desc->semantics))return finish(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_SEMANTICS,
        "semantics","explicit basis owner requires signed/unsigned exact-wide arithmetic");
    if(modes&~out.supported_execution_modes)return finish(RNS8_UNSUPPORTED_BACKEND,RNS8_SUPPORT_EXECUTION_MODE,
        "execution_modes","this operation has no explicit ordered-basis execution adapter");
    if(ctx->hip_failure_poisoned)return finish(RNS8_BACKEND_FAILURE,RNS8_SUPPORT_RUNTIME,"context","context has a poisoned failure lease");
    BasisStorage storage;const auto status=basis_plan_storage(*data,*desc,storage);
    out.required_device_bytes=storage.device;out.required_host_bytes=storage.host;
    if(status!=RNS8_SUCCESS && status!=RNS8_WORKSPACE_TOO_SMALL) {
      out.resource_status=status==RNS8_RANGE_ERROR?2:3;
      return finish(status,status==RNS8_RANGE_ERROR?RNS8_SUPPORT_RESOURCE_OVERFLOW:RNS8_SUPPORT_BAD_DESCRIPTOR,
          "problem","basis plan rejected its shape, semantics, flags, prefix or address domain");
    }
    char build[65]{};const auto build_status=rns8_get_pipeline_build_identity(build,sizeof(build));if(build_status!=RNS8_SUCCESS)return build_status;
    std::string identity="basis-runtime-provider-v1;build="+std::string(build)+";backend="+std::to_string(desc->backend)+
        ";operation="+std::to_string(operation)+";modes="+std::to_string(modes)+";semantics="+std::to_string(desc->semantics)+
        ";shape="+std::to_string(desc->m)+","+std::to_string(desc->n)+","+std::to_string(desc->k)+
        ";prefix="+std::to_string(desc->prefix)+";representation="+std::to_string(data->packet.version)+";moduli=";
    for(unsigned i=0;i<data->packet.count;++i)identity+=std::to_string(data->packet.moduli[i])+",";
    identity+=";target="+std::string(out.target_id)+";devicebudget="+std::to_string(desc->max_device_bytes)+
        ";hostbudget="+std::to_string(desc->max_host_bytes);
    const auto digest=sha256(identity);text(out.provider_identity_sha256,sizeof(out.provider_identity_sha256),digest.c_str());
    if(status==RNS8_WORKSPACE_TOO_SMALL)return finish(status,RNS8_SUPPORT_BUDGET,"budgets","plan payload exceeds its explicit device/host budgets");
    return finish(RNS8_SUCCESS,RNS8_SUPPORT_COMPATIBLE,"","explicit ordered CPU/direct-HIP owner; complete current basis/range/generation binding still required");
  });
}
rns8_status rns8_create_basis_plan(rns8_context* ctx,const rns8_basis* basis,const rns8_basis_plan_desc* desc,rns8_basis_plan** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!ctx || !desc || ctx->backend!=desc->backend)return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    if(ctx->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
    auto data=basis_data(basis);BasisStorage storage;
    const auto status=basis_plan_storage(*data,*desc,storage);if(status!=RNS8_SUCCESS)return status;
    auto plan=std::make_unique<rns8_basis_plan>();plan->ctx=ctx;plan->desc=*desc;plan->basis=std::move(data);
    plan->max_cells=storage.cells;plan->packet_target=storage.target;plan->packet_source=storage.source;
    plan->native_upload=storage.native;plan->scratch=storage.scratch;
    plan->host_words.resize(size_t(storage.cells*3));
    if(desc->backend==RNS8_BACKEND_CPU_REFERENCE)plan->host_residues.resize(size_t(storage.cells*plan->basis->packet.count));
    if(plan->host_words.capacity()>desc->max_host_bytes/8 ||
        plan->host_residues.capacity()>desc->max_host_bytes-uint64_t(plan->host_words.capacity())*8)return RNS8_WORKSPACE_TOO_SMALL;
    if(storage.device) {
      if(!direct_hip_compiled())return RNS8_UNSUPPORTED_BACKEND;
      const auto allocated=plan->device.allocate(ctx->device_id,size_t(storage.device));if(allocated!=RNS8_SUCCESS)return allocated;
    }
    ++ctx->prepared_references;*out=plan.release();return RNS8_SUCCESS;
  });
}
rns8_status rns8_destroy_basis_plan(rns8_basis_plan* plan) {
  if(!plan)return RNS8_SUCCESS;
  std::unique_lock<std::mutex> lock(plan->mutex,std::try_to_lock);if(!lock.owns_lock() || plan->lease_work.in_flight)return RNS8_BUSY;
  plan->retiring=true;plan->device.mark_completed();
  const auto status=plan->device.reset();if(status!=RNS8_SUCCESS)return status;
  --plan->ctx->prepared_references;lock.unlock();delete plan;return RNS8_SUCCESS;
}
rns8_status rns8_get_basis_plan_info(const rns8_basis_plan* plan,rns8_basis_info* basis,rns8_basis_resources* resources) {
  return guard_api([&]() -> rns8_status {
    if(!plan || !basis || !resources || !valid_abi(basis->struct_size,basis->abi_version,sizeof(*basis)) ||
        !valid_abi(resources->struct_size,resources->abi_version,sizeof(*resources)))return RNS8_INVALID_ARGUMENT;
    BasisStorage storage;const auto status=basis_plan_storage(*plan->basis,plan->desc,storage);if(status!=RNS8_SUCCESS)return status;
    const auto described=describe_basis(*plan->basis,plan->desc.prefix,basis);if(described!=RNS8_SUCCESS)return described;
    rns8_basis_resources info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    info.required_device_bytes=storage.device;info.required_host_bytes=storage.host;info.ordered_basis_identity=plan->basis->packet.identity;
    info.prefix=plan->desc.prefix;info.representation=1;*resources=info;return RNS8_SUCCESS;
  });
}
namespace {
rns8_status basis_plan_ready(const rns8_basis_plan& plan) {
  if(plan.retiring || plan.lease_work.in_flight)return RNS8_BUSY;
  return plan.ctx->hip_failure_poisoned?RNS8_BACKEND_FAILURE:RNS8_SUCCESS;
}
rns8_status basis_matrix_contract(const rns8_basis_plan& plan,const rns8_matrix& matrix,bool current,bool same_basis=true) {
  if(matrix.in_flight_writer || (!current && matrix.in_flight_readers))return RNS8_BUSY;
  if(matrix.generation_exhausted)return RNS8_RANGE_ERROR;
  const auto data=matrix_basis(matrix);AddressDomain bytes;
  if(matrix.backend!=plan.ctx->backend || matrix.desc.semantics!=plan.desc.semantics || matrix.desc.bound_kind!=RNS8_BOUND_NONE ||
      matrix.desc.logical_layout!=RNS8_LAYOUT_ROW_MAJOR || !matrix.prefix || matrix.prefix>data->packet.count ||
      matrix.desc.max_prefix!=matrix.prefix || matrix.materialized_rns_schedule ||
      (same_basis && !equal_basis(*plan.basis,*data)) ||
      !checked_host_plane_address(matrix.desc.rows,matrix.desc.cols,matrix.desc.cols,matrix.prefix,1,bytes) ||
      bytes.byte_extent/matrix.prefix>plan.max_cells)return RNS8_INVALID_ARGUMENT;
  if(plan.ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
    if(matrix.hip_device_id!=plan.ctx->device_id || !matrix.hip_residues || matrix.hip_residue_bytes!=bytes.byte_extent ||
        (current && !matrix.device_residues_current))return RNS8_INVALID_ARGUMENT;
  } else if(matrix.residues.size()!=bytes.byte_extent || matrix.hip_residues || (current && !matrix.host_residues_current))return RNS8_INVALID_ARGUMENT;
  if(current && (!matrix.exact_range_prefix || matrix.exact_range_prefix>matrix.prefix ||
      !exact_range_fits(matrix.desc.semantics,matrix.exact_max_magnitude,data->products[matrix.exact_range_prefix])))return RNS8_RANGE_ERROR;
  return RNS8_SUCCESS;
}
void* basis_device(rns8_basis_plan& plan,uint64_t offset) {return static_cast<uint8_t*>(plan.device.get())+offset;}
void* basis_device_flag(rns8_basis_plan& plan) {return basis_device(plan,plan.scratch+plan.max_cells*24);}
rns8_status upload_basis_packet(rns8_basis_plan& plan,const std::shared_ptr<const ExperimentalBasis>& source={}) {
  if(!plan.target_uploaded) {
    const auto status=hip_direct_copy_host_to_device(plan.ctx->device_id,basis_device(plan,plan.packet_target),&plan.basis->packet,sizeof(BasisPacket));
    if(status!=RNS8_SUCCESS)return status;plan.target_uploaded=true;
  }
  if(source && (!plan.source_basis || !equal_basis(*source,*plan.source_basis))) {
    const auto status=hip_direct_copy_host_to_device(plan.ctx->device_id,basis_device(plan,plan.packet_source),&source->packet,sizeof(BasisPacket));
    if(status!=RNS8_SUCCESS){plan.source_basis.reset();return status;}plan.source_basis=source;
  }
  return RNS8_SUCCESS;
}
template<class Operation>
rns8_status basis_transaction(rns8_basis_plan& plan,const rns8_matrix* a,const rns8_matrix* b,rns8_matrix* c,
    bool unary_write,Operation&& operation) {
  size_t token=0;auto status=a && b && c?begin_synchronous_lease(*plan.ctx,*a,*b,*c,plan.lease_work,token):
      begin_synchronous_matrix_lease(*plan.ctx,c?*c:*const_cast<rns8_matrix*>(a),plan.lease_work,token,unary_write);
  if(status!=RNS8_SUCCESS)return status;
  bool written=false;plan.device.mark_pending();
  struct Finish {
    rns8_basis_plan& plan;size_t token;bool& written;rns8_status status=RNS8_INTERNAL_ERROR;
    ~Finish(){if(finish_synchronous_lease(*plan.ctx,token,status,written))plan.device.mark_completed();}
  } finish{plan,token,written};
  status=operation(written);finish.status=status;
  if(status!=RNS8_SUCCESS){plan.target_uploaded=false;plan.source_basis.reset();}
  return status;
}
rns8_status commit_basis_stage(rns8_basis_plan& plan,rns8_matrix& matrix,uint32_t prefix,bool& written) {
  written=true;clear_residue_current(matrix);clear_native_current(matrix);clear_byte_limb_current(matrix);
  const auto bytes=uint64_t(matrix.desc.rows)*matrix.desc.cols*prefix;
  if(plan.ctx->backend==RNS8_BACKEND_HIP_DIRECT)
    return hip_direct_copy_device_to_device(plan.ctx->device_id,matrix.hip_residues,basis_device(plan,plan.scratch),size_t(bytes));
  std::copy_n(plan.host_residues.data(),size_t(bytes),matrix.residues.data());return RNS8_SUCCESS;
}
void publish_basis_value(rns8_basis_plan& plan,rns8_matrix& matrix,uint32_t prefix,cpp_int& bound,bool imported=false,uint64_t version=0) {
  if(plan.ctx->backend==RNS8_BACKEND_HIP_DIRECT)mark_device_residues_current(matrix,prefix);
  else mark_host_residues_current(matrix,prefix);
  clear_native_current(matrix);matrix.exact_max_magnitude.swap(bound);matrix.exact_range_prefix=prefix;
  matrix.exact_axis_ranges.rows.clear();matrix.exact_axis_ranges.cols.clear();publish_matrix_value(matrix,imported,version);
}
rns8_status reconstruct_basis_cpu(const rns8_matrix& source,const ExperimentalBasis& basis,uint64_t cell,cpp_int& value) {
  const uint64_t cells=uint64_t(source.desc.rows)*source.desc.cols;value=0;
  for(uint32_t p=0;p<source.exact_range_prefix;++p) {
    const auto modulus=basis.packet.moduli[p];const int32_t residue=source.residues[size_t(uint64_t(p)*cells+cell)];
    if(residue < -int32_t(modulus/2) || residue>int32_t((modulus-1)/2))return RNS8_INVALID_ARGUMENT;
    const auto canonical=uint32_t(residue<0?residue+modulus:residue);
    const auto remainder=static_cast<uint32_t>(value%modulus);
    const auto digit=((canonical+modulus-remainder)*basis.packet.inverse_prefix[p])%modulus;
    value+=basis.products[p]*digit;
  }
  const auto& product=basis.products[source.exact_range_prefix];
  if(source.desc.semantics==RNS8_EXACT_WIDE_SIGNED && value>=(product+1)/2)value-=product;
  return boost::multiprecision::abs(value)>source.exact_max_magnitude?RNS8_RANGE_ERROR:RNS8_SUCCESS;
}
template<class T>
rns8_status pack_basis(rns8_basis_plan* plan,rns8_matrix* matrix,const T* source,int64_t ld,uint64_t version) {
  return guard_api([&]() -> rns8_status {
    if(!plan || !matrix || !source || reinterpret_cast<uintptr_t>(source)%alignof(T))return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(plan->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    auto status=basis_plan_ready(*plan);if(status!=RNS8_SUCCESS)return status;
    status=basis_matrix_contract(*plan,*matrix,false);if(status!=RNS8_SUCCESS)return status;
    constexpr bool sign=std::is_same_v<T,int64_t>;
    if((matrix->desc.semantics==RNS8_EXACT_WIDE_SIGNED)!=sign)return RNS8_INVALID_ARGUMENT;
    AddressDomain span;
    if(!checked_host_plane_address(matrix->desc.rows,matrix->desc.cols,ld,1,sizeof(T),span))return RNS8_RANGE_ERROR;
    if(version && matrix->imported_source && matrix->source_version==version &&
        matrix->exact_range_prefix==matrix->prefix && (plan->ctx->backend==RNS8_BACKEND_HIP_DIRECT?matrix->device_residues_current:matrix->host_residues_current))return RNS8_SUCCESS;
    uint64_t maximum=0;const auto cells=uint64_t(matrix->desc.rows)*matrix->desc.cols;
    for(int64_t row=0;row<matrix->desc.rows;++row)for(int64_t col=0;col<matrix->desc.cols;++col) {
      const auto value=source[row*ld+col];const auto word=uint64_t(value);plan->host_words[size_t(uint64_t(row)*matrix->desc.cols+col)]=word;
      const auto magnitude=sign && (word>>63)?uint64_t(0)-word:word;maximum=std::max(maximum,magnitude);
    }
    cpp_int bound=maximum;if(!exact_range_fits(matrix->desc.semantics,bound,plan->basis->products[matrix->prefix]))return RNS8_RANGE_ERROR;
    status=basis_transaction(*plan,nullptr,nullptr,matrix,true,[&](bool& written) -> rns8_status {
      if(plan->ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
        auto result=upload_basis_packet(*plan);if(result!=RNS8_SUCCESS)return result;
        result=hip_direct_copy_host_to_device(plan->ctx->device_id,basis_device(*plan,plan->native_upload),plan->host_words.data(),size_t(cells*8));if(result!=RNS8_SUCCESS)return result;
        result=hip_direct_basis_pack(plan->ctx->device_id,basis_device(*plan,plan->native_upload),basis_device(*plan,plan->scratch),cells,
            basis_device(*plan,plan->packet_target),matrix->prefix,sign);if(result!=RNS8_SUCCESS)return result;
      } else for(uint32_t p=0;p<matrix->prefix;++p)for(uint64_t cell=0;cell<cells;++cell) {
        const auto word=plan->host_words[size_t(cell)];const bool negative=sign && word>>63;
        const auto modulus=uint32_t(plan->basis->packet.moduli[p]);auto residue=uint32_t((negative?uint64_t(0)-word:word)%modulus);
        if(negative && residue)residue=modulus-residue;
        plan->host_residues[size_t(uint64_t(p)*cells+cell)]=basis_center(residue,modulus);
      }
      return commit_basis_stage(*plan,*matrix,matrix->prefix,written);
    });
    if(status==RNS8_SUCCESS)publish_basis_value(*plan,*matrix,matrix->prefix,bound,true,version);
    return status;
  });
}
}
rns8_status rns8_pack_basis_i64(rns8_basis_plan* plan,rns8_matrix* matrix,const int64_t* source,int64_t ld,uint64_t version) {return pack_basis(plan,matrix,source,ld,version);}
rns8_status rns8_pack_basis_u64(rns8_basis_plan* plan,rns8_matrix* matrix,const uint64_t* source,int64_t ld,uint64_t version) {return pack_basis(plan,matrix,source,ld,version);}
rns8_status rns8_gemm_basis(rns8_basis_plan* plan,const rns8_matrix* a,const rns8_matrix* b,rns8_matrix* c) {
  return guard_api([&]() -> rns8_status {
    if(!plan || !a || !b || !c || a==c || b==c)return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(plan->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    auto status=basis_plan_ready(*plan);if(status!=RNS8_SUCCESS)return status;
    for(const auto* input:{a,b}){status=basis_matrix_contract(*plan,*input,true);if(status!=RNS8_SUCCESS)return status;}
    status=basis_matrix_contract(*plan,*c,false);if(status!=RNS8_SUCCESS)return status;
    const auto& d=plan->desc;
    if(a->desc.rows!=d.m || a->desc.cols!=d.k || b->desc.rows!=d.k || b->desc.cols!=d.n || c->desc.rows!=d.m || c->desc.cols!=d.n ||
        a->exact_range_prefix<d.prefix || b->exact_range_prefix<d.prefix || c->prefix<d.prefix)return RNS8_INVALID_ARGUMENT;
    cpp_int bound=cpp_int(d.k)*a->exact_max_magnitude*b->exact_max_magnitude;
    if(!exact_range_fits(d.semantics,bound,plan->basis->products[d.prefix]))return RNS8_RANGE_ERROR;
    status=basis_transaction(*plan,a,b,c,false,[&](bool& written) -> rns8_status {
      if(d.backend==RNS8_BACKEND_HIP_DIRECT) {
        const auto result=hip_direct_basis_gemm(plan->ctx->device_id,a->hip_residues,b->hip_residues,basis_device(*plan,plan->scratch),
            d.m,d.n,d.k,plan->basis->packet.moduli,d.prefix);if(result!=RNS8_SUCCESS)return result;
      } else for(uint32_t p=0;p<d.prefix;++p)for(int64_t row=0;row<d.m;++row)for(int64_t col=0;col<d.n;++col) {
        const auto modulus=int32_t(plan->basis->packet.moduli[p]);int32_t residue=0;
        for(int64_t begin=0;begin<d.k;) {
          const auto end=begin+std::min<int64_t>(65536,d.k-begin);int32_t sum=0;
          for(auto q=begin;q<end;++q)sum+=int32_t(a->residues[size_t((uint64_t(p)*d.m+row)*d.k+q)])*
              int32_t(b->residues[size_t((uint64_t(p)*d.k+q)*d.n+col)]);
          residue=int32_t((int64_t(residue)+sum)%modulus);begin=end;
        }
        if(residue<0)residue+=modulus;
        plan->host_residues[size_t((uint64_t(p)*d.m+row)*d.n+col)]=basis_center(uint32_t(residue),uint32_t(modulus));
      }
      return commit_basis_stage(*plan,*c,d.prefix,written);
    });
    if(status==RNS8_SUCCESS)publish_basis_value(*plan,*c,d.prefix,bound);
    return status;
  });
}
namespace {
class PreparedBasisGemm final:public PreparedTransfer {
 public:
  const rns8_matrix *a=nullptr,*b=nullptr;
  rns8_basis_plan_desc desc{};
  std::shared_ptr<const ExperimentalBasis> basis;
  uint64_t published_generation=0;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  hipStream_t stream=nullptr;
#endif
  const rns8_matrix* input_a() const noexcept override{return a;}
  const rns8_matrix* input_b() const noexcept override{return b;}
  bool writes_matrix() const noexcept override{return true;}
  bool reads_matrix() const noexcept override{return true;}
  bool produces_resident() const noexcept override{return true;}
  uint32_t output_prefix() const noexcept override{return desc.prefix;}
  const char* operation_name() const noexcept override{return "ordered_basis_gemm";}
  void* sync_stream() const noexcept override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    return stream;
#else
    return nullptr;
#endif
  }
  rns8_status validate(const cpp_int* pending,uint32_t,cpp_int& bound) override {
    if(pending)return RNS8_UNSUPPORTED_BACKEND;
    if(ctx->hip_failure_poisoned)return RNS8_BACKEND_FAILURE;
    if(a==matrix || b==matrix)return RNS8_INVALID_ARGUMENT;
    for(const auto* value:{a,b,static_cast<const rns8_matrix*>(matrix)}) {
      if(value->in_flight_writer || (value==matrix && value->in_flight_readers))return RNS8_BUSY;
      if(value->generation_exhausted || (value==matrix && value->content_generation==UINT64_MAX))return RNS8_RANGE_ERROR;
      if(!basis_storage_matches(*value,RNS8_BACKEND_HIP_DIRECT) || value->hip_device_id!=ctx->device_id ||
          value->desc.semantics!=desc.semantics || value->prefix<desc.prefix ||
          !equal_basis(*stored_basis(*value),*basis))return RNS8_INVALID_ARGUMENT;
    }
    if(a->desc.rows!=desc.m || a->desc.cols!=desc.k || b->desc.rows!=desc.k || b->desc.cols!=desc.n ||
        matrix->desc.rows!=desc.m || matrix->desc.cols!=desc.n)return RNS8_INVALID_ARGUMENT;
    for(const auto* value:{a,b}) {
      if(value->in_flight_readers==UINT64_MAX)return RNS8_RANGE_ERROR;
      if(!value->device_residues_current || value->exact_range_prefix<desc.prefix || value->exact_range_prefix>value->prefix ||
          value->materialized_rns_prefix<desc.prefix ||
          !exact_range_fits(desc.semantics,value->exact_max_magnitude,basis->products[value->exact_range_prefix]))return RNS8_RANGE_ERROR;
    }
    bound=cpp_int(desc.k)*a->exact_max_magnitude*b->exact_max_magnitude;
    return exact_range_fits(desc.semantics,bound,basis->products[desc.prefix])?RNS8_SUCCESS:RNS8_RANGE_ERROR;
  }
  rns8_status enqueue(void* target) override {
    return hip_direct_enqueue_basis_gemm(ctx->device_id,a->hip_residues,b->hip_residues,matrix->hip_residues,
        desc.m,desc.n,desc.k,basis->packet.moduli,desc.prefix,target);
  }
  void reserve() noexcept override {
    ++a->in_flight_readers;if(a!=b)++b->in_flight_readers;
    clear_residue_current(*matrix);clear_native_current(*matrix);clear_byte_limb_current(*matrix);
    ++matrix->prepared_in_flight_writers;matrix->in_flight_writer=true;published_generation=0;
  }
  rns8_status finish(rns8_status operation,cpp_int& bound) noexcept override {
    --a->in_flight_readers;if(a!=b)--b->in_flight_readers;
    --matrix->prepared_in_flight_writers;matrix->in_flight_writer=matrix->prepared_in_flight_writers!=0;
    if(operation==RNS8_SUCCESS) {
      mark_output_device_residues_current(*matrix);matrix->exact_max_magnitude.swap(bound);
      matrix->exact_range_prefix=desc.prefix;matrix->materialized_rns_prefix=desc.prefix;
      matrix->exact_axis_ranges.rows.clear();matrix->exact_axis_ranges.cols.clear();
      published_generation=matrix->content_generation;
    }
    return operation;
  }
  bool output_current() const noexcept override {
    return published_generation && matrix->content_generation==published_generation && matrix->device_residues_current;
  }
  rns8_status close() override {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(stream){hip_scoped_device selected(ctx->device_id);if(selected.status()!=hipSuccess || hipStreamDestroy(stream)!=hipSuccess)return RNS8_BACKEND_FAILURE;stream=nullptr;}
#endif
    return RNS8_SUCCESS;
  }
  ~PreparedBasisGemm() override {(void)close();}
};
}
rns8_status rns8_prepare_basis_gemm_hip(rns8_basis_plan* plan,const rns8_matrix* a,const rns8_matrix* b,
    rns8_matrix* c,rns8_prepared_execution** out) {
  return guard_api([&]() -> rns8_status {
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!plan || !a || !b || !c)return RNS8_INVALID_ARGUMENT;
    if(plan->desc.backend!=RNS8_BACKEND_HIP_DIRECT || !direct_hip_compiled())return RNS8_UNSUPPORTED_BACKEND;
    std::unique_lock<std::mutex> lock(plan->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    const auto ready=basis_plan_ready(*plan);if(ready!=RNS8_SUCCESS)return ready;
    auto owner=std::make_unique<PreparedBasisGemm>();owner->ctx=plan->ctx;owner->matrix=c;owner->a=a;owner->b=b;
    owner->desc=plan->desc;owner->basis=plan->basis;
    cpp_int bound;const auto valid=owner->validate(nullptr,0,bound);if(valid!=RNS8_SUCCESS)return valid;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    hip_scoped_device selected(plan->ctx->device_id);
    if(selected.status()!=hipSuccess || hipStreamCreateWithFlags(&owner->stream,hipStreamNonBlocking)!=hipSuccess)return RNS8_BACKEND_FAILURE;
#endif
    return create_transfer_prepared(std::move(owner),out);
  });
}
rns8_status rns8_export_basis_limbs(rns8_basis_plan* plan,const rns8_matrix* matrix,uint64_t* destination,int64_t ld,uint32_t limbs) {
  return guard_api([&]() -> rns8_status {
    if(!plan || !matrix || !destination || reinterpret_cast<uintptr_t>(destination)%8 || !limbs || limbs>3)return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(plan->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    auto status=basis_plan_ready(*plan);if(status!=RNS8_SUCCESS)return status;
    status=basis_matrix_contract(*plan,*matrix,true);if(status!=RNS8_SUCCESS)return status;
    AddressDomain span;if(!checked_host_plane_address(matrix->desc.rows,matrix->desc.cols,ld,1,uint64_t(limbs)*8,span))return RNS8_RANGE_ERROR;
    for(const auto storage:{std::pair<const void*,uint64_t>{plan->host_words.data(),uint64_t(plan->host_words.capacity())*8},
        {plan->host_residues.data(),plan->host_residues.capacity()},
        {matrix->residues.data(),matrix->residues.size()},{matrix->hip_residues,matrix->hip_residue_bytes},
        {matrix->native_i64.data(),uint64_t(matrix->native_i64.size())*8},{matrix->native_u64.data(),uint64_t(matrix->native_u64.size())*8}})
      if(storage.second && !nonoverlapping_byte_ranges(destination,span.byte_extent,storage.first,storage.second))return RNS8_INVALID_ARGUMENT;
    const bool sign=matrix->desc.semantics==RNS8_EXACT_WIDE_SIGNED;
    const unsigned bits=64*limbs-(sign?1:0);
    if(matrix->exact_max_magnitude>=(cpp_int(1)<<bits))return RNS8_RANGE_ERROR;
    const auto cells=uint64_t(matrix->desc.rows)*matrix->desc.cols;
    status=basis_transaction(*plan,matrix,nullptr,nullptr,false,[&](bool&) -> rns8_status {
      if(plan->desc.backend==RNS8_BACKEND_HIP_DIRECT) {
        auto result=upload_basis_packet(*plan);if(result!=RNS8_SUCCESS)return result;
        BasisMagnitude bound;basis_words(matrix->exact_max_magnitude,bound.words);
        result=hip_direct_basis_export(plan->ctx->device_id,matrix->hip_residues,basis_device(*plan,plan->scratch),cells,
            basis_device(*plan,plan->packet_target),matrix->exact_range_prefix,sign,limbs,bound,basis_device_flag(*plan),&plan->device_status);
        if(result!=RNS8_SUCCESS)return result;
        return hip_direct_copy_device_to_host(plan->ctx->device_id,plan->host_words.data(),basis_device(*plan,plan->scratch),size_t(cells*limbs*8));
      }
      cpp_int value;
      for(uint64_t cell=0;cell<cells;++cell) {
        const auto result=reconstruct_basis_cpu(*matrix,*plan->basis,cell,value);if(result!=RNS8_SUCCESS)return result;
        if(value<0)value+=cpp_int(1)<<(64*limbs);
        for(uint32_t limb=0;limb<limbs;++limb){plan->host_words[size_t(cell*limbs+limb)]=static_cast<uint64_t>(value&UINT64_MAX);value>>=64;}
      }
      return RNS8_SUCCESS;
    });
    if(status!=RNS8_SUCCESS)return status;
    const auto row_bytes=size_t(uint64_t(matrix->desc.cols)*limbs*8);
    for(int64_t row=0;row<matrix->desc.rows;++row)
      std::memcpy(destination+uint64_t(row)*ld*limbs,plan->host_words.data()+uint64_t(row)*matrix->desc.cols*limbs,row_bytes);
    return RNS8_SUCCESS;
  });
}
rns8_status rns8_convert_basis_values(rns8_basis_plan* plan,const rns8_matrix* source,rns8_matrix* destination) {
  return guard_api([&]() -> rns8_status {
    if(!plan || !source || !destination || source==destination)return RNS8_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(plan->mutex,std::try_to_lock);if(!lock.owns_lock())return RNS8_BUSY;
    auto status=basis_plan_ready(*plan);if(status!=RNS8_SUCCESS)return status;
    status=basis_matrix_contract(*plan,*source,true,false);if(status!=RNS8_SUCCESS)return status;
    status=basis_matrix_contract(*plan,*destination,false);if(status!=RNS8_SUCCESS)return status;
    if(source->desc.rows!=destination->desc.rows || source->desc.cols!=destination->desc.cols)return RNS8_INVALID_ARGUMENT;
    cpp_int bound=source->exact_max_magnitude;
    if(!exact_range_fits(destination->desc.semantics,bound,plan->basis->products[destination->prefix]))return RNS8_RANGE_ERROR;
    const auto from=matrix_basis(*source);const auto cells=uint64_t(source->desc.rows)*source->desc.cols;
    const bool sign=source->desc.semantics==RNS8_EXACT_WIDE_SIGNED;
    status=basis_transaction(*plan,source,source,destination,false,[&](bool& written) -> rns8_status {
      if(plan->desc.backend==RNS8_BACKEND_HIP_DIRECT) {
        auto result=upload_basis_packet(*plan,from);if(result!=RNS8_SUCCESS)return result;
        BasisMagnitude magnitude;basis_words(bound,magnitude.words);
        result=hip_direct_basis_convert(plan->ctx->device_id,source->hip_residues,basis_device(*plan,plan->scratch),cells,
            basis_device(*plan,plan->packet_source),basis_device(*plan,plan->packet_target),source->exact_range_prefix,destination->prefix,
            sign,magnitude,basis_device_flag(*plan),&plan->device_status);
        if(result!=RNS8_SUCCESS)return result;
      } else {
        cpp_int value;
        for(uint64_t cell=0;cell<cells;++cell) {
          const auto result=reconstruct_basis_cpu(*source,*from,cell,value);if(result!=RNS8_SUCCESS)return result;
          for(uint32_t p=0;p<destination->prefix;++p)
            plan->host_residues[size_t(uint64_t(p)*cells+cell)]=centered_residue(value,plan->basis->packet.moduli[p]);
        }
      }
      return commit_basis_stage(*plan,*destination,destination->prefix,written);
    });
    if(status==RNS8_SUCCESS)publish_basis_value(*plan,*destination,destination->prefix,bound);
    return status;
  });
}
