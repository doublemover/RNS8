struct rns8_crt_export {
  rns8_context* ctx=nullptr;
  const rns8_plan* plan=nullptr;
  const rns8_matrix* source=nullptr;
  rns8_crt_export_desc desc{};
  rns8::detail::checked_crt::Decode decode{};
  rns8::detail::checked_crt::Resources resources{};
  boost::multiprecision::cpp_int envelope;
  const rns8::detail::balanced_crt::Plan* balanced=nullptr;
  const rns8::detail::group_crt::Plan* groups=nullptr;
  uint64_t instance=0,generation=0,successful_exports=0;
  std::string certificate;
  std::unique_ptr<uint64_t[]> staged,partials;
  rns8::detail::hip_direct_device_buffer device;
  bool pending=false;
};

namespace {
using CrtWide=rns8::detail::balanced_crt::Wide;
CrtWide checked_crt_wide(boost::multiprecision::cpp_int value) {
  CrtWide result{};
  for(auto& limb:result.limb) {limb=static_cast<uint64_t>(value&UINT64_MAX);value>>=64;}
  return result;
}
std::string checked_crt_binding(const rns8_crt_export& owner,const rns8_matrix& source) {
  std::string text="checked-integer-crt-v1;"+owner.plan->arithmetic_guard->checked.sha256;
  for(auto value:{uint64_t(owner.desc.algorithm),uint64_t(owner.desc.mode),uint64_t(owner.desc.limb_count),
      uint64_t(owner.desc.producer_order),uint64_t(owner.plan->prefix),source.matrix_instance_id,source.content_generation})
    text+=';'+std::to_string(value);
  text+=";bound="+owner.envelope.str();
  text+=";build="+owner.plan->arithmetic_guard->certificate.instruction.compiler_build;
  for(uint32_t p=0;p<owner.plan->prefix;++p)text+=';'+std::to_string(rns8::detail::garner::kConstants.moduli[p]);
  if(owner.groups)for(uint32_t g=0;g<owner.groups->group_count;++g) {
    text+=';'+std::to_string(owner.groups->groups[g].channel_mask);
    text+=';'+std::to_string(owner.groups->groups[g].product);
  }
  return rns8::detail::sha256(text);
}
rns8_status checked_crt_source(const rns8_crt_export& owner,const rns8_matrix& source) {
  using namespace rns8::detail;
  if(source.in_flight_writer)return RNS8_BUSY;
  const auto status=validate_export_matrix(*owner.ctx,*owner.plan,source,owner.plan->desc.semantics,
      RNS8_BOUND_NONE,owner.plan->prefix);
  if(status!=RNS8_SUCCESS)return status;
  if(source.generation_exhausted || !source.content_generation || source.exact_range_prefix<owner.plan->prefix ||
      source.exact_max_magnitude<0 || source.exact_max_magnitude>owner.envelope)
    return RNS8_RANGE_ERROR;
  return RNS8_SUCCESS;
}
CrtWide checked_crt_cpu_root(const rns8_crt_export& owner,uint64_t cell,uint64_t cells) {
  using namespace rns8::detail;
  using namespace balanced_crt;
  const auto* residues=owner.source->residues.data();
  if(owner.desc.algorithm>=RNS8_CRT_GROUP_LOCAL) {
    uint64_t groups[4]{};
    for(uint32_t g=0;g<owner.groups->group_count;++g)
      groups[g]=owner.desc.algorithm==RNS8_CRT_GROUP_SPLIT?owner.partials[uint64_t(g)*cells+cell]:
          group_crt::reconstruct_group(residues,cells,cell,owner.groups->groups[g]);
    return group_crt::merge_groups(groups,*owner.groups);
  }
  Wide root{},values[20]{};
  for(uint32_t p=0;p<owner.plan->prefix;++p) {
    const uint32_t modulus=garner::kConstants.moduli[p];
    const int stored=residues[uint64_t(p)*cells+cell];
    const auto residue=stored<0?uint32_t(stored+int(modulus)):uint32_t(stored);
    if(owner.desc.algorithm==RNS8_CRT_BALANCED) {values[p].limb[0]=residue;continue;}
    Wide weight{};uint32_t coefficient=0;
    if(owner.desc.algorithm==RNS8_CRT_DIRECT) {
      coefficient=residue*direct_crt::kConstants.inverses[owner.plan->prefix][p]%modulus;
      for(unsigned i=0;i<3;++i)weight.limb[i]=direct_crt::kConstants.weights[owner.plan->prefix][p][i];
    } else {
      const auto current=direct_crt::modulo_small(root.limb,modulus);
      coefficient=((residue+modulus-current)%modulus)*garner::kConstants.inverses[p]%modulus;
      for(unsigned i=0;i<3;++i)weight.limb[i]=garner::kConstants.products[p][i];
    }
    Wide factor{};factor.limb[0]=coefficient;
    root=add(root,multiply_bounded(weight,factor));
  }
  if(owner.balanced) {
    for(uint32_t i=0;i<owner.balanced->count;++i) {
      const auto& node=owner.balanced->merges[i];values[node.left]=merge(values[node.left],values[node.right],node);
    }
    return values[0];
  }
  if(owner.desc.algorithm==RNS8_CRT_DIRECT)
    while(compare(root,owner.decode.product)>=0)root=subtract(root,owner.decode.product);
  return root;
}
} // namespace

rns8_status rns8_prepare_crt_export(rns8_context* ctx,const rns8_plan* plan,const rns8_matrix* source,
    const rns8_crt_export_desc* desc,rns8_crt_export** out) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
    if(!ctx || !plan || !source || !desc || !valid_abi(desc->struct_size,desc->abi_version,sizeof(*desc)) ||
        desc->algorithm>RNS8_CRT_GROUP_SPLIT || desc->mode>2 || !desc->limb_count || desc->limb_count>32)
      return RNS8_INVALID_ARGUMENT;
    if(!checked_crt_backend_supported(plan->backend))return RNS8_UNSUPPORTED_BACKEND;
    if(!exact_wide_semantics(plan->desc.semantics) ||
        (desc->mode==0 && plan->desc.semantics!=RNS8_EXACT_WIDE_SIGNED) ||
        (desc->mode==1 && plan->desc.semantics!=RNS8_EXACT_WIDE_UNSIGNED))return RNS8_INVALID_ARGUMENT;
    if(plan->desc.m<=0 || plan->desc.n<=0 || plan->desc.m>INT64_MAX/plan->desc.n)return RNS8_RANGE_ERROR;
    const auto cells=plan->desc.m*plan->desc.n;
    const bool gpu=plan->backend==RNS8_BACKEND_HIP_DIRECT;
    const auto resources=checked_crt::resources(cells,plan->prefix,desc->limb_count,desc->algorithm,gpu,desc->max_scratch_bytes);
    if(resources.status!=RNS8_SUCCESS)return resources.status;
    const uint32_t groups=(plan->prefix+7)/8;
    if((desc->algorithm!=RNS8_CRT_GROUP_SPLIT && desc->producer_order) || desc->producer_order>=2*groups)
      return RNS8_INVALID_ARGUMENT;
    if(!plan->arithmetic_guard || ctx->prepared_references==UINT64_MAX || plan->prepared_references==UINT64_MAX ||
        source->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
    auto owner=std::make_unique<rns8_crt_export>();
    owner->ctx=ctx;owner->plan=plan;owner->desc=*desc;owner->resources=resources;owner->envelope=source->exact_max_magnitude;
    const auto valid=checked_crt_source(*owner,*source);if(valid!=RNS8_SUCCESS)return valid;
    owner->decode.product=checked_crt_wide(modulus_product(plan->prefix));
    owner->decode.half=checked_crt_wide(modulus_product(plan->prefix)/2);
    owner->decode.bound=checked_crt_wide(owner->envelope);owner->decode.limbs=desc->limb_count;
    owner->decode.mode=desc->mode;owner->decode.signed_source=plan->desc.semantics==RNS8_EXACT_WIDE_SIGNED;
    if(desc->algorithm==RNS8_CRT_BALANCED)owner->balanced=&balanced_crt::plan(plan->prefix);
    if(desc->algorithm>=RNS8_CRT_GROUP_LOCAL)owner->groups=&group_crt::plan(plan->prefix);
    owner->staged.reset(new uint64_t[size_t(resources.output_bytes/8)]);
    if(gpu) {
      const auto allocated=owner->device.allocate(ctx->device_id,size_t(resources.device_bytes));
      if(allocated!=RNS8_SUCCESS)return allocated;
    } else if(resources.group_bytes)owner->partials.reset(new uint64_t[size_t(resources.group_bytes/8)]);
    owner->source=source;owner->instance=source->matrix_instance_id;owner->generation=source->content_generation;
    owner->certificate=checked_crt_binding(*owner,*source);
    ++ctx->prepared_references;++plan->prepared_references;++source->prepared_references;
    *out=owner.release();return RNS8_SUCCESS;
  });
}

rns8_status rns8_rebind_crt_export(rns8_crt_export* owner,const rns8_matrix* source) {
  return guard_api([&]() -> rns8_status {
    if(!owner || !source)return RNS8_INVALID_ARGUMENT;
    if(owner->pending)return RNS8_BUSY;
    const auto valid=checked_crt_source(*owner,*source);if(valid!=RNS8_SUCCESS)return valid;
    if(source!=owner->source && source->prepared_references==UINT64_MAX)return RNS8_RANGE_ERROR;
    auto certificate=checked_crt_binding(*owner,*source); // All fallible work precedes rebinding.
    if(source!=owner->source) {++source->prepared_references;--owner->source->prepared_references;}
    owner->source=source;owner->instance=source->matrix_instance_id;owner->generation=source->content_generation;
    owner->certificate.swap(certificate);return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_crt_export_info(const rns8_crt_export* owner,rns8_crt_export_info* info) {
  return guard_api([&]() -> rns8_status {
    if(!owner || !info || !rns8::detail::valid_abi(info->struct_size,info->abi_version,sizeof(*info)))return RNS8_INVALID_ARGUMENT;
    *info={};info->struct_size=sizeof(*info);info->abi_version=RNS8_ABI_VERSION;
    info->algorithm=owner->desc.algorithm;info->mode=owner->desc.mode;info->limb_count=owner->desc.limb_count;
    info->prefix=owner->plan->prefix;info->producer_order=owner->desc.producer_order;
    info->source_instance=owner->instance;info->source_generation=owner->generation;info->successful_exports=owner->successful_exports;
    info->retained_host_bytes=owner->resources.host_bytes;
    info->retained_device_bytes=owner->device.bytes();info->group_scratch_bytes=owner->resources.group_bytes;
    for(unsigned i=0;i<3;++i)info->magnitude_bound[i]=owner->decode.bound.limb[i];
    if(owner->groups) {
      info->group_count=owner->groups->group_count;
      for(uint32_t g=0;g<info->group_count;++g) {
        info->group_products[g]=owner->groups->groups[g].product;
        info->group_channel_masks[g]=owner->groups->groups[g].channel_mask;
      }
    }
    std::copy(owner->certificate.begin(),owner->certificate.end(),info->certificate_sha256);return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_crt_limbs(rns8_crt_export* owner,uint64_t* dst,int64_t ld) {
  return guard_api([&]() -> rns8_status {
    using namespace rns8::detail;
    if(!owner || !dst)return RNS8_INVALID_ARGUMENT;
    if(owner->pending || owner->source->in_flight_writer)return RNS8_BUSY;
    const auto& plan=*owner->plan;const auto& source=*owner->source;
    if(!valid_limb_export_access(plan.desc.m,plan.desc.n,ld,owner->desc.limb_count) ||
        source.matrix_instance_id!=owner->instance || source.content_generation!=owner->generation)
      return RNS8_INVALID_ARGUMENT;
    const auto valid=checked_crt_source(*owner,source);if(valid!=RNS8_SUCCESS)return valid;
    if(source.in_flight_readers==UINT64_MAX || owner->successful_exports==UINT64_MAX)return RNS8_RANGE_ERROR;
    PreparedExportReadLease read(source,owner->pending);
    const auto cells=uint64_t(plan.desc.m)*uint64_t(plan.desc.n);
    if(owner->device.get()) {
      auto* roots=static_cast<uint64_t*>(owner->device.get());
      auto* output=reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(roots)+owner->resources.root_bytes);
      auto* groups=reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(output)+owner->resources.output_bytes);
      auto* status_word=reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(groups)+owner->resources.group_bytes);
      owner->pending=true;
      auto status=hip_direct_zero(owner->ctx->device_id,status_word,4);
      if(status==RNS8_SUCCESS)switch(owner->desc.algorithm) {
        case RNS8_CRT_GARNER: {
          void* unused=nullptr;std::size_t unused_bytes=0;
          status=hip_direct_export_exact_wide_unsigned_limbs_to_device(owner->ctx->device_id,source.hip_residues,
              roots,&unused,&unused_bytes,plan.desc.m,plan.desc.n,plan.prefix,3);break;
        }
        case RNS8_CRT_DIRECT:status=hip_direct_export_canonical_direct_crt_device(owner->ctx->device_id,source.hip_residues,roots,int64_t(cells),plan.prefix);break;
        case RNS8_CRT_BALANCED:status=hip_direct_export_canonical_balanced_crt_device(owner->ctx->device_id,source.hip_residues,roots,int64_t(cells),plan.prefix);break;
        default:status=hip_direct_export_canonical_group_crt_device(owner->ctx->device_id,source.hip_residues,roots,int64_t(cells),plan.prefix,
            owner->desc.algorithm==RNS8_CRT_GROUP_SPLIT,owner->resources.group_bytes?groups:nullptr,size_t(owner->resources.group_bytes),owner->desc.producer_order);break;
      }
      if(status==RNS8_SUCCESS)status=hip_direct_decode_crt_roots_device(owner->ctx->device_id,roots,output,int64_t(cells),owner->decode,status_word);
      uint32_t result=0;
      if(status==RNS8_SUCCESS)status=hip_direct_copy_device_to_host(owner->ctx->device_id,&result,status_word,4);
      if(status==RNS8_SUCCESS && !result)status=hip_direct_copy_device_to_host(owner->ctx->device_id,owner->staged.get(),output,owner->resources.output_bytes);
      const auto drained=hip_direct_synchronize(owner->ctx->device_id);
      if(drained!=RNS8_SUCCESS) {owner->ctx->hip_failure_poisoned=true;return drained;}
      owner->pending=false;
      if(status!=RNS8_SUCCESS)return status;
      if(result)return RNS8_RANGE_ERROR;
    } else {
      if(owner->desc.algorithm==RNS8_CRT_GROUP_SPLIT)
        for(uint32_t physical=0;physical<owner->groups->group_count;++physical) {
          const auto g=group_crt::producer_group(physical,owner->desc.producer_order,owner->groups->group_count);
          for(uint64_t cell=0;cell<cells;++cell)owner->partials[uint64_t(g)*cells+cell]=
              group_crt::reconstruct_group(source.residues.data(),cells,cell,owner->groups->groups[g]);
        }
      for(uint64_t cell=0;cell<cells;++cell)
        if(!checked_crt::decode(checked_crt_cpu_root(*owner,cell,cells),owner->decode,
            owner->staged.get()+cell*owner->desc.limb_count))return RNS8_RANGE_ERROR;
    }
    for(int64_t row=0;row<plan.desc.m;++row)
      std::copy_n(owner->staged.get()+uint64_t(row)*plan.desc.n*owner->desc.limb_count,
          size_t(plan.desc.n)*owner->desc.limb_count,dst+uint64_t(row)*ld*owner->desc.limb_count);
    ++owner->successful_exports;return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_crt_export(rns8_crt_export* owner) {
  if(!owner)return RNS8_SUCCESS;
  if(owner->pending) {
    const auto status=rns8::detail::hip_direct_synchronize(owner->ctx->device_id);
    if(status!=RNS8_SUCCESS)return status;
    owner->pending=false;--owner->source->in_flight_readers;
  }
  const auto released=owner->device.reset();if(released!=RNS8_SUCCESS)return released;
  --owner->ctx->prepared_references;--owner->plan->prepared_references;--owner->source->prepared_references;
  delete owner;return RNS8_SUCCESS;
}
