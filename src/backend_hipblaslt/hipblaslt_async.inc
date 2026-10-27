namespace rns8::detail {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
extern "C" int rns8_hipblaslt_pack_transpose_centered_on_stream(const int8_t*,int8_t*,int,int,int,int,int,int,void*);
extern "C" int rns8_hipblaslt_reduce_i32_to_centered_on_stream(const int32_t*,int8_t*,int,int,int,int,int,uint32_t,int,void*);
namespace {
struct HipblasLtAsync {
  rns8_context* ctx=nullptr;const rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
  std::array<matmul_descriptors,2> descriptors;
  std::array<hipblasLtMatmulAlgo_t,2> algorithms{};
  std::array<padded_block_shape,2> shapes{};
  uint32_t count=0;bool warmup_pending=false;
};
hipblasStatus_t async_matmul(HipblasLtAsync& owner,unsigned slot,void* stream) {
  auto& d=owner.descriptors[slot];const auto& shape=owner.shapes[slot];auto& w=*owner.workspace;
  auto* a=static_cast<int8_t*>(w.hipblaslt_workspace);
  auto* b=a+uint64_t(shape.m)*shape.k;
  auto* library=b+uint64_t(shape.n)*shape.k;
  const auto library_bytes=w.hipblaslt_workspace_bytes-(uint64_t(shape.m)+shape.n)*shape.k;
  const int32_t alpha=1,beta=0;
  return hipblasLtMatmul(static_cast<hipblasLtHandle_t>(owner.ctx->hipblaslt_handle),d.op,&alpha,b,d.a,a,d.b,
      &beta,w.hipblaslt_int32_scratch,d.c,w.hipblaslt_int32_scratch,d.d,&owner.algorithms[slot],
      library,library_bytes,static_cast<hipStream_t>(stream));
}
rns8_status close_async_descriptors(HipblasLtAsync& owner) {
  for(auto& d:owner.descriptors){
    if(d.preference){if(hipblasLtMatmulPreferenceDestroy(d.preference)!=HIPBLAS_STATUS_SUCCESS)return RNS8_BACKEND_FAILURE;d.preference=nullptr;}
    for(auto* layout:{&d.d,&d.c,&d.b,&d.a})if(*layout){if(hipblasLtMatrixLayoutDestroy(*layout)!=HIPBLAS_STATUS_SUCCESS)return RNS8_BACKEND_FAILURE;*layout=nullptr;}
    if(d.op){if(hipblasLtMatmulDescDestroy(d.op)!=HIPBLAS_STATUS_SUCCESS)return RNS8_BACKEND_FAILURE;d.op=nullptr;}
  }
  return RNS8_SUCCESS;
}
}
#endif
rns8_status hipblaslt_destroy_async(void* pointer) {
  if(!pointer)return RNS8_SUCCESS;
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  auto* owner=static_cast<HipblasLtAsync*>(pointer);hip_scoped_device selected(owner->ctx->device_id);
  if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  if(owner->warmup_pending){if(hipDeviceSynchronize()!=hipSuccess)return RNS8_BACKEND_FAILURE;
    owner->warmup_pending=false;owner->workspace->in_flight=false;}
  const auto status=close_async_descriptors(*owner);if(status!=RNS8_SUCCESS)return status;
  delete owner;return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hipblaslt_prepare_async(rns8_context* ctx,const rns8_plan* plan,rns8_workspace* workspace,void** out) {
  if(!out)return RNS8_INVALID_ARGUMENT;*out=nullptr;
  if(!ctx || !plan || !workspace)return RNS8_INVALID_ARGUMENT;
  if(workspace->in_flight)return RNS8_BUSY;
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  if(!ctx->hipblaslt_handle || !checked_workspace(plan->desc.m,plan->desc.n,plan->desc.k,
      workspace->hipblaslt_int32_scratch_bytes,workspace->hipblaslt_workspace_bytes))return RNS8_INVALID_ARGUMENT;
  hip_scoped_device selected(ctx->device_id);if(selected.status()!=hipSuccess)return RNS8_BACKEND_FAILURE;
  auto owner=std::make_unique<HipblasLtAsync>();owner->ctx=ctx;owner->plan=plan;owner->workspace=workspace;
  const int64_t first=std::min<int64_t>(65536,plan->desc.k),tail=plan->desc.k%65536;
  owner->count=tail && tail!=first?2:1;
  for(unsigned slot=0;slot<owner->count;++slot){
    auto& shape=owner->shapes[slot];auto& d=owner->descriptors[slot];
    if(!padded_block_shape_for(plan->desc.m,plan->desc.n,slot?tail:first,shape))return RNS8_RANGE_ERROR;
    auto status=create_matmul_descriptors(shape,shape.n,d);
    const uint64_t packed=(uint64_t(shape.m)+shape.n)*shape.k;
    if(packed>workspace->hipblaslt_workspace_bytes)return RNS8_WORKSPACE_TOO_SMALL;
    const uint64_t library_bytes=workspace->hipblaslt_workspace_bytes-packed;
    if(status==HIPBLAS_STATUS_SUCCESS)status=create_matmul_preference(size_t(library_bytes),d);
    hipblasLtMatmulHeuristicResult_t heuristic{};int count=0;
    if(status==HIPBLAS_STATUS_SUCCESS)status=hipblasLtMatmulAlgoGetHeuristic(static_cast<hipblasLtHandle_t>(ctx->hipblaslt_handle),
        d.op,d.a,d.b,d.c,d.d,d.preference,1,&heuristic,&count);
    if(status!=HIPBLAS_STATUS_SUCCESS)return status_from_hipblas(status);
    if(count!=1 || heuristic.state!=HIPBLAS_STATUS_SUCCESS || heuristic.workspaceSize>library_bytes)return RNS8_UNSUPPORTED_BACKEND;
    owner->algorithms[slot]=heuristic.algo;
  }
  // Warm both geometries outside capture. Only owned pack/scratch storage is
  // touched; actual user operands and result currentness remain unchanged.
  workspace->in_flight=true;owner->warmup_pending=true;*out=owner.release();
  auto* retained=static_cast<HipblasLtAsync*>(*out);rns8_status result=RNS8_SUCCESS;
  for(unsigned slot=0;slot<retained->count;++slot){
    const auto& shape=retained->shapes[slot];
    if(hipMemsetAsync(workspace->hipblaslt_workspace,0,(uint64_t(shape.m)+shape.n)*shape.k,nullptr)!=hipSuccess){result=RNS8_BACKEND_FAILURE;break;}
    result=status_from_hipblas(async_matmul(*retained,slot,nullptr));if(result!=RNS8_SUCCESS)break;
  }
  if(hipDeviceSynchronize()!=hipSuccess){ctx->hip_failure_poisoned=true;return result==RNS8_SUCCESS?RNS8_BACKEND_FAILURE:result;}
  retained->warmup_pending=false;workspace->in_flight=false;
  if(result!=RNS8_SUCCESS){const auto closed=hipblaslt_destroy_async(retained);if(closed==RNS8_SUCCESS)*out=nullptr;return result;}
  return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status hipblaslt_enqueue_async(void* pointer,const void* a,const void* b,void* c,
    int64_t lda,int64_t ldb,int64_t ldc,void* stream) {
  if(!pointer || !a || !b || !c || !stream)return RNS8_INVALID_ARGUMENT;
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  auto& owner=*static_cast<HipblasLtAsync*>(pointer);const auto& p=*owner.plan;auto& w=*owner.workspace;
  if(owner.warmup_pending)return RNS8_BUSY;
  const auto m=p.desc.m,n=p.desc.n,k=p.desc.k;
  const bool finite=p.desc.semantics==RNS8_FINITE_RING_U8 || p.desc.semantics==RNS8_FINITE_FIELD_U8;
  const uint32_t prefix=finite?1:p.prefix;
  if(!qualified_i32_gemm_address(m,n,k,lda,ldb,ldc,prefix))return RNS8_UNSUPPORTED_BACKEND;
  for(uint32_t plane=0;plane<prefix;++plane)for(int64_t first=0;first<k;){
    const auto count=std::min<int64_t>(65536,k-first);const unsigned slot=count<65536 && owner.count==2?1:0;
    const auto& shape=owner.shapes[slot];auto* packed_a=static_cast<int8_t*>(w.hipblaslt_workspace);
    auto* packed_b=packed_a+uint64_t(shape.m)*shape.k;
    auto error=hipError_t(rns8_hipblaslt_pack_transpose_centered_on_stream(
        static_cast<const int8_t*>(a)+uint64_t(plane)*m*lda+first,packed_a,int(m),int(count),int(lda),shape.k,shape.m,shape.k,stream));
    if(error==hipSuccess)error=hipError_t(rns8_hipblaslt_pack_transpose_centered_on_stream(
        static_cast<const int8_t*>(b)+uint64_t(plane)*k*ldb+first*ldb,packed_b,int(count),int(n),int(ldb),shape.n,shape.k,shape.n,stream));
    if(error!=hipSuccess)return RNS8_BACKEND_FAILURE;
    const auto status=status_from_hipblas(async_matmul(owner,slot,stream));if(status!=RNS8_SUCCESS)return status;
    const uint16_t modulus=finite?uint16_t(p.desc.finite_modulus):kHipblasLtDefaultModuli[plane];
    error=hipError_t(rns8_hipblaslt_reduce_i32_to_centered_on_stream(static_cast<const int32_t*>(w.hipblaslt_int32_scratch),
        static_cast<int8_t*>(c)+uint64_t(plane)*m*ldc,int(m),int(n),shape.n,int(ldc),modulus,modulus_reciprocal_u32(modulus),first!=0,stream));
    if(error!=hipSuccess)return RNS8_BACKEND_FAILURE;first+=count;
  }
  return RNS8_SUCCESS;
#else
  (void)lda;(void)ldb;(void)ldc;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
} // namespace rns8::detail
