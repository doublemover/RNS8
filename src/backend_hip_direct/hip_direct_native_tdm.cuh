#include "backend_common/native_tdm.hpp"
#if defined(__gfx1250__) && __has_builtin(__builtin_amdgcn_tensor_load_to_lds) && \
    __has_builtin(__builtin_amdgcn_s_wait_tensorcnt) && __has_builtin(__builtin_amdgcn_s_wait_dscnt)
#define RNS8_DEVICE_TDM 1
#else
#define RNS8_DEVICE_TDM 0
#endif
#if defined(__gfx1250__) && __has_builtin(__builtin_amdgcn_wmma_i32_16x16x64_iu8)
#define RNS8_DEVICE_TDM_WMMA 1
#else
#define RNS8_DEVICE_TDM_WMMA 0
#endif
#if RNS8_DEVICE_TDM && __has_attribute(cluster_dims)
#define RNS8_TDM_CLUSTER(N) __attribute__((cluster_dims(N,1,1)))
#define RNS8_DEVICE_TDM_CLUSTER RNS8_DEVICE_TDM
#else
#define RNS8_TDM_CLUSTER(N)
#define RNS8_DEVICE_TDM_CLUSTER 0
#endif
__global__ void rns8_native_tdm_capability_kernel(int* flags) {
  *flags=RNS8_DEVICE_TDM|(RNS8_DEVICE_TDM_CLUSTER<<1)|(RNS8_DEVICE_TDM_WMMA<<2);
}
#if RNS8_DEVICE_TDM
__device__ void rns8_native_tdm_load(const uint64_t* source,uint64_t* destination,
    unsigned rows,unsigned cols,unsigned tile_rows,unsigned tile_cols,uint64_t stride,unsigned mask,unsigned* status) {
  using v4u=unsigned __attribute__((ext_vector_type(4)));
  using v8u=unsigned __attribute__((ext_vector_type(8)));
  rns8::detail::TdmDescriptor2D record;
  const bool valid=rns8::detail::make_native_tdm_descriptor(reinterpret_cast<uint64_t>(source),
      uint32_t(reinterpret_cast<uintptr_t>(destination)),rows,cols,tile_rows,tile_cols,stride,mask,record);
  if(!valid){atomicOr(status,1u);return;}
  v4u d0{},zero4{};v8u d1{};
  for(unsigned i=0;i<4;++i)d0[i]=record.group0[i];for(unsigned i=0;i<8;++i)d1[i]=record.group1[i];
#if __clang_major__ >= 23
  v8u zero8{};__builtin_amdgcn_tensor_load_to_lds(d0,d1,zero4,zero4,zero8,0);
#else
  __builtin_amdgcn_tensor_load_to_lds(d0,d1,zero4,zero4,0);
#endif
}
#endif
template<unsigned Mode,unsigned Group,unsigned Cluster>
__global__ RNS8_TDM_CLUSTER(Cluster) __launch_bounds__(256) void rns8_native_tdm_kernel(
    const uint64_t* a,const uint64_t* b,int8_t* c,rns8::detail::NativeTdmProblem p,unsigned* status) {
  if constexpr(Mode!=0) {
#if !RNS8_DEVICE_TDM
    return;
#endif
  }
  __shared__ uint64_t raw[2][2048];
  const unsigned thread=threadIdx.x;
  const uint64_t rowbase=uint64_t(blockIdx.y)*16,colbase=uint64_t(p.cluster_axis?blockIdx.x:blockIdx.x/Cluster)*16;
  const unsigned first=(p.cluster_axis?blockIdx.z:blockIdx.z*Cluster+blockIdx.x%Cluster)*Group;
  int carry[Group]{},matrix_carry[8]{};
  const auto load=[&](unsigned slot,uint64_t inner) {
    if constexpr(Mode==0) {
      for(unsigned item=thread;item<2048;item+=256) {
        const uint64_t row=item<1024?rowbase+item/64:inner+(item-1024)/16;
        const uint64_t col=item<1024?inner+item%64:colbase+(item-1024)%16;
        raw[slot][item]=item<1024?(row<p.m && col<p.k?a[row*p.lda+col]:0):(row<p.k && col<p.n?b[row*p.ldb+col]:0);
      }
    } else {
#if RNS8_DEVICE_TDM
      if(thread<warpSize) {
        // Initialize all masked destinations before the issuing wave starts
        // TDM. Clearing the other slot does not touch any current reader.
        for(unsigned item=thread;item<2048;item+=warpSize)raw[slot][item]=0;
        __builtin_amdgcn_s_wait_dscnt(0);asm volatile("":::"memory");
        const unsigned mask=Mode==2?(1u<<Cluster)-1:0;
        rns8_native_tdm_load(a+rowbase*p.lda+inner,raw[slot],unsigned(p.m-rowbase),unsigned(p.k-inner),16,64,p.lda,mask,status);
        if(colbase<p.n)rns8_native_tdm_load(b+inner*p.ldb+colbase,raw[slot]+1024,unsigned(p.k-inner),unsigned(p.n-colbase),64,16,p.ldb,p.cluster_axis?0:mask,status);
      }
#endif
    }
  };
  const auto ready=[&]() {
#if RNS8_DEVICE_TDM
    if constexpr(Mode!=0)if(thread<warpSize)__builtin_amdgcn_s_wait_tensorcnt(0);
#endif
    __syncthreads();
  };
  load(0,0);
  for(uint64_t inner=0;inner<p.k;inner+=64) {
    const unsigned slot=unsigned((inner/64)&1);ready();
    if(inner+64<p.k)load(slot^1,inner+64);
    if(p.compute) {
#if RNS8_DEVICE_TDM_WMMA
      const unsigned channel=thread/32,lane=thread%32;
      if(channel<Group && first+channel<p.prefix) {
        using v8i=int __attribute__((ext_vector_type(8)));using Layout=rns8::detail::TdmWmmaLayout;
        const int modulus=rns8_default_moduli_device[first+channel];const auto reducer=rns8::detail::finite_u8::native64_reducer(modulus);
        const auto centered=[&](uint64_t bits){int x=int(rns8::detail::finite_u8::canonical_native_u64(bits,reducer));
          if(p.signed_input && (bits>>63)){x-=int(reducer.signed_bias);if(x<0)x+=modulus;}if(x>=(modulus+1)/2)x-=modulus;return uint8_t(x);};
        v8i ap{},bp{},acc{};
        for(unsigned word=0;word<8;++word){uint32_t av=0,bv=0;
          for(unsigned byte=0;byte<4;++byte){const auto ac=Layout::a(lane,word*4+byte),bc=Layout::b(lane,word*4+byte);
            av|=uint32_t(centered(raw[slot][ac.row*64+ac.col]))<<(byte*8);
            bv|=uint32_t(centered(raw[slot][1024+bc.row*16+bc.col]))<<(byte*8);}
          ap[word]=__builtin_bit_cast(int,av);bp[word]=__builtin_bit_cast(int,bv);}
        acc=__builtin_amdgcn_wmma_i32_16x16x64_iu8(true,ap,true,bp,acc,false,false);
        for(unsigned reg=0;reg<8;++reg)matrix_carry[reg]=(matrix_carry[reg]+acc[reg])%modulus;
      }
#endif
    }else for(unsigned channel=0;channel<Group;++channel)if(first+channel<p.prefix) {
      const unsigned modulus=rns8_default_moduli_device[first+channel];
      const auto reducer=rns8::detail::finite_u8::native64_reducer(modulus);
      int sum=carry[channel];
      for(unsigned k=0;k<64;++k) {
        const uint64_t av=raw[slot][(thread/16)*64+k],bv=raw[slot][1024+k*16+thread%16];
        int ar=int(rns8::detail::finite_u8::canonical_native_u64(av,reducer));
        int br=int(rns8::detail::finite_u8::canonical_native_u64(bv,reducer));
        if(p.signed_input && (av>>63)){ar-=int(reducer.signed_bias);if(ar<0)ar+=modulus;}
        if(p.signed_input && (bv>>63)){br-=int(reducer.signed_bias);if(br<0)br+=modulus;}
        if(ar>=int((modulus+1)/2))ar-=modulus;if(br>=int((modulus+1)/2))br-=modulus;
        sum+=ar*br;
      }
      carry[channel]=sum%int(modulus);
    }
    __syncthreads(); // every local reader is finished before slot reuse
#if RNS8_DEVICE_TDM_CLUSTER
    if constexpr(Mode==2) {
      if(thread<warpSize)__builtin_amdgcn_s_barrier_signal(-3);
      __builtin_amdgcn_s_barrier_wait(-3); // every wave participates
    }
#endif
  }
  if(p.compute) {
    const unsigned channel=thread/32,lane=thread%32;
    if(channel<Group && first+channel<p.prefix)for(unsigned reg=0;reg<8;++reg){const auto cell=rns8::detail::TdmWmmaLayout::c(lane,reg);
      const uint64_t row=rowbase+cell.row,col=colbase+cell.col;const int modulus=rns8_default_moduli_device[first+channel];int value=matrix_carry[reg];
      if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;
      if(row<p.m && col<p.n)c[(uint64_t(first+channel)*p.m+row)*p.ldc+col]=int8_t(value);}
    return;
  }
  const uint64_t row=rowbase+thread/16,col=colbase+thread%16;
  if(row<p.m && col<p.n)for(unsigned channel=0;channel<Group;++channel)if(first+channel<p.prefix) {
    const int modulus=rns8_default_moduli_device[first+channel];int value=carry[channel];
    if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;
    c[(uint64_t(first+channel)*p.m+row)*p.ldc+col]=int8_t(value);
  }
}
extern "C" int rns8_hip_native_tdm_capability(int* flags) {
  hipLaunchKernelGGL(rns8_native_tdm_capability_kernel,dim3(1),dim3(1),0,nullptr,flags);return int(hipGetLastError());
}
template<unsigned Cluster>
__global__ RNS8_TDM_CLUSTER(Cluster) void rns8_native_tdm_litmus_kernel(const uint64_t* input,uint64_t* output,
    unsigned rows,unsigned cols,unsigned ld,unsigned* status) {
#if RNS8_DEVICE_TDM
  __shared__ uint64_t tile[512];const unsigned thread=threadIdx.x;
  if(thread<warpSize) {
    for(unsigned i=thread;i<512;i+=warpSize)tile[i]=0;
    __builtin_amdgcn_s_wait_dscnt(0);asm volatile("":::"memory");
    rns8_native_tdm_load(input,tile,rows,cols,16,32,ld,Cluster>1?(1u<<Cluster)-1:0,status);
    __builtin_amdgcn_s_wait_tensorcnt(0);
  }
  __syncthreads();
#if RNS8_DEVICE_TDM_CLUSTER
  if constexpr(Cluster>1){if(thread<warpSize)__builtin_amdgcn_s_barrier_signal(-3);__builtin_amdgcn_s_barrier_wait(-3);}
#endif
  for(unsigned i=thread;i<512;i+=blockDim.x)output[uint64_t(blockIdx.x)*512+i]=tile[i];
#else
  (void)input;(void)output;(void)rows;(void)cols;(void)ld;(void)status;
#endif
}
extern "C" int rns8_hip_native_tdm_litmus(const uint64_t* input,uint64_t* output,unsigned rows,unsigned cols,unsigned ld,
    unsigned cluster,unsigned* status) {
#define RNS8_TDM_LITMUS(C) case C:hipLaunchKernelGGL((rns8_native_tdm_litmus_kernel<C>),dim3(C),dim3(256),0,nullptr,input,output,rows,cols,ld,status);break
  switch(cluster){RNS8_TDM_LITMUS(1);RNS8_TDM_LITMUS(2);RNS8_TDM_LITMUS(4);default:return int(hipErrorInvalidValue);}
#undef RNS8_TDM_LITMUS
  return int(hipGetLastError());
}
template<unsigned Mode,unsigned Group,unsigned Cluster>
int rns8_launch_native_tdm(const uint64_t* a,const uint64_t* b,int8_t* c,rns8::detail::NativeTdmProblem p,unsigned* status,hipStream_t stream) {
  const unsigned groups=(p.prefix+Group-1)/Group;
  const unsigned cols=unsigned((p.n+15)/16);
  hipLaunchKernelGGL((rns8_native_tdm_kernel<Mode,Group,Cluster>),dim3(p.cluster_axis?((cols+Cluster-1)/Cluster)*Cluster:cols*Cluster,unsigned((p.m+15)/16),p.cluster_axis?groups:(groups+Cluster-1)/Cluster),
      dim3(256),0,stream,a,b,c,p,status);return int(hipGetLastError());
}
extern "C" int rns8_hip_native_tdm(const uint64_t* a,const uint64_t* b,int8_t* c,rns8::detail::NativeTdmProblem p,unsigned* status,void* stream) {
  const auto s=static_cast<hipStream_t>(stream);
#define RNS8_TDM_GROUP(G) case G:if(p.mode==0)return rns8_launch_native_tdm<0,G,1>(a,b,c,p,status,s); \
  if(p.mode==1)return rns8_launch_native_tdm<1,G,1>(a,b,c,p,status,s); \
  if(p.cluster==2)return rns8_launch_native_tdm<2,G,2>(a,b,c,p,status,s); \
  return rns8_launch_native_tdm<2,G,4>(a,b,c,p,status,s)
  switch(p.group){RNS8_TDM_GROUP(1);RNS8_TDM_GROUP(2);RNS8_TDM_GROUP(4);default:return int(hipErrorInvalidValue);}
#undef RNS8_TDM_GROUP
}
#undef RNS8_TDM_CLUSTER
#undef RNS8_DEVICE_TDM
#undef RNS8_DEVICE_TDM_CLUSTER

#undef RNS8_DEVICE_TDM_WMMA
