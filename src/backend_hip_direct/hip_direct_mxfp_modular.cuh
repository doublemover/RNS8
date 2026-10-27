#include "backend_common/mxfp_modular.hpp"
#if defined(__gfx950__) && __has_builtin(__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
#define RNS8_MXFP_AVAILABLE 1
#else
#define RNS8_MXFP_AVAILABLE 0
#endif
__global__ void rns8_mxfp_capability(int* supported){*supported=RNS8_MXFP_AVAILABLE;}
template<unsigned A,unsigned B,bool Materialized>
__global__ __launch_bounds__(64) void rns8_mxfp_modular_kernel(const int8_t* a,const int8_t* b,int8_t* c,
    int m,int n,int k,int lda,int ldb,int ldc,int modulus,unsigned* status,int32_t* planes,
    const uint8_t* a_scales,const uint8_t* b_scales) {
#if RNS8_MXFP_AVAILABLE
  namespace f=rns8::detail::mxfp_modular;
  constexpr auto af=f::format(A),bf=f::format(B);constexpr unsigned count=af.digits*bf.digits;
  using i32x8=int __attribute__((ext_vector_type(8)));using f32x4=float __attribute__((ext_vector_type(4)));
  const unsigned lane=threadIdx.x;const int64_t rowbase=int64_t(blockIdx.y)*16,colbase=int64_t(blockIdx.x)*16;
  int carry[4]{},products[count][4]{};
  for(int64_t inner=0;inner<k;inner+=128) {
    // All 32-value scale granules of this instruction use the same exponent.
    // Replicated bytes make OPSEL and scale-lane arrangement unambiguous.
    const int sa=int(a_scales[inner/128])-127,sb=int(b_scales[inner/128])-127;
    i32x8 ap[af.digits]{},bp[bf.digits]{};
    for(unsigned item=0;item<32;++item) {
      const auto ac=f::Layout::a(lane,item,A),bc=f::Layout::b(lane,item,B);
      const int av=rowbase+ac.row<m && inner+ac.col<k?a[(rowbase+ac.row)*lda+inner+ac.col]:0;
      const int bv=inner+bc.row<k && colbase+bc.col<n?b[(inner+bc.row)*ldb+colbase+bc.col]:0;
      for(unsigned d=0;d<af.digits;++d) {
        const uint64_t code=f::encode(f::digit(av,d,A),A,sa);const unsigned bit=item*af.bits,word=bit/32,shift=bit%32;
        ap[d][word]|=int(uint32_t(code<<shift));if(shift+af.bits>32)ap[d][word+1]|=int(code>>(32-shift));
      }
      for(unsigned d=0;d<bf.digits;++d) {
        const uint64_t code=f::encode(f::digit(bv,d,B),B,sb);const unsigned bit=item*bf.bits,word=bit/32,shift=bit%32;
        bp[d][word]|=int(uint32_t(code<<shift));if(shift+bf.bits>32)bp[d][word+1]|=int(code>>(32-shift));
      }
    }
    for(unsigned left=0;left<af.digits;++left)for(unsigned right=0;right<bf.digits;++right) {
      // A fresh accumulator per instruction is a mandatory scale barrier.
      f32x4 partial{};
      partial=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(ap[left],bp[right],partial,A,B,
          0,int(uint32_t(sa+127)*0x01010101u),0,int(uint32_t(sb+127)*0x01010101u));
      const unsigned term=left*bf.digits+right;const int weight=1<<(left*af.radix_bits+right*bf.radix_bits);
      constexpr int bound=128*(1<<(af.radix_bits-1))*(1<<(bf.radix_bits-1));
      for(unsigned reg=0;reg<4;++reg) {
        int integer=0;
        if(!rns8::detail::fp8_modular::integer_partial(partial[reg],integer,bound)){atomicOr(status,1u);continue;}
        if constexpr(Materialized)products[term][reg]=(products[term][reg]+integer%modulus)%modulus;
        else carry[reg]=(carry[reg]+(integer%modulus)*weight)%modulus;
      }
    }
  }
  for(unsigned reg=0;reg<4;++reg) {
    const auto cell=f::Layout::c(lane,reg);const int64_t row=rowbase+cell.row,col=colbase+cell.col;
    if(row>=m || col>=n)continue;
    if constexpr(Materialized)for(unsigned term=0;term<count;++term)planes[(uint64_t(term)*m+row)*n+col]=products[term][reg];
    else {int value=carry[reg];if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;c[row*ldc+col]=int8_t(value);}
  }
#else
  (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)lda;(void)ldb;(void)ldc;(void)modulus;(void)status;(void)planes;(void)a_scales;(void)b_scales;
#endif
}
__global__ void rns8_mxfp_recombine(const int32_t* planes,int8_t* c,uint64_t m,uint64_t n,uint64_t ldc,int modulus,unsigned format) {
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(cell>=m*n)return;
  namespace f=rns8::detail::mxfp_modular;int value=0;
  for(unsigned term=0;term<f::terms(format);++term)value=(value+planes[uint64_t(term)*m*n+cell]*f::weight(term,format))%modulus;
  if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;c[(cell/n)*ldc+cell%n]=int8_t(value);
}
extern "C" int rns8_hip_mxfp_capability(int* supported) {
  hipLaunchKernelGGL(rns8_mxfp_capability,dim3(1),dim3(1),0,nullptr,supported);return int(hipGetLastError());
}
extern "C" int rns8_hip_mxfp_modular(const int8_t* a,const int8_t* b,int8_t* c,int m,int n,int k,
    int lda,int ldb,int ldc,int modulus,unsigned* status,int32_t* planes,uint32_t format,
    const uint8_t* a_scales,const uint8_t* b_scales) {
  const auto grid=dim3((uint64_t(n)+15)/16,(uint64_t(m)+15)/16);
#define RNS8_MX_LAUNCH(F,A,B) case F: \
  if(planes)hipLaunchKernelGGL((rns8_mxfp_modular_kernel<A,B,true>),grid,dim3(64),0,nullptr,a,b,c,m,n,k,lda,ldb,ldc,modulus,status,planes,a_scales,b_scales); \
  else hipLaunchKernelGGL((rns8_mxfp_modular_kernel<A,B,false>),grid,dim3(64),0,nullptr,a,b,c,m,n,k,lda,ldb,ldc,modulus,status,planes,a_scales,b_scales);break
  switch(format){RNS8_MX_LAUNCH(3,4,4);RNS8_MX_LAUNCH(4,2,2);RNS8_MX_LAUNCH(5,3,3);RNS8_MX_LAUNCH(6,0,0);
    RNS8_MX_LAUNCH(7,0,4);RNS8_MX_LAUNCH(8,4,0);RNS8_MX_LAUNCH(9,2,4);RNS8_MX_LAUNCH(10,4,2);default:return int(hipErrorInvalidValue);}
#undef RNS8_MX_LAUNCH
  auto error=hipGetLastError();
  if(error==hipSuccess && planes){hipLaunchKernelGGL(rns8_mxfp_recombine,dim3((uint64_t(m)*n+255)/256),dim3(256),0,nullptr,planes,c,uint64_t(m),uint64_t(n),uint64_t(ldc),modulus,format);error=hipGetLastError();}
  return int(error);
}
#undef RNS8_MXFP_AVAILABLE
