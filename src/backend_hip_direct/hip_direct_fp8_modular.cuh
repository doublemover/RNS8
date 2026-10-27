#include "backend_common/fp8_modular.hpp"
#include "backend_common/mfma_i8_layout.hpp"
#if defined(__gfx942__) && __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8)
#define RNS8_FP8_MODULAR_AVAILABLE 1
#define RNS8_FP8_OCP 0
#define RNS8_FP8_WAVE 64
#define RNS8_FP8_REGS 4
#define RNS8_FP8_STEP 32
#elif defined(__gfx1201__) && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12)
#define RNS8_FP8_MODULAR_AVAILABLE 2
#define RNS8_FP8_OCP 1
#define RNS8_FP8_WAVE 32
#define RNS8_FP8_REGS 8
#define RNS8_FP8_STEP 16
#else
#define RNS8_FP8_MODULAR_AVAILABLE 0
#define RNS8_FP8_WAVE 64
#endif
__global__ void rns8_fp8_modular_capability_kernel(int* supported){*supported=RNS8_FP8_MODULAR_AVAILABLE;}
template<bool Materialized,bool Three>
__global__ __launch_bounds__(RNS8_FP8_WAVE) void rns8_fp8_modular_kernel(const int8_t* a,const int8_t* b,int8_t* c,
    int m,int n,int k,int lda,int ldb,int ldc,int modulus,unsigned* status,int32_t* planes) {
#if RNS8_FP8_MODULAR_AVAILABLE
  namespace f=rns8::detail::fp8_modular;
#if RNS8_FP8_OCP
  using Layout=f::Rdna4Layout;
#else
  using Layout=rns8::detail::MfmaI8Layout<32>;
#endif
  using f32vec=float __attribute__((ext_vector_type(RNS8_FP8_REGS)));
  constexpr unsigned terms=Three?3:4;
  const unsigned lane=threadIdx.x;const int64_t rowbase=int64_t(blockIdx.y)*16,colbase=int64_t(blockIdx.x)*16;
  int carry[RNS8_FP8_REGS]{},products[terms][RNS8_FP8_REGS]{};
  for(int64_t slab=0;slab<k;slab+=f::safe_k) {
    f32vec accum[terms]{};const int64_t end=slab+f::safe_k<k?slab+f::safe_k:k;
    for(int64_t inner=slab;inner<end;inner+=RNS8_FP8_STEP) {
      uint64_t ap[Three?3:2]{},bp[Three?3:2]{};
      for(unsigned element=0;element<8;++element) {
        const auto ac=Layout::a(lane,element),bc=Layout::b(lane,element);
        const int av=rowbase+ac.row<m && inner+ac.col<k?a[(rowbase+ac.row)*lda+inner+ac.col]:0;
        const int bv=inner+bc.row<k && colbase+bc.col<n?b[(inner+bc.row)*ldb+colbase+bc.col]:0;
        for(unsigned digit=0;digit<(Three?3u:2u);++digit) {
          ap[digit]|=uint64_t(f::encode(f::component(av,digit,Three),RNS8_FP8_OCP))<<(8*element);
          bp[digit]|=uint64_t(f::encode(f::component(bv,digit,Three),RNS8_FP8_OCP))<<(8*element);
        }
      }
      for(unsigned term=0;term<terms;++term) {
        const auto left=ap[Three?term:term/2],right=bp[Three?term:term%2];
#if RNS8_FP8_OCP
        using i32x2=int __attribute__((ext_vector_type(2)));
        accum[term]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            __builtin_bit_cast(i32x2,left),__builtin_bit_cast(i32x2,right),accum[term]);
#else
        accum[term]=__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            __builtin_bit_cast(long long,left),__builtin_bit_cast(long long,right),accum[term],0,0,0);
#endif
      }
    }
    for(unsigned reg=0;reg<RNS8_FP8_REGS;++reg)for(unsigned term=0;term<terms;++term) {
      const float partial=accum[term][reg];
      // Check before conversion: NaN, infinity, nonintegral values or a bound
      // contradiction invalidate publication rather than invoking float->int UB.
      int integer=0;if(!f::integer_partial(partial,integer,f::partial_bound(term,Three))){atomicOr(status,1u);continue;}
      if constexpr(Materialized)products[term][reg]=(products[term][reg]+integer%modulus)%modulus;
      else carry[reg]=(carry[reg]+(integer%modulus)*f::weight(term,Three))%modulus;
    }
  }
  for(unsigned reg=0;reg<RNS8_FP8_REGS;++reg) {
    const auto cell=Layout::c(lane,reg);const int64_t row=rowbase+cell.row,col=colbase+cell.col;
    if(row<m && col<n){if constexpr(Materialized)for(unsigned term=0;term<terms;++term)planes[(uint64_t(term)*m+row)*n+col]=products[term][reg];
      else {int value=carry[reg];if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;c[row*ldc+col]=int8_t(value);}}
  }
#else
  (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)lda;(void)ldb;(void)ldc;(void)modulus;(void)status;(void)planes;
#endif
}
extern "C" int rns8_hip_fp8_modular_capability(int* supported) {
  hipLaunchKernelGGL(rns8_fp8_modular_capability_kernel,dim3(1),dim3(1),0,nullptr,supported);return int(hipGetLastError());
}
__global__ void rns8_fp8_modular_recombine(const int32_t* planes,int8_t* c,uint64_t m,uint64_t n,uint64_t ldc,int modulus,bool three) {
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(cell>=m*n)return;
  namespace f=rns8::detail::fp8_modular;const unsigned terms=three?3:4;int value=0;
  for(unsigned term=0;term<terms;++term)value=(value+planes[uint64_t(term)*m*n+cell]*f::weight(term,three))%modulus;
  if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;c[(cell/n)*ldc+cell%n]=int8_t(value);
}
template<bool Three> int launch_fp8(const int8_t* a,const int8_t* b,int8_t* c,int m,int n,int k,int lda,int ldb,int ldc,int modulus,unsigned* status,int32_t* planes,uint32_t format) {
  if(planes)hipLaunchKernelGGL((rns8_fp8_modular_kernel<true,Three>),dim3((uint64_t(n)+15)/16,(uint64_t(m)+15)/16),dim3(format==2?32:64),0,nullptr,a,b,c,m,n,k,lda,ldb,ldc,modulus,status,planes);
  else hipLaunchKernelGGL((rns8_fp8_modular_kernel<false,Three>),dim3((uint64_t(n)+15)/16,(uint64_t(m)+15)/16),dim3(format==2?32:64),0,nullptr,a,b,c,m,n,k,lda,ldb,ldc,modulus,status,planes);
  auto error=hipGetLastError();
  if(error==hipSuccess && planes){hipLaunchKernelGGL(rns8_fp8_modular_recombine,dim3((uint64_t(m)*n+255)/256),dim3(256),0,nullptr,planes,c,uint64_t(m),uint64_t(n),uint64_t(ldc),modulus,Three);error=hipGetLastError();}
  return int(error);
}
extern "C" int rns8_hip_fp8_modular(const int8_t* a,const int8_t* b,int8_t* c,int m,int n,int k,int lda,int ldb,int ldc,int modulus,unsigned* status,int32_t* planes,bool three,uint32_t format) {
  return three?launch_fp8<true>(a,b,c,m,n,k,lda,ldb,ldc,modulus,status,planes,format):launch_fp8<false>(a,b,c,m,n,k,lda,ldb,ldc,modulus,status,planes,format);
}
#undef RNS8_FP8_MODULAR_AVAILABLE
#undef RNS8_FP8_OCP
#undef RNS8_FP8_WAVE
#undef RNS8_FP8_REGS
#undef RNS8_FP8_STEP
