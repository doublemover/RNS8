#include "backend_common/fp8_modular.hpp"
#include "backend_common/mfma_i8_layout.hpp"
#if defined(__gfx942__) && __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8)
#define RNS8_FP8_MODULAR_AVAILABLE 1
#else
#define RNS8_FP8_MODULAR_AVAILABLE 0
#endif
__global__ void rns8_fp8_modular_capability_kernel(int* supported){*supported=RNS8_FP8_MODULAR_AVAILABLE;}
template<bool Materialized>
__global__ __launch_bounds__(64) void rns8_fp8_modular_kernel(const int8_t* a,const int8_t* b,int8_t* c,
    int m,int n,int k,int lda,int ldb,int ldc,int modulus,unsigned* status,int32_t* planes) {
#if RNS8_FP8_MODULAR_AVAILABLE
  using Layout=rns8::detail::MfmaI8Layout<32>;namespace f=rns8::detail::fp8_modular;
  using f32x4=float __attribute__((ext_vector_type(4)));
  const unsigned lane=threadIdx.x;const int64_t rowbase=int64_t(blockIdx.y)*16,colbase=int64_t(blockIdx.x)*16;
  int carry[4]{},products[4][4]{};
  for(int64_t slab=0;slab<k;slab+=f::safe_k) {
    f32x4 accum[4]{};const int64_t end=slab+f::safe_k<k?slab+f::safe_k:k;
    for(int64_t inner=slab;inner<end;inner+=32) {
      uint64_t ap[2]{},bp[2]{};
      for(unsigned element=0;element<8;++element) {
        const auto ac=Layout::a(lane,element),bc=Layout::b(lane,element);
        const int av=rowbase+ac.row<m && inner+ac.col<k?a[(rowbase+ac.row)*lda+inner+ac.col]:0;
        const int bv=inner+bc.row<k && colbase+bc.col<n?b[(inner+bc.row)*ldb+colbase+bc.col]:0;
        ap[0]|=uint64_t(f::encode(f::low(av)))<<(8*element);ap[1]|=uint64_t(f::encode(f::high(av)))<<(8*element);
        bp[0]|=uint64_t(f::encode(f::low(bv)))<<(8*element);bp[1]|=uint64_t(f::encode(f::high(bv)))<<(8*element);
      }
      for(unsigned x=0;x<2;++x)for(unsigned y=0;y<2;++y)
        accum[x*2+y]=__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(__builtin_bit_cast(long long,ap[x]),
            __builtin_bit_cast(long long,bp[y]),accum[x*2+y],0,0,0);
    }
    const int weights[4]={1,16,16,256};
    for(unsigned reg=0;reg<4;++reg)for(unsigned term=0;term<4;++term) {
      const float partial=accum[term][reg];
      // Check before conversion: NaN, infinity, nonintegral values or a bound
      // contradiction invalidate publication rather than invoking float->int UB.
      int integer=0;if(!f::integer_partial(partial,integer)){atomicOr(status,1u);continue;}
      if constexpr(Materialized)products[term][reg]=(products[term][reg]+integer%modulus)%modulus;
      else carry[reg]=(carry[reg]+(integer%modulus)*weights[term])%modulus;
    }
  }
  for(unsigned reg=0;reg<4;++reg) {
    const auto cell=Layout::c(lane,reg);const int64_t row=rowbase+cell.row,col=colbase+cell.col;
    if(row<m && col<n){if constexpr(Materialized)for(unsigned term=0;term<4;++term)planes[(uint64_t(term)*m+row)*n+col]=products[term][reg];
      else {int value=carry[reg];if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;c[row*ldc+col]=int8_t(value);}}
  }
#else
  (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)lda;(void)ldb;(void)ldc;(void)modulus;(void)status;(void)planes;
#endif
}
extern "C" int rns8_hip_fp8_modular_capability(int* supported) {
  hipLaunchKernelGGL(rns8_fp8_modular_capability_kernel,dim3(1),dim3(1),0,nullptr,supported);return int(hipGetLastError());
}
__global__ void rns8_fp8_modular_recombine(const int32_t* planes,int8_t* c,uint64_t m,uint64_t n,uint64_t ldc,int modulus) {
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(cell>=m*n)return;
  const int weights[4]={1,16,16,256};int value=0;
  for(unsigned term=0;term<4;++term)value=(value+planes[uint64_t(term)*m*n+cell]*weights[term])%modulus;
  if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;c[(cell/n)*ldc+cell%n]=int8_t(value);
}
extern "C" int rns8_hip_fp8_modular(const int8_t* a,const int8_t* b,int8_t* c,int m,int n,int k,int lda,int ldb,int ldc,int modulus,unsigned* status,int32_t* planes) {
  if(planes)hipLaunchKernelGGL((rns8_fp8_modular_kernel<true>),dim3((uint64_t(n)+15)/16,(uint64_t(m)+15)/16),dim3(64),0,nullptr,a,b,c,m,n,k,lda,ldb,ldc,modulus,status,planes);
  else hipLaunchKernelGGL((rns8_fp8_modular_kernel<false>),dim3((uint64_t(n)+15)/16,(uint64_t(m)+15)/16),dim3(64),0,nullptr,a,b,c,m,n,k,lda,ldb,ldc,modulus,status,planes);
  auto error=hipGetLastError();
  if(error==hipSuccess && planes){hipLaunchKernelGGL(rns8_fp8_modular_recombine,dim3((uint64_t(m)*n+255)/256),dim3(256),0,nullptr,planes,c,uint64_t(m),uint64_t(n),uint64_t(ldc),modulus);error=hipGetLastError();}
  return int(error);
}
#undef RNS8_FP8_MODULAR_AVAILABLE
