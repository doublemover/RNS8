#include "backend_common/wrap64_signed_radix.hpp"
#include "backend_common/mfma_i8_layout.hpp"
#if defined(__gfx90a__) && __has_builtin(__builtin_amdgcn_mfma_i32_16x16x16i8)
#define RNS8_WRAP_MFMA_K 16
#elif defined(__gfx942__) && __has_builtin(__builtin_amdgcn_mfma_i32_16x16x32_i8)
#define RNS8_WRAP_MFMA_K 32
#elif defined(__gfx950__) && __has_builtin(__builtin_amdgcn_mfma_i32_16x16x64_i8)
#define RNS8_WRAP_MFMA_K 64
#else
#define RNS8_WRAP_MFMA_K 0
#endif
__global__ void rns8_wrap_mfma_capability_kernel(int* width){*width=RNS8_WRAP_MFMA_K;}
__global__ __launch_bounds__(64) void rns8_wrap_signed_radix_mfma_kernel(const uint8_t* a,const uint8_t* b,uint8_t* c,int m,int n,int k){
#if RNS8_WRAP_MFMA_K
  using Layout=rns8::detail::MfmaI8Layout<RNS8_WRAP_MFMA_K>;
  using i32x4=int __attribute__((ext_vector_type(4)));
  const unsigned lane=threadIdx.x;const int64_t rowbase=int64_t(blockIdx.y)*16,colbase=int64_t(blockIdx.x)*16;
  uint64_t result[4]{};
  for(int64_t base=0;base<k;base+=RNS8_WRAP_MFMA_K){
    // Canonical cells loaded/re-coded exactly once per instruction tile. Every
    // lane owns distinct A/B coordinates; MFMA performs the wave communication.
    uint64_t av[Layout::input_bytes_per_lane]{},bv[Layout::input_bytes_per_lane]{};
    for(unsigned e=0;e<Layout::input_bytes_per_lane;++e){const auto x=Layout::a(lane,e),y=Layout::b(lane,e);uint64_t aa=0,bb=0;
      if(rowbase+x.row<m && base+x.col<k)for(unsigned byte=0;byte<8;++byte)aa|=uint64_t(a[((rowbase+x.row)*k+base+x.col)*8+byte])<<(8*byte);
      if(base+y.row<k && colbase+y.col<n)for(unsigned byte=0;byte<8;++byte)bb|=uint64_t(b[((base+y.row)*n+colbase+y.col)*8+byte])<<(8*byte);
      av[e]=rns8::detail::wrap64_signed_recode(aa);bv[e]=rns8::detail::wrap64_signed_recode(bb);}
    for(unsigned i=0;i<8;++i)for(unsigned j=0;i+j<8;++j){
      uint32_t aw[4]{},bw[4]{};for(unsigned e=0;e<Layout::input_bytes_per_lane;++e){aw[e/4]|=uint32_t((av[e]>>(8*i))&255)<<(8*(e%4));bw[e/4]|=uint32_t((bv[e]>>(8*j))&255)<<(8*(e%4));}
      i32x4 acc{};
#if RNS8_WRAP_MFMA_K == 16
      acc=__builtin_amdgcn_mfma_i32_16x16x16i8(__builtin_bit_cast(int,aw[0]),__builtin_bit_cast(int,bw[0]),acc,0,0,0);
#elif RNS8_WRAP_MFMA_K == 32
      const uint64_t ap=uint64_t(aw[0])+(uint64_t(aw[1])<<32),bp=uint64_t(bw[0])+(uint64_t(bw[1])<<32);
      acc=__builtin_amdgcn_mfma_i32_16x16x32_i8(__builtin_bit_cast(long long,ap),__builtin_bit_cast(long long,bp),acc,0,0,0);
#elif RNS8_WRAP_MFMA_K == 64
      i32x4 ap{},bp{};for(unsigned w=0;w<4;++w){ap[w]=__builtin_bit_cast(int,aw[w]);bp[w]=__builtin_bit_cast(int,bw[w]);}
      acc=__builtin_amdgcn_mfma_i32_16x16x64_i8(ap,bp,acc,0,0,0);
#endif
      // Each accumulator was reset; at most64*16384 <2^31. Fold each pair
      // immediately, never sum a whole diagonal in signed INT32.
      for(unsigned reg=0;reg<4;++reg)result[reg]+=uint64_t(int64_t(acc[reg]))<<(8*(i+j));
    }
  }
  for(unsigned reg=0;reg<4;++reg){const auto cell=Layout::c(lane,reg);const int64_t row=rowbase+cell.row,col=colbase+cell.col;if(row<m && col<n)for(unsigned byte=0;byte<8;++byte)c[(row*n+col)*8+byte]=uint8_t(result[reg]>>(8*byte));}
#else
  (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;
#endif
}
extern "C" int rns8_hip_wrap_mfma_capability(int* width){hipLaunchKernelGGL(rns8_wrap_mfma_capability_kernel,dim3(1),dim3(1),0,nullptr,width);return int(hipGetLastError());}
extern "C" int rns8_hip_wrap_signed_radix_mfma(const uint8_t* a,const uint8_t* b,uint8_t* c,int m,int n,int k){hipLaunchKernelGGL(rns8_wrap_signed_radix_mfma_kernel,dim3((uint64_t(n)+15)/16,(uint64_t(m)+15)/16),dim3(64),0,nullptr,a,b,c,m,n,k);return int(hipGetLastError());}
#undef RNS8_WRAP_MFMA_K

// Same canonical-byte boundary, independent native-U64 vector-ALU control.
// One lane owns one output; there is no digit convolution or matrix instruction.
__global__ void rns8_wrap_native_u64_kernel(const uint8_t* a,const uint8_t* b,uint8_t* c,int m,int n,int k){
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(cell>=uint64_t(m)*n)return;
  const uint64_t row=cell/n,col=cell%n;uint64_t sum=0;
  for(int t=0;t<k;++t){uint64_t av=0,bv=0;
    for(unsigned byte=0;byte<8;++byte){av|=uint64_t(a[(row*k+t)*8+byte])<<(8*byte);bv|=uint64_t(b[(uint64_t(t)*n+col)*8+byte])<<(8*byte);}
    sum+=av*bv;
  }
  for(unsigned byte=0;byte<8;++byte)c[cell*8+byte]=uint8_t(sum>>(8*byte));
}
extern "C" int rns8_hip_wrap_native_u64(const uint8_t* a,const uint8_t* b,uint8_t* c,int m,int n,int k){
  hipLaunchKernelGGL(rns8_wrap_native_u64_kernel,dim3((uint64_t(m)*n+255)/256),dim3(256),0,nullptr,a,b,c,m,n,k);return int(hipGetLastError());
}
