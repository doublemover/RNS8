#include "backend_common/mfma_f64_layout.hpp"
#if (defined(__gfx90a__) || defined(__gfx942__)) && __has_builtin(__builtin_amdgcn_mfma_f64_16x16x4f64)
#define RNS8_WORD_FP64_MFMA 1
#else
#define RNS8_WORD_FP64_MFMA 0
#endif
__global__ void rns8_word_mod_fp64_mfma_capability_kernel(int* supported){*supported=RNS8_WORD_FP64_MFMA;}
__global__ __launch_bounds__(64) void rns8_word_mod_fp64_mfma_kernel(const uint64_t* a,const uint64_t* b,uint64_t* c,
    int m,int n,int k,int lda,int ldb,int ldc,uint64_t modulus,rns8::detail::word_mod::FloatingSchedule schedule){
#if RNS8_WORD_FP64_MFMA
  using namespace rns8::detail;using namespace word_mod;
  using f64x4=double __attribute__((ext_vector_type(4)));
  const auto ac=MfmaF64Layout::a(threadIdx.x),bc=MfmaF64Layout::b(threadIdx.x);
  const int64_t rowbase=int64_t(blockIdx.y)*16,colbase=int64_t(blockIdx.x)*16;
  uint64_t result[4]{};
  for(int64_t begin=0;begin<k;begin+=schedule.slab_k){
    const int64_t end=begin+schedule.slab_k<k?begin+schedule.slab_k:k;
    for(unsigned i=0;i<schedule.left_digits;++i)for(unsigned j=0;j<schedule.right_digits;++j){
      f64x4 sums{};
      for(int64_t q=begin;q<end;q+=4){uint64_t av=0,bv=0;
        if(rowbase+ac.row<m && q+ac.col<end)av=a[(rowbase+ac.row)*lda+q+ac.col];
        if(q+bc.row<end && colbase+bc.col<n)bv=b[(q+bc.row)*ldb+colbase+bc.col];
        for(unsigned d=0;d<i;++d)av/=schedule.alpha;
        for(unsigned d=0;d<j;++d)bv/=schedule.beta;
        // Canonical admission and the host exact integer bound guarantee each
        // product and every nonnegative accumulator are integers <=2^53.
        // Padding is exact zero, not an additional unbounded K contribution.
        sums=__builtin_amdgcn_mfma_f64_16x16x4f64(double(av%schedule.alpha),double(bv%schedule.beta),sums,0,0,0);
      }
      for(unsigned reg=0;reg<4;++reg){const uint64_t coefficient=static_cast<uint64_t>(sums[reg]);
        result[reg]=add(result[reg],multiply(coefficient%modulus,schedule.weights[4*i+j],modulus),modulus);}
    }
  }
  for(unsigned reg=0;reg<4;++reg){const auto z=MfmaF64Layout::c(threadIdx.x,reg);if(rowbase+z.row<m && colbase+z.col<n)c[(rowbase+z.row)*ldc+colbase+z.col]=result[reg];}
#else
  (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)lda;(void)ldb;(void)ldc;(void)modulus;(void)schedule;
#endif
}

// Exact expanded GEMM. Axis order is explicit: word-major groups whole matrix
// copies; cell-major groups digit tuples. Both form all (i,j) coefficient dots.
__global__ __launch_bounds__(64) void rns8_word_mod_fp64_concat_kernel(const uint64_t* a,const uint64_t* b,uint64_t* products,
    int m,int n,int k,int lda,int ldb,uint64_t modulus,rns8::detail::word_mod::FloatingSchedule schedule){
#if RNS8_WORD_FP64_MFMA
  using namespace rns8::detail;using namespace word_mod;using f64x4=double __attribute__((ext_vector_type(4)));
  const uint64_t rows=uint64_t(m)*schedule.left_digits,cols=uint64_t(n)*schedule.right_digits;
  const uint64_t rowbase=uint64_t(blockIdx.y)*16,colbase=uint64_t(blockIdx.x)*16;
  const auto ax=MfmaF64Layout::a(threadIdx.x),bx=MfmaF64Layout::b(threadIdx.x);
  const uint64_t er=rowbase+ax.row,ec=colbase+bx.col;
  const uint64_t ar=schedule.reserved==1?er%m:er/schedule.left_digits;
  const unsigned ai=schedule.reserved==1?unsigned(er/m):unsigned(er%schedule.left_digits);
  const uint64_t bc=schedule.reserved==1?ec%n:ec/schedule.right_digits;
  const unsigned bj=schedule.reserved==1?unsigned(ec/n):unsigned(ec%schedule.right_digits);
  uint64_t result[4]{};
  for(int64_t begin=0;begin<k;begin+=schedule.slab_k){const int64_t end=begin+schedule.slab_k<k?begin+schedule.slab_k:k;f64x4 sums{};
    for(int64_t q=begin;q<end;q+=4){uint64_t av=0,bv=0;
      if(er<rows && q+ax.col<end)av=a[ar*lda+q+ax.col];
      if(ec<cols && q+bx.row<end)bv=b[(q+bx.row)*ldb+bc];
      if(er<rows)for(unsigned i=0;i<ai;++i)av/=schedule.alpha;
      if(ec<cols)for(unsigned j=0;j<bj;++j)bv/=schedule.beta;
      sums=__builtin_amdgcn_mfma_f64_16x16x4f64(double(av%schedule.alpha),double(bv%schedule.beta),sums,0,0,0);
    }
    for(unsigned reg=0;reg<4;++reg){const auto z=MfmaF64Layout::c(threadIdx.x,reg);const auto r=rowbase+z.row,c=colbase+z.col;
      if(r<rows && c<cols){const unsigned i=schedule.reserved==1?unsigned(r/m):unsigned(r%schedule.left_digits),j=schedule.reserved==1?unsigned(c/n):unsigned(c%schedule.right_digits);
        result[reg]=add(result[reg],multiply(static_cast<uint64_t>(sums[reg])%modulus,schedule.weights[4*i+j],modulus),modulus);}}
  }
  for(unsigned reg=0;reg<4;++reg){const auto z=MfmaF64Layout::c(threadIdx.x,reg);const auto r=rowbase+z.row,c=colbase+z.col;if(r<rows && c<cols)products[r*cols+c]=result[reg];}
#else
  (void)a;(void)b;(void)products;(void)m;(void)n;(void)k;(void)lda;(void)ldb;(void)modulus;(void)schedule;
#endif
}
__global__ void rns8_word_mod_fp64_concat_merge(const uint64_t* products,uint64_t* c,int m,int n,int ldc,uint64_t modulus,rns8::detail::word_mod::FloatingSchedule s){
  using namespace rns8::detail::word_mod;const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(cell>=uint64_t(m)*n)return;
  const uint64_t row=cell/n,col=cell%n,cols=uint64_t(n)*s.right_digits;uint64_t result=0;
  for(unsigned i=0;i<s.left_digits;++i)for(unsigned j=0;j<s.right_digits;++j){const uint64_t r=s.reserved==1?uint64_t(i)*m+row:row*s.left_digits+i,q=s.reserved==1?uint64_t(j)*n+col:col*s.right_digits+j;result=add(result,products[r*cols+q],modulus);}
  c[row*ldc+col]=result;
}
extern "C" int rns8_hip_word_mod_fp64_concat(const void* a,const void* b,void* products,int m,int n,int k,int lda,int ldb,uint64_t modulus,rns8::detail::word_mod::FloatingSchedule s){
  hipLaunchKernelGGL(rns8_word_mod_fp64_concat_kernel,dim3((uint64_t(n)*s.right_digits+15)/16,(uint64_t(m)*s.left_digits+15)/16),dim3(64),0,nullptr,static_cast<const uint64_t*>(a),static_cast<const uint64_t*>(b),static_cast<uint64_t*>(products),m,n,k,lda,ldb,modulus,s);return int(hipGetLastError());
}
extern "C" int rns8_hip_word_mod_fp64_concat_merge(const void* products,void* c,int m,int n,int ldc,uint64_t modulus,rns8::detail::word_mod::FloatingSchedule s){
  hipLaunchKernelGGL(rns8_word_mod_fp64_concat_merge,dim3((uint64_t(m)*n+255)/256),dim3(256),0,nullptr,static_cast<const uint64_t*>(products),static_cast<uint64_t*>(c),m,n,ldc,modulus,s);return int(hipGetLastError());
}
extern "C" int rns8_hip_word_mod_fp64_mfma_capability(int* supported){hipLaunchKernelGGL(rns8_word_mod_fp64_mfma_capability_kernel,dim3(1),dim3(1),0,nullptr,supported);return int(hipGetLastError());}
extern "C" int rns8_hip_word_mod_fp64_mfma(const void* a,const void* b,void* c,int m,int n,int k,int lda,int ldb,int ldc,uint64_t modulus,rns8::detail::word_mod::FloatingSchedule schedule){
  hipLaunchKernelGGL(rns8_word_mod_fp64_mfma_kernel,dim3((uint64_t(n)+15)/16,(uint64_t(m)+15)/16),dim3(64),0,nullptr,static_cast<const uint64_t*>(a),static_cast<const uint64_t*>(b),static_cast<uint64_t*>(c),m,n,k,lda,ldb,ldc,modulus,schedule);return int(hipGetLastError());
}
#undef RNS8_WORD_FP64_MFMA
