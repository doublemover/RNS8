// Generic scalar control: every logical coordinate and address product is u64.
// Fixed grid avoids narrowing a shape-derived grid dimension.
__global__ void rns8_wide_gemm_kernel(const int8_t* a,const int8_t* b,int8_t* c,
    uint64_t m,uint64_t n,uint64_t k,uint64_t lda,uint64_t ldb,uint64_t ldc,uint64_t count) {
  const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
  for (uint64_t logical=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; logical<count; logical+=stride) {
    const uint64_t col=logical%n, row=(logical/n)%m, plane=logical/(m*n);
    const int modulus=rns8_default_moduli_device[plane];
    int sum=0;
    for (uint64_t base=0; base<k;) {
      const uint64_t length=k-base<65536 ? k-base : 65536;
      for (uint64_t offset=0; offset<length; ++offset) {
        const uint64_t q=base+offset;
        sum+=int(a[(plane*m+row)*lda+q])*int(b[(plane*k+q)*ldb+col]);
      }
      sum%=modulus;
      base+=length;
    }
    if (sum<0) sum+=modulus;
    if (sum>=(modulus+1)/2) sum-=modulus;
    c[(plane*m+row)*ldc+col]=static_cast<int8_t>(sum);
  }
}
