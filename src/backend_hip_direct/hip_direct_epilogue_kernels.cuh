// Small/skinny scalar baseline, not a matrix-engine claim. Each output has one
// owner. Coefficients are lowered on the host once per channel; zero channel
// coefficients uniformly omit their dot or addend reads without weakening API
// currentness/ownership/proof admission.
__global__ void rns8_gemm_rns_axpby_kernel(
    const int8_t* a,const int8_t* b,const int8_t* d,int8_t* c,
    int m,int n,int k,int lda,int ldb,int ldd,int ldc,rns8::detail::ResidueAxpbyTable coefficients) {
  const int64_t cell=int64_t{blockIdx.x}*blockDim.x+threadIdx.x;
  if(cell>=int64_t{m}*n)return;
  const int plane=blockIdx.y;
  const int64_t row=cell/n,col=cell%n;
  const auto coefficient=coefficients.channel[plane];
  int8_t product=0,addend=0;
  if(coefficient.alpha!=0)
    product=rns8::detail::centered_residue_dot(
        a+int64_t{plane}*m*lda+row*lda,b+int64_t{plane}*k*ldb+col,k,ldb,coefficient.modulus);
  if(coefficient.beta!=0)addend=d[(int64_t{plane}*m+row)*ldd+col];
  c[(int64_t{plane}*m+row)*ldc+col]=rns8::detail::centered_residue_axpby_reduced(product,addend,coefficient);
}

__global__ void rns8_axpby_residues_kernel(
    const int8_t* x,const int8_t* d,int8_t* c,
    int m,int n,int ldx,int ldd,int ldc,rns8::detail::ResidueAxpbyTable coefficients) {
  const int64_t cell=int64_t{blockIdx.x}*blockDim.x+threadIdx.x;
  if(cell>=int64_t{m}*n)return;
  const int64_t row=cell/n,col=cell%n,plane=blockIdx.y;
  const auto coefficient=coefficients.channel[plane];
  const int8_t product=coefficient.alpha?x[(plane*m+row)*ldx+col]:int8_t{0};
  const int8_t addend=coefficient.beta?d[(plane*m+row)*ldd+col]:int8_t{0};
  c[(plane*m+row)*ldc+col]=rns8::detail::centered_residue_axpby_reduced(product,addend,coefficient);
}
