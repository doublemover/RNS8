__device__ __constant__ rns8::detail::rank48::slp::Plan rns8_rank48_slp_u=rns8::detail::rank48::slp::u;
__device__ __constant__ rns8::detail::rank48::slp::Plan rns8_rank48_slp_v=rns8::detail::rank48::slp::v;
__device__ __constant__ rns8::detail::rank48::slp::Plan rns8_rank48_slp_w=rns8::detail::rank48::slp::w;
template<bool Lhs>
__global__ void rns8_rank48_slp_transform_kernel(const int8_t* source,int8_t* output,
    int rows,int cols,int ld,int hr,int hc,int prefix,int selected) {
  const int64_t cell=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,cells=int64_t(hr)*hc;
  if(cell>=cells)return;
  const int plane=blockIdx.y,q=rns8_default_moduli_device[plane];
  const auto& plan=Lhs?rns8_rank48_slp_u:rns8_rank48_slp_v;
  int8_t slots[rns8::detail::rank48::slp::capacity]{};
  if(q&1) {
    for(int block=0;block<16;++block){
      const int64_t row=cell/hc+int64_t(block/4)*hr,col=cell%hc+int64_t(block%4)*hc;
      if(row<rows && col<cols)slots[block]=rns8_strassen_center(source[(int64_t(plane)*rows+row)*ld+col],q);
    }
    rns8::detail::rank48::slp::evaluate(plan,slots,q);
  }
  if(selected>=0)output[int64_t(plane)*cells+cell]=slots[plan.results[selected]];
  else for(int term=0;term<48;++term)output[(int64_t(term)*prefix+plane)*cells+cell]=slots[plan.results[term]];
}
__global__ void rns8_rank48_slp_combine_kernel(const int8_t* products,int8_t* stage,
    int m,int n,int hm,int hn,int prefix) {
  const int64_t cell=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,cells=int64_t(hm)*hn;
  if(cell>=cells)return;
  const int plane=blockIdx.y,q=rns8_default_moduli_device[plane];
  int8_t slots[rns8::detail::rank48::slp::capacity]{};
  if(q&1){
    for(int term=0;term<48;++term)slots[term]=products[(int64_t(term)*prefix+plane)*cells+cell];
    rns8::detail::rank48::slp::evaluate(rns8_rank48_slp_w,slots,q);
  }
  for(int block=0;block<16;++block){
    const int64_t row=cell/hn+int64_t(block/4)*hm,col=cell%hn+int64_t(block%4)*hn;
    if(row<m && col<n)stage[(int64_t(plane)*m+row)*n+col]=slots[rns8_rank48_slp_w.results[block]];
  }
}
