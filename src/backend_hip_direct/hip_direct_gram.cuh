#include "backend_common/gram_plan.hpp"
__device__ __forceinline__ int8_t rns8_gram_load(const int8_t* x,int64_t plane,int64_t index,
    int64_t q,int rows,int cols,int ld,bool transposed) {
  return x[transposed?(plane*rows+q)*ld+index:(plane*rows+index)*ld+q];
}
// Coalesced lower-triangle stores through a padded transpose tile. The upper
// triangle is complete before this separately launched same-stream kernel.
__global__ void rns8_gram_mirror_kernel(int8_t* c,const rns8::detail::GramTile* jobs,int n,int ld) {
  __shared__ int8_t tile[16][17];const auto job=jobs[blockIdx.x];const int r=threadIdx.x/16,t=threadIdx.x%16,plane=blockIdx.y;
  const int row=int(job.row)*16+r,col=int(job.col)*16+t;
  tile[r][t]=row<n && col<n && row<=col?c[(int64_t(plane)*n+row)*ld+col]:0;
  __syncthreads();const int dst_row=int(job.col)*16+r,dst_col=int(job.row)*16+t;
  if(dst_row<n && dst_col<n && dst_row>dst_col)c[(int64_t(plane)*n+dst_row)*ld+dst_col]=tile[t][r];
}
template<bool Wmma>
__global__ void rns8_gram_kernel(const int8_t* x,int8_t* c,const rns8::detail::GramTile* jobs,
                                int rows,int cols,int ld,int ldc,int modulus,bool transposed,bool separate_mirror) {
  const auto job=jobs[blockIdx.x];
  const int64_t n=transposed?cols:rows,k=transposed?rows:cols,plane=blockIdx.y;
  const int qmod=modulus?modulus:rns8_default_moduli_device[plane];
  if constexpr(!Wmma) {
    const int64_t row=int64_t(job.row)*16+threadIdx.x/16,col=int64_t(job.col)*16+threadIdx.x%16;
    if(row>=n || col>=n || row>col) return;
    int sum=0;
    for(int64_t q=0;q<k;++q)
      sum=(sum+int(rns8_gram_load(x,plane,row,q,rows,cols,ld,transposed))*
                   int(rns8_gram_load(x,plane,col,q,rows,cols,ld,transposed)))%qmod;
    if(sum<0) sum+=qmod;if(sum>=(qmod+1)/2) sum-=qmod;
    c[(plane*n+row)*ldc+col]=int8_t(sum);
    if(row!=col && !separate_mirror) c[(plane*n+col)*ldc+row]=int8_t(sum);
  } else {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
    using fragment=int __attribute__((ext_vector_type(4)));
    using accumulator=int __attribute__((ext_vector_type(8)));
    const int lane=threadIdx.x,local=lane&15;
    const int64_t ar=int64_t(job.row)*16+local,bc=int64_t(job.col)*16+local;
    accumulator acc{};
    for(int64_t base=0;base<k;base+=16) {
      fragment av{},bv{};
      for(int word=0;word<4;++word) {
        uint32_t aw=0,bw=0;
        for(int byte=0;byte<4;++byte) {
          const int64_t q=base+word*4+byte;
          if(ar<n && q<k) aw|=uint32_t(uint8_t(rns8_gram_load(x,plane,ar,q,rows,cols,ld,transposed)))<<(8*byte);
          if(bc<n && q<k) bw|=uint32_t(uint8_t(rns8_gram_load(x,plane,bc,q,rows,cols,ld,transposed)))<<(8*byte);
        }
        av[word]=__builtin_bit_cast(int,aw);bv[word]=__builtin_bit_cast(int,bw);
      }
      acc=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,av,true,bv,acc,false);
      for(int reg=0;reg<8;++reg) acc[reg]%=qmod;
    }
    for(int reg=0;reg<8;++reg) {
      const int64_t row=int64_t(job.row)*16+2*reg+(lane>>4),col=bc;
      if(row>=n || col>=n || row>col) continue;
      int sum=acc[reg];if(sum<0) sum+=qmod;if(sum>=(qmod+1)/2) sum-=qmod;
      c[(plane*n+row)*ldc+col]=int8_t(sum);
      if(row!=col && !separate_mirror) c[(plane*n+col)*ldc+row]=int8_t(sum);
    }
#endif
  }
}
extern "C" int rns8_hip_gram(const int8_t* x,int8_t* c,const rns8::detail::GramTile* jobs,
    int count,int rows,int cols,int ld,int ldc,int prefix,int modulus,int transposed,int wmma,int separate_mirror) {
  const dim3 grid(count,prefix);
  if(wmma) hipLaunchKernelGGL((rns8_gram_kernel<true>),grid,dim3(32),0,nullptr,
      x,c,jobs,rows,cols,ld,ldc,modulus,transposed!=0,separate_mirror!=0);
  else hipLaunchKernelGGL((rns8_gram_kernel<false>),grid,dim3(256),0,nullptr,
      x,c,jobs,rows,cols,ld,ldc,modulus,transposed!=0,separate_mirror!=0);
  auto status=hipGetLastError();if(status==hipSuccess && separate_mirror){hipLaunchKernelGGL(rns8_gram_mirror_kernel,grid,dim3(256),0,nullptr,c,jobs,transposed?cols:rows,ldc);status=hipGetLastError();}
  return int(status);
}

#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
#define RNS8_GRAM_NATIVE_WMMA 1
#else
#define RNS8_GRAM_NATIVE_WMMA 0
#endif
// One physical raw panel on the diagonal; two elsewhere. Both logical operand
// roles are encoded from those retained raw cells with distinct WMMA mappings.
template<bool Wmma>
__global__ void rns8_gram_native_kernel(const int64_t* x,int8_t* c,const rns8::detail::GramTile* jobs,
    int rows,int cols,int ld,int ldc,bool transposed,bool separate_mirror){
  const auto job=jobs[blockIdx.x];const int plane=blockIdx.y,modulus=rns8_default_moduli_device[plane];
  const auto reducer=rns8::detail::finite_u8::native64_reducer(uint16_t(modulus));
  const int64_t order=transposed?cols:rows,k=transposed?rows:cols;
  __shared__ int64_t panels[2][256];
  const int lane=threadIdx.x,local=lane&15;const bool diagonal=job.row==job.col;
  const int64_t scalar_row=int64_t(job.row)*16+lane/16,scalar_col=int64_t(job.col)*16+lane%16;
  int scalar_sum=0;
#if RNS8_GRAM_NATIVE_WMMA
  using fragment=int __attribute__((ext_vector_type(4)));using accumulator=int __attribute__((ext_vector_type(8)));accumulator acc{};
#endif
  for(int64_t base=0;base<k;base+=16){
    for(int index=lane;index<256;index+=blockDim.x){const int local_row=index/16;const int64_t q=base+index%16;
      for(int role=0;role<(diagonal?1:2);++role){const int64_t row=int64_t(role?job.col:job.row)*16+local_row;
        panels[role][index]=row<order && q<k ? x[transposed?q*ld+row:row*ld+q]:0;}}
    __syncthreads();
    if constexpr(!Wmma){
      if(scalar_row<order && scalar_col<order && scalar_row<=scalar_col)
        for(int q=0;q<16;++q){const int a=rns8_native_wmma_center(panels[0][(lane/16)*16+q],reducer),b=rns8_native_wmma_center(panels[diagonal?0:1][(lane%16)*16+q],reducer);scalar_sum=(scalar_sum+a*b)%modulus;}
    }else{
#if RNS8_GRAM_NATIVE_WMMA
      fragment av{},bv{};for(int word=0;word<4;++word){uint32_t aa=0,bb=0;for(int byte=0;byte<4;++byte){const int q=word*4+byte;
        aa|=uint32_t(uint8_t(rns8_native_wmma_center(panels[0][local*16+q],reducer)))<<(8*byte);
        bb|=uint32_t(uint8_t(rns8_native_wmma_center(panels[diagonal?0:1][local*16+q],reducer)))<<(8*byte);}
        av[word]=__builtin_bit_cast(int,aa);bv[word]=__builtin_bit_cast(int,bb);}
      acc=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,av,true,bv,acc,false);
      for(int r=0;r<8;++r)acc[r]%=modulus;
#endif
    }
    __syncthreads();
  }
  if constexpr(!Wmma){if(scalar_row<order && scalar_col<order && scalar_row<=scalar_col){if(scalar_sum<0)scalar_sum+=modulus;if(scalar_sum>=(modulus+1)/2)scalar_sum-=modulus;
    c[(int64_t(plane)*order+scalar_row)*ldc+scalar_col]=int8_t(scalar_sum);if(scalar_row!=scalar_col && !separate_mirror)c[(int64_t(plane)*order+scalar_col)*ldc+scalar_row]=int8_t(scalar_sum);}}
  else{
#if RNS8_GRAM_NATIVE_WMMA
    const int64_t col=int64_t(job.col)*16+local;
    for(int reg=0;reg<8;++reg){const int64_t row=int64_t(job.row)*16+2*reg+(lane>>4);if(row>=order || col>=order || row>col)continue;
      int value=acc[reg];if(value<0)value+=modulus;if(value>=(modulus+1)/2)value-=modulus;
      c[(int64_t(plane)*order+row)*ldc+col]=int8_t(value);if(row!=col && !separate_mirror)c[(int64_t(plane)*order+col)*ldc+row]=int8_t(value);}
#endif
  }
}
__global__ void rns8_gram_transpose_kernel(const int8_t* source,int8_t* target,int rows,int cols,int prefix){
  const uint64_t cell=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,plane_cells=uint64_t(rows)*cols;
  if(cell>=plane_cells*prefix)return;const auto plane=cell/plane_cells,index=cell%plane_cells,row=index/cols,col=index%cols;
  target[plane*plane_cells+col*rows+row]=source[cell];
}
__global__ void rns8_gram_export_once_kernel(const int8_t* residues,uint64_t* output,const rns8::detail::GramTile* jobs,
    int order,int prefix,int limbs,int* status){
  const auto job=jobs[blockIdx.x];const int row=int(job.row)*16+threadIdx.x/16,col=int(job.col)*16+threadIdx.x%16;
  if(row>=order || col>=order || row>col)return;const int cell=row*order+col;
  rns8_u192_device value{},product{};rns8_reconstruct_canonical_wide_device(residues,cell,order*order,prefix,&value,&product);
  const bool negative=rns8_u192_centered_is_negative_device(value,product);
  const bool fits=negative?rns8_u192_signed_negative_magnitude_fits_limbs_device(rns8_u192_sub_device(product,value),limbs):rns8_u192_signed_positive_fits_limbs_device(value,limbs);
  if(!fits){atomicCAS(status,0,5);return;}
  rns8_export_exact_wide_signed_limbs_device(output,cell,value,product,limbs,status);
  // Only this independent cell owns its mirror. Reconstruction is performed once.
  if(row!=col)for(int limb=0;limb<limbs;++limb)output[(int64_t(col)*order+row)*limbs+limb]=output[int64_t(cell)*limbs+limb];
}
extern "C" int rns8_hip_gram_native(const void* x,void* c,const rns8::detail::GramTile* jobs,int count,int rows,int cols,int ld,int ldc,int prefix,int transposed,int wmma,int separate_mirror){
  if(wmma)hipLaunchKernelGGL((rns8_gram_native_kernel<true>),dim3(count,prefix),dim3(32),0,nullptr,static_cast<const int64_t*>(x),static_cast<int8_t*>(c),jobs,rows,cols,ld,ldc,transposed!=0,separate_mirror!=0);
  else hipLaunchKernelGGL((rns8_gram_native_kernel<false>),dim3(count,prefix),dim3(256),0,nullptr,static_cast<const int64_t*>(x),static_cast<int8_t*>(c),jobs,rows,cols,ld,ldc,transposed!=0,separate_mirror!=0);
  auto status=hipGetLastError();if(status==hipSuccess && separate_mirror){hipLaunchKernelGGL(rns8_gram_mirror_kernel,dim3(count,prefix),dim3(256),0,nullptr,static_cast<int8_t*>(c),jobs,transposed?cols:rows,ldc);status=hipGetLastError();}
  return int(status);
}
extern "C" int rns8_hip_gram_transpose(const void* x,void* target,int rows,int cols,int prefix){hipLaunchKernelGGL(rns8_gram_transpose_kernel,dim3((uint64_t(rows)*cols*prefix+255)/256),dim3(256),0,nullptr,static_cast<const int8_t*>(x),static_cast<int8_t*>(target),rows,cols,prefix);return int(hipGetLastError());}
extern "C" int rns8_hip_gram_export_once(const void* x,void* target,const rns8::detail::GramTile* jobs,int count,int order,int prefix,int limbs,int* status){hipLaunchKernelGGL(rns8_gram_export_once_kernel,dim3(count),dim3(256),0,nullptr,static_cast<const int8_t*>(x),static_cast<uint64_t*>(target),jobs,order,prefix,limbs,status);return int(hipGetLastError());}

#undef RNS8_GRAM_NATIVE_WMMA
