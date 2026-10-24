__device__ __forceinline__ void rns8_block_sparse_scalar_tile(
    const int8_t *a, const uint32_t *ao, const uint32_t *ai, const int8_t *b,
    const uint32_t *bo, const uint32_t *bi, int8_t *c, int m, int n, int k,
    int ldc, int prefix, int tile_cols) {
  __shared__ int8_t at[1024], bt[1024];
  const int lane = threadIdx.x, lr = lane / 16, lc = lane % 16;
  const int64_t rt = blockIdx.x / tile_cols, ct = blockIdx.x % tile_cols;
  const int64_t row = rt * 16 + lr, col = ct * 16 + lc;
  const int plane = blockIdx.y, modulus = rns8_default_moduli_device[plane];
  uint32_t x = ao[rt], y = bo[ct], xe = ao[rt + 1], ye = bo[ct + 1];
  int sum = 0;
  while (x < xe && y < ye) {
    if (ai[x] < bi[y]) {
      ++x;
      continue;
    }
    if (bi[y] < ai[x]) {
      ++y;
      continue;
    }
    const uint64_t ap = (uint64_t{x} * prefix + plane) * 1024;
    const uint64_t bp = (uint64_t{y} * prefix + plane) * 1024;
    for (int cell = lane; cell < 1024; cell += 256) {
      at[cell] = a[ap + cell];
      bt[cell] = b[bp + cell];
    }
    __syncthreads();
    const int64_t start = uint64_t{ai[x]} * 64;
    const int extent = k - start < 64 ? static_cast<int>(k - start) : 64;
    for (int q = 0; q < extent; ++q)
      sum += static_cast<int>(at[lr * 64 + q]) * bt[q * 16 + lc];
    // At most 64*128*128 plus a carried residue: no cancellation assumption.
    sum %= modulus;
    __syncthreads();
    ++x;
    ++y;
  }
  if (row < m && col < n) {
    if (sum < 0)
      sum += modulus;
    if (sum >= (modulus + 1) / 2)
      sum -= modulus;
    c[(static_cast<int64_t>(plane) * m + row) * ldc + col] =
        static_cast<int8_t>(sum);
  }
}

__device__ __forceinline__ void rns8_block_sparse_wmma_tile(
    const int8_t* a,const uint32_t* ao,const uint32_t* ai,const int8_t* b,
    const uint32_t* bo,const uint32_t* bi,int8_t* c,int m,int n,int ldc,int prefix,int tile_cols) {
#if RNS8_DEVICE_NATIVE_WMMA
  // One complete wave owns the 16x16 output; no divergent lane participates
  // in the intrinsic. The packed blocks already contain masked zero tails.
  if(threadIdx.x>=32) return;
  using fragment=int __attribute__((ext_vector_type(4)));
  using accumulator=int __attribute__((ext_vector_type(8)));
  const int lane=threadIdx.x,local=lane&15,plane=blockIdx.y;
  const int64_t rt=blockIdx.x/tile_cols,ct=blockIdx.x%tile_cols;
  const int modulus=rns8_default_moduli_device[plane];
  accumulator acc{};
  uint32_t x=ao[rt],y=bo[ct],xe=ao[rt+1],ye=bo[ct+1];
  while(x<xe && y<ye) {
    if(ai[x]<bi[y]) {++x;continue;}
    if(bi[y]<ai[x]) {++y;continue;}
    const uint64_t ap=(uint64_t(x)*prefix+plane)*1024,bp=(uint64_t(y)*prefix+plane)*1024;
    for(int kk=0;kk<64;kk+=16) {
      fragment av{},bv{};
      for(int word=0;word<4;++word) {
        uint32_t aw=0,bw=0;
        for(int byte=0;byte<4;++byte) {
          const int q=kk+word*4+byte;
          aw|=uint32_t(uint8_t(a[ap+local*64+q]))<<(8*byte);
          bw|=uint32_t(uint8_t(b[bp+q*16+local]))<<(8*byte);
        }
        av[word]=int(aw);bv[word]=int(bw);
      }
      acc=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,av,true,bv,acc,false);
    }
    // Only 64 products between reductions, including the carried residue.
    for(int reg=0;reg<8;++reg) acc[reg]%=modulus;
    ++x;++y;
  }
  const int64_t col=ct*16+local;
  for(int reg=0;reg<8;++reg) {
    const int64_t row=rt*16+2*reg+(lane>>4);
    if(row<m && col<n) {
      int value=acc[reg]%modulus;if(value<0) value+=modulus;
      if(value>=(modulus+1)/2) value-=modulus;
      c[(int64_t(plane)*m+row)*ldc+col]=int8_t(value);
    }
  }
#else
  (void)a;(void)ao;(void)ai;(void)b;(void)bo;(void)bi;(void)c;
  (void)m;(void)n;(void)ldc;(void)prefix;(void)tile_cols;
#endif
}
template<int Method>
__global__ void rns8_block_sparse_gemm_kernel(
    const int8_t* a,const uint32_t* ao,const uint32_t* ai,const int8_t* b,
    const uint32_t* bo,const uint32_t* bi,int8_t* c,int m,int n,int k,int ldc,int prefix,
    int tile_cols,uint32_t scalar_limit) {
  if constexpr(Method==0) {
    rns8_block_sparse_scalar_tile(a,ao,ai,b,bo,bi,c,m,n,k,ldc,prefix,tile_cols);
  } else if constexpr(Method==1) {
    rns8_block_sparse_wmma_tile(a,ao,ai,b,bo,bi,c,m,n,ldc,prefix,tile_cols);
  } else {
    // Block-uniform dispatch. This prices work by exact block intersections,
    // not numerical pruning. A caller chooses the crossover explicitly.
    __shared__ uint32_t intersections;
    if(threadIdx.x==0) {
      const int64_t rt=blockIdx.x/tile_cols,ct=blockIdx.x%tile_cols;
      uint32_t x=ao[rt],y=bo[ct],xe=ao[rt+1],ye=bo[ct+1],count=0;
      while(x<xe && y<ye) {
        if(ai[x]<bi[y]) ++x;
        else if(bi[y]<ai[x]) ++y;
        else {++count;++x;++y;}
      }
      intersections=count;
    }
    __syncthreads();
    if(intersections<=scalar_limit)
      rns8_block_sparse_scalar_tile(a,ao,ai,b,bo,bi,c,m,n,k,ldc,prefix,tile_cols);
    else rns8_block_sparse_wmma_tile(a,ao,ai,b,bo,bi,c,m,n,ldc,prefix,tile_cols);
  }
}
