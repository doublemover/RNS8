__global__ void rns8_convert_residue_layout_kernel(
    const int8_t* src,rns8::detail::ResidueLayout source,int8_t* dst,rns8::detail::ResidueLayout destination,
    uint32_t first,uint32_t end) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t cells = source.rows * source.cols;
  if (index >= cells * (end-first)) return;
  const auto plane = first+static_cast<uint32_t>(index / cells);
  const uint64_t cell = index % cells, row = cell / source.cols, col = cell % source.cols;
  dst[destination.offset(plane, row, col)] = src[source.offset(plane, row, col)];
}

// Coalesced descriptor-aware staging into 16x32 and 32x16 LDS tiles. The
// accumulator is reduced every 32 terms: 32*128*128+255 fits signed INT32.
// This is a packed-byte/vector consumer, not a matrix-instruction claim.
__global__ __launch_bounds__(256) void rns8_gemm_residue_layout_kernel(
    const int8_t* a,rns8::detail::ResidueLayout al,const int8_t* b,rns8::detail::ResidueLayout bl,
    int8_t* c,rns8::detail::ResidueLayout cl,uint32_t prefix,rns8::detail::ResidueLayoutModuli moduli) {
  __shared__ int8_t sa[16][32],sb[32][16];
  const uint32_t plane=blockIdx.z,lane=threadIdx.y*16+threadIdx.x;
  const uint64_t row0=uint64_t(blockIdx.y)*16,col0=uint64_t(blockIdx.x)*16;
  const uint64_t row=row0+threadIdx.y,col=col0+threadIdx.x;
  const int modulus=moduli.values[plane];int sum=0;
  for(uint64_t base=0;base<al.cols;base+=32) {
    for(unsigned i=lane;i<512;i+=256) {
      const uint64_t ar=row0+i/32,aq=base+i%32,bq=base+i/16,bc=col0+i%16;
      sa[i/32][i%32]=ar<al.rows && aq<al.cols?a[al.offset(plane,ar,aq)]:0;
      sb[i/16][i%16]=bq<bl.rows && bc<bl.cols?b[bl.offset(plane,bq,bc)]:0;
    }
    __syncthreads();
#pragma unroll
    for(unsigned q=0;q<32;q+=4) {
      if constexpr(rns8::detail::kResidueScalarControl) {
#pragma unroll
        for(unsigned byte=0;byte<4;++byte)
          sum+=int(sa[threadIdx.y][q+byte])*int(sb[q+byte][threadIdx.x]);
      } else {
        uint32_t av=0,bv=0;
#pragma unroll
        for(unsigned byte=0;byte<4;++byte) {
          av|=uint32_t(uint8_t(sa[threadIdx.y][q+byte]))<<(8*byte);
          bv|=uint32_t(uint8_t(sb[q+byte][threadIdx.x]))<<(8*byte);
        }
        sum=rns8::detail::signed_byte_dot4(av,bv,sum);
      }
    }
    sum%=modulus;
    __syncthreads();
  }
  if(row<cl.rows && col<cl.cols && plane<prefix) {
    if(sum<0)sum+=modulus;if(sum>=(modulus+1)/2)sum-=modulus;
    c[cl.offset(plane,row,col)]=static_cast<int8_t>(sum);
  }
}
