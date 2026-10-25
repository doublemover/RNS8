// Four waves own the four 16x16 quadrants of a 32x32 output tile.
// Staged modes share raw native inputs, not encoded residues, across waves.
template<class T, int Depth>
__device__ __forceinline__ void rns8_native_stage_tile(
    const T* a, const T* b, T* sa, T* sb, int m, int n, int k, int lda, int ldb,
    int64_t row_base, int64_t col_base, int64_t k_base) {
  for (int index=threadIdx.x; index<32*Depth; index+=128) {
    const int row=index/Depth, q=index%Depth;
    sa[row*(Depth+1)+q] = row_base+row<m && k_base+q<k ? a[(row_base+row)*lda+k_base+q] : T{};
    const int bq=index/32, col=index%32;
    sb[bq*33+col] = k_base+bq<k && col_base+col<n ? b[(k_base+bq)*ldb+col_base+col] : T{};
  }
}

template<class T, int Group, int Depth, int Buffers, bool Cooperative = false>
__global__ __launch_bounds__(128) void rns8_native_tiled_wmma_kernel(
    const T* a, const T* b, int8_t* c, int m, int n, int k, int lda, int ldb, int ldc, int prefix) {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  const int lane=threadIdx.x&31, wave=threadIdx.x>>5;
  const int64_t row_base=int64_t(blockIdx.y)*32, col_base=int64_t(blockIdx.x)*32;
  const int first_plane=blockIdx.z*Group;
  const int wave_row=(wave/2)*16, wave_col=(wave%2)*16;
  if constexpr (Buffers==0) {
    rns8_native_grouped_wmma_tile<T,Group,false,Cooperative>(a,b,c,m,n,k,lda,ldb,ldc,prefix,
        row_base+wave_row,col_base+wave_col,first_plane);
  } else {
    using fragment = int __attribute__((ext_vector_type(4)));
    using accumulator = int __attribute__((ext_vector_type(8)));
    __shared__ T raw_a[Buffers][32*(Depth+1)];
    __shared__ T raw_b[Buffers][Depth*33];
    accumulator acc[Group]{};
    rns8::detail::finite_u8::Native64Reducer reducers[Group]{};
#pragma unroll
    for (int channel=0; channel<Group; ++channel)
      if (first_plane+channel<prefix)
        reducers[channel]=rns8::detail::finite_u8::native64_reducer(rns8_default_moduli_device[first_plane+channel]);
    rns8_native_stage_tile<T,Depth>(a,b,raw_a[0],raw_b[0],m,n,k,lda,ldb,row_base,col_base,0);
    __syncthreads();
    int current=0;
    for (int64_t base=0; base<k; base+=Depth) {
      const bool has_next=base+Depth<k;
      if constexpr (Buffers==2) {
        // Alternate storage is disjoint from the current reads. No barrier is
        // needed here: the end-of-iteration barrier completes next-tile stores.
        if (has_next) rns8_native_stage_tile<T,Depth>(a,b,raw_a[current^1],raw_b[current^1],
            m,n,k,lda,ldb,row_base,col_base,base+Depth);
      }
#pragma unroll
      for (int sub=0; sub<Depth; sub+=16) {
        fragment av[Group]{}, bv[Group]{};
        for (int word=0; word<4; ++word) {
          uint32_t ap[Group]{}, bp[Group]{};
          for (int byte=0; byte<4; ++byte) {
            const int q=sub+word*4+byte;
            const T va=raw_a[current][(wave_row+(lane&15))*(Depth+1)+q];
            const T vb=raw_b[current][q*33+wave_col+(lane&15)];
#pragma unroll
            for (int channel=0; channel<Group; ++channel) if (first_plane+channel<prefix) {
              ap[channel] |= uint32_t(uint8_t(rns8_native_wmma_center(va,reducers[channel])))<<(8*byte);
              bp[channel] |= uint32_t(uint8_t(rns8_native_wmma_center(vb,reducers[channel])))<<(8*byte);
            }
          }
          for (int channel=0; channel<Group; ++channel) {
            av[channel][word]=static_cast<int>(ap[channel]); bv[channel][word]=static_cast<int>(bp[channel]);
          }
        }
        for (int channel=0; channel<Group; ++channel) if (first_plane+channel<prefix) {
          acc[channel]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,av[channel],true,bv[channel],acc[channel],false);
          // Depth divides 65536. Padding contributes zeros and never extends a
          // live accumulator beyond the established 65536*128*128+255 bound.
          if ((base+sub+16)%65536==0)
            for (int reg=0; reg<8; ++reg) acc[channel][reg]%=static_cast<int>(reducers[channel].modulus);
        }
      }
      __syncthreads(); // all current readers and alternate stores have completed
      if constexpr (Buffers==1) {
        if (has_next) {
          rns8_native_stage_tile<T,Depth>(a,b,raw_a[0],raw_b[0],m,n,k,lda,ldb,row_base,col_base,base+Depth);
          __syncthreads();
        }
      } else current^=1;
    }
    const int64_t col=col_base+wave_col+(lane&15);
    for (int channel=0; channel<Group; ++channel) if (first_plane+channel<prefix) {
      const int modulus=reducers[channel].modulus;
      for (int reg=0; reg<8; ++reg) {
        const int64_t row=row_base+wave_row+2*reg+(lane>>4);
        if (row<m && col<n) {
          int value=acc[channel][reg]%modulus;
          if (value<0) value+=modulus;
          if (value>=(modulus+1)/2) value-=modulus;
          c[(int64_t(first_plane+channel)*m+row)*ldc+col]=static_cast<int8_t>(value);
        }
      }
    }
  }
#else
  (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)lda;(void)ldb;(void)ldc;(void)prefix;
#endif
}

// A producer owns a complete packed word, eliminating inter-thread byte stores.
// No raw LDS copy: each raw scalar is held only until its channel encodings are
// written. One producer/consumer barrier and one reuse barrier per depth panel.
template<class T, int Group, int Depth>
__global__ __launch_bounds__(128) void rns8_native_encoded_lds_wmma_kernel(
    const T* a,const T* b,int8_t* c,int m,int n,int k,int lda,int ldb,int ldc,int prefix) {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  using fragment = int __attribute__((ext_vector_type(4)));
  using accumulator = int __attribute__((ext_vector_type(8)));
  constexpr int words=Depth/4, stride=words+1;
  __shared__ uint32_t encoded[Group][2][32*stride];
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5,first=blockIdx.z*Group;
  const int64_t row0=int64_t(blockIdx.y)*32,col0=int64_t(blockIdx.x)*32;
  accumulator acc[Group]{};
  rns8::detail::finite_u8::Native64Reducer reducers[Group]{};
#pragma unroll
  for(int channel=0;channel<Group;++channel) if(first+channel<prefix)
    reducers[channel]=rns8::detail::finite_u8::native64_reducer(rns8_default_moduli_device[first+channel]);
  for(int64_t base=0;base<k;base+=Depth) {
    for(int item=threadIdx.x;item<2*32*words;item+=128) {
      const int role=item/(32*words),axis=(item/words)%32,word=item%words;
      uint32_t packed[Group]{};
#pragma unroll
      for(int byte=0;byte<4;++byte) {
        const int64_t q=base+word*4+byte;
        const T raw=role==0 ? (row0+axis<m && q<k ? a[(row0+axis)*lda+q] : T{}) :
                             (col0+axis<n && q<k ? b[q*ldb+col0+axis] : T{});
#pragma unroll
        for(int channel=0;channel<Group;++channel) if(first+channel<prefix)
          packed[channel] |= uint32_t(uint8_t(rns8_native_wmma_center(raw,reducers[channel])))<<(8*byte);
      }
#pragma unroll
      for(int channel=0;channel<Group;++channel)
        encoded[channel][role][rns8::detail::native_encoded_word(axis,word,Depth)]=packed[channel];
    }
    __syncthreads();
#pragma unroll
    for(int sub=0;sub<Depth;sub+=16) {
#pragma unroll
      for(int channel=0;channel<Group;++channel) if(first+channel<prefix) {
        fragment av{},bv{};
#pragma unroll
        for(int word=0;word<4;++word) {
          av[word]=static_cast<int>(encoded[channel][0][rns8::detail::native_encoded_word((wave/2)*16+(lane&15),sub/4+word,Depth)]);
          bv[word]=static_cast<int>(encoded[channel][1][rns8::detail::native_encoded_word((wave%2)*16+(lane&15),sub/4+word,Depth)]);
        }
        acc[channel]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,av,true,bv,acc[channel],false);
        if((base+sub+16)%65536==0)
          for(int reg=0;reg<8;++reg) acc[channel][reg]%=static_cast<int>(reducers[channel].modulus);
      }
    }
    __syncthreads(); // Never overwrite encoded words while another wave reads.
  }
  const int64_t col=col0+(wave%2)*16+(lane&15);
  for(int channel=0;channel<Group;++channel) if(first+channel<prefix) {
    const int modulus=reducers[channel].modulus;
    for(int reg=0;reg<8;++reg) {
      const int64_t row=row0+(wave/2)*16+2*reg+(lane>>4);
      if(row<m && col<n) {
        int value=acc[channel][reg]%modulus;
        if(value<0)value+=modulus;
        if(value>=(modulus+1)/2)value-=modulus;
        c[(int64_t(first+channel)*m+row)*ldc+col]=static_cast<int8_t>(value);
      }
    }
  }
#else
  (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)lda;(void)ldb;(void)ldc;(void)prefix;
#endif
}

template<class T,int Group,bool Cooperative>
__global__ __launch_bounds__(32) void rns8_native_conversion_wave_kernel(
    const T* a,const T* b,int8_t* c,int m,int n,int k,int lda,int ldb,int ldc,int prefix) {
  rns8_native_grouped_wmma_tile<T,Group,false,Cooperative>(a,b,c,m,n,k,lda,ldb,ldc,prefix,
      int64_t(blockIdx.y)*16,int64_t(blockIdx.x)*16,blockIdx.z*Group);
}
