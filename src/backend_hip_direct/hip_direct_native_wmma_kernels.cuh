#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
#define RNS8_DEVICE_NATIVE_WMMA 1
#else
#define RNS8_DEVICE_NATIVE_WMMA 0
#endif
__global__ void rns8_native_wmma_capability_kernel(int* supported) {
  *supported = RNS8_DEVICE_NATIVE_WMMA;
}

__device__ __forceinline__ int8_t rns8_native_wmma_center(
    int64_t value, const rns8::detail::finite_u8::Native64Reducer& reducer) {
  return rns8::detail::finite_u8::center_native_i64(value, reducer);
}
__device__ __forceinline__ int8_t rns8_native_wmma_center(
    uint64_t value, const rns8::detail::finite_u8::Native64Reducer& reducer) {
  return rns8::detail::finite_u8::center_native_u64(value, reducer);
}

// Each wave owns one output tile and Group consecutive default-basis planes.
// Native loads are outside the channel loop; channels have separate encoded
// fragments and accumulators. No INT32 or packed-input global workspace exists.
template <typename NativeT, int Group, bool EncodedB = false, bool Cooperative = false>
__device__ __forceinline__ void rns8_native_grouped_wmma_tile(
    const NativeT* a, const NativeT* b, int8_t* c,
    int m, int n, int k, int lda, int ldb, int ldc, int prefix,
    int64_t tile_row, int64_t tile_col, int first_plane,
    const uint32_t* encoded_b = nullptr, uint64_t encoded_k_words = 0) {
#if RNS8_DEVICE_NATIVE_WMMA
  using fragment = int __attribute__((ext_vector_type(4)));
  using accumulator = int __attribute__((ext_vector_type(8)));
  namespace layout = rns8::detail::fragment_layout;
  const int lane = threadIdx.x & 31;
  const int64_t ar = tile_row + layout::row(layout::a_coordinate(lane,0,0));
  const int64_t bc = tile_col + layout::col(layout::b_coordinate(lane,0,0));
  accumulator acc[Group]{};
  rns8::detail::finite_u8::Native64Reducer reducers[Group]{};
#pragma unroll
  for (int channel = 0; channel < Group; ++channel)
    if (first_plane + channel < prefix)
      reducers[channel] = rns8::detail::finite_u8::native64_reducer(
          rns8_default_moduli_device[first_plane + channel]);
  for (int64_t kk = 0; kk < k; kk += 16) {
    fragment av[Group]{}, bv[Group]{};
    if constexpr (Cooperative) {
      static_assert(!EncodedB, "cooperative native ownership cannot reinterpret encoded B");
      const bool owns_a = rns8::detail::native_lane_owns_a(lane);
      for (int word = 0; word < 4; ++word) {
        uint32_t owned[Group]{};
        for (int byte = 0; byte < 4; ++byte) {
          const int64_t q = kk + word * 4 + byte;
          // Exactly one selected native scalar and one common reducer stream.
          // All 32 lanes remain active, including initialized zero tail lanes.
          const NativeT raw = owns_a ? (ar < m && q < k ? a[ar*lda+q] : NativeT{}) :
                                      (bc < n && q < k ? b[q*ldb+bc] : NativeT{});
#pragma unroll
          for (int channel = 0; channel < Group; ++channel)
            if (first_plane + channel < prefix)
              owned[channel] |= uint32_t(uint8_t(rns8_native_wmma_center(raw, reducers[channel]))) << (8*byte);
        }
#pragma unroll
        for (int channel = 0; channel < Group; ++channel) {
          const uint32_t partner = __shfl_xor(owned[channel], 16, 32);
          av[channel][word] = static_cast<int>(owns_a ? owned[channel] : partner);
          bv[channel][word] = static_cast<int>(owns_a ? partner : owned[channel]);
        }
      }
    } else {
    for (int word = 0; word < 4; ++word) {
      uint32_t ap[Group]{}, bp[Group]{};
      for (int byte = 0; byte < 4; ++byte) {
        const int64_t q = kk + layout::col(layout::a_coordinate(lane,word,byte));
        const NativeT raw_a = ar < m && q < k ? a[ar * lda + q] : NativeT{};
        NativeT raw_b{};
        if constexpr (!EncodedB) {
          const int64_t bq = kk + layout::row(layout::b_coordinate(lane,word,byte));
          raw_b = bc < n && bq < k ? b[bq * ldb + bc] : NativeT{};
        }
#pragma unroll
        for (int channel = 0; channel < Group; ++channel) {
          const int plane = first_plane + channel;
          if (plane >= prefix) continue;
          ap[channel] |= static_cast<uint32_t>(static_cast<uint8_t>(
              rns8_native_wmma_center(raw_a, reducers[channel]))) << (8 * byte);
          if constexpr (!EncodedB)
            bp[channel] |= static_cast<uint32_t>(static_cast<uint8_t>(
                rns8_native_wmma_center(raw_b, reducers[channel]))) << (8 * byte);
        }
      }
      for (int channel = 0; channel < Group; ++channel) {
        av[channel][word] = static_cast<int>(ap[channel]);
        if constexpr (EncodedB) {
          const int plane=first_plane+channel;
          bv[channel][word]=plane<prefix && bc<n
              ? static_cast<int>(encoded_b[(uint64_t(plane)*n+bc)*encoded_k_words+uint64_t(kk)/4+word]) : 0;
        } else bv[channel][word] = static_cast<int>(bp[channel]);
      }
    }
    }
    for (int channel = 0; channel < Group; ++channel) {
      if (first_plane + channel >= prefix) continue;
      acc[channel] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
          true, av[channel], true, bv[channel], acc[channel], false);
      // |every partial| <= 65536*128*128+255 < INT32_MAX.
      // Signed remainder carries at most 255 into the next safe slab.
      if (((kk + 16) % 65536) == 0)
        for (int reg = 0; reg < 8; ++reg) acc[channel][reg] %= static_cast<int>(reducers[channel].modulus);
    }
  }
  const int64_t col = tile_col + layout::col(layout::c_coordinate(lane,0));
  for (int channel = 0; channel < Group; ++channel) {
    const int plane = first_plane + channel;
    if (plane >= prefix) continue;
    const int modulus = static_cast<int>(reducers[channel].modulus);
    for (int reg = 0; reg < 8; ++reg) {
      const int64_t row = tile_row + layout::row(layout::c_coordinate(lane,reg));
      if (row < m && col < n) {
        int value = acc[channel][reg] % modulus;
        if (value < 0) value += modulus;
        if (value >= (modulus + 1) / 2) value -= modulus;
        c[(static_cast<int64_t>(plane) * m + row) * ldc + col] = static_cast<int8_t>(value);
      }
    }
  }
#else
  (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
  (void)lda; (void)ldb; (void)ldc; (void)prefix;
  (void)encoded_b; (void)encoded_k_words;
#endif
}
template <typename NativeT, int Group>
__global__ __launch_bounds__(32) void rns8_native_grouped_wmma_kernel(
    const NativeT* a, const NativeT* b, int8_t* c,
    int m, int n, int k, int lda, int ldb, int ldc, int prefix) {
  rns8_native_grouped_wmma_tile<NativeT, Group>(a, b, c, m, n, k, lda, ldb, ldc, prefix,
      static_cast<int64_t>(blockIdx.y) * 16, static_cast<int64_t>(blockIdx.x) * 16, blockIdx.z * Group);
}
template<class T,int Group>
__device__ __forceinline__ void rns8_native_queue_group_tile(
    const rns8::detail::NativeWmmaTask& task,const rns8::detail::NativeWmmaWork& work) {
  const int64_t row=int64_t(work.tile_row)*16,col=int64_t(work.tile_col)*16;
  if(task.encoded_b)
    rns8_native_grouped_wmma_tile<T,Group,true>(static_cast<const T*>(task.a),nullptr,task.c,
        task.m,task.n,task.k,task.lda,task.ldb,task.ldc,task.prefix,row,col,work.first_plane,
        task.encoded_b,task.encoded_k_words);
  else
    rns8_native_grouped_wmma_tile<T,Group>(static_cast<const T*>(task.a),static_cast<const T*>(task.b),task.c,
        task.m,task.n,task.k,task.lda,task.ldb,task.ldc,task.prefix,row,col,work.first_plane);
}
template<class T>
__device__ __forceinline__ void rns8_native_queue_tile(
    const rns8::detail::NativeWmmaTask& task,const rns8::detail::NativeWmmaWork& work) {
  if(task.group==1) rns8_native_queue_group_tile<T,1>(task,work);
  else if(task.group==2) rns8_native_queue_group_tile<T,2>(task,work);
  else rns8_native_queue_group_tile<T,4>(task,work);
}
// Group zero retains heterogeneous dispatch. Uniform queues instantiate only
// one tile body, so a group-one job need not reserve group-four registers.
template <typename NativeT, int Group>
__global__ __launch_bounds__(32) void rns8_native_wmma_queue_kernel(
    const rns8::detail::NativeWmmaTask* tasks, const rns8::detail::NativeWmmaWork* work,
    uint32_t count, uint32_t* cursor) {
#if RNS8_DEVICE_NATIVE_WMMA
  __shared__ uint32_t next;
  while (true) {
    if (!threadIdx.x) next = atomicAdd(cursor, 1u);
    __syncthreads();
    if (next >= count) return;
    const auto item = work[next];
    const auto task = tasks[item.task];
    if constexpr (Group != 0) {
      rns8_native_grouped_wmma_tile<NativeT, Group>(
          static_cast<const NativeT*>(task.a), static_cast<const NativeT*>(task.b), task.c,
          task.m, task.n, task.k, task.lda, task.ldb, task.ldc, task.prefix,
          static_cast<int64_t>(item.tile_row) * 16, static_cast<int64_t>(item.tile_col) * 16,
          item.first_plane);
    } else {
      if (task.unsigned_inputs) rns8_native_queue_tile<uint64_t>(task, item);
      else rns8_native_queue_tile<int64_t>(task, item);
    }
    __syncthreads(); // Every lane finishes before the next shared queue claim.
  }
#else
  (void)tasks; (void)work; (void)count; (void)cursor;
#endif
}
// Compact immutable scheduling uses the identical tile arithmetic. One wave
// claims a bounded chunk; near the tail singleton claims reduce monopolization.
// The host admits uint32 counter overshoot for workers*maximum_chunk, not just
// actual work. No inter-workgroup barrier or speculative counter wrap is used.
template<class NativeT,int Group,bool Completion=false>
__global__ __launch_bounds__(32) void rns8_native_wmma_schedule_kernel(
    const rns8::detail::NativeWmmaTask* tasks,const rns8::detail::NativeWmmaWork* work,
    const rns8::detail::NativeWmmaTaskRange* ranges,const rns8::detail::NativeWmmaRoundPhase* phases,
    rns8::detail::NativeWmmaScheduleHeader header,uint32_t* cursor,
    uint32_t* completed=nullptr,uint64_t* ticks=nullptr,const uint64_t* start=nullptr) {
#if RNS8_DEVICE_NATIVE_WMMA
  __shared__ uint32_t first,chunk,valid;
  __shared__ rns8::detail::NativeWmmaWork item;
  while(true) {
    if(!threadIdx.x) {
      const uint32_t observed=atomicAdd(cursor,0u);
      const uint64_t tail=uint64_t(gridDim.x)*header.claim_chunk;
      chunk=uint64_t(observed)+tail>=header.work_count?1u:header.claim_chunk;
      first=atomicAdd(cursor,chunk);
    }
    __syncthreads();
    if(first>=header.work_count)return;
    for(uint32_t offset=0;offset<chunk && uint64_t(first)+offset<header.work_count;++offset) {
      if(!threadIdx.x) {
        if(header.mode==0){item=work[first+offset];valid=1;}
        else valid=rns8::detail::decode_native_wmma_schedule(header,ranges,phases,tasks,first+offset,item);
      }
      __syncthreads();
      if(!valid)return; // host immutable-directory admission makes this unreachable
      const auto task=tasks[item.task];
      if constexpr(Group!=0) {
        rns8_native_grouped_wmma_tile<NativeT,Group>(static_cast<const NativeT*>(task.a),
            static_cast<const NativeT*>(task.b),task.c,task.m,task.n,task.k,task.lda,task.ldb,task.ldc,task.prefix,
            int64_t(item.tile_row)*16,int64_t(item.tile_col)*16,item.first_plane);
      } else if(task.unsigned_inputs)rns8_native_queue_tile<uint64_t>(task,item);
      else rns8_native_queue_tile<int64_t>(task,item);
      // Each writer publishes its own output stores before the completion
      // counter. No workgroup waits for another workgroup to be scheduled.
      if constexpr(Completion) __threadfence();
      __syncthreads();
      if constexpr(Completion) if(!threadIdx.x) {
        const uint32_t required=uint32_t(uint64_t((task.m+15)/16)*((task.n+15)/16)*
            ((task.prefix+task.group-1)/task.group));
        if(atomicAdd(completed+item.task,1u)+1==required)
          ticks[item.task]=uint64_t(wall_clock64())-*start;
      }
      __syncthreads();
    }
  }
#else
  (void)tasks;(void)work;(void)ranges;(void)phases;(void)header;(void)cursor;(void)completed;(void)ticks;(void)start;
#endif
}
// Start is a preceding stream-ordered kernel, so every CTA uses one epoch.
__global__ void rns8_native_queue_completion_begin(uint32_t count,uint32_t* completed,
    uint64_t* ticks,uint64_t* start) {
#if RNS8_DEVICE_NATIVE_WMMA
  for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x){completed[i]=0;ticks[i]=0;}
  if(!blockIdx.x && !threadIdx.x)*start=uint64_t(wall_clock64());
#else
  (void)count;(void)completed;(void)ticks;(void)start;
#endif
}
// Plane/column/K-word order: each B fragment word is a single aligned load.
// Native loads during preparation remain explicit and are paid once per handle.
template<class T>
__global__ void rns8_encode_native_b_kernel(const T* b,uint32_t* encoded,
    int k,int n,int ldb,int prefix,uint64_t k_words) {
  const uint64_t total=uint64_t(prefix)*n*k_words;
  for(uint64_t word=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;word<total;
      word+=uint64_t(gridDim.x)*blockDim.x) {
    const uint64_t col=(word/k_words)%n,plane=word/(k_words*n),q0=(word%k_words)*4;
    const auto reducer=rns8::detail::finite_u8::native64_reducer(rns8_default_moduli_device[plane]);
    uint32_t packed=0;
    for(int byte=0;byte<4;++byte) {
      const uint64_t q=q0+byte;
      const T raw=q<uint64_t(k)?b[q*ldb+col]:T{};
      packed|=uint32_t(uint8_t(rns8_native_wmma_center(raw,reducer)))<<(8*byte);
    }
    encoded[word]=packed;
  }
}
template<class T,int Group>
__global__ __launch_bounds__(32) void rns8_native_encoded_b_wmma_kernel(
    const T* a,const uint32_t* b,int8_t* c,int m,int n,int k,int lda,int ldc,
    int prefix,uint64_t k_words) {
  rns8_native_grouped_wmma_tile<T,Group,true>(a,nullptr,c,m,n,k,lda,n,ldc,prefix,
      int64_t(blockIdx.y)*16,int64_t(blockIdx.x)*16,blockIdx.z*Group,b,k_words);
}
#undef RNS8_DEVICE_NATIVE_WMMA
