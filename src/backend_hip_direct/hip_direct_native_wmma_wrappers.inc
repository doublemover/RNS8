extern "C" int rns8_hip_native_wmma_capability(int* supported) {
  hipLaunchKernelGGL(rns8_native_wmma_capability_kernel, dim3(1), dim3(1), 0, nullptr, supported);
  return static_cast<int>(hipGetLastError());
}
template <typename NativeT, int Group>
int rns8_launch_native_wmma_queue(
    const rns8::detail::NativeWmmaTask* tasks, const rns8::detail::NativeWmmaWork* work,
    uint32_t count, uint32_t* cursor, uint32_t workers) {
  hipLaunchKernelGGL((rns8_native_wmma_queue_kernel<NativeT, Group>), dim3(workers), dim3(32), 0, nullptr,
                    tasks, work, count, cursor);
  return static_cast<int>(hipGetLastError());
}
template <typename NativeT>
int rns8_select_native_wmma_queue(
    const rns8::detail::NativeWmmaTask* tasks, const rns8::detail::NativeWmmaWork* work,
    uint32_t count, uint32_t* cursor, uint32_t workers, uint32_t group) {
  switch (group) {
    case 1: return rns8_launch_native_wmma_queue<NativeT, 1>(tasks, work, count, cursor, workers);
    case 2: return rns8_launch_native_wmma_queue<NativeT, 2>(tasks, work, count, cursor, workers);
    case 4: return rns8_launch_native_wmma_queue<NativeT, 4>(tasks, work, count, cursor, workers);
    default: return static_cast<int>(hipErrorInvalidValue);
  }
}
extern "C" int rns8_hip_native_wmma_queue(
    const rns8::detail::NativeWmmaTask* tasks, const rns8::detail::NativeWmmaWork* work,
    uint32_t count, uint32_t* cursor, uint32_t workers, uint32_t uniform_group, bool unsigned_inputs) {
  if (!uniform_group)
    return rns8_launch_native_wmma_queue<int64_t, 0>(tasks, work, count, cursor, workers);
  return unsigned_inputs
      ? rns8_select_native_wmma_queue<uint64_t>(tasks, work, count, cursor, workers, uniform_group)
      : rns8_select_native_wmma_queue<int64_t>(tasks, work, count, cursor, workers, uniform_group);
}
template<class T,int Group>
int rns8_launch_native_wmma_schedule(const rns8::detail::NativeWmmaTask* tasks,
    const rns8::detail::NativeWmmaWork* work,const rns8::detail::NativeWmmaTaskRange* ranges,
    const rns8::detail::NativeWmmaRoundPhase* phases,rns8::detail::NativeWmmaScheduleHeader header,
    uint32_t* cursor,uint32_t workers,hipStream_t stream=nullptr) {
  hipLaunchKernelGGL((rns8_native_wmma_schedule_kernel<T,Group>),dim3(workers),dim3(32),0,stream,
      tasks,work,ranges,phases,header,cursor,static_cast<uint32_t*>(nullptr),static_cast<uint64_t*>(nullptr),static_cast<const uint64_t*>(nullptr));
  return static_cast<int>(hipGetLastError());
}
extern "C" int rns8_hip_native_wmma_queue_schedule(const rns8::detail::NativeWmmaTask* tasks,
    const rns8::detail::NativeWmmaWork* work,const rns8::detail::NativeWmmaTaskRange* ranges,
    const rns8::detail::NativeWmmaRoundPhase* phases,rns8::detail::NativeWmmaScheduleHeader header,
    uint32_t* cursor,uint32_t workers,uint32_t group,bool unsigned_inputs) {
  if(header.abi_version!=1 || header.mode>2 || !header.claim_chunk || header.claim_chunk>32 || !workers)
    return static_cast<int>(hipErrorInvalidValue);
#define RNS8_QUEUE_SCHEDULE(G) return unsigned_inputs ? rns8_launch_native_wmma_schedule<uint64_t,G>(tasks,work,ranges,phases,header,cursor,workers) : rns8_launch_native_wmma_schedule<int64_t,G>(tasks,work,ranges,phases,header,cursor,workers)
  switch(group) {case 0:RNS8_QUEUE_SCHEDULE(0);case 1:RNS8_QUEUE_SCHEDULE(1);case 2:RNS8_QUEUE_SCHEDULE(2);case 4:RNS8_QUEUE_SCHEDULE(4);default:return static_cast<int>(hipErrorInvalidValue);}
#undef RNS8_QUEUE_SCHEDULE
}
template <typename NativeT, int Group>
int rns8_launch_native_wmma_group(
    const void* a, const void* b, int8_t* c, int m, int n, int k,
    int lda, int ldb, int ldc, int prefix,hipStream_t stream=nullptr) {
  hipLaunchKernelGGL((rns8_native_grouped_wmma_kernel<NativeT, Group>),
      dim3((static_cast<uint64_t>(n) + 15) / 16, (static_cast<uint64_t>(m) + 15) / 16,
           (prefix + Group - 1) / Group), dim3(32), 0, stream,
      static_cast<const NativeT*>(a), static_cast<const NativeT*>(b), c, m, n, k, lda, ldb, ldc, prefix);
  return static_cast<int>(hipGetLastError());
}
template <typename NativeT>
int rns8_select_native_wmma_group(
    const void* a, const void* b, int8_t* c, int m, int n, int k,
    int lda, int ldb, int ldc, int prefix, int group,hipStream_t stream=nullptr) {
  switch (group) {
    case 1: return rns8_launch_native_wmma_group<NativeT, 1>(a, b, c, m, n, k, lda, ldb, ldc, prefix,stream);
    case 2: return rns8_launch_native_wmma_group<NativeT, 2>(a, b, c, m, n, k, lda, ldb, ldc, prefix,stream);
    case 4: return rns8_launch_native_wmma_group<NativeT, 4>(a, b, c, m, n, k, lda, ldb, ldc, prefix,stream);
    default: return static_cast<int>(hipErrorInvalidValue);
  }
}
extern "C" int rns8_hip_native_wmma_grouped(
    const void* a, const void* b, int8_t* c, int m, int n, int k,
    int lda, int ldb, int ldc, int prefix, int group, bool unsigned_inputs) {
  return unsigned_inputs
      ? rns8_select_native_wmma_group<uint64_t>(a, b, c, m, n, k, lda, ldb, ldc, prefix, group)
      : rns8_select_native_wmma_group<int64_t>(a, b, c, m, n, k, lda, ldb, ldc, prefix, group);
}

extern "C" int rns8_hip_encode_native_b(const void* b,void* encoded,int k,int n,int ldb,
    int prefix,uint64_t k_words,bool unsigned_inputs) {
  const uint64_t words=uint64_t(prefix)*n*k_words;
  const unsigned blocks=static_cast<unsigned>(std::min(uint64_t{1024},(words+255)/256));
  if(unsigned_inputs)
    hipLaunchKernelGGL((rns8_encode_native_b_kernel<uint64_t>),dim3(blocks),dim3(256),0,nullptr,
        static_cast<const uint64_t*>(b),static_cast<uint32_t*>(encoded),k,n,ldb,prefix,k_words);
  else
    hipLaunchKernelGGL((rns8_encode_native_b_kernel<int64_t>),dim3(blocks),dim3(256),0,nullptr,
        static_cast<const int64_t*>(b),static_cast<uint32_t*>(encoded),k,n,ldb,prefix,k_words);
  return static_cast<int>(hipGetLastError());
}
template<class T,int Group>
int rns8_launch_encoded_b_wmma(const void* a,const void* b,void* c,int m,int n,int k,
    int lda,int ldc,int prefix,uint64_t k_words,hipStream_t stream=nullptr) {
  hipLaunchKernelGGL((rns8_native_encoded_b_wmma_kernel<T,Group>),
      dim3((uint64_t(n)+15)/16,(uint64_t(m)+15)/16,(prefix+Group-1)/Group),dim3(32),0,stream,
      static_cast<const T*>(a),static_cast<const uint32_t*>(b),static_cast<int8_t*>(c),
      m,n,k,lda,ldc,prefix,k_words);
  return static_cast<int>(hipGetLastError());
}
template<class T>
int rns8_select_encoded_b_wmma(const void* a,const void* b,void* c,int m,int n,int k,
    int lda,int ldc,int prefix,uint64_t k_words,int group,hipStream_t stream=nullptr) {
  switch(group) {
    case 1:return rns8_launch_encoded_b_wmma<T,1>(a,b,c,m,n,k,lda,ldc,prefix,k_words,stream);
    case 2:return rns8_launch_encoded_b_wmma<T,2>(a,b,c,m,n,k,lda,ldc,prefix,k_words,stream);
    case 4:return rns8_launch_encoded_b_wmma<T,4>(a,b,c,m,n,k,lda,ldc,prefix,k_words,stream);
    default:return static_cast<int>(hipErrorInvalidValue);
  }
}
extern "C" int rns8_hip_native_encoded_b_wmma(const void* a,const void* b,void* c,
    int m,int n,int k,int lda,int ldc,int prefix,uint64_t k_words,int group,bool unsigned_inputs) {
  return unsigned_inputs?rns8_select_encoded_b_wmma<uint64_t>(a,b,c,m,n,k,lda,ldc,prefix,k_words,group)
      :rns8_select_encoded_b_wmma<int64_t>(a,b,c,m,n,k,lda,ldc,prefix,k_words,group);
}

extern "C" int rns8_hip_native_wmma_queue_completion(const rns8::detail::NativeWmmaTask* tasks,
    const rns8::detail::NativeWmmaWork* work,const rns8::detail::NativeWmmaTaskRange* ranges,
    const rns8::detail::NativeWmmaRoundPhase* phases,rns8::detail::NativeWmmaScheduleHeader header,
    uint32_t* cursor,uint32_t workers,uint32_t* completed,uint64_t* ticks,uint64_t* start) {
  if(!tasks || !cursor || !workers || workers>1024 || !completed || !ticks || !start ||
      header.abi_version!=1 || header.mode>2 || !header.task_count || header.task_count>4096 ||
      !header.claim_chunk || header.claim_chunk>32 || (header.mode==0?!work:!ranges))return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(rns8_native_queue_completion_begin,dim3((header.task_count+255)/256),dim3(256),0,nullptr,
      header.task_count,completed,ticks,start);
  auto status=hipGetLastError();if(status!=hipSuccess)return int(status);
  // General dispatch deliberately isolates diagnostic completion from the
  // production uniform-specialization measurements.
  hipLaunchKernelGGL((rns8_native_wmma_schedule_kernel<int64_t,0,true>),dim3(workers),dim3(32),0,nullptr,
      tasks,work,ranges,phases,header,cursor,completed,ticks,start);
  return int(hipGetLastError());
}

extern "C" int rns8_hip_native_wmma_queue_schedule_on_stream(const rns8::detail::NativeWmmaTask* tasks,
    const rns8::detail::NativeWmmaWork* work,const rns8::detail::NativeWmmaTaskRange* ranges,
    const rns8::detail::NativeWmmaRoundPhase* phases,rns8::detail::NativeWmmaScheduleHeader header,
    uint32_t* cursor,uint32_t workers,uint32_t group,bool unsigned_inputs,void* stream) {
  if(!tasks || !cursor || !stream || header.abi_version!=1 || header.mode>2 || !header.claim_chunk || header.claim_chunk>32 || !workers)
    return static_cast<int>(hipErrorInvalidValue);
#define RNS8_QUEUE_STREAM(G) return unsigned_inputs ? rns8_launch_native_wmma_schedule<uint64_t,G>(tasks,work,ranges,phases,header,cursor,workers,static_cast<hipStream_t>(stream)) : rns8_launch_native_wmma_schedule<int64_t,G>(tasks,work,ranges,phases,header,cursor,workers,static_cast<hipStream_t>(stream))
  switch(group){case 0:RNS8_QUEUE_STREAM(0);case 1:RNS8_QUEUE_STREAM(1);case 2:RNS8_QUEUE_STREAM(2);case 4:RNS8_QUEUE_STREAM(4);default:return static_cast<int>(hipErrorInvalidValue);}
#undef RNS8_QUEUE_STREAM
}
extern "C" int rns8_hip_native_wmma_grouped_on_stream(const void* a,const void* b,int8_t* c,
    int m,int n,int k,int lda,int ldb,int ldc,int prefix,int group,bool u,void* stream) {
  return u?rns8_select_native_wmma_group<uint64_t>(a,b,c,m,n,k,lda,ldb,ldc,prefix,group,static_cast<hipStream_t>(stream)):
      rns8_select_native_wmma_group<int64_t>(a,b,c,m,n,k,lda,ldb,ldc,prefix,group,static_cast<hipStream_t>(stream));
}
extern "C" int rns8_hip_native_encoded_b_wmma_on_stream(const void* a,const void* b,void* c,
    int m,int n,int k,int lda,int ldc,int prefix,uint64_t k_words,int group,bool u,void* stream) {
  return u?rns8_select_encoded_b_wmma<uint64_t>(a,b,c,m,n,k,lda,ldc,prefix,k_words,group,static_cast<hipStream_t>(stream)):
      rns8_select_encoded_b_wmma<int64_t>(a,b,c,m,n,k,lda,ldc,prefix,k_words,group,static_cast<hipStream_t>(stream));
}
