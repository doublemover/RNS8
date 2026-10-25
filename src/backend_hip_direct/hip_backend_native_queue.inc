#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_wmma_queue(const NativeWmmaTask*, const NativeWmmaWork*, uint32_t, uint32_t*, uint32_t, uint32_t, bool);
extern "C" int rns8_hip_native_wmma_queue_schedule(const NativeWmmaTask*,const NativeWmmaWork*,
    const NativeWmmaTaskRange*,const NativeWmmaRoundPhase*,NativeWmmaScheduleHeader,
    uint32_t*,uint32_t,uint32_t,bool);
extern "C" int rns8_hip_native_wmma_queue_completion(const NativeWmmaTask*,const NativeWmmaWork*,
    const NativeWmmaTaskRange*,const NativeWmmaRoundPhase*,NativeWmmaScheduleHeader,uint32_t*,uint32_t,
    uint32_t*,uint64_t*,uint64_t*);
#endif
rns8_status NativeWmmaPreparedQueue::prepare(int device_id,
    const std::vector<NativeWmmaTask>& tasks, uint32_t workers, uint64_t max_scratch_bytes,
    NativeWmmaScheduleConfig config) {
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return RNS8_BUSY;
  NativeWmmaSchedulePlan schedule;
  // Transactional replacement keeps the old allocation until upload succeeds.
  // Charge both allocations against the preparation budget at that peak.
  const auto retained = static_cast<uint64_t>(storage_.bytes());
  const auto available = max_scratch_bytes >= retained ? max_scratch_bytes - retained : 0;
  auto status = build_native_wmma_schedule(tasks, workers, available, config, schedule);
  const auto bytes=schedule.device_bytes;
  if (status != RNS8_SUCCESS) return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  // Allocate the ordinary-launch control descriptors before replacing any state.
  // These are metadata only; operand contents are never retained on the host.
  std::vector<NativeWmmaTask> host_tasks;
  try {host_tasks=tasks;} catch(const std::bad_alloc&) {return RNS8_INTERNAL_ERROR;}
  status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer storage;
  status = storage.allocate(device_id, static_cast<std::size_t>(bytes));
  if (status != RNS8_SUCCESS) return status;
  auto* device_tasks = static_cast<NativeWmmaTask*>(storage.get());
  auto* payload=reinterpret_cast<unsigned char*>(device_tasks+tasks.size());
  auto* device_work=config.mode==NativeWmmaScheduleMode::explicit_round_robin?reinterpret_cast<NativeWmmaWork*>(payload):nullptr;
  auto* ranges=device_work?nullptr:reinterpret_cast<NativeWmmaTaskRange*>(payload);
  auto* phases=device_work?nullptr:reinterpret_cast<NativeWmmaRoundPhase*>(ranges+schedule.ranges.size());
  auto* cursor=device_work?reinterpret_cast<uint32_t*>(device_work+schedule.explicit_work.size()):
      reinterpret_cast<uint32_t*>(phases+schedule.phases.size());
  auto error = timed_hip_operation("native_queue_prepare_capability", [&]() {
    const auto launch = static_cast<hipError_t>(rns8_hip_native_wmma_capability(reinterpret_cast<int*>(cursor)));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t supported = 0;
  status = hip_direct_copy_device_to_host(device_id, &supported, cursor, sizeof(supported));
  if (status != RNS8_SUCCESS) return status;
  if (!supported) return RNS8_UNSUPPORTED_BACKEND;
  error = timed_hip_operation("native_queue_prepare_upload", [&]() {
    auto upload = tracked_sync_copy(device_tasks, tasks.data(), tasks.size() * sizeof(NativeWmmaTask), hipMemcpyHostToDevice);
    if (upload == hipSuccess && device_work)
      upload = tracked_sync_copy(device_work,schedule.explicit_work.data(),schedule.explicit_work.size()*sizeof(NativeWmmaWork),hipMemcpyHostToDevice);
    if(upload==hipSuccess && ranges)
      upload=tracked_sync_copy(ranges,schedule.ranges.data(),schedule.ranges.size()*sizeof(NativeWmmaTaskRange),hipMemcpyHostToDevice);
    if(upload==hipSuccess && !schedule.phases.empty())
      upload=tracked_sync_copy(phases,schedule.phases.data(),schedule.phases.size()*sizeof(NativeWmmaRoundPhase),hipMemcpyHostToDevice);
    return upload == hipSuccess ? hipDeviceSynchronize() : upload;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  materialized_host_bytes_=schedule.transient_host_capacity_bytes()+sizeof(schedule.header)+
      (host_tasks.capacity()+host_tasks_.capacity())*sizeof(NativeWmmaTask);
  peak_host_payload_bytes_=std::max(materialized_host_bytes_,schedule.peak_host_payload_bytes+sizeof(schedule.header)+host_tasks_.capacity()*sizeof(NativeWmmaTask));
  peak_device_bytes_=retained+bytes;
  storage_ = std::move(storage);
  host_tasks_.swap(host_tasks);
  device_id_ = device_id;
  tasks_ = device_tasks; work_ = device_work; cursor_ = cursor;
  ranges_=ranges;phases_=phases;header_=schedule.header;
  count_ = header_.work_count;
  workers_ = std::min(workers, count_);
  uniform_group_ = native_wmma_uniform_group(tasks);
  unsigned_inputs_ = tasks.front().unsigned_inputs != 0;
  poisoned_ = false;
  return RNS8_SUCCESS;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status NativeWmmaPreparedQueue::execute(bool& outputs_may_be_written) {
  outputs_may_be_written = false;
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return RNS8_BUSY;
  if (!storage_) return RNS8_INVALID_ARGUMENT;
  if (poisoned_) return RNS8_BACKEND_FAILURE;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id_);
  if (status != RNS8_SUCCESS) return status;
  const auto error = timed_hip_operation("native_queue_prepared_execute", [&]() {
    auto launch = hipMemsetAsync(cursor_, 0, sizeof(uint32_t), nullptr);
    if (launch == hipSuccess) {
      outputs_may_be_written = true;
      launch = static_cast<hipError_t>(header_.mode==0 && header_.claim_chunk==1
          ? rns8_hip_native_wmma_queue(tasks_,work_,count_,cursor_,workers_,uniform_group_,unsigned_inputs_)
          : rns8_hip_native_wmma_queue_schedule(tasks_,work_,ranges_,phases_,header_,cursor_,workers_,uniform_group_,unsigned_inputs_));
      // Drain the submitted cursor reset even when launch validation fails.
      // A failed synchronization still poisons the owner below.
      const auto completion = hipDeviceSynchronize();
      return launch == hipSuccess ? completion : launch;
    }
    return launch;
  });
  if (error != hipSuccess) { poisoned_ = true; return RNS8_BACKEND_FAILURE; }
  return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status NativeWmmaPreparedQueue::execute_ordinary(bool& outputs_may_be_written) {
  outputs_may_be_written = false;
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return RNS8_BUSY;
  if (!storage_) return RNS8_INVALID_ARGUMENT;
  if (poisoned_) return RNS8_BACKEND_FAILURE;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id_);
  if (status != RNS8_SUCCESS) return status;
  const auto error = timed_hip_operation("native_queue_prepared_ordinary_grids", [&]() {
    auto launch = hipSuccess;
    for (const auto& task : host_tasks_) {
      outputs_may_be_written = true;
      launch = static_cast<hipError_t>(task.encoded_b
          ? rns8_hip_native_encoded_b_wmma(task.a,task.encoded_b,task.c,int(task.m),int(task.n),int(task.k),
              int(task.lda),int(task.ldc),task.prefix,task.encoded_k_words,task.group,task.unsigned_inputs!=0)
          : rns8_hip_native_wmma_grouped(task.a,task.b,task.c,int(task.m),int(task.n),int(task.k),
              int(task.lda),int(task.ldb),int(task.ldc),task.prefix,task.group,task.unsigned_inputs!=0));
      if (launch != hipSuccess) break;
    }
    // Drain earlier grids even if a later launch failed. The whole batch is
    // poisoned on either launch or completion failure; no partial publication.
    const auto completion = hipDeviceSynchronize();
    return launch == hipSuccess ? completion : launch;
  });
  if (error != hipSuccess) { poisoned_ = true; return RNS8_BACKEND_FAILURE; }
  return RNS8_SUCCESS;
#else
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status NativeWmmaPreparedQueue::execute_completion(NativeWmmaCompletionTelemetry& result,
    bool& outputs_may_be_written) {
  outputs_may_be_written=false;
  std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
  if(!lock.owns_lock())return RNS8_BUSY;
  if(!storage_)return RNS8_INVALID_ARGUMENT;
  if(poisoned_)return RNS8_BACKEND_FAILURE;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id_);if(status!=RNS8_SUCCESS)return status;
  NativeWmmaCompletionTelemetry receipt;
  try {receipt.completed_work.resize(header_.task_count);receipt.completion_ticks.resize(header_.task_count);}
  catch(const std::bad_alloc&) {return RNS8_INTERNAL_ERROR;}
  if(hipDeviceGetAttribute(&receipt.wall_clock_rate_khz,hipDeviceAttributeWallClockRate,device_id_)!=hipSuccess ||
      receipt.wall_clock_rate_khz<=0)return RNS8_UNSUPPORTED_BACKEND;
  receipt.diagnostic_device_bytes=sizeof(uint64_t)*(header_.task_count+1)+sizeof(uint32_t)*header_.task_count;
  hip_direct_device_buffer telemetry;
  status=telemetry.allocate(device_id_,receipt.diagnostic_device_bytes);if(status!=RNS8_SUCCESS)return status;
  auto* start=static_cast<uint64_t*>(telemetry.get());
  auto* ticks=start+1;auto* completed=reinterpret_cast<uint32_t*>(ticks+header_.task_count);
  auto launch=hipMemsetAsync(cursor_,0,sizeof(uint32_t),nullptr);
  if(launch==hipSuccess) {
    outputs_may_be_written=true;
    launch=static_cast<hipError_t>(rns8_hip_native_wmma_queue_completion(tasks_,work_,ranges_,phases_,header_,
        cursor_,workers_,completed,ticks,start));
  }
  const auto completion=hipDeviceSynchronize();
  if(launch!=hipSuccess || completion!=hipSuccess){poisoned_=true;return RNS8_BACKEND_FAILURE;}
  status=hip_direct_copy_device_to_host(device_id_,receipt.completed_work.data(),completed,header_.task_count*sizeof(uint32_t));
  if(status==RNS8_SUCCESS)status=hip_direct_copy_device_to_host(device_id_,receipt.completion_ticks.data(),ticks,header_.task_count*sizeof(uint64_t));
  if(status!=RNS8_SUCCESS){poisoned_=true;return status;}
  for(uint32_t i=0;i<header_.task_count;++i) {
    const auto& task=host_tasks_[i];
    const uint64_t required=uint64_t((task.m+15)/16)*((task.n+15)/16)*((task.prefix+task.group-1)/task.group);
    if(receipt.completed_work[i]!=required || !receipt.completion_ticks[i]){poisoned_=true;return RNS8_BACKEND_FAILURE;}
  }
  result=std::move(receipt);return RNS8_SUCCESS;
#else
  (void)result;return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_gemm_native_wmma_queue(
    int device_id, const std::vector<NativeWmmaTask>& tasks, uint32_t workers,
    uint64_t max_scratch_bytes, bool& outputs_may_be_written) {
  outputs_may_be_written = false;
  std::vector<NativeWmmaWork> work;
  uint64_t bytes = 0;
  auto status = build_native_wmma_queue(tasks, workers, max_scratch_bytes, work, bytes);
  if (status != RNS8_SUCCESS) return status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  const auto uniform_group = native_wmma_uniform_group(tasks);
  status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS) return status;
  hip_direct_device_buffer storage;
  status = storage.allocate(device_id, static_cast<std::size_t>(bytes));
  if (status != RNS8_SUCCESS) return status;
  auto* device_tasks = static_cast<NativeWmmaTask*>(storage.get());
  auto* device_work = reinterpret_cast<NativeWmmaWork*>(device_tasks + tasks.size());
  auto* cursor = reinterpret_cast<uint32_t*>(device_work + work.size());
  auto error = timed_hip_operation("native_queue_capability", [&]() {
    auto launch = static_cast<hipError_t>(rns8_hip_native_wmma_capability(reinterpret_cast<int*>(cursor)));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  uint32_t supported = 0;
  status = hip_direct_copy_device_to_host(device_id, &supported, cursor, sizeof(supported));
  if (status != RNS8_SUCCESS) return status;
  if (!supported) return RNS8_UNSUPPORTED_BACKEND;
  error = timed_hip_operation("native_queue_descriptor_upload", [&]() {
    auto upload = tracked_sync_copy(device_tasks, tasks.data(), tasks.size() * sizeof(NativeWmmaTask), hipMemcpyHostToDevice);
    if (upload == hipSuccess) upload = tracked_sync_copy(device_work, work.data(), work.size() * sizeof(NativeWmmaWork), hipMemcpyHostToDevice);
    if (upload == hipSuccess) upload = hipMemset(cursor, 0, sizeof(uint32_t));
    return upload == hipSuccess ? hipDeviceSynchronize() : upload;
  });
  if (error != hipSuccess) return RNS8_BACKEND_FAILURE;
  outputs_may_be_written = true;
  error = timed_hip_operation("native_queue_persistent_wmma", [&]() {
    auto launch = static_cast<hipError_t>(rns8_hip_native_wmma_queue(device_tasks, device_work,
        static_cast<uint32_t>(work.size()), cursor, std::min(workers, static_cast<uint32_t>(work.size())),
        uniform_group, tasks.front().unsigned_inputs != 0));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_native_wmma_queue_schedule_on_stream(const NativeWmmaTask*,const NativeWmmaWork*,
    const NativeWmmaTaskRange*,const NativeWmmaRoundPhase*,NativeWmmaScheduleHeader,uint32_t*,uint32_t,uint32_t,bool,void*);
extern "C" int rns8_hip_native_wmma_grouped_on_stream(const void*,const void*,int8_t*,int,int,int,int,int,int,int,int,bool,void*);
extern "C" int rns8_hip_native_encoded_b_wmma_on_stream(const void*,const void*,void*,int,int,int,int,int,int,uint64_t,int,bool,void*);
#endif
rns8_status NativeWmmaPreparedQueue::enqueue_borrowed_stream(void* stream,bool ordinary,bool& written) {
  written=false;std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
  if(!lock.owns_lock())return RNS8_BUSY;
  if(!stream || !storage_)return RNS8_INVALID_ARGUMENT;
  if(poisoned_)return RNS8_BACKEND_FAILURE;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status=set_hip_device(device_id_);if(status!=RNS8_SUCCESS)return status;
  hipError_t launch=hipSuccess;
  if(ordinary) {
    for(const auto& task:host_tasks_) {
      written=true;
      launch=static_cast<hipError_t>(task.encoded_b?
        rns8_hip_native_encoded_b_wmma_on_stream(task.a,task.encoded_b,task.c,int(task.m),int(task.n),int(task.k),
          int(task.lda),int(task.ldc),task.prefix,task.encoded_k_words,task.group,task.unsigned_inputs!=0,stream):
        rns8_hip_native_wmma_grouped_on_stream(task.a,task.b,task.c,int(task.m),int(task.n),int(task.k),
          int(task.lda),int(task.ldb),int(task.ldc),task.prefix,task.group,task.unsigned_inputs!=0,stream));
      if(launch!=hipSuccess)break;
    }
  } else {
    launch=hipMemsetAsync(cursor_,0,sizeof(uint32_t),static_cast<hipStream_t>(stream));
    if(launch==hipSuccess) {
      written=true;
      launch=static_cast<hipError_t>(rns8_hip_native_wmma_queue_schedule_on_stream(tasks_,work_,ranges_,phases_,header_,
        cursor_,workers_,uniform_group_,unsigned_inputs_,stream));
    }
  }
  if(launch!=hipSuccess){poisoned_=true;return RNS8_BACKEND_FAILURE;}
  return RNS8_SUCCESS;
#else
  (void)ordinary;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
rns8_status NativeWmmaPreparedQueue::close() {
  std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
  if(!lock.owns_lock())return RNS8_BUSY;
  if(poisoned_ && storage_ && hip_direct_synchronize(device_id_)!=RNS8_SUCCESS)return RNS8_BACKEND_FAILURE;
  const auto status=storage_.reset();if(status!=RNS8_SUCCESS)return status;
  host_tasks_.clear();tasks_=nullptr;work_=nullptr;ranges_=nullptr;phases_=nullptr;cursor_=nullptr;
  count_=workers_=uniform_group_=0;device_id_=-1;header_={};poisoned_=false;
  materialized_host_bytes_=peak_device_bytes_=peak_host_payload_bytes_=0;
  return RNS8_SUCCESS;
}
