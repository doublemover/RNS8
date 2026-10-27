namespace {

constexpr uint16_t kHipblasLtDefaultModuli[RNS8_DEFAULT_MODULUS_COUNT] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

constexpr uint64_t kReciprocalScale = 1ull << 32u;

uint32_t modulus_reciprocal_u32(uint16_t modulus) {
  return static_cast<uint32_t>(kReciprocalScale / static_cast<uint32_t>(modulus));
}

bool checked_common_shape(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  if (m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < n || ldc < n) {
    return false;
  }
  return m <= std::numeric_limits<int>::max() && n <= std::numeric_limits<int>::max() &&
         k <= std::numeric_limits<int>::max() && lda <= std::numeric_limits<int>::max() &&
         ldb <= std::numeric_limits<int>::max() && ldc <= std::numeric_limits<int>::max();
}

bool checked_workspace(int64_t m, int64_t n, int64_t k, std::size_t scratch_bytes, std::size_t workspace_bytes) {
  std::size_t required_scratch = 0;
  std::size_t required_workspace = 0;
  if (!hipblaslt_baseline_workspace_requirements(m, n, k, required_scratch, required_workspace)) {
    return false;
  }
  return scratch_bytes >= required_scratch && workspace_bytes >= required_workspace;
}

#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
struct padded_block_shape {
  int m = 0;
  int n = 0;
  int k = 0;
};

bool padded_block_shape_for(int64_t m, int64_t n, int64_t k_block, padded_block_shape& out) {
  uint64_t padded_m = 0;
  uint64_t padded_n = 0;
  uint64_t padded_k = 0;
  if (!hipblaslt_round_up_aligned(static_cast<uint64_t>(m), padded_m) ||
      !hipblaslt_round_up_aligned(static_cast<uint64_t>(n), padded_n) ||
      !hipblaslt_round_up_aligned(static_cast<uint64_t>(k_block), padded_k)) {
    return false;
  }
  out.m = static_cast<int>(padded_m);
  out.n = static_cast<int>(padded_n);
  out.k = static_cast<int>(padded_k);
  return true;
}

rns8_status set_hip_device(int device_id) {
  if (device_id < 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  const hipError_t err = hipSetDevice(device_id);
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

rns8_status status_from_hipblas(hipblasStatus_t status) {
  switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
      return RNS8_SUCCESS;
    case HIPBLAS_STATUS_NOT_SUPPORTED:
    case HIPBLAS_STATUS_ARCH_MISMATCH:
      return RNS8_UNSUPPORTED_BACKEND;
    case HIPBLAS_STATUS_INVALID_VALUE:
    case HIPBLAS_STATUS_INVALID_ENUM:
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
      return RNS8_INVALID_ARGUMENT;
    case HIPBLAS_STATUS_ALLOC_FAILED:
      return RNS8_INTERNAL_ERROR;
    case HIPBLAS_STATUS_NOT_INITIALIZED:
    case HIPBLAS_STATUS_MAPPING_ERROR:
    case HIPBLAS_STATUS_EXECUTION_FAILED:
    case HIPBLAS_STATUS_INTERNAL_ERROR:
    case HIPBLAS_STATUS_UNKNOWN:
      return RNS8_BACKEND_FAILURE;
  }
  return RNS8_BACKEND_FAILURE;
}

hipError_t create_recorded_timing_event_pair(hip_unique_event_pair& events) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    const hipError_t status = events.create_and_record_start();
    if (status == hipSuccess) {
      return hipSuccess;
    }
    events.reset();
    (void)hipDeviceSynchronize();
  }
  return hipErrorUnknown;
}

void record_synchronized_host_timing(const char* label, std::chrono::steady_clock::time_point start) {
  (void)hipDeviceSynchronize();
  const auto stop = std::chrono::steady_clock::now();
  const double microseconds = std::chrono::duration<double, std::micro>(stop - start).count();
  hip_direct_timing_record_failure(label,"synchronized_host_fallback",-1);
  hip_direct_timing_record_sample((std::string("host_synchronized_fallback:")+label).c_str(), microseconds);
}

bool record_elapsed_event_sample(const char* label, hipEvent_t start, hipEvent_t stop) {
  hipError_t sync_status = hipEventSynchronize(stop);
  if (sync_status != hipSuccess) {
    (void)hipDeviceSynchronize();
    sync_status = hipEventSynchronize(stop);
  }
  if (sync_status != hipSuccess) {
    return false;
  }
  float milliseconds = 0.0f;
  hipError_t elapsed_status = hipEventElapsedTime(&milliseconds, start, stop);
  if (elapsed_status != hipSuccess) {
    (void)hipDeviceSynchronize();
    elapsed_status = hipEventElapsedTime(&milliseconds, start, stop);
  }
  if (elapsed_status != hipSuccess || milliseconds < 0.0f) {
    return false;
  }
  hip_direct_timing_record_sample(label, static_cast<double>(milliseconds) * 1000.0);
  return true;
}

template <typename Fn>
hipblasStatus_t timed_hipblaslt_operation(const char* label, Fn&& fn) {
  if (!hip_direct_timing_enabled()) {
    return fn();
  }
  const auto host_start = std::chrono::steady_clock::now();
  hip_unique_event_pair events;
  if (create_recorded_timing_event_pair(events) != hipSuccess) {
    const hipblasStatus_t status = fn();
    if (status == HIPBLAS_STATUS_SUCCESS) {
      record_synchronized_host_timing(label, host_start);
    }
    return status;
  }
  const hipblasStatus_t status = fn();
  if (status == HIPBLAS_STATUS_SUCCESS) {
    // hipBLASLt may enqueue work in a way that is not reliably delimited by
    // default-stream stop events on small Windows RDNA3 captures. This path is
    // active only for benchmark event timing; normal runtime calls bypass it.
    (void)hipDeviceSynchronize();
    hipError_t err = events.record_stop();
    if (err != hipSuccess) {
      (void)hipDeviceSynchronize();
      err = events.record_stop();
    }
    if (err == hipSuccess) {
      if (record_elapsed_event_sample(label, events.start(), events.stop())) {
        return status;
      }
    }
    record_synchronized_host_timing(label, host_start);
  }
  return status;
}

template <typename Fn>
hipError_t timed_hip_operation(const char* label, Fn&& fn) {
  if (!hip_direct_timing_enabled()) {
    return fn();
  }
  const auto host_start = std::chrono::steady_clock::now();
  hip_unique_event_pair events;
  if (create_recorded_timing_event_pair(events) != hipSuccess) {
    const hipError_t status = fn();
    if (status == hipSuccess) {
      record_synchronized_host_timing(label, host_start);
    }
    return status;
  }
  const hipError_t status = fn();
  if (status == hipSuccess) {
    hipError_t err = events.record_stop();
    if (err != hipSuccess) {
      (void)hipDeviceSynchronize();
      err = events.record_stop();
    }
    if (err == hipSuccess) {
      if (record_elapsed_event_sample(label, events.start(), events.stop())) {
        return status;
      }
    }
    record_synchronized_host_timing(label, host_start);
  }
  return status;
}

struct matmul_descriptors {
  hipblasLtMatmulDesc_t op = nullptr;
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;

  matmul_descriptors() = default;
  matmul_descriptors(const matmul_descriptors&) = delete;
  matmul_descriptors& operator=(const matmul_descriptors&) = delete;

  ~matmul_descriptors() {
    if (preference) {
      ++metadata_counters.preference_destroy_calls;
      (void)hipblasLtMatmulPreferenceDestroy(preference);
    }
    if (d) {
      ++metadata_counters.descriptor_destroy_calls;
      (void)hipblasLtMatrixLayoutDestroy(d);
    }
    if (c) {
      ++metadata_counters.descriptor_destroy_calls;
      (void)hipblasLtMatrixLayoutDestroy(c);
    }
    if (b) {
      ++metadata_counters.descriptor_destroy_calls;
      (void)hipblasLtMatrixLayoutDestroy(b);
    }
    if (a) {
      ++metadata_counters.descriptor_destroy_calls;
      (void)hipblasLtMatrixLayoutDestroy(a);
    }
    if (op) {
      ++metadata_counters.descriptor_destroy_calls;
      (void)hipblasLtMatmulDescDestroy(op);
    }
  }
};

using matmul_algorithm_cache_key=HipblasLtMetadataKey;

struct matmul_algorithm_cache_entry {
  matmul_algorithm_cache_key key{};
  hipblasLtMatmulAlgo_t algo{};
  int algorithm_index=-1;
  bool identity_attempted=false;
};

std::mutex& matmul_algorithm_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<matmul_algorithm_cache_entry>& matmul_algorithm_cache_entries() {
  static std::vector<matmul_algorithm_cache_entry> entries;
  return entries;
}

int observe_algorithm_index(const hipblasLtMatmulAlgo_t& algo) {
#if defined(RNS8_HIPBLASLT_ALGO_INDEX_AVAILABLE) && RNS8_HIPBLASLT_ALGO_INDEX_AVAILABLE
  auto& inventory=hipblaslt_metadata_state::algorithm_inventory;
  if(inventory.index_queries==UINT64_MAX)inventory.complete=false;
  else ++inventory.index_queries;
  return rns8_hipblaslt_algorithm_index(&algo);
#else
  (void)algo;return -1;
#endif
}

bool find_cached_matmul_algorithm(const matmul_algorithm_cache_key& key, hipblasLtMatmulAlgo_t& algo,int& index) {
  std::lock_guard<std::mutex> lock(matmul_algorithm_cache_mutex());
  auto& entries = matmul_algorithm_cache_entries();
  const auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) { return entry.key == key; });
  if (it == entries.end()) {
    return false;
  }
  algo = it->algo;
  if(hipblaslt_metadata_state::algorithm_observations_enabled && !it->identity_attempted) {
    it->algorithm_index=observe_algorithm_index(it->algo);
    it->identity_attempted=true;
  }
  index=it->algorithm_index;
  return true;
}

void remember_matmul_algorithm(const matmul_algorithm_cache_key& key, const hipblasLtMatmulAlgo_t& algo,int index,bool attempted) {
  std::lock_guard<std::mutex> lock(matmul_algorithm_cache_mutex());
  auto& entries = matmul_algorithm_cache_entries();
  const auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) { return entry.key == key; });
  if (it != entries.end()) {
    it->algo = algo;
    it->algorithm_index=index;it->identity_attempted=attempted;
    return;
  }
  entries.push_back(matmul_algorithm_cache_entry{key, algo,index,attempted});
}

hipblasStatus_t create_matmul_descriptors(
    const padded_block_shape& shape,
    int64_t scratch_ld,
    matmul_descriptors& descriptors) {
  ++metadata_counters.descriptor_create_calls;
  hipblasStatus_t status = hipblasLtMatmulDescCreate(&descriptors.op, HIPBLAS_COMPUTE_32I, HIP_R_32I);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  const hipblasOperation_t op_n = HIPBLAS_OP_N;
  ++metadata_counters.attribute_set_calls;
  status = hipblasLtMatmulDescSetAttribute(descriptors.op, HIPBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  ++metadata_counters.attribute_set_calls;
  status = hipblasLtMatmulDescSetAttribute(descriptors.op, HIPBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  ++metadata_counters.descriptor_create_calls;
  status = hipblasLtMatrixLayoutCreate(
      &descriptors.a, HIP_R_8I, static_cast<uint64_t>(shape.n), static_cast<uint64_t>(shape.k), shape.n);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  ++metadata_counters.descriptor_create_calls;
  status = hipblasLtMatrixLayoutCreate(
      &descriptors.b, HIP_R_8I, static_cast<uint64_t>(shape.k), static_cast<uint64_t>(shape.m), shape.k);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  ++metadata_counters.descriptor_create_calls;
  status = hipblasLtMatrixLayoutCreate(
      &descriptors.c, HIP_R_32I, static_cast<uint64_t>(shape.n), static_cast<uint64_t>(shape.m), scratch_ld);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  ++metadata_counters.descriptor_create_calls;
  status = hipblasLtMatrixLayoutCreate(
      &descriptors.d, HIP_R_32I, static_cast<uint64_t>(shape.n), static_cast<uint64_t>(shape.m), scratch_ld);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  ++metadata_counters.descriptor_sets_created;
  return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t create_matmul_preference(std::size_t workspace_bytes,matmul_descriptors& descriptors) {
  if(descriptors.preference)return HIPBLAS_STATUS_SUCCESS;
  ++metadata_counters.preference_create_calls;
  auto status=hipblasLtMatmulPreferenceCreate(&descriptors.preference);
  if(status!=HIPBLAS_STATUS_SUCCESS)return status;
  const uint64_t max_workspace=workspace_bytes;
  ++metadata_counters.preference_attribute_calls;
  return hipblasLtMatmulPreferenceSetAttribute(descriptors.preference,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&max_workspace,sizeof(max_workspace));
}

struct matmul_call_metadata {
  HipblasLtCallMetadataDirectory directory;
  std::array<matmul_descriptors,2> descriptors;
  bool submitted=false,drained=false;
  // Error paths may have already enqueued earlier planes. Keep their metadata
  // alive through a drain before member destruction; ordinary success uses the
  // existing one final synchronization, not another per-plane synchronization.
  ~matmul_call_metadata(){if(submitted&&!drained)(void)hipDeviceSynchronize();}
};

hipblasStatus_t run_matmul_block(
    int device_id,
    hipblasLtHandle_t handle,
    const int8_t* packed_a_t,
    const int8_t* packed_b_t,
    int32_t* scratch,
    void* workspace,
    std::size_t workspace_bytes,
    const padded_block_shape& shape,
    matmul_call_metadata& call) {
  ++metadata_counters.matmul_block_calls;
  matmul_descriptors per_block;
  const int64_t scratch_ld=shape.n;
  const matmul_algorithm_cache_key cache_key{device_id,shape.m,shape.n,shape.k,int(scratch_ld),workspace_bytes};
  matmul_descriptors* selected=&per_block;
  hipblasStatus_t status=HIPBLAS_STATUS_SUCCESS;
  if(metadata_policy==HipblasLtMetadataPolicy::per_call) {
    int slot=call.directory.find(cache_key);
    if(slot>=0)++metadata_counters.descriptor_set_hits;
    else {
      slot=call.directory.free_slot();
      if(slot<0)return HIPBLAS_STATUS_INTERNAL_ERROR; // proven full/tail geometry cap
      status=create_matmul_descriptors(shape,scratch_ld,call.descriptors[slot]);
      if(status!=HIPBLAS_STATUS_SUCCESS)return status;
      if(!call.directory.publish(unsigned(slot),cache_key))return HIPBLAS_STATUS_INTERNAL_ERROR;
      const uint64_t live=uint64_t(call.directory.ready[0])+uint64_t(call.directory.ready[1]);
      metadata_counters.maximum_live_sets=std::max(metadata_counters.maximum_live_sets,live);
    }
    selected=&call.descriptors[slot];
  } else {
    status=create_matmul_descriptors(shape,scratch_ld,per_block);
    if(status!=HIPBLAS_STATUS_SUCCESS)return status;
    // Exact old-path control includes the otherwise unnecessary preference.
    status=create_matmul_preference(workspace_bytes,per_block);
    if(status!=HIPBLAS_STATUS_SUCCESS)return status;
    metadata_counters.maximum_live_sets=std::max(metadata_counters.maximum_live_sets,uint64_t{1});
  }
  auto& descriptors=*selected;
  hipblasLtMatmulAlgo_t algo{};
  int algorithm_index=-1;
  if (!find_cached_matmul_algorithm(cache_key, algo,algorithm_index)) {
    status=create_matmul_preference(workspace_bytes,descriptors);
    if(status!=HIPBLAS_STATUS_SUCCESS)return status;
    hipblasLtMatmulHeuristicResult_t heuristic{};
    int heuristic_count = 0;
    ++metadata_counters.heuristic_calls;
    status = hipblasLtMatmulAlgoGetHeuristic(
        handle,
        descriptors.op,
        descriptors.a,
        descriptors.b,
        descriptors.c,
        descriptors.d,
        descriptors.preference,
        1,
        &heuristic,
        &heuristic_count);
    if (status == HIPBLAS_STATUS_SUCCESS && heuristic_count > 0 && heuristic.state == HIPBLAS_STATUS_SUCCESS) {
      algo = heuristic.algo;
      const bool observe=hipblaslt_metadata_state::algorithm_observations_enabled;
      if(observe)algorithm_index=observe_algorithm_index(algo);
      remember_matmul_algorithm(cache_key, algo,algorithm_index,observe);
    } else if (status == HIPBLAS_STATUS_SUCCESS || status == HIPBLAS_STATUS_NOT_SUPPORTED) {
      return HIPBLAS_STATUS_NOT_SUPPORTED;
    } else {
      return status;
    }
  } else ++metadata_counters.algorithm_cache_hits;
  if(hipblaslt_metadata_state::algorithm_observations_enabled)
    hipblaslt_metadata_state::algorithm_inventory.record(cache_key,algorithm_index);

  const int32_t alpha = 1;
  const int32_t beta = 0;
  call.submitted=true;
  return timed_hipblaslt_operation("hipblaslt_int8_i32_matmul", [&]() {
    return hipblasLtMatmul(
        handle,
        descriptors.op,
        &alpha,
        packed_b_t,
        descriptors.a,
        packed_a_t,
        descriptors.b,
        &beta,
        scratch,
        descriptors.c,
        scratch,
        descriptors.d,
        &algo,
        workspace,
        workspace_bytes,
        nullptr);
  });
}

#endif
