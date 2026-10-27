namespace {

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
constexpr std::size_t kPinnedExportStagingMinBytes = 64u * 1024u;
thread_local hip_pinned_host_buffer g_pinned_export_staging;
#endif
std::atomic<uint64_t> g_hip_direct_allocate_calls{0};
std::atomic<uint64_t> g_hip_direct_free_calls{0};
std::atomic<uint64_t> g_hip_direct_allocated_bytes{0};
std::atomic<uint64_t> g_hip_direct_h2d_calls{0};
std::atomic<uint64_t> g_hip_direct_h2d_bytes{0};
std::atomic<uint64_t> g_hip_direct_d2h_calls{0};
std::atomic<uint64_t> g_hip_direct_d2h_bytes{0};
std::atomic<uint64_t> g_hip_direct_d2d_calls{0};
std::atomic<uint64_t> g_hip_direct_d2d_bytes{0};

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
// Completed synchronous API payload, excluding padding in a 2D pitch. These
// counters do not measure physical bus traffic or replayed graph transfers.
void record_sync_copy(hipMemcpyKind kind, uint64_t bytes) {
  auto* calls = kind==hipMemcpyHostToDevice ? &g_hip_direct_h2d_calls :
      kind==hipMemcpyDeviceToHost ? &g_hip_direct_d2h_calls :
      kind==hipMemcpyDeviceToDevice ? &g_hip_direct_d2d_calls : nullptr;
  auto* payload = kind==hipMemcpyHostToDevice ? &g_hip_direct_h2d_bytes :
      kind==hipMemcpyDeviceToHost ? &g_hip_direct_d2h_bytes :
      kind==hipMemcpyDeviceToDevice ? &g_hip_direct_d2d_bytes : nullptr;
  if(calls) { calls->fetch_add(1,std::memory_order_relaxed); payload->fetch_add(bytes,std::memory_order_relaxed); }
}
hipError_t tracked_sync_copy(void* dst,const void* src,size_t bytes,hipMemcpyKind kind) {
  const auto status=hipMemcpy(dst,src,bytes,kind);
  if(status==hipSuccess) record_sync_copy(kind,bytes);
  return status;
}
hipError_t tracked_sync_copy_2d(void* dst,size_t dpitch,const void* src,size_t spitch,
                              size_t width,size_t height,hipMemcpyKind kind) {
  if(height && width>SIZE_MAX/height) return hipErrorInvalidValue;
  const auto status=hipMemcpy2D(dst,dpitch,src,spitch,width,height,kind);
  if(status==hipSuccess) record_sync_copy(kind,uint64_t(width)*height);
  return status;
}
#endif


constexpr uint32_t kKnownTileScheduleFlags =
    RNS8_TILE_SCHEDULE_ZERO_OUTPUT | RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT;

void mark_device_residue_output_current(rns8_matrix* matrix, uint32_t written_prefix = 0, const rns8_plan* plan = nullptr) {
  if (!matrix) {
    return;
  }
  api::mark_output_device_residues_current(*matrix, plan);
  if (written_prefix) matrix->materialized_rns_prefix = written_prefix;
}

}  // namespace

AllocationObservationLedger* hip_direct_allocation_ledger(int device_id) noexcept {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(device_id<0)return nullptr;
  struct Registry {
    std::mutex mutex;
    std::atomic<bool> failed{false};
    std::map<int,std::unique_ptr<AllocationObservationLedger>> devices;
  };
  // Reachable process-lifetime metadata also serves late TLS/deferred cleanup;
  // destruction order must not resurrect a ledger after its owners retire.
  static Registry* const registry=[]() noexcept -> Registry* {try{return new Registry;}catch(...){return nullptr;}}();
  if(!registry || registry->failed.load())return nullptr;
  try {
    std::lock_guard<std::mutex> lock(registry->mutex);
    auto found=registry->devices.find(device_id);
    if(found!=registry->devices.end())return found->second.get();
    auto ledger=std::make_unique<AllocationObservationLedger>();auto* result=ledger.get();
    registry->devices.emplace(device_id,std::move(ledger));return result;
  }catch(...){registry->failed=true;return nullptr;}
#else
  (void)device_id;return nullptr;
#endif
}

void hip_direct_allocation_counters_reset() {
  g_hip_direct_allocate_calls.store(0, std::memory_order_relaxed);
  g_hip_direct_free_calls.store(0, std::memory_order_relaxed);
  g_hip_direct_allocated_bytes.store(0, std::memory_order_relaxed);
  g_hip_direct_h2d_calls.store(0, std::memory_order_relaxed);
  g_hip_direct_h2d_bytes.store(0, std::memory_order_relaxed);
  g_hip_direct_d2h_calls.store(0, std::memory_order_relaxed);
  g_hip_direct_d2h_bytes.store(0, std::memory_order_relaxed);
  g_hip_direct_d2d_calls.store(0, std::memory_order_relaxed);
  g_hip_direct_d2d_bytes.store(0, std::memory_order_relaxed);

}

hip_direct_allocation_counters hip_direct_allocation_counters_snapshot() {
  hip_direct_allocation_counters counters{};
  counters.allocate_calls = g_hip_direct_allocate_calls.load(std::memory_order_relaxed);
  counters.free_calls = g_hip_direct_free_calls.load(std::memory_order_relaxed);
  counters.allocated_bytes = g_hip_direct_allocated_bytes.load(std::memory_order_relaxed);
  counters.h2d_calls = g_hip_direct_h2d_calls.load(std::memory_order_relaxed);
  counters.h2d_bytes = g_hip_direct_h2d_bytes.load(std::memory_order_relaxed);
  counters.d2h_calls = g_hip_direct_d2h_calls.load(std::memory_order_relaxed);
  counters.d2h_bytes = g_hip_direct_d2h_bytes.load(std::memory_order_relaxed);
  counters.d2d_calls = g_hip_direct_d2d_calls.load(std::memory_order_relaxed);
  counters.d2d_bytes = g_hip_direct_d2d_bytes.load(std::memory_order_relaxed);
  return counters;
}

