namespace {

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
constexpr std::size_t kPinnedExportStagingMinBytes = 64u * 1024u;
thread_local hip_pinned_host_buffer g_pinned_export_staging;
#endif
std::atomic<uint64_t> g_hip_direct_allocate_calls{0};
std::atomic<uint64_t> g_hip_direct_free_calls{0};
std::atomic<uint64_t> g_hip_direct_allocated_bytes{0};

constexpr uint32_t kKnownTileScheduleFlags =
    RNS8_TILE_SCHEDULE_ZERO_OUTPUT | RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT;

void mark_device_residue_output_current(rns8_matrix* matrix) {
  if (!matrix) {
    return;
  }
  api::mark_output_device_residues_current(*matrix);
}

}  // namespace

void hip_direct_allocation_counters_reset() {
  g_hip_direct_allocate_calls.store(0, std::memory_order_relaxed);
  g_hip_direct_free_calls.store(0, std::memory_order_relaxed);
  g_hip_direct_allocated_bytes.store(0, std::memory_order_relaxed);
}

hip_direct_allocation_counters hip_direct_allocation_counters_snapshot() {
  hip_direct_allocation_counters counters{};
  counters.allocate_calls = g_hip_direct_allocate_calls.load(std::memory_order_relaxed);
  counters.free_calls = g_hip_direct_free_calls.load(std::memory_order_relaxed);
  counters.allocated_bytes = g_hip_direct_allocated_bytes.load(std::memory_order_relaxed);
  return counters;
}

