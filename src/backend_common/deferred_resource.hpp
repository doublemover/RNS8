#pragma once

namespace rns8::detail {
// Intrusive retry nodes are allocated before runtime acquisition. Drains can
// requeue failed destruction without allocating or changing resource identity.
enum class hip_deferred_allocation_kind {
  device,
  pinned_host,
  event,
  stream,
  graph,
  graph_exec,
  library_handle
};
struct hip_deferred_buffer_cleanup {
  hip_deferred_allocation_kind kind = hip_deferred_allocation_kind::device;
  int device_id = -1;
  void* pointer = nullptr;
  bool (*release_library_handle)(int, void*) noexcept = nullptr;
  hip_deferred_buffer_cleanup* next = nullptr;
};
} // namespace rns8::detail
