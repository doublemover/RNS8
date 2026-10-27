#pragma once
#include "core/hip_handle_owner.hpp"
#if (defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP) || defined(__HIPCC__)
#include <hip/hip_runtime_api.h>
namespace rns8::detail {
void hip_direct_defer_buffer_cleanup(hip_deferred_buffer_cleanup*) noexcept;
void hip_direct_timing_flush_pending_events();
class hip_scoped_device {
 public:
  explicit hip_scoped_device(int device) noexcept {
    status_ = hipGetDevice(&previous_);
    if (status_ == hipSuccess && previous_ != device) {
      status_ = hipSetDevice(device);
      changed_ = status_ == hipSuccess;
    }
  }
  hip_scoped_device(const hip_scoped_device&) = delete;
  hip_scoped_device& operator=(const hip_scoped_device&) = delete;
  ~hip_scoped_device() {
    if (changed_) (void)hipSetDevice(previous_);
  }
  hipError_t status() const noexcept { return status_; }

 private:
  int previous_ = -1;
  bool changed_ = false;
  hipError_t status_ = hipSuccess;
};
template <class Handle, hip_deferred_allocation_kind Kind>
struct hip_runtime_handle_operations {
  using handle_type = Handle;
  using status_type = hipError_t;
  static constexpr auto success = hipSuccess, out_of_memory = hipErrorOutOfMemory,
                        invalid_argument = hipErrorInvalidValue;
  static constexpr auto kind = Kind;
  static hipError_t current_device(int& device) noexcept { return hipGetDevice(&device); }
  static hipError_t destroy(int device, Handle handle) noexcept {
    hip_scoped_device selected(device);
    if (selected.status() != hipSuccess) return selected.status();
    if constexpr (Kind == hip_deferred_allocation_kind::event)
      return hipEventDestroy(handle);
    else if constexpr (Kind == hip_deferred_allocation_kind::stream)
      return hipStreamDestroy(handle);
    else if constexpr (Kind == hip_deferred_allocation_kind::graph)
      return hipGraphDestroy(handle);
    else
      return hipGraphExecDestroy(handle);
  }
  static void defer(hip_deferred_buffer_cleanup* record) noexcept {
    hip_direct_defer_buffer_cleanup(record);
  }
};
class hip_unique_event
    : public basic_hip_handle<
          hip_runtime_handle_operations<hipEvent_t, hip_deferred_allocation_kind::event>> {
  using Base = basic_hip_handle<
      hip_runtime_handle_operations<hipEvent_t, hip_deferred_allocation_kind::event>>;

 public:
  hipError_t create(unsigned flags = hipEventDefault) {
    return Base::create(
        [flags](hipEvent_t* out) noexcept { return hipEventCreateWithFlags(out, flags); });
  }
};
class hip_unique_stream
    : public basic_hip_handle<
          hip_runtime_handle_operations<hipStream_t, hip_deferred_allocation_kind::stream>> {
  using Base = basic_hip_handle<
      hip_runtime_handle_operations<hipStream_t, hip_deferred_allocation_kind::stream>>;

 public:
  hipError_t create_non_blocking() {
    return Base::create([](hipStream_t* out) noexcept {
      return hipStreamCreateWithFlags(out, hipStreamNonBlocking);
    });
  }
};
class hip_unique_graph
    : public basic_hip_handle<
          hip_runtime_handle_operations<hipGraph_t, hip_deferred_allocation_kind::graph>> {
 public:
  hipError_t end_capture(hipStream_t stream) noexcept {
    return receive([stream](hipGraph_t* out) noexcept { return hipStreamEndCapture(stream, out); });
  }
};
class hip_unique_graph_exec
    : public basic_hip_handle<
          hip_runtime_handle_operations<hipGraphExec_t, hip_deferred_allocation_kind::graph_exec>> {
  using Base = basic_hip_handle<
      hip_runtime_handle_operations<hipGraphExec_t, hip_deferred_allocation_kind::graph_exec>>;

 public:
  hipError_t instantiate(hipGraph_t graph) {
    return Base::create([graph](hipGraphExec_t* out) noexcept {
      return hipGraphInstantiate(out, graph, nullptr, nullptr, 0);
    });
  }
};
// End owned capture on exception paths too. The graph's retry node must have
// been reserved before this scope begins; unwinding performs no allocation.
class hip_capture_scope {
 public:
  hip_capture_scope(hipStream_t stream, hip_unique_graph& graph) noexcept
      : stream_(stream), graph_(graph) {}
  hip_capture_scope(const hip_capture_scope&) = delete;
  hip_capture_scope& operator=(const hip_capture_scope&) = delete;
  ~hip_capture_scope() {
    if (active_) (void)graph_.end_capture(stream_);
  }
  hipError_t begin() noexcept {
    const auto status = hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
    active_ = status == hipSuccess;
    return status;
  }
  hipError_t end() noexcept {
    active_ = false;
    return graph_.end_capture(stream_);
  }

 private:
  hipStream_t stream_;
  hip_unique_graph& graph_;
  bool active_ = false;
};
class hip_unique_event_pair {
 public:
  hipError_t create() {
    auto status = reset();
    if (status != hipSuccess) return status;
    status = start_.create();
    if (status != hipSuccess) return status;
    return stop_.create();
  }
  hipError_t create_and_record_start(hipStream_t stream = nullptr) {
    const auto status = create();
    return status == hipSuccess ? record_start(stream) : status;
  }
  hipError_t record_start(hipStream_t stream = nullptr) const {
    return hipEventRecord(start_.get(), stream);
  }
  hipError_t record_stop(hipStream_t stream = nullptr) const {
    return hipEventRecord(stop_.get(), stream);
  }
  hipEvent_t start() const noexcept { return start_.get(); }
  hipEvent_t stop() const noexcept { return stop_.get(); }
  hip_unique_event take_start() noexcept { return std::move(start_); }
  hip_unique_event take_stop() noexcept { return std::move(stop_); }
  hipError_t reset() noexcept {
    const auto stop = stop_.reset(), start = start_.reset();
    return stop != hipSuccess ? stop : start;
  }

 private:
  hip_unique_event start_, stop_;
};

// Keep the accelerator timing retry policy shared. A failed destroy leaves
// ownership intact and stops replacement instead of leaking another event pair.
inline bool hip_create_timing_events_with_retry(hip_unique_event_pair& events) {
  for (int attempt = 0; attempt < 8; ++attempt) {
    if (events.create_and_record_start() == hipSuccess) return true;
    if (events.reset() != hipSuccess) return false;
    if (attempt == 0) hip_direct_timing_flush_pending_events();
    if (attempt + 1 < 8) (void)hipDeviceSynchronize();
  }
  return false;
}
}  // namespace rns8::detail
#endif
