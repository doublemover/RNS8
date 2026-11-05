#ifndef RNS8_CORE_HIP_RESOURCES_HPP
#define RNS8_CORE_HIP_RESOURCES_HPP

#include <cstddef>
#include <utility>
#include <new>

#include "backend_hip_direct/hip_backend.hpp"
#include "backend_common/allocation_observations.hpp"
#include "core/hip_runtime_handles.hpp"
#include "core/hip_device_buffer.hpp"

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace rns8::detail {

// Capability/status kernels can fail before a context-owned operation exists.
// This owner keeps their small scratch allocation alive across an uncertain drain.
class hip_probe_buffer {
public:
  hip_probe_buffer() = default;
  hip_probe_buffer(const hip_probe_buffer&) = delete;
  hip_probe_buffer& operator=(const hip_probe_buffer&) = delete;
  ~hip_probe_buffer() {
    if (!cleanup_)
      return;
    if (cleanup_->pointer &&
        (pending_ || hip_direct_free(cleanup_->device_id, cleanup_->pointer) != RNS8_SUCCESS))
      hip_direct_defer_buffer_cleanup(cleanup_);
    else
      delete cleanup_;
  }
  rns8_status allocate(int device_id, std::size_t bytes) {
    if (cleanup_)
      return RNS8_INVALID_ARGUMENT;
    cleanup_ = new (std::nothrow) hip_deferred_buffer_cleanup;
    if (!cleanup_)
      return RNS8_INTERNAL_ERROR;
    cleanup_->device_id = device_id;
    return hip_direct_allocate(device_id, bytes, &cleanup_->pointer);
  }
  void* get() const noexcept { return cleanup_ ? cleanup_->pointer : nullptr; }
  void mark_pending() noexcept { pending_ = true; }
  void mark_completed() noexcept { pending_ = false; }

private:
  hip_deferred_buffer_cleanup* cleanup_ = nullptr;
  bool pending_ = false;
};

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP

class hip_pinned_host_buffer {
public:
  hip_pinned_host_buffer() = default;
  hip_pinned_host_buffer(const hip_pinned_host_buffer&) = delete;
  hip_pinned_host_buffer& operator=(const hip_pinned_host_buffer&) = delete;
  hip_pinned_host_buffer(hip_pinned_host_buffer&& other) noexcept { move_from(other); }
  hip_pinned_host_buffer& operator=(hip_pinned_host_buffer&& other) noexcept {
    if (this != &other) {
      retire();
      move_from(other);
    }
    return *this;
  }
  ~hip_pinned_host_buffer() { retire(); }
  hipError_t reset() noexcept {
    if (pending_)
      return hipErrorNotReady;
    if (!cleanup_)
      return hipSuccess;
    hip_scoped_device selected(cleanup_->device_id);
    if (selected.status() != hipSuccess)
      return selected.status();
    AllocationObservationLedger::Transaction observation(
        hip_direct_allocation_ledger(cleanup_->device_id));
    const auto status = hipHostFree(cleanup_->pointer);
    if (status != hipSuccess)
      return status;
    observation.released(cleanup_->pointer, AllocationObservationDomain::pinned_host);
    delete cleanup_;
    cleanup_ = nullptr;
    capacity_ = 0;
    return hipSuccess;
  }
  void mark_pending() noexcept { pending_ = true; }
  void mark_completed() noexcept { pending_ = false; }
  bool pending() const noexcept { return pending_; }
  hipError_t ensure(int device_id, std::size_t bytes, bool portable = false) {
    if (pending_)
      return hipErrorNotReady;
    if (!bytes || device_id < 0)
      return hipErrorInvalidValue;
    if (cleanup_ && cleanup_->device_id == device_id && capacity_ >= bytes && portable_ == portable)
      return hipSuccess;
    const auto release_status = reset();
    if (release_status != hipSuccess)
      return release_status;
    hip_scoped_device selected(device_id);
    if (selected.status() != hipSuccess)
      return selected.status();
    auto* record = new (std::nothrow) hip_deferred_buffer_cleanup;
    if (!record)
      return hipErrorOutOfMemory;
    record->kind = hip_deferred_allocation_kind::pinned_host;
    record->device_id = device_id;
    AllocationObservationLedger::Transaction observation(hip_direct_allocation_ledger(device_id));
    const auto status = hipHostMalloc(&record->pointer, bytes,
                                      portable ? hipHostMallocPortable : hipHostMallocDefault);
    if (status != hipSuccess) {
      delete record;
      return status;
    }
    observation.allocated(record->pointer, bytes, AllocationObservationDomain::pinned_host);
    cleanup_ = record;
    capacity_ = bytes;
    portable_ = portable;
    return hipSuccess;
  }
  void* get() const noexcept { return cleanup_ ? cleanup_->pointer : nullptr; }
  std::size_t capacity() const noexcept { return capacity_; }
  int device_id() const noexcept { return cleanup_ ? cleanup_->device_id : -1; }

private:
  void retire() noexcept {
    if (reset() != hipSuccess) {
      hip_direct_defer_buffer_cleanup(cleanup_);
      cleanup_ = nullptr;
      capacity_ = 0;
      pending_ = false;
    }
  }
  void move_from(hip_pinned_host_buffer& other) noexcept {
    cleanup_ = other.cleanup_;
    capacity_ = other.capacity_;
    pending_ = other.pending_;
    portable_ = other.portable_;
    other.cleanup_ = nullptr;
    other.capacity_ = 0;
    other.pending_ = false;
  }
  hip_deferred_buffer_cleanup* cleanup_ = nullptr;
  std::size_t capacity_ = 0;
  bool pending_ = false;
  bool portable_ = false;
};
#endif

} // namespace rns8::detail

#endif
