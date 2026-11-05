#pragma once
#include "rns8/rns8.h"
#include "core/hip_handle_owner.hpp"
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace rns8::detail {
rns8_status hip_direct_allocate(int, std::size_t, void**);
rns8_status hip_direct_free(int, void*);
void hip_direct_defer_buffer_cleanup(hip_deferred_buffer_cleanup*) noexcept;

// Failed frees keep their allocation identity. Destruction/move replacement
// transfers that identity to the existing device-drain retry queue; allocation
// replacement instead reports failure while leaving the old owner intact.
struct hip_device_buffer_operations {
  static rns8_status allocate(int device, std::size_t bytes, void** pointer) {
    return hip_direct_allocate(device, bytes, pointer);
  }
  static rns8_status release(int device, void* pointer) { return hip_direct_free(device, pointer); }
  static void defer(hip_deferred_buffer_cleanup* record) noexcept {
    hip_direct_defer_buffer_cleanup(record);
  }
};

template <class Operations = hip_device_buffer_operations> class basic_hip_device_buffer {
public:
  basic_hip_device_buffer() = default;
  basic_hip_device_buffer(const basic_hip_device_buffer&) = delete;
  basic_hip_device_buffer& operator=(const basic_hip_device_buffer&) = delete;
  basic_hip_device_buffer(basic_hip_device_buffer&& other) noexcept { move_from(other); }
  basic_hip_device_buffer& operator=(basic_hip_device_buffer&& other) noexcept {
    if (this != &other) {
      retire();
      move_from(other);
    }
    return *this;
  }
  ~basic_hip_device_buffer() { retire(); }
  rns8_status allocate(int device_id, std::size_t bytes) {
    if (device_id < 0 || !bytes)
      return RNS8_INVALID_ARGUMENT;
    const auto release_status = reset();
    if (release_status != RNS8_SUCCESS)
      return release_status;
    auto* record = new (std::nothrow) hip_deferred_buffer_cleanup;
    if (!record)
      return RNS8_INTERNAL_ERROR;
    record->device_id = device_id;
    const auto status = Operations::allocate(device_id, bytes, &record->pointer);
    if (status != RNS8_SUCCESS) {
      delete record;
      return status;
    }
    cleanup_ = record;
    bytes_ = bytes;
    return RNS8_SUCCESS;
  }
  rns8_status reset() noexcept {
    if (!cleanup_)
      return RNS8_SUCCESS;
    if (pending_)
      return RNS8_BUSY;
    const auto status = Operations::release(cleanup_->device_id, cleanup_->pointer);
    if (status != RNS8_SUCCESS)
      return status;
    delete cleanup_;
    cleanup_ = nullptr;
    bytes_ = 0;
    return RNS8_SUCCESS;
  }
  void mark_pending() noexcept { pending_ = true; }
  void mark_completed() noexcept {
    pending_ = false;
    release_host_payload();
  }
  rns8_status retain_host_payload(std::shared_ptr<const void> payload) noexcept {
    if (!cleanup_ || !payload)
      return RNS8_INVALID_ARGUMENT;
    if (cleanup_->host_payload)
      return RNS8_BUSY;
    cleanup_->host_payload = std::move(payload);
    return RNS8_SUCCESS;
  }
  // The caller has observed completion of every copy using this host payload.
  // Other work may still use the device allocation, so keep pending_ unchanged.
  void release_host_payload() noexcept {
    if (cleanup_)
      cleanup_->host_payload.reset();
  }
  bool pending() const noexcept { return pending_; }
  void* get() const noexcept { return cleanup_ ? cleanup_->pointer : nullptr; }
  std::size_t bytes() const noexcept { return bytes_; }
  int device_id() const noexcept { return cleanup_ ? cleanup_->device_id : -1; }
  explicit operator bool() const noexcept { return get() != nullptr; }
  // Keep dispatch aliases non-owning while retaining exception-safe
  // allocation records in their enclosing API owner.
  rns8_status allocate_alias(int device_id, std::size_t bytes, void*& alias) {
    const auto status = allocate(device_id, bytes);
    alias = get();
    return status;
  }
  rns8_status reset_alias(void*& alias, std::size_t& bytes) noexcept {
    if (alias != get())
      return RNS8_INTERNAL_ERROR;
    const auto status = reset();
    if (status == RNS8_SUCCESS) {
      alias = nullptr;
      bytes = 0;
    }
    return status;
  }

private:
  void retire() noexcept {
    if (reset() != RNS8_SUCCESS) {
      // The node was allocated before device ownership; failure cleanup cannot
      // allocate or discard the only handle needed for a later successful free.
      Operations::defer(cleanup_);
      cleanup_ = nullptr;
      bytes_ = 0;
    }
  }
  void move_from(basic_hip_device_buffer& other) noexcept {
    cleanup_ = other.cleanup_;
    bytes_ = other.bytes_;
    pending_ = other.pending_;
    other.cleanup_ = nullptr;
    other.bytes_ = 0;
    other.pending_ = false;
  }
  hip_deferred_buffer_cleanup* cleanup_ = nullptr;
  std::size_t bytes_ = 0;
  bool pending_ = false;
};
using hip_direct_device_buffer = basic_hip_device_buffer<>;

} // namespace rns8::detail
