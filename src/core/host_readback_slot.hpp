#pragma once

#include "backend_common/deferred_resource.hpp"
#include "rns8/status.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

namespace rns8::detail {
// Reuse a small ordinary host destination, rather than letting a failed DMA
// retain a pointer into an expired stack frame. Retry uses the device's existing
// completion queue; this slot never establishes completion itself.
class HostReadbackSlot {
public:
  using Defer = void (*)(hip_deferred_buffer_cleanup*) noexcept;
  explicit HostReadbackSlot(Defer defer) noexcept : defer_(defer) {}
  HostReadbackSlot(const HostReadbackSlot&) = delete;
  HostReadbackSlot& operator=(const HostReadbackSlot&) = delete;
  ~HostReadbackSlot() {
    if (pending_)
      fail();
  }
  rns8_status begin(int device_id, std::size_t bytes) {
    if (pending_)
      return RNS8_BUSY;
    if (!defer_ || device_id < 0 || bytes == 0 || bytes > capacity)
      return RNS8_INVALID_ARGUMENT;
    if (!record_ || record_->device_id != device_id) {
      try {
        auto next = std::make_unique<hip_deferred_buffer_cleanup>();
        auto payload = std::make_shared<std::array<unsigned char, capacity>>();
        next->kind = hip_deferred_allocation_kind::host_payload;
        next->device_id = device_id;
        next->pointer = payload->data();
        next->host_payload = std::move(payload);
        record_ = std::move(next);
      } catch (const std::bad_alloc&) {
        return RNS8_INTERNAL_ERROR;
      }
    }
    bytes_ = bytes;
    pending_ = true;
    return RNS8_SUCCESS;
  }
  void* data() const noexcept { return record_ ? record_->pointer : nullptr; }
  rns8_status complete(void* destination) noexcept {
    if (!pending_ || !destination)
      return RNS8_INVALID_ARGUMENT;
    std::memcpy(destination, record_->pointer, bytes_);
    pending_ = false;
    return RNS8_SUCCESS;
  }
  void fail() noexcept {
    if (pending_)
      defer_(record_.release());
    pending_ = false;
    bytes_ = 0;
  }
  // Covers scalar flags and the maintained two/four-word admission proofs.
  static constexpr std::size_t capacity = 4 * sizeof(uint64_t);

private:
  Defer defer_;
  std::unique_ptr<hip_deferred_buffer_cleanup> record_;
  std::size_t bytes_ = 0;
  bool pending_ = false;
};
} // namespace rns8::detail
