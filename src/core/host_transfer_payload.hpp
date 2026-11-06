#pragma once

#include "backend_common/deferred_resource.hpp"
#include "rns8/status.h"
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace rns8::detail {
// A DMA destination already has shared storage. Reserve its retry node before
// submission; uncertain return/exception transfers that owner without copying.
class HostTransferPayload {
public:
  using Defer = void (*)(hip_deferred_buffer_cleanup*) noexcept;
  explicit HostTransferPayload(Defer defer) noexcept : defer_(defer) {}
  HostTransferPayload(const HostTransferPayload&) = delete;
  HostTransferPayload& operator=(const HostTransferPayload&) = delete;
  ~HostTransferPayload() {
    if (record_)
      defer_(record_.release());
  }
  rns8_status begin(int device_id, std::shared_ptr<void> destination, size_t bytes) noexcept {
    if (record_)
      return RNS8_BUSY;
    if (!defer_ || device_id < 0 || !destination || !bytes)
      return RNS8_INVALID_ARGUMENT;
    auto next = std::unique_ptr<hip_deferred_buffer_cleanup>(new (std::nothrow)
                                                                 hip_deferred_buffer_cleanup);
    if (!next)
      return RNS8_INTERNAL_ERROR;
    next->kind = hip_deferred_allocation_kind::host_payload;
    next->device_id = device_id;
    next->pointer = destination.get();
    next->host_payload = std::move(destination);
    record_ = std::move(next);
    return RNS8_SUCCESS;
  }
  void* data() const noexcept { return record_ ? record_->pointer : nullptr; }
  // Only a successful synchronous copy establishes completion here.
  void complete() noexcept { record_.reset(); }

private:
  Defer defer_;
  std::unique_ptr<hip_deferred_buffer_cleanup> record_;
};
} // namespace rns8::detail
