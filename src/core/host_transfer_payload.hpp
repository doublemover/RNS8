#pragma once

#include "backend_common/deferred_resource.hpp"
#include "backend_common/host_upload.hpp"
#include "rns8/status.h"
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace rns8::detail {
// Reuse only observed-complete pageable storage. A failed transfer must discard
// the cache's reference while its HostTransferPayload retains the old bytes.
class HostTransferStorage {
public:
  HostTransferStorage() = default;
  HostTransferStorage(const HostTransferStorage&) = delete;
  HostTransferStorage& operator=(const HostTransferStorage&) = delete;
  rns8_status reserve(size_t bytes) noexcept {
    if (!bytes)
      return RNS8_INVALID_ARGUMENT;
    if (storage_.use_count() > 1)
      return RNS8_BUSY;
    if (capacity_ >= bytes)
      return RNS8_SUCCESS;
    try {
      std::shared_ptr<unsigned char[]> next(new unsigned char[bytes]);
      storage_ = std::move(next);
      capacity_ = bytes;
      return RNS8_SUCCESS;
    } catch (const std::bad_alloc&) {
      return RNS8_INTERNAL_ERROR;
    }
  }
  void* data() const noexcept { return storage_.get(); }
  // Copy only accessible logical bytes before submission. The caller can then
  // return even if HIP cannot establish completion of its private source.
  rns8_status snapshot(const void* source, const HostUploadPlan& plan) noexcept {
    if (!source || !valid_host_upload_plan(plan))
      return RNS8_INVALID_ARGUMENT;
    // Snapshotting this cache into itself is not a supported borrowed input.
    if (storage_ && !nonoverlapping_byte_ranges(source, plan.source_span_bytes, data(), capacity_))
      return RNS8_INVALID_ARGUMENT;
    const auto status = reserve(plan.device_bytes);
    if (status != RNS8_SUCCESS)
      return status;
    if (plan.contiguous()) {
      std::memcpy(data(), source, plan.device_bytes);
    } else {
      const auto* src = static_cast<const unsigned char*>(source);
      auto* dst = static_cast<unsigned char*>(data());
      for (size_t row = 0; row < plan.rows; ++row)
        std::memcpy(dst + row * plan.row_bytes, src + row * plan.source_pitch_bytes,
                    plan.row_bytes);
    }
    return RNS8_SUCCESS;
  }
  size_t capacity() const noexcept { return capacity_; }
  std::shared_ptr<void> owner() const noexcept { return {storage_, storage_.get()}; }
  void discard() noexcept {
    storage_.reset();
    capacity_ = 0;
  }
  void trim(size_t maximum_cached_bytes) noexcept {
    if (capacity_ > maximum_cached_bytes)
      discard();
  }

private:
  std::shared_ptr<unsigned char[]> storage_;
  size_t capacity_ = 0;
};

// A DMA payload already has shared storage. Reserve its retry node before
// submission; uncertain return/exception transfers that owner without copying.
template <class Host = void> class BasicHostTransferPayload {
  static_assert(std::is_void_v<Host>);

public:
  using Defer = void (*)(hip_deferred_buffer_cleanup*) noexcept;
  explicit BasicHostTransferPayload(Defer defer) noexcept : defer_(defer) {}
  BasicHostTransferPayload(const BasicHostTransferPayload&) = delete;
  BasicHostTransferPayload& operator=(const BasicHostTransferPayload&) = delete;
  ~BasicHostTransferPayload() {
    if (record_)
      defer_(record_.release());
  }
  rns8_status begin(int device_id, std::shared_ptr<Host> destination, size_t bytes) noexcept {
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
    // The retry record erases pointer type; data() preserves source constness.
    next->pointer = const_cast<void*>(static_cast<const void*>(destination.get()));
    next->host_payload = std::move(destination);
    record_ = std::move(next);
    return RNS8_SUCCESS;
  }
  Host* data() const noexcept { return record_ ? record_->pointer : nullptr; }
  // Only a successful synchronous copy establishes completion here.
  void complete() noexcept { record_.reset(); }

private:
  Defer defer_;
  std::unique_ptr<hip_deferred_buffer_cleanup> record_;
};
using HostTransferPayload = BasicHostTransferPayload<void>;
using HostUploadPayload = BasicHostTransferPayload<const void>;
} // namespace rns8::detail
