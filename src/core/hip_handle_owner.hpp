#pragma once
#include <new>
#include <type_traits>
#include <utility>

#include "backend_common/deferred_resource.hpp"

namespace rns8::detail {
// Allocate retry metadata before acquiring a handle. Failed destruction and
// move replacement cannot allocate or discard the only remaining handle copy.
template <class Operations>
class basic_hip_handle {
 public:
  using handle_type = typename Operations::handle_type;
  using status_type = typename Operations::status_type;
  basic_hip_handle() = default;
  basic_hip_handle(const basic_hip_handle&) = delete;
  basic_hip_handle& operator=(const basic_hip_handle&) = delete;
  basic_hip_handle(basic_hip_handle&& other) noexcept
      : record_(std::exchange(other.record_, nullptr)) {}
  basic_hip_handle& operator=(basic_hip_handle&& other) noexcept {
    if (this != &other) {
      retire();
      record_ = std::exchange(other.record_, nullptr);
    }
    return *this;
  }
  ~basic_hip_handle() { retire(); }
  status_type reset() noexcept {
    if (!record_) return Operations::success;
    if (record_->pointer) {
      const auto status = Operations::destroy(record_->device_id, get());
      if (status != Operations::success) return status;
    }
    delete record_;
    record_ = nullptr;
    return Operations::success;
  }
  status_type reserve() noexcept {
    const auto status = reset();
    if (status != Operations::success) return status;
    auto* record = new (std::nothrow) hip_deferred_buffer_cleanup;
    if (!record) return Operations::out_of_memory;
    const auto selected = Operations::current_device(record->device_id);
    if (selected != Operations::success) {
      delete record;
      return selected;
    }
    record->kind = Operations::kind;
    record_ = record;
    return Operations::success;
  }
  // Reserve before BeginCapture; finishing a capture must not allocate
  // metadata.
  template <class Receive>
  status_type receive(Receive&& fn) noexcept {
    static_assert(std::is_nothrow_invocable_r_v<status_type, Receive, handle_type*>,
                  "handle acquisition must not throw after writing its output");
    if (!record_ || record_->pointer) return Operations::invalid_argument;
    handle_type handle = nullptr;
    const auto status = fn(&handle);
    record_->pointer = handle;  // retain nonnull partial acquisition on error
    return status;
  }
  template <class Create>
  status_type create(Create&& fn) noexcept {
    const auto status = reserve();
    return status == Operations::success ? receive(std::forward<Create>(fn)) : status;
  }
  handle_type get() const noexcept {
    return record_ ? static_cast<handle_type>(record_->pointer) : nullptr;
  }
  explicit operator bool() const noexcept { return get() != nullptr; }
  // Transfer only to a caller that explicitly owns checked destruction.
  handle_type release() noexcept {
    const auto handle = get();
    delete record_;
    record_ = nullptr;
    return handle;
  }

 private:
  void retire() noexcept {
    if (reset() != Operations::success) {
      Operations::defer(record_);
      record_ = nullptr;
    }
  }
  hip_deferred_buffer_cleanup* record_ = nullptr;
};
}  // namespace rns8::detail
