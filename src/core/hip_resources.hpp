#ifndef RNS8_CORE_HIP_RESOURCES_HPP
#define RNS8_CORE_HIP_RESOURCES_HPP

#include <cstddef>
#include <utility>
#include <new>

#include "backend_hip_direct/hip_backend.hpp"

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#  include <hip/hip_runtime_api.h>
#endif

namespace rns8::detail {

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
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
  ~hip_scoped_device() { if (changed_) (void)hipSetDevice(previous_); }
  hipError_t status() const noexcept { return status_; }
 private:
  int previous_ = -1;
  bool changed_ = false;
  hipError_t status_ = hipSuccess;
};
#endif

// Failed frees keep their allocation identity. Destruction/move replacement
// transfers that identity to the existing device-drain retry queue; allocation
// replacement instead reports failure while leaving the old owner intact.
struct hip_device_buffer_operations {
  static rns8_status allocate(int device, std::size_t bytes, void** pointer) {
    return hip_direct_allocate(device, bytes, pointer);
  }
  static rns8_status release(int device, void* pointer) { return hip_direct_free(device, pointer); }
  static void defer(hip_deferred_buffer_cleanup* record) noexcept { hip_direct_defer_buffer_cleanup(record); }
};

template<class Operations = hip_device_buffer_operations>
class basic_hip_device_buffer {
 public:
  basic_hip_device_buffer() = default;
  basic_hip_device_buffer(const basic_hip_device_buffer&) = delete;
  basic_hip_device_buffer& operator=(const basic_hip_device_buffer&) = delete;
  basic_hip_device_buffer(basic_hip_device_buffer&& other) noexcept { move_from(other); }
  basic_hip_device_buffer& operator=(basic_hip_device_buffer&& other) noexcept {
    if(this!=&other) { retire(); move_from(other); }
    return *this;
  }
  ~basic_hip_device_buffer() { retire(); }
  rns8_status allocate(int device_id,std::size_t bytes) {
    if(device_id<0 || !bytes) return RNS8_INVALID_ARGUMENT;
    const auto release_status=reset();
    if(release_status!=RNS8_SUCCESS) return release_status;
    auto* record=new(std::nothrow) hip_deferred_buffer_cleanup;
    if(!record) return RNS8_INTERNAL_ERROR;
    record->device_id=device_id;
    const auto status=Operations::allocate(device_id,bytes,&record->pointer);
    if(status!=RNS8_SUCCESS) {delete record;return status;}
    cleanup_=record;bytes_=bytes;return RNS8_SUCCESS;
  }
  rns8_status reset() noexcept {
    if(!cleanup_) return RNS8_SUCCESS;
    if(pending_)return RNS8_BUSY;
    const auto status=Operations::release(cleanup_->device_id,cleanup_->pointer);
    if(status!=RNS8_SUCCESS) return status;
    delete cleanup_;cleanup_=nullptr;bytes_=0;return RNS8_SUCCESS;
  }
  void mark_pending() noexcept {pending_=true;}
  void mark_completed() noexcept {pending_=false;}
  bool pending() const noexcept {return pending_;}
  void* get() const noexcept {return cleanup_?cleanup_->pointer:nullptr;}
  std::size_t bytes() const noexcept {return bytes_;}
  explicit operator bool() const noexcept {return get()!=nullptr;}
  // Keep legacy dispatch aliases non-owning while retaining exception-safe
  // allocation records in their enclosing API owner.
  rns8_status allocate_alias(int device_id, std::size_t bytes, void*& alias) {
    const auto status = allocate(device_id, bytes);
    alias = get();
    return status;
  }
  rns8_status reset_alias(void*& alias, std::size_t& bytes) noexcept {
    if (alias != get()) return RNS8_INTERNAL_ERROR;
    const auto status = reset();
    if (status == RNS8_SUCCESS) { alias = nullptr; bytes = 0; }
    return status;
  }
 private:
  void retire() noexcept {
    if(reset()!=RNS8_SUCCESS) {
      // The node was allocated before device ownership; failure cleanup cannot
      // allocate or discard the only handle needed for a later successful free.
      Operations::defer(cleanup_);cleanup_=nullptr;bytes_=0;
    }
  }
  void move_from(basic_hip_device_buffer& other) noexcept {
    cleanup_=other.cleanup_;bytes_=other.bytes_;pending_=other.pending_;
    other.cleanup_=nullptr;other.bytes_=0;other.pending_=false;
  }
  hip_deferred_buffer_cleanup* cleanup_=nullptr;
  std::size_t bytes_=0;
  bool pending_=false;
};
using hip_direct_device_buffer = basic_hip_device_buffer<>;

// Capability/status kernels can fail before a context-owned operation exists.
// This owner keeps their small scratch allocation alive across an uncertain drain.
class hip_probe_buffer {
 public:
  hip_probe_buffer()=default;
  hip_probe_buffer(const hip_probe_buffer&)=delete;
  hip_probe_buffer& operator=(const hip_probe_buffer&)=delete;
  ~hip_probe_buffer() {
    if(!cleanup_) return;
    if(cleanup_->pointer && (pending_ || hip_direct_free(cleanup_->device_id,cleanup_->pointer)!=RNS8_SUCCESS))
      hip_direct_defer_buffer_cleanup(cleanup_);
    else delete cleanup_;
  }
  rns8_status allocate(int device_id,std::size_t bytes) {
    if(cleanup_) return RNS8_INVALID_ARGUMENT;
    cleanup_=new(std::nothrow) hip_deferred_buffer_cleanup;
    if(!cleanup_) return RNS8_INTERNAL_ERROR;
    cleanup_->device_id=device_id;
    return hip_direct_allocate(device_id,bytes,&cleanup_->pointer);
  }
  void* get() const noexcept {return cleanup_?cleanup_->pointer:nullptr;}
  void mark_pending() noexcept {pending_=true;}
  void mark_completed() noexcept {pending_=false;}
 private:
  hip_deferred_buffer_cleanup* cleanup_=nullptr;
  bool pending_=false;
};

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
class hip_unique_event {
 public:
  hip_unique_event() = default;
  explicit hip_unique_event(hipEvent_t event) : event_(event) {}
  hip_unique_event(const hip_unique_event&) = delete;
  hip_unique_event& operator=(const hip_unique_event&) = delete;

  hip_unique_event(hip_unique_event&& other) noexcept : event_(other.release()) {}

  hip_unique_event& operator=(hip_unique_event&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  ~hip_unique_event() {
    reset();
  }

  hipError_t create() {
    reset();
    return hipEventCreate(&event_);
  }

  void reset(hipEvent_t event = nullptr) noexcept {
    if (event_) {
      (void)hipEventDestroy(event_);
    }
    event_ = event;
  }

  hipEvent_t release() noexcept {
    hipEvent_t event = event_;
    event_ = nullptr;
    return event;
  }

  hipEvent_t get() const noexcept {
    return event_;
  }

  explicit operator bool() const noexcept {
    return event_ != nullptr;
  }

 private:
  hipEvent_t event_ = nullptr;
};

class hip_unique_event_pair {
 public:
  hip_unique_event_pair() = default;
  hip_unique_event_pair(const hip_unique_event_pair&) = delete;
  hip_unique_event_pair& operator=(const hip_unique_event_pair&) = delete;
  hip_unique_event_pair(hip_unique_event_pair&&) noexcept = default;
  hip_unique_event_pair& operator=(hip_unique_event_pair&&) noexcept = default;

  hipError_t create() {
    reset();
    hipError_t status = start_.create();
    if (status != hipSuccess) {
      return status;
    }
    status = stop_.create();
    if (status != hipSuccess) {
      reset();
    }
    return status;
  }

  hipError_t create_and_record_start(hipStream_t stream = nullptr) {
    hipError_t status = create();
    if (status != hipSuccess) {
      return status;
    }
    status = record_start(stream);
    if (status != hipSuccess) {
      reset();
    }
    return status;
  }

  hipError_t record_start(hipStream_t stream = nullptr) const {
    return hipEventRecord(start_.get(), stream);
  }

  hipError_t record_stop(hipStream_t stream = nullptr) const {
    return hipEventRecord(stop_.get(), stream);
  }

  hipEvent_t start() const noexcept {
    return start_.get();
  }

  hipEvent_t stop() const noexcept {
    return stop_.get();
  }

  hipEvent_t release_start() noexcept {
    return start_.release();
  }

  hipEvent_t release_stop() noexcept {
    return stop_.release();
  }

  void reset() noexcept {
    stop_.reset();
    start_.reset();
  }

 private:
  hip_unique_event start_;
  hip_unique_event stop_;
};

class hip_unique_stream {
 public:
  hip_unique_stream() = default;
  hip_unique_stream(const hip_unique_stream&) = delete;
  hip_unique_stream& operator=(const hip_unique_stream&) = delete;

  hip_unique_stream(hip_unique_stream&& other) noexcept : stream_(other.release()) {}

  hip_unique_stream& operator=(hip_unique_stream&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  ~hip_unique_stream() {
    reset();
  }

  hipError_t create_non_blocking() {
    reset();
    return hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
  }

  void reset(hipStream_t stream = nullptr) noexcept {
    if (stream_) {
      (void)hipStreamDestroy(stream_);
    }
    stream_ = stream;
  }

  hipStream_t release() noexcept {
    hipStream_t stream = stream_;
    stream_ = nullptr;
    return stream;
  }

  hipStream_t get() const noexcept {
    return stream_;
  }

  explicit operator bool() const noexcept {
    return stream_ != nullptr;
  }

 private:
  hipStream_t stream_ = nullptr;
};

class hip_pinned_host_buffer {
 public:
  hip_pinned_host_buffer() = default;
  hip_pinned_host_buffer(const hip_pinned_host_buffer&) = delete;
  hip_pinned_host_buffer& operator=(const hip_pinned_host_buffer&) = delete;
  hip_pinned_host_buffer(hip_pinned_host_buffer&& other) noexcept {move_from(other);}
  hip_pinned_host_buffer& operator=(hip_pinned_host_buffer&& other) noexcept {
    if(this!=&other) {retire();move_from(other);} return *this;
  }
  ~hip_pinned_host_buffer() {retire();}
  hipError_t reset() noexcept {
    if(pending_) return hipErrorNotReady;
    if(!cleanup_) return hipSuccess;
    hip_scoped_device selected(cleanup_->device_id);
    if(selected.status()!=hipSuccess) return selected.status();
    const auto status=hipHostFree(cleanup_->pointer);
    if(status!=hipSuccess) return status;
    delete cleanup_;cleanup_=nullptr;capacity_=0;return hipSuccess;
  }
  void mark_pending() noexcept {pending_=true;}
  void mark_completed() noexcept {pending_=false;}
  bool pending() const noexcept {return pending_;}
  hipError_t ensure(int device_id,std::size_t bytes) {
    if(pending_) return hipErrorNotReady;
    if(!bytes || device_id<0) return hipErrorInvalidValue;
    if(cleanup_ && cleanup_->device_id==device_id && capacity_>=bytes) return hipSuccess;
    const auto release_status=reset();
    if(release_status!=hipSuccess) return release_status;
    hip_scoped_device selected(device_id);
    if(selected.status()!=hipSuccess) return selected.status();
    auto* record=new(std::nothrow) hip_deferred_buffer_cleanup;
    if(!record) return hipErrorOutOfMemory;
    record->kind=hip_deferred_allocation_kind::pinned_host;record->device_id=device_id;
    const auto status=hipHostMalloc(&record->pointer,bytes,hipHostMallocDefault);
    if(status!=hipSuccess) {delete record;return status;}
    cleanup_=record;capacity_=bytes;return hipSuccess;
  }
  void* get() const noexcept {return cleanup_?cleanup_->pointer:nullptr;}
  std::size_t capacity() const noexcept {return capacity_;}
  int device_id() const noexcept {return cleanup_?cleanup_->device_id:-1;}
 private:
  void retire() noexcept {
    if(reset()!=hipSuccess) {
      hip_direct_defer_buffer_cleanup(cleanup_);cleanup_=nullptr;capacity_=0;pending_=false;
    }
  }
  void move_from(hip_pinned_host_buffer& other) noexcept {
    cleanup_=other.cleanup_;capacity_=other.capacity_;pending_=other.pending_;
    other.cleanup_=nullptr;other.capacity_=0;other.pending_=false;
  }
  hip_deferred_buffer_cleanup* cleanup_=nullptr;
  std::size_t capacity_=0;
  bool pending_=false;
};
#endif

}  // namespace rns8::detail

#endif
