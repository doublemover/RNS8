#ifndef RNS8_CORE_HIP_RESOURCES_HPP
#define RNS8_CORE_HIP_RESOURCES_HPP

#include <cstddef>
#include <utility>

#include "backend_hip_direct/hip_backend.hpp"

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
#  include <hip/hip_runtime_api.h>
#endif

namespace rns8::detail {

class hip_direct_device_buffer {
 public:
  hip_direct_device_buffer() = default;
  hip_direct_device_buffer(const hip_direct_device_buffer&) = delete;
  hip_direct_device_buffer& operator=(const hip_direct_device_buffer&) = delete;

  hip_direct_device_buffer(hip_direct_device_buffer&& other) noexcept {
    move_from(other);
  }

  hip_direct_device_buffer& operator=(hip_direct_device_buffer&& other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }

  ~hip_direct_device_buffer() {
    reset();
  }

  rns8_status allocate(int device_id, std::size_t bytes) {
    reset();
    device_id_ = device_id;
    bytes_ = bytes;
    return hip_direct_allocate(device_id_, bytes_, &ptr_);
  }

  rns8_status reset() noexcept {
    rns8_status status = RNS8_SUCCESS;
    if (ptr_) {
      status = hip_direct_free(device_id_, ptr_);
    }
    ptr_ = nullptr;
    bytes_ = 0;
    device_id_ = -1;
    return status;
  }

  void* get() const noexcept {
    return ptr_;
  }

  std::size_t bytes() const noexcept {
    return bytes_;
  }

  explicit operator bool() const noexcept {
    return ptr_ != nullptr;
  }

 private:
  void move_from(hip_direct_device_buffer& other) noexcept {
    device_id_ = other.device_id_;
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.device_id_ = -1;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }

  int device_id_ = -1;
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
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

  hip_pinned_host_buffer(hip_pinned_host_buffer&& other) noexcept {
    move_from(other);
  }

  hip_pinned_host_buffer& operator=(hip_pinned_host_buffer&& other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }

  ~hip_pinned_host_buffer() {
    reset();
  }

  void reset() noexcept {
    if (ptr_) {
      (void)hipHostFree(ptr_);
    }
    device_id_ = -1;
    ptr_ = nullptr;
    capacity_ = 0;
  }

  hipError_t ensure(int device_id, std::size_t bytes) {
    if (bytes == 0) {
      return hipErrorInvalidValue;
    }
    if (ptr_ && device_id_ == device_id && capacity_ >= bytes) {
      return hipSuccess;
    }
    reset();
    void* ptr = nullptr;
    const hipError_t status = hipHostMalloc(&ptr, bytes, hipHostMallocDefault);
    if (status != hipSuccess) {
      return status;
    }
    device_id_ = device_id;
    ptr_ = ptr;
    capacity_ = bytes;
    return hipSuccess;
  }

  void* get() const noexcept {
    return ptr_;
  }

  std::size_t capacity() const noexcept {
    return capacity_;
  }

  int device_id() const noexcept {
    return device_id_;
  }

 private:
  void move_from(hip_pinned_host_buffer& other) noexcept {
    device_id_ = other.device_id_;
    ptr_ = other.ptr_;
    capacity_ = other.capacity_;
    other.device_id_ = -1;
    other.ptr_ = nullptr;
    other.capacity_ = 0;
  }

  int device_id_ = -1;
  void* ptr_ = nullptr;
  std::size_t capacity_ = 0;
};
#endif

}  // namespace rns8::detail

#endif
