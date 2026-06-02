#ifndef RNS8_RNS8_HPP
#define RNS8_RNS8_HPP

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rns8/rns8.h"

namespace rns8 {

class Error final : public std::runtime_error {
 public:
  explicit Error(rns8_status status)
      : std::runtime_error(rns8_status_string(status)), status_(status) {}

  rns8_status status() const noexcept { return status_; }

 private:
  rns8_status status_;
};

inline void check(rns8_status status) {
  if (status != RNS8_SUCCESS) {
    throw Error(status);
  }
}

class Context final {
 public:
  explicit Context(int device_id = -1, rns8_backend_kind backend = RNS8_BACKEND_CPU_REFERENCE) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = backend;
    check(rns8_create_context(device_id, &options, &handle_));
  }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Context& operator=(Context&& other) noexcept {
    if (this != &other) {
      rns8_destroy_context(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~Context() { rns8_destroy_context(handle_); }

  rns8_context* get() const noexcept { return handle_; }

  rns8_device_info device_info() const {
    rns8_device_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_device_info(handle_, &info));
    return info;
  }

 private:
  rns8_context* handle_ = nullptr;
};

class Plan final {
 public:
  Plan(Context& context, const rns8_gemm_desc& desc) {
    check(rns8_create_plan(context.get(), &desc, &handle_));
  }

  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  Plan(Plan&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Plan& operator=(Plan&& other) noexcept {
    if (this != &other) {
      rns8_destroy_plan(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~Plan() { rns8_destroy_plan(handle_); }

  rns8_plan* get() const noexcept { return handle_; }

  rns8_plan_schedule_info schedule_info() const {
    rns8_plan_schedule_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_plan_schedule_info(handle_, &info));
    return info;
  }

  std::vector<rns8_plan_tile_schedule_entry> tile_schedule() const {
    uint64_t count = 0;
    check(rns8_get_plan_tile_schedule(handle_, nullptr, 0, &count));
    std::vector<rns8_plan_tile_schedule_entry> entries(static_cast<std::size_t>(count));
    if (count != 0) {
      check(rns8_get_plan_tile_schedule(handle_, entries.data(), count, &count));
    }
    return entries;
  }

 private:
  rns8_plan* handle_ = nullptr;
};

class Matrix final {
 public:
  Matrix(Context& context, const rns8_matrix_desc& desc) {
    check(rns8_create_matrix(context.get(), &desc, &handle_));
  }

  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  Matrix(Matrix&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Matrix& operator=(Matrix&& other) noexcept {
    if (this != &other) {
      rns8_destroy_matrix(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~Matrix() { rns8_destroy_matrix(handle_); }

  rns8_matrix* get() const noexcept { return handle_; }

 private:
  rns8_matrix* handle_ = nullptr;
};

class Workspace final {
 public:
  Workspace(Context& context, const Plan& plan) {
    check(rns8_create_workspace(context.get(), plan.get(), &handle_));
  }

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  Workspace(Workspace&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Workspace& operator=(Workspace&& other) noexcept {
    if (this != &other) {
      rns8_destroy_workspace(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~Workspace() { rns8_destroy_workspace(handle_); }

  rns8_workspace* get() const noexcept { return handle_; }

 private:
  rns8_workspace* handle_ = nullptr;
};

}  // namespace rns8

#endif
