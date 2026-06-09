#ifndef RNS8_RNS8_HPP
#define RNS8_RNS8_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
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

  rns8_plan_backend_info backend_info() const {
    rns8_plan_backend_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_plan_backend_info(handle_, &info));
    return info;
  }

  rns8_plan_packing_info packing_info() const {
    rns8_plan_packing_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_plan_packing_info(handle_, &info));
    return info;
  }

  rns8_grouped_dispatch_contract_info grouped_dispatch_contract_info(uint32_t task_count) const {
    rns8_grouped_dispatch_contract_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_grouped_dispatch_contract_info(handle_, task_count, &info));
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

  rns8_matrix_storage_info storage_info() const {
    rns8_matrix_storage_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_matrix_storage_info(handle_, &info));
    return info;
  }

  rns8_resident_lifetime_info resident_lifetime_info(
      rns8_resident_matrix_role role = RNS8_RESIDENT_MATRIX_ROLE_UNKNOWN) const {
    rns8_resident_lifetime_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_resident_lifetime_info(handle_, nullptr, nullptr, role, &info));
    return info;
  }

 private:
  rns8_matrix* handle_ = nullptr;
};

inline rns8_resident_lifetime_info resident_lifetime_info(
    const Matrix& matrix,
    const Plan& plan,
    rns8_resident_matrix_role role) {
  rns8_resident_lifetime_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  check(rns8_get_resident_lifetime_info(matrix.get(), plan.get(), nullptr, role, &info));
  return info;
}

inline rns8_prepack_cache_key_info prepack_cache_key_info(
    const Plan& plan,
    const Matrix& matrix,
    rns8_operand_role operand_role) {
  rns8_prepack_cache_key_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  check(rns8_get_prepack_cache_key_info(plan.get(), matrix.get(), operand_role, &info));
  return info;
}

class PrepackCache final {
 public:
  PrepackCache(Context& context, const Plan& plan, const Matrix& matrix, rns8_operand_role operand_role) {
    check(rns8_create_prepack_cache(context.get(), plan.get(), matrix.get(), operand_role, &handle_));
  }

  PrepackCache(Context& context, const Plan& plan, const int64_t* values, int64_t ld, uint64_t source_version) {
    check(rns8_create_b_prepack_cache_i64(context.get(), plan.get(), values, ld, source_version, &handle_));
  }

  PrepackCache(Context& context, const Plan& plan, const uint64_t* values, int64_t ld, uint64_t source_version) {
    check(rns8_create_b_prepack_cache_u64(context.get(), plan.get(), values, ld, source_version, &handle_));
  }

  PrepackCache(const PrepackCache&) = delete;
  PrepackCache& operator=(const PrepackCache&) = delete;

  PrepackCache(PrepackCache&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  PrepackCache& operator=(PrepackCache&& other) noexcept {
    if (this != &other) {
      rns8_destroy_prepack_cache(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~PrepackCache() { rns8_destroy_prepack_cache(handle_); }

  rns8_prepack_cache* get() const noexcept { return handle_; }

  rns8_prepack_cache_info info() const {
    rns8_prepack_cache_info out{};
    out.struct_size = sizeof(out);
    out.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_prepack_cache_info(handle_, &out));
    return out;
  }

 private:
  rns8_prepack_cache* handle_ = nullptr;
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

inline rns8_resident_lifetime_info resident_lifetime_info(
    const Matrix& matrix,
    const Plan& plan,
    const Workspace& workspace,
    rns8_resident_matrix_role role) {
  rns8_resident_lifetime_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  check(rns8_get_resident_lifetime_info(matrix.get(), plan.get(), workspace.get(), role, &info));
  return info;
}

inline rns8_grouped_gemm_task grouped_gemm_task(
    const Matrix& a,
    const Matrix& b,
    Matrix& c,
    Workspace& workspace) {
  return {sizeof(rns8_grouped_gemm_task), RNS8_ABI_VERSION, a.get(), b.get(), c.get(), workspace.get()};
}

struct DirtyRegion {
  int64_t row_offset = 0;
  int64_t col_offset = 0;
  int64_t row_extent = 0;
  int64_t col_extent = 0;

  rns8_dirty_region c_region() const noexcept {
    return {
        sizeof(rns8_dirty_region),
        RNS8_ABI_VERSION,
        0,
        row_offset,
        col_offset,
        row_extent,
        col_extent};
  }
};

class ResultCache final {
 public:
  ResultCache(Context& context, const Plan& plan, uint32_t max_dirty_regions = 64) {
    rns8_result_cache_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = RNS8_ABI_VERSION;
    desc.flags = RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES |
                 RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY |
                 RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE |
                 RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN;
    desc.max_dirty_regions = max_dirty_regions;
    check(rns8_create_result_cache(context.get(), plan.get(), &desc, &handle_));
  }

  ResultCache(const ResultCache&) = delete;
  ResultCache& operator=(const ResultCache&) = delete;

  ResultCache(ResultCache&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  ResultCache& operator=(ResultCache&& other) noexcept {
    if (this != &other) {
      rns8_destroy_result_cache(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~ResultCache() { rns8_destroy_result_cache(handle_); }

  rns8_result_cache* get() const noexcept { return handle_; }

  rns8_result_cache_info info() const {
    rns8_result_cache_info out{};
    out.struct_size = sizeof(out);
    out.abi_version = RNS8_ABI_VERSION;
    check(rns8_get_result_cache_info(handle_, &out));
    return out;
  }

  void invalidate() { check(rns8_invalidate_result_cache(handle_)); }

 private:
  rns8_result_cache* handle_ = nullptr;
};

inline uint32_t checked_grouped_task_count(std::size_t size) {
  if (size > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
    throw Error(RNS8_INVALID_ARGUMENT);
  }
  return static_cast<uint32_t>(size);
}

inline void gemm_rns_grouped(
    Context& context,
    const Plan& plan,
    const std::vector<rns8_grouped_gemm_task>& tasks) {
  check(rns8_gemm_rns_grouped(
      context.get(),
      plan.get(),
      tasks.data(),
      checked_grouped_task_count(tasks.size())));
}

inline void gemm_finite_u8_grouped(
    Context& context,
    const Plan& plan,
    uint16_t modulus,
    const std::vector<rns8_grouped_gemm_task>& tasks) {
  check(rns8_gemm_finite_u8_grouped(
      context.get(),
      plan.get(),
      modulus,
      tasks.data(),
      checked_grouped_task_count(tasks.size())));
}

inline void gemm_rns_prepacked_b(
    Context& context,
    const Plan& plan,
    const Matrix& a,
    const PrepackCache& b,
    Matrix& c,
    Workspace& workspace) {
  check(rns8_gemm_rns_prepacked_b(context.get(), plan.get(), a.get(), b.get(), c.get(), workspace.get()));
}

inline std::vector<rns8_dirty_region> c_dirty_regions(const std::vector<DirtyRegion>& regions) {
  std::vector<rns8_dirty_region> out;
  out.reserve(regions.size());
  for (const DirtyRegion& region : regions) {
    out.push_back(region.c_region());
  }
  return out;
}

inline uint32_t checked_dirty_region_count(std::size_t size) {
  if (size > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
    throw Error(RNS8_INVALID_ARGUMENT);
  }
  return static_cast<uint32_t>(size);
}

inline void gemm_rns_incremental(
    Context& context,
    const Plan& plan,
    const Matrix& a,
    const Matrix& b,
    Matrix& c,
    Workspace& workspace,
    ResultCache& cache,
    const std::vector<DirtyRegion>& dirty_regions = {}) {
  std::vector<rns8_dirty_region> regions = c_dirty_regions(dirty_regions);
  check(rns8_gemm_rns_incremental(
      context.get(),
      plan.get(),
      a.get(),
      b.get(),
      c.get(),
      workspace.get(),
      cache.get(),
      regions.empty() ? nullptr : regions.data(),
      checked_dirty_region_count(regions.size())));
}

inline void gemm_finite_u8_incremental(
    Context& context,
    const Plan& plan,
    uint16_t modulus,
    const Matrix& a,
    const Matrix& b,
    Matrix& c,
    Workspace& workspace,
    ResultCache& cache,
    const std::vector<DirtyRegion>& dirty_regions = {}) {
  std::vector<rns8_dirty_region> regions = c_dirty_regions(dirty_regions);
  check(rns8_gemm_finite_u8_incremental(
      context.get(),
      plan.get(),
      modulus,
      a.get(),
      b.get(),
      c.get(),
      workspace.get(),
      cache.get(),
      regions.empty() ? nullptr : regions.data(),
      checked_dirty_region_count(regions.size())));
}

}  // namespace rns8

#endif
