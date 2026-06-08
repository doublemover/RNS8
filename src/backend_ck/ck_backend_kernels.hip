#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <exception>

#include <hip/hip_runtime.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

#include "../backend_common/finite_u8_reducer.hpp"

struct rns8_plan_tile_schedule_entry {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t flags;
  uint64_t tile_row;
  uint64_t tile_col;
  int64_t row_offset;
  int64_t col_offset;
  int64_t row_extent;
  int64_t col_extent;
  uint32_t required_prefix;
  uint32_t selected_prefix;
  uint32_t group_index;
  uint32_t range_bit_length;
};

namespace rns8::detail {
bool hip_direct_timing_enabled();
void hip_direct_timing_flush_pending_events();
void hip_direct_timing_record_sample(const char* label, double microseconds);
void hip_direct_timing_record_pending_event_with_alias(
    const char* label,
    const char* alias,
    void* start_event,
    void* stop_event);
}  // namespace rns8::detail

#include "ck_backend_timing_workspace.inc"
#include "ck_backend_gemm_kernels.inc"
#include "ck_backend_schedule_helpers.inc"
#include "ck_backend_wrappers.inc"
