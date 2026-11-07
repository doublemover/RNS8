#include "../backend_common/packed_gemm_layout.hpp"
#include "../backend_common/hip_backend_timing.hpp"

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
#include "input_encoding.hpp"
#include "implementation_contract.hpp"

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

#include "ck_backend_timing_workspace.inc"
#include "ck_backend_gemm_kernels.inc"
#include "ck_backend_schedule_helpers.inc"
#include "ck_backend_wrappers.inc"
