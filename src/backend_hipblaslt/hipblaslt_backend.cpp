#include "backend_hipblaslt/hipblaslt_backend.hpp"

#include "backend_hip_direct/hip_backend.hpp"
#include "core/hip_resources.hpp"
#include "core/internal.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
#  include <hip/hip_runtime_api.h>
#  include <hipblaslt/hipblaslt.h>

extern "C" int rns8_hipblaslt_reduce_i32_to_centered_device(
    const int32_t* scratch,
    int8_t* residues,
    int rows,
    int cols,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int accumulate);

extern "C" int rns8_hipblaslt_pack_transpose_centered_device(
    const int8_t* src,
    int8_t* dst,
    int src_rows,
    int src_cols,
    int src_ld,
    int dst_rows,
    int dst_cols,
    int dst_ld);

extern "C" int rns8_hipblaslt_reduce_i32_to_centered_strided_device(
    const int32_t* scratch,
    int8_t* residues,
    int rows,
    int cols,
    int scratch_ld,
    int ldc,
    int modulus,
    uint32_t modulus_reciprocal,
    int accumulate);
#endif

namespace rns8::detail {

#include "hipblaslt_timing_matmul_cache.inc"
#include "hipblaslt_prepack_cache.inc"
#include "hipblaslt_gemm_plane.inc"
#include "hipblaslt_context.inc"
#include "hipblaslt_rns_gemm.inc"
#include "hipblaslt_finite_gemm.inc"
