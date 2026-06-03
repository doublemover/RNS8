#include "core/internal.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <string>

#include "backend_hip_direct/hip_backend.hpp"
#include "backend_hipblaslt/hipblaslt_backend.hpp"
#include "backend_ck/ck_backend.hpp"
#include "backend_wmma/wmma_backend.hpp"
#include "backend_wrap64/wrap64_hip.hpp"
#include "backend_vector_alu/vector_alu_backend.hpp"
#include "core/accelerator_backend.hpp"
#include "core/autotune_cache.hpp"
#include "core/backend_common.hpp"

namespace {

template <typename Fn>
rns8_status guard_api(Fn&& fn) {
  try {
    return fn();
  } catch (const std::bad_alloc&) {
    return RNS8_INTERNAL_ERROR;
  } catch (...) {
    return RNS8_INTERNAL_ERROR;
  }
}

template <typename Fn>
rns8_status run_timed_api_status(const char* label, Fn&& fn) {
  rns8_status status = RNS8_SUCCESS;
  const int code = rns8::detail::run_timed_device_code(label, [&]() {
    status = fn();
    return status == RNS8_SUCCESS ? 0 : 3;
  });
  if (code != 0 && status == RNS8_SUCCESS) {
    return RNS8_BACKEND_FAILURE;
  }
  return status;
}

rns8_backend_kind effective_backend(rns8_backend_kind requested, rns8_backend_kind default_backend) {
  return requested == RNS8_BACKEND_AUTO ? default_backend : requested;
}

bool backend_supports_semantics(rns8_backend_kind backend, rns8_semantics semantics) {
  switch (backend) {
    case RNS8_BACKEND_CPU_REFERENCE:
      return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
             semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED ||
             semantics == RNS8_FINITE_RING_U8 || semantics == RNS8_FINITE_FIELD_U8;
    case RNS8_BACKEND_HIP_DIRECT:
      return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
             semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED ||
             semantics == RNS8_WRAP_U64_MOD_2_64 || semantics == RNS8_FINITE_RING_U8 ||
             semantics == RNS8_FINITE_FIELD_U8;
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64;
#else
      return false;
#endif
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return semantics == RNS8_WRAP_U64_MOD_2_64;
    case RNS8_BACKEND_HIPBLASLT:
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
             semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED ||
             semantics == RNS8_FINITE_RING_U8 || semantics == RNS8_FINITE_FIELD_U8;
#else
      return false;
#endif
    case RNS8_BACKEND_AUTO:
      return false;
    case RNS8_BACKEND_CK:
    case RNS8_BACKEND_WMMA:
      return rns8::detail::accelerator_backend_supports_semantics(backend, semantics);
  }
  return false;
}

bool known_backend_kind(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_AUTO:
    case RNS8_BACKEND_CPU_REFERENCE:
    case RNS8_BACKEND_HIP_DIRECT:
    case RNS8_BACKEND_HIPBLASLT:
    case RNS8_BACKEND_CK:
    case RNS8_BACKEND_WMMA:
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
      return true;
  }
  return false;
}

const char* backend_name(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_AUTO:
      return "auto";
    case RNS8_BACKEND_CPU_REFERENCE:
      return "cpu-reference";
    case RNS8_BACKEND_HIP_DIRECT:
      return "hip-direct";
    case RNS8_BACKEND_HIPBLASLT:
      return "hipblaslt";
    case RNS8_BACKEND_CK:
      return "ck";
    case RNS8_BACKEND_WMMA:
      return "wmma";
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return "wrap64-byte-limb";
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
      return "hip-vector-alu-int64";
  }
  return "unknown";
}

bool accelerator_backend(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIPBLASLT || backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_WMMA;
}

uint32_t direct_hip_compiled() {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  return 1;
#else
  return 0;
#endif
}

uint32_t hipblaslt_backend_compiled() {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
  return 1;
#else
  return 0;
#endif
}

bool hip_resident_rns_backend(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIP_DIRECT || backend == RNS8_BACKEND_HIPBLASLT ||
         (rns8::detail::accelerator_backend_kind(backend) &&
          rns8::detail::accelerator_backend_compiled(backend));
}

bool native_vector_backend(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
}

bool hip_device_backend(rns8_backend_kind backend) {
  return hip_resident_rns_backend(backend) || native_vector_backend(backend);
}

bool context_accepts_backend(const rns8_context& ctx, rns8_backend_kind backend) {
  if (ctx.backend == backend) {
    return true;
  }
  return ctx.auto_backend_selection && ctx.backend == RNS8_BACKEND_HIP_DIRECT &&
         hip_device_backend(backend);
}

bool matrix_backend_compatible_with_plan(
    const rns8_context& ctx,
    const rns8_matrix& matrix,
    rns8_backend_kind plan_backend) {
  if (matrix.backend == plan_backend) {
    return true;
  }
  return ctx.auto_backend_selection && ctx.backend == RNS8_BACKEND_HIP_DIRECT &&
         matrix.backend == RNS8_BACKEND_HIP_DIRECT && hip_device_backend(plan_backend);
}

void set_text(char* dst, std::size_t dst_size, const char* text) {
  rns8::detail::copy_c_string(dst, dst_size, text ? text : "");
}

void set_text(char* dst, std::size_t dst_size, const std::string& text) {
  rns8::detail::copy_c_string(dst, dst_size, text);
}

void fill_backend_capability_info(rns8_backend_kind backend, rns8_backend_capability_info& info) {
  const uint64_t struct_size = info.struct_size;
  const uint32_t abi_version = info.abi_version;
  info = {};
  info.struct_size = struct_size;
  info.abi_version = abi_version;
  info.backend = backend;
  set_text(info.backend_name, sizeof(info.backend_name), backend_name(backend));

  if (backend == RNS8_BACKEND_AUTO) {
    set_text(info.status, sizeof(info.status), "context_default_selector");
    set_text(info.detail, sizeof(info.detail), "AUTO is a context default selector, not an accelerator backend.");
    return;
  }

  info.is_accelerator = accelerator_backend(backend) ? 1u : 0u;
  switch (backend) {
    case RNS8_BACKEND_CPU_REFERENCE:
      info.is_available = 1;
      info.is_correctness_backend = 1;
      info.supports_bounded_rns = 1;
      info.supports_exact_wide_rns = 1;
      info.supports_finite_u8 = 1;
      info.compiled_kernel_available = 1;
      info.exact_differential_validated = 1;
      set_text(info.selected_kernel, sizeof(info.selected_kernel), "cpu_reference_scalar_rns_gemm_v1");
      set_text(info.epilogue_mode, sizeof(info.epilogue_mode), "host_reference_reconstruction");
      set_text(info.workspace_mode, sizeof(info.workspace_mode), "host_reference_workspace");
      set_text(info.isa_evidence, sizeof(info.isa_evidence), "not_applicable_cpu");
      set_text(info.status, sizeof(info.status), "implemented_correctness_backend");
      set_text(info.detail, sizeof(info.detail), "Portable CPU reference backend and exact oracle.");
      break;
    case RNS8_BACKEND_HIP_DIRECT:
      info.is_available = direct_hip_compiled();
      info.is_correctness_backend = direct_hip_compiled();
      info.supports_bounded_rns = direct_hip_compiled();
      info.supports_exact_wide_rns = direct_hip_compiled();
      info.supports_finite_u8 = direct_hip_compiled();
      info.supports_wrap64 = direct_hip_compiled();
      info.compiled_kernel_available = direct_hip_compiled();
      info.exact_differential_validated = direct_hip_compiled();
      set_text(info.selected_kernel, sizeof(info.selected_kernel), "direct_hip_tiled_rns_gemm_v1");
      set_text(info.library_name, sizeof(info.library_name), "HIP runtime");
      set_text(info.epilogue_mode, sizeof(info.epilogue_mode), "fused_centered_residue_correctness");
      set_text(info.workspace_mode, sizeof(info.workspace_mode), "resident_device_buffers");
      set_text(info.isa_evidence, sizeof(info.isa_evidence), "rns8_hip_direct_reciprocal_isa_gate");
      set_text(
          info.status,
          sizeof(info.status),
          direct_hip_compiled() ? "implemented_correctness_backend" : "not_compiled_in_this_build");
      set_text(
          info.detail,
          sizeof(info.detail),
          "Direct HIP correctness backend; not an optimized matrix-engine accelerator.");
      break;
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
      info.is_available = rns8::detail::vector_alu_compiled() ? 1u : 0u;
      info.is_correctness_backend = rns8::detail::vector_alu_compiled() ? 1u : 0u;
      info.supports_bounded_rns = rns8::detail::vector_alu_compiled() ? 1u : 0u;
      info.compiled_kernel_available = rns8::detail::vector_alu_compiled() ? 1u : 0u;
      info.exact_differential_validated = rns8::detail::vector_alu_compiled() ? 1u : 0u;
      info.performance_validated = 0;
      set_text(info.selected_kernel, sizeof(info.selected_kernel), "hip_vector_alu_i64_u64_exact_192b_v1");
      set_text(info.library_name, sizeof(info.library_name), "HIP runtime");
      set_text(info.epilogue_mode, sizeof(info.epilogue_mode), "direct_int64_export");
      set_text(info.workspace_mode, sizeof(info.workspace_mode), "native_device_i64_u64_buffers");
      set_text(info.isa_evidence, sizeof(info.isa_evidence), "source_level_192bit_limb_accumulator_no_matrix_engine");
      set_text(
          info.status,
          sizeof(info.status),
          rns8::detail::vector_alu_compiled() ? "implemented_native_bounded_vector_backend"
                                              : "not_compiled_in_this_build");
      set_text(
          info.detail,
          sizeof(info.detail),
          "Bounded i64/u64 native-vector HIP backend using original integer storage, not resident RNS residues.");
      break;
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      info.is_available = 1;
      info.is_correctness_backend = 1;
      info.supports_wrap64 = 1;
      info.compiled_kernel_available = 1;
      info.exact_differential_validated = 1;
      set_text(info.selected_kernel, sizeof(info.selected_kernel), "cpu_wrap64_byte_limb_reference_v1");
      set_text(info.epilogue_mode, sizeof(info.epilogue_mode), "low64_wrap_export");
      set_text(info.workspace_mode, sizeof(info.workspace_mode), "host_byte_limb_reference_workspace");
      set_text(info.isa_evidence, sizeof(info.isa_evidence), "not_applicable_cpu");
      set_text(info.status, sizeof(info.status), "implemented_correctness_backend");
      set_text(info.detail, sizeof(info.detail), "Strict mod 2^64 CPU byte-limb reference backend.");
      break;
    case RNS8_BACKEND_HIPBLASLT:
      info.requires_feature_detection = 1;
      info.is_available = hipblaslt_backend_compiled();
      info.is_correctness_backend = hipblaslt_backend_compiled();
      info.is_matrix_engine_backend = hipblaslt_backend_compiled();
      info.supports_bounded_rns = hipblaslt_backend_compiled();
      info.supports_exact_wide_rns = hipblaslt_backend_compiled();
      info.supports_finite_u8 = hipblaslt_backend_compiled();
      info.compiled_kernel_available = hipblaslt_backend_compiled();
      info.exact_differential_validated = hipblaslt_backend_compiled();
      info.performance_validated = 0;
      info.enable_flag_fail_fast = hipblaslt_backend_compiled() ? 0u : 1u;
      info.candidate_evidence_only = hipblaslt_backend_compiled() ? 0u : 1u;
      set_text(
          info.selected_kernel,
          sizeof(info.selected_kernel),
          hipblaslt_backend_compiled() ? "hipblaslt_int8_i32_scratch_reduce_baseline_v1" : "not_implemented");
      set_text(info.library_name, sizeof(info.library_name), "hipBLASLt");
      set_text(
          info.library_version,
          sizeof(info.library_version),
          hipblaslt_backend_compiled() ? "runtime_queried_in_context" : "");
      set_text(info.enable_flag, sizeof(info.enable_flag), "RNS8_ENABLE_HIPBLASLT");
      set_text(
          info.epilogue_mode,
          sizeof(info.epilogue_mode),
          hipblaslt_backend_compiled() ? "separate_i32_scratch_residue_reduce" : "not_implemented");
      set_text(
          info.workspace_mode,
          sizeof(info.workspace_mode),
          hipblaslt_backend_compiled() ? "resident_device_buffers_with_hipblaslt_scratch" : "not_implemented");
      set_text(
          info.isa_evidence,
          sizeof(info.isa_evidence),
          hipblaslt_backend_compiled() ? "hipblaslt_library_int8_matmul_baseline" : "not_validated");
      set_text(
          info.status,
          sizeof(info.status),
          hipblaslt_backend_compiled() ? "implemented_baseline_backend" : "not_implemented_evidence_only");
      set_text(
          info.detail,
          sizeof(info.detail),
          hipblaslt_backend_compiled()
              ? "hipBLASLt INT8->INT32 GEMM baseline with separate HIP centered-residue reduction; no adaptive per-tile support."
              : "Reserved baseline accelerator; enable flag stays fail-fast until exact kernels and differentials exist.");
      break;
    case RNS8_BACKEND_CK:
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      info.is_available = 1;
      info.is_correctness_backend = 1;
      info.requires_feature_detection = 1;
      info.supports_bounded_rns = 1;
      info.supports_exact_wide_rns = 1;
      info.supports_finite_u8 = 1;
      info.compiled_kernel_available = 1;
      info.exact_differential_validated = 1;
      info.is_matrix_engine_backend = 1;
      set_text(info.selected_kernel, sizeof(info.selected_kernel), "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1");
      set_text(info.library_name, sizeof(info.library_name), "Composable Kernel");
      set_text(info.library_version, sizeof(info.library_version), "repo-local release/rocm-rel-7.1");
      set_text(info.enable_flag, sizeof(info.enable_flag), "RNS8_ENABLE_CK");
      set_text(info.epilogue_mode, sizeof(info.epilogue_mode), "ck_fused_i32_to_centered_residue");
      set_text(
          info.workspace_mode,
          sizeof(info.workspace_mode),
          "resident_device_buffers_with_ck_canonical_pack_workspace");
      set_text(
          info.isa_evidence,
          sizeof(info.isa_evidence),
          "ck_wmma_cshuffle_int8_matrix_isa_gate_no_int32_global_store_no_divide");
      set_text(info.status, sizeof(info.status), "implemented_opt_in_ck_backend");
      set_text(
          info.detail,
          sizeof(info.detail),
          "Opt-in CK backend using WMMA CShuffle int8 GEMM with fused centered-residue epilogue.");
#else
      rns8::detail::fill_disabled_accelerator_capability(backend, info);
#endif
      break;
    case RNS8_BACKEND_WMMA:
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      info.is_available = 1;
      info.is_correctness_backend = 1;
      info.requires_feature_detection = 1;
      info.supports_bounded_rns = 1;
      info.supports_exact_wide_rns = 1;
      info.supports_finite_u8 = 1;
      info.compiled_kernel_available = 1;
      info.exact_differential_validated = 1;
      info.performance_validated = 0;
      info.is_matrix_engine_backend = 1;
      set_text(info.selected_kernel, sizeof(info.selected_kernel), "rocwmma_i8_i32_signed_hot_residue_v1");
      set_text(info.library_name, sizeof(info.library_name), "rocWMMA");
      set_text(info.library_version, sizeof(info.library_version), "repo-local release/rocm-rel-7.1");
      set_text(info.enable_flag, sizeof(info.enable_flag), "RNS8_ENABLE_ROCWMMA");
      set_text(info.epilogue_mode, sizeof(info.epilogue_mode), "rocwmma_fused_i32_to_centered_residue");
      set_text(info.workspace_mode, sizeof(info.workspace_mode), "resident_device_buffers_with_rocwmma_pack_workspace");
      set_text(
          info.isa_evidence,
          sizeof(info.isa_evidence),
          "rocwmma_i8_wmma_isa_gate_no_int32_global_store_no_divide");
      set_text(info.status, sizeof(info.status), "implemented_opt_in_rocwmma_backend");
      set_text(
          info.detail,
          sizeof(info.detail),
          "Opt-in rocWMMA backend using signed int8 WMMA GEMM with fused centered-residue reduction.");
#else
      rns8::detail::fill_disabled_accelerator_capability(backend, info);
#endif
      break;
    case RNS8_BACKEND_AUTO:
      break;
  }
}

bool uses_rns_storage(rns8_semantics semantics) {
  return semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64 ||
         semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED;
}

bool uses_finite_storage(rns8_semantics semantics) {
  return semantics == RNS8_FINITE_RING_U8 || semantics == RNS8_FINITE_FIELD_U8;
}

rns8_matrix_desc make_matrix_desc(
    int64_t rows,
    int64_t cols,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix,
    uint32_t tile_m = 128,
    uint32_t tile_n = 128) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  desc.tile_m = tile_m;
  desc.tile_n = tile_n;
  desc.max_prefix = prefix;
  return desc;
}

bool valid_matrix_access(int64_t rows, int64_t cols, int64_t ld) {
  if (rows <= 0 || cols <= 0 || ld < cols) {
    return false;
  }
  return rows <= std::numeric_limits<int64_t>::max() / ld;
}

bool valid_api_tile_size(uint32_t value) {
  return value == 0 || ((value >= 64 && value <= 512) && (value & (value - 1u)) == 0);
}

bool finite_backend_supports(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_CPU_REFERENCE || backend == RNS8_BACKEND_HIP_DIRECT ||
         backend == RNS8_BACKEND_HIPBLASLT || backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_WMMA;
}

rns8_status validate_finite_u8_oneshot_contract(
    const rns8_context& ctx,
    const rns8_gemm_desc& desc,
    rns8_semantics expected_semantics,
    uint16_t modulus,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  if (!rns8::detail::valid_abi(desc.struct_size, desc.abi_version, sizeof(desc)) ||
      desc.semantics != expected_semantics) {
    return RNS8_INVALID_ARGUMENT;
  }
  const bool valid_modulus = rns8::detail::valid_finite_modulus_for_semantics(expected_semantics, modulus);
  if (!valid_modulus || desc.m <= 0 || desc.n <= 0 || desc.k <= 0 || desc.bound_kind != RNS8_BOUND_NONE ||
      desc.bound != 0 || desc.max_prefix != 0 || desc.finite_modulus != modulus || desc.flags != 0 ||
      desc.tile_bounds || desc.tile_bounds_count != 0 || !valid_api_tile_size(desc.tile_m) ||
      !valid_api_tile_size(desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!valid_matrix_access(desc.m, desc.k, lda) || !valid_matrix_access(desc.k, desc.n, ldb) ||
      !valid_matrix_access(desc.m, desc.n, ldc)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_backend_kind requested = effective_backend(desc.requested_backend, ctx.backend);
  if (requested != ctx.backend || !finite_backend_supports(requested)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  return RNS8_SUCCESS;
}

struct resident_oneshot_state {
  rns8_plan* plan = nullptr;
  rns8_matrix* A = nullptr;
  rns8_matrix* B = nullptr;
  rns8_matrix* C = nullptr;
  rns8_workspace* workspace = nullptr;

  resident_oneshot_state() = default;
  resident_oneshot_state(const resident_oneshot_state&) = delete;
  resident_oneshot_state& operator=(const resident_oneshot_state&) = delete;

  ~resident_oneshot_state() {
    rns8_destroy_workspace(workspace);
    rns8_destroy_matrix(C);
    rns8_destroy_matrix(B);
    rns8_destroy_matrix(A);
    rns8_destroy_plan(plan);
  }
};

rns8_status create_resident_oneshot_state(
    rns8_context* ctx,
    const rns8_gemm_desc& desc,
    resident_oneshot_state& state) {
  rns8_status status = rns8_create_plan(ctx, &desc, &state.plan);
  if (status != RNS8_SUCCESS) {
    return status;
  }

  const rns8_matrix_desc a_desc =
      make_matrix_desc(desc.m, desc.k, desc.semantics, desc.bound_kind, state.plan->prefix,
                       state.plan->desc.tile_m, state.plan->desc.tile_n);
  const rns8_matrix_desc b_desc =
      make_matrix_desc(desc.k, desc.n, desc.semantics, desc.bound_kind, state.plan->prefix,
                       state.plan->desc.tile_m, state.plan->desc.tile_n);
  const rns8_matrix_desc c_desc =
      make_matrix_desc(desc.m, desc.n, desc.semantics, desc.bound_kind, state.plan->prefix,
                       state.plan->desc.tile_m, state.plan->desc.tile_n);

  status = rns8_create_matrix(ctx, &a_desc, &state.A);
  if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &b_desc, &state.B);
  if (status == RNS8_SUCCESS) status = rns8_create_matrix(ctx, &c_desc, &state.C);
  if (status == RNS8_SUCCESS) status = rns8_create_workspace(ctx, state.plan, &state.workspace);
  return status;
}

rns8_status finite_u8_oneshot_resident(
    rns8_context* ctx,
    const rns8_gemm_desc& desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc) {
  resident_oneshot_state state;
  rns8_status status = create_resident_oneshot_state(ctx, desc, state);
  if (status == RNS8_SUCCESS) status = rns8_pack_finite_u8(ctx, state.A, modulus, A, lda, 1);
  if (status == RNS8_SUCCESS) status = rns8_pack_finite_u8(ctx, state.B, modulus, B, ldb, 2);
  if (status == RNS8_SUCCESS) {
    status = rns8_gemm_finite_u8(ctx, state.plan, modulus, state.A, state.B, state.C, state.workspace);
  }
  if (status == RNS8_SUCCESS) status = rns8_export_finite_u8(ctx, state.plan, modulus, state.C, C, ldc);
  return status;
}

bool valid_limb_export_access(int64_t rows, int64_t cols, int64_t ld, uint32_t limb_count) {
  if (!valid_matrix_access(rows, cols, ld) || limb_count == 0 || limb_count > 32) {
    return false;
  }
  const auto max = std::numeric_limits<int64_t>::max();
  if (ld > max / static_cast<int64_t>(limb_count)) {
    return false;
  }
  const int64_t limb_ld = ld * static_cast<int64_t>(limb_count);
  return rows <= max / limb_ld;
}

rns8_status validate_typed_oneshot_contract(
    const rns8_context& ctx,
    const rns8_gemm_desc& desc,
    rns8_semantics expected_semantics,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  if (!rns8::detail::valid_abi(desc.struct_size, desc.abi_version, sizeof(desc)) ||
      desc.semantics != expected_semantics) {
    return RNS8_INVALID_ARGUMENT;
  }
  const uint32_t prefix =
      desc.max_prefix == 0 ? rns8::detail::default_prefix_for_semantics(desc.semantics) : desc.max_prefix;
  const rns8_status validation = rns8::detail::validate_gemm_desc(desc, prefix);
  if (validation != RNS8_SUCCESS) {
    return validation;
  }
  if (!valid_matrix_access(desc.m, desc.k, lda) || !valid_matrix_access(desc.k, desc.n, ldb) ||
      !valid_matrix_access(desc.m, desc.n, ldc)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const rns8_backend_kind requested = effective_backend(desc.requested_backend, ctx.backend);
  if (requested != ctx.backend || !backend_supports_semantics(requested, desc.semantics)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  return RNS8_SUCCESS;
}

uint64_t ceil_div_i64_u32(int64_t value, uint32_t divisor) {
  const auto unsigned_value = static_cast<uint64_t>(value);
  return (unsigned_value + static_cast<uint64_t>(divisor) - 1u) / static_cast<uint64_t>(divisor);
}

boost::multiprecision::cpp_int schedule_required_range(const rns8_gemm_desc& desc) {
  using boost::multiprecision::cpp_int;
  switch (desc.semantics) {
    case RNS8_BOUNDED_I64:
      return cpp_int(2) * cpp_int(desc.bound);
    case RNS8_BOUNDED_U64:
      return cpp_int(desc.bound);
    case RNS8_EXACT_WIDE_SIGNED:
      return cpp_int(desc.k) * (cpp_int(1) << 127u);
    case RNS8_EXACT_WIDE_UNSIGNED:
      return cpp_int(desc.k) * (cpp_int(1) << 128u);
    case RNS8_WRAP_U64_MOD_2_64:
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return 0;
  }
  return 0;
}

bool is_per_tile_bound_kind(rns8_bound_kind bound_kind) {
  return bound_kind == RNS8_BOUND_PER_TILE_MAX_ABS || bound_kind == RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
}

boost::multiprecision::cpp_int bounded_range_from_bound(rns8_semantics semantics, uint64_t bound) {
  using boost::multiprecision::cpp_int;
  return semantics == RNS8_BOUNDED_I64 ? cpp_int(2) * cpp_int(bound) : cpp_int(bound);
}

rns8_plan_tile_schedule_entry make_tile_schedule_entry(
    const rns8_plan& plan,
    uint64_t index,
    uint32_t required_prefix,
    uint32_t selected_prefix,
    uint32_t group_index,
    uint32_t range_bit_length) {
  const uint64_t tile_row = index / plan.schedule_tile_cols;
  const uint64_t tile_col = index % plan.schedule_tile_cols;
  const int64_t row_offset = static_cast<int64_t>(tile_row * static_cast<uint64_t>(plan.desc.tile_m));
  const int64_t col_offset = static_cast<int64_t>(tile_col * static_cast<uint64_t>(plan.desc.tile_n));
  rns8_plan_tile_schedule_entry entry{};
  entry.struct_size = sizeof(entry);
  entry.abi_version = RNS8_ABI_VERSION;
  entry.flags = plan.schedule_flags;
  entry.tile_row = tile_row;
  entry.tile_col = tile_col;
  entry.row_offset = row_offset;
  entry.col_offset = col_offset;
  entry.row_extent = std::min<int64_t>(plan.desc.tile_m, plan.desc.m - row_offset);
  entry.col_extent = std::min<int64_t>(plan.desc.tile_n, plan.desc.n - col_offset);
  entry.required_prefix = required_prefix;
  entry.selected_prefix = selected_prefix;
  entry.group_index = group_index;
  entry.range_bit_length = range_bit_length;
  return entry;
}

rns8_status configure_plan_schedule(rns8_plan& plan) {
  const uint32_t tile_m = plan.desc.tile_m == 0 ? 128u : plan.desc.tile_m;
  const uint32_t tile_n = plan.desc.tile_n == 0 ? 128u : plan.desc.tile_n;
  plan.schedule_tile_rows = ceil_div_i64_u32(plan.desc.m, tile_m);
  plan.schedule_tile_cols = ceil_div_i64_u32(plan.desc.n, tile_n);
  if (plan.schedule_tile_cols != 0 &&
      plan.schedule_tile_rows > std::numeric_limits<uint64_t>::max() / plan.schedule_tile_cols) {
    return RNS8_RANGE_ERROR;
  }
  plan.schedule_tile_count = plan.schedule_tile_rows * plan.schedule_tile_cols;
  plan.tile_bounds.clear();
  plan.tile_schedule.clear();

  const boost::multiprecision::cpp_int required_range = schedule_required_range(plan.desc);
  plan.schedule_range_bit_length = rns8::detail::bit_length(required_range);
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64 || uses_finite_storage(plan.desc.semantics)) {
    plan.schedule_min_required_prefix = 0;
    plan.schedule_max_required_prefix = 0;
    plan.schedule_min_selected_prefix = 0;
    plan.schedule_max_selected_prefix = 0;
    plan.schedule_prefix_group_count = 0;
    plan.schedule_adaptive_prefix_active = 0;
    plan.schedule_adaptive_skip_active = 0;
    return RNS8_SUCCESS;
  }

  if (is_per_tile_bound_kind(plan.desc.bound_kind)) {
    if (!plan.desc.tile_bounds || plan.desc.tile_bounds_count != plan.schedule_tile_count) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan.schedule_tile_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return RNS8_RANGE_ERROR;
    }
    plan.tile_bounds.assign(plan.desc.tile_bounds, plan.desc.tile_bounds + plan.desc.tile_bounds_count);
    plan.desc.tile_bounds = nullptr;
    plan.desc.tile_bounds_count = static_cast<uint64_t>(plan.tile_bounds.size());
    plan.tile_schedule.reserve(plan.tile_bounds.size());

    uint32_t min_required = std::numeric_limits<uint32_t>::max();
    uint32_t max_required = 0;
    uint32_t min_selected = std::numeric_limits<uint32_t>::max();
    uint32_t max_selected = 0;
    uint32_t max_range_bits = 0;
    std::vector<uint32_t> groups;
    groups.reserve(plan.tile_bounds.size());

    for (uint64_t index = 0; index < plan.schedule_tile_count; ++index) {
      const uint64_t bound = plan.tile_bounds[static_cast<std::size_t>(index)];
      if (plan.desc.semantics == RNS8_BOUNDED_I64 && bound > (uint64_t{1} << 63u)) {
        return RNS8_INVALID_ARGUMENT;
      }
      const boost::multiprecision::cpp_int range = bounded_range_from_bound(plan.desc.semantics, bound);
      const uint32_t range_bits = rns8::detail::bit_length(range);
      const uint32_t required_prefix = rns8::detail::required_prefix_for_range(range);
      if (required_prefix == 0 || required_prefix > plan.prefix) {
        return RNS8_RANGE_ERROR;
      }
      const uint32_t selected_prefix = required_prefix;
      min_required = std::min(min_required, required_prefix);
      max_required = std::max(max_required, required_prefix);
      min_selected = std::min(min_selected, selected_prefix);
      max_selected = std::max(max_selected, selected_prefix);
      max_range_bits = std::max(max_range_bits, range_bits);
      if (std::find(groups.begin(), groups.end(), selected_prefix) == groups.end()) {
        groups.push_back(selected_prefix);
      }
      plan.tile_schedule.push_back(
          make_tile_schedule_entry(plan, index, required_prefix, selected_prefix, 0, range_bits));
    }

    std::sort(groups.begin(), groups.end());
    for (auto& entry : plan.tile_schedule) {
      entry.group_index = static_cast<uint32_t>(
          std::lower_bound(groups.begin(), groups.end(), entry.selected_prefix) - groups.begin());
    }
    plan.schedule_min_required_prefix = min_required == std::numeric_limits<uint32_t>::max() ? 0 : min_required;
    plan.schedule_max_required_prefix = max_required;
    plan.schedule_min_selected_prefix = min_selected == std::numeric_limits<uint32_t>::max() ? 0 : min_selected;
    plan.schedule_max_selected_prefix = max_selected;
    plan.schedule_prefix_group_count = static_cast<uint32_t>(groups.size());
    plan.schedule_range_bit_length = max_range_bits;
    plan.schedule_adaptive_prefix_active = groups.size() > 1 ? 1u : 0u;
    plan.schedule_adaptive_skip_active = max_selected < plan.prefix ? 1u : 0u;
    return RNS8_SUCCESS;
  }

  const uint32_t required_prefix = rns8::detail::required_prefix_for_range(required_range);
  if (required_prefix == 0 || required_prefix > plan.prefix) {
    return RNS8_RANGE_ERROR;
  }
  plan.schedule_min_required_prefix = required_prefix;
  plan.schedule_max_required_prefix = required_prefix;
  plan.schedule_min_selected_prefix = plan.prefix;
  plan.schedule_max_selected_prefix = plan.prefix;
  plan.schedule_prefix_group_count = 1;
  plan.schedule_adaptive_prefix_active = 0;
  plan.schedule_adaptive_skip_active = 0;
  return RNS8_SUCCESS;
}

const char* semantics_name_for_key(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
      return "bounded_i64";
    case RNS8_BOUNDED_U64:
      return "bounded_u64";
    case RNS8_EXACT_WIDE_SIGNED:
      return "exact_wide_signed";
    case RNS8_EXACT_WIDE_UNSIGNED:
      return "exact_wide_unsigned";
    case RNS8_WRAP_U64_MOD_2_64:
      return "wrap_u64_mod_2_64";
    case RNS8_FINITE_RING_U8:
      return "finite_ring_u8";
    case RNS8_FINITE_FIELD_U8:
      return "finite_field_u8";
  }
  return "unknown";
}

std::string selected_kernel_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_CPU_REFERENCE) {
    return uses_finite_storage(plan.desc.semantics) ? "cpu_reference_finite_u8_gemm_v1"
                                                    : "cpu_reference_scalar_rns_gemm_v1";
  }
  if (plan.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
    return "cpu_wrap64_byte_limb_reference_v1";
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      return "direct_hip_wrap64_byte_gemm36_tiled_2d_v3";
    }
    if (uses_finite_storage(plan.desc.semantics)) {
      if (plan.desc.finite_modulus == 256) {
        return "direct_hip_tiled_finite_u8_gemm_mod256_v1";
      }
      if (plan.desc.finite_modulus == 255) {
        return "direct_hip_tiled_finite_u8_gemm_mod255_v1";
      }
      if (plan.desc.finite_modulus == 251) {
        return "direct_hip_tiled_finite_u8_gemm_mod251_v1";
      }
      return "direct_hip_tiled_finite_u8_gemm_v1";
    }
    return "direct_hip_tiled_rns_gemm_v1";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return plan.desc.semantics == RNS8_BOUNDED_I64 ? "hip_vector_alu_i64_exact_192b_v1"
                                                   : "hip_vector_alu_u64_exact_192b_v1";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return "hipblaslt_int8_i32_scratch_reduce_baseline_v1";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    if (!plan.tile_schedule.empty()) {
      return "ck_wmma_cshuffle_tiled_i8_i32_centered_epilogue_v1";
    }
    if (uses_finite_storage(plan.desc.semantics)) {
      return "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1";
    }
    return "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1";
  }
  if (plan.backend == RNS8_BACKEND_WMMA) {
    if (!plan.tile_schedule.empty()) {
      return "rocwmma_i8_i32_signed_tiled_hot_residue_v1";
    }
    if (uses_finite_storage(plan.desc.semantics)) {
      return "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1";
    }
    return "rocwmma_i8_i32_signed_hot_residue_v1";
  }
  return "not_implemented";
}

std::string epilogue_mode_for_plan(const rns8_plan& plan) {
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "low64_wrap_export";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "direct_int64_export";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "separate_i32_scratch_reduce_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "separate_i32_scratch_reduce_rns_output";
    }
    return "separate_i32_scratch_reduce_then_crt_export";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "ck_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "ck_fused_i32_to_centered_residue_rns_output";
    }
    return "ck_fused_i32_to_centered_residue_then_crt_export";
  }
  if (plan.backend == RNS8_BACKEND_WMMA) {
    if (uses_finite_storage(plan.desc.semantics)) {
      return "rocwmma_fused_i32_to_centered_residue_then_canonical_u8_export";
    }
    if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
      return "rocwmma_fused_i32_to_centered_residue_rns_output";
    }
    return "rocwmma_fused_i32_to_centered_residue_then_crt_export";
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return "fused_centered_residue_then_canonical_u8_export";
  }
  if (plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    return "fused_centered_residue_rns_output";
  }
  return "fused_centered_residue_then_crt_export";
}

std::string workspace_mode_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    return plan.tile_schedule.empty() ? "resident_device_buffers"
                                      : "resident_device_buffers_with_tiled_schedule";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "native_device_i64_u64_buffers";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return "resident_device_buffers_with_hipblaslt_scratch";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    return "resident_device_buffers_with_ck_canonical_pack_workspace";
  }
  if (plan.backend == RNS8_BACKEND_WMMA) {
    return "resident_device_buffers_with_rocwmma_pack_workspace";
  }
  if (plan.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
    return "host_byte_limb_reference_workspace";
  }
  return "host_reference_workspace";
}

std::string isa_evidence_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      return "wrap64_byte_gemm36_isa_gate_no_variable_divide_no_matrix_engine";
    }
    if (uses_finite_storage(plan.desc.semantics) &&
        (plan.desc.finite_modulus == 251 || plan.desc.finite_modulus == 255 || plan.desc.finite_modulus == 256)) {
      return "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide";
    }
    return "rns8_hip_direct_reciprocal_isa_gate";
  }
  if (plan.backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
    return "source_level_192bit_limb_accumulator_no_matrix_engine";
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return "hipblaslt_library_int8_matmul_baseline";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    return "ck_wmma_cshuffle_int8_matrix_isa_gate_no_int32_global_store_no_divide";
  }
  if (plan.backend == RNS8_BACKEND_WMMA) {
    return "rocwmma_i8_wmma_isa_gate_no_int32_global_store_no_divide";
  }
  return "not_applicable_cpu";
}

uint64_t workspace_required_bytes_for_plan(const rns8_plan& plan) {
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    std::size_t scratch_bytes = 0;
    std::size_t workspace_bytes = 0;
    if (!rns8::detail::hipblaslt_baseline_workspace_requirements(
            plan.desc.m, plan.desc.n, plan.desc.k, scratch_bytes, workspace_bytes)) {
      return 0;
    }
    if (scratch_bytes > std::numeric_limits<uint64_t>::max() - workspace_bytes) {
      return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(scratch_bytes) + static_cast<uint64_t>(workspace_bytes);
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    int64_t max_m = plan.desc.m;
    int64_t max_n = plan.desc.n;
    if (!plan.tile_schedule.empty()) {
      max_m = 0;
      max_n = 0;
      for (const auto& entry : plan.tile_schedule) {
        max_m = std::max(max_m, entry.row_extent);
        max_n = std::max(max_n, entry.col_extent);
      }
    }
    std::size_t a_pack_bytes = 0;
    std::size_t b_pack_bytes = 0;
    std::size_t temp_c_bytes = 0;
    std::size_t total_bytes = 0;
    if (!rns8::detail::ck_workspace_requirements(
            max_m, max_n, plan.desc.k, a_pack_bytes, b_pack_bytes, temp_c_bytes, total_bytes)) {
      return 0;
    }
    return static_cast<uint64_t>(total_bytes);
  }
  if (plan.backend == RNS8_BACKEND_WMMA) {
    int64_t max_m = plan.desc.m;
    int64_t max_n = plan.desc.n;
    if (!plan.tile_schedule.empty()) {
      max_m = 0;
      max_n = 0;
      for (const auto& entry : plan.tile_schedule) {
        max_m = std::max(max_m, entry.row_extent);
        max_n = std::max(max_n, entry.col_extent);
      }
    }
    std::size_t a_pack_bytes = 0;
    std::size_t b_pack_bytes = 0;
    std::size_t total_bytes = 0;
    if (!rns8::detail::wmma_workspace_requirements(
            max_m, max_n, plan.desc.k, a_pack_bytes, b_pack_bytes, total_bytes)) {
      return 0;
    }
    return static_cast<uint64_t>(total_bytes);
  }
  if (plan.backend != RNS8_BACKEND_HIP_DIRECT || plan.tile_schedule.empty()) {
    return 0;
  }
  return static_cast<uint64_t>(plan.tile_schedule.size()) * sizeof(rns8_plan_tile_schedule_entry);
}

bool accelerator_workspace_shape_for_plan(const rns8_plan& plan, int64_t& max_m, int64_t& max_n) {
  max_m = plan.desc.m;
  max_n = plan.desc.n;
  if (!plan.tile_schedule.empty()) {
    max_m = 0;
    max_n = 0;
    for (const auto& entry : plan.tile_schedule) {
      max_m = std::max(max_m, entry.row_extent);
      max_n = std::max(max_n, entry.col_extent);
    }
  }
  return max_m > 0 && max_n > 0 && plan.desc.k > 0;
}

bool hipblaslt_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_pack_bytes,
    uint64_t& b_pack_bytes,
    uint64_t& accumulator_bytes,
    uint64_t& library_workspace_bytes,
    uint64_t& total_bytes) {
  a_pack_bytes = 0;
  b_pack_bytes = 0;
  accumulator_bytes = 0;
  library_workspace_bytes = 0;
  total_bytes = 0;
  std::size_t scratch_bytes = 0;
  std::size_t workspace_bytes = 0;
  if (!rns8::detail::hipblaslt_baseline_workspace_requirements(
          plan.desc.m, plan.desc.n, plan.desc.k, scratch_bytes, workspace_bytes)) {
    return false;
  }
  uint64_t padded_m = 0;
  uint64_t padded_n = 0;
  uint64_t padded_k = 0;
  const uint64_t max_k_block =
      static_cast<uint64_t>(plan.desc.k) < static_cast<uint64_t>(RNS8_SAFE_INT32_K_BLOCK)
          ? static_cast<uint64_t>(plan.desc.k)
          : static_cast<uint64_t>(RNS8_SAFE_INT32_K_BLOCK);
  if (!rns8::detail::hipblaslt_round_up_aligned(static_cast<uint64_t>(plan.desc.m), padded_m) ||
      !rns8::detail::hipblaslt_round_up_aligned(static_cast<uint64_t>(plan.desc.n), padded_n) ||
      !rns8::detail::hipblaslt_round_up_aligned(max_k_block, padded_k)) {
    return false;
  }
  if (!rns8::detail::checked_mul_u64(padded_m, padded_k, a_pack_bytes) ||
      !rns8::detail::checked_mul_u64(padded_n, padded_k, b_pack_bytes) ||
      workspace_bytes < a_pack_bytes || workspace_bytes - a_pack_bytes < b_pack_bytes) {
    return false;
  }
  accumulator_bytes = static_cast<uint64_t>(scratch_bytes);
  library_workspace_bytes = static_cast<uint64_t>(workspace_bytes) - a_pack_bytes - b_pack_bytes;
  if (accumulator_bytes > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(workspace_bytes)) {
    return false;
  }
  total_bytes = accumulator_bytes + static_cast<uint64_t>(workspace_bytes);
  return total_bytes == plan.backend_workspace_required_bytes;
}

bool ck_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_pack_bytes,
    uint64_t& b_pack_bytes,
    uint64_t& accumulator_bytes,
    uint64_t& total_bytes) {
  a_pack_bytes = 0;
  b_pack_bytes = 0;
  accumulator_bytes = 0;
  total_bytes = 0;
  int64_t max_m = 0;
  int64_t max_n = 0;
  if (!accelerator_workspace_shape_for_plan(plan, max_m, max_n)) {
    return false;
  }
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t temp_bytes = 0;
  std::size_t total_workspace = 0;
  if (!rns8::detail::ck_workspace_requirements(
          max_m, max_n, plan.desc.k, a_bytes, b_bytes, temp_bytes, total_workspace)) {
    return false;
  }
  a_pack_bytes = static_cast<uint64_t>(a_bytes);
  b_pack_bytes = static_cast<uint64_t>(b_bytes);
  accumulator_bytes = static_cast<uint64_t>(temp_bytes);
  total_bytes = static_cast<uint64_t>(total_workspace);
  return total_bytes == plan.backend_workspace_required_bytes;
}

bool wmma_pack_workspace_breakdown(
    const rns8_plan& plan,
    uint64_t& a_pack_bytes,
    uint64_t& b_pack_bytes,
    uint64_t& total_bytes) {
  a_pack_bytes = 0;
  b_pack_bytes = 0;
  total_bytes = 0;
  int64_t max_m = 0;
  int64_t max_n = 0;
  if (!accelerator_workspace_shape_for_plan(plan, max_m, max_n)) {
    return false;
  }
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t total_workspace = 0;
  if (!rns8::detail::wmma_workspace_requirements(max_m, max_n, plan.desc.k, a_bytes, b_bytes, total_workspace)) {
    return false;
  }
  a_pack_bytes = static_cast<uint64_t>(a_bytes);
  b_pack_bytes = static_cast<uint64_t>(b_bytes);
  total_bytes = static_cast<uint64_t>(total_workspace);
  return total_bytes == plan.backend_workspace_required_bytes;
}

const char* persistent_layout_version_for_semantics(rns8_semantics semantics) {
  if (semantics == RNS8_WRAP_U64_MOD_2_64) {
    return "wrap64_byte_limb_v1";
  }
  if (uses_finite_storage(semantics)) {
    return "finite_u8_centered_residue_v1";
  }
  return "rns_centered_residue_planes_v1";
}

const char* native_layout_version_for_semantics(rns8_semantics semantics) {
  if (semantics == RNS8_BOUNDED_I64) {
    return "native_i64_rowmajor_v1";
  }
  if (semantics == RNS8_BOUNDED_U64) {
    return "native_u64_rowmajor_v1";
  }
  return "not_applicable";
}

const char* persistent_layout_version_for_plan(const rns8_plan& plan) {
  return native_vector_backend(plan.backend) ? native_layout_version_for_semantics(plan.desc.semantics)
                                             : persistent_layout_version_for_semantics(plan.desc.semantics);
}

const char* storage_scope_for_matrix(const rns8_matrix& matrix) {
  if (native_vector_backend(matrix.backend)) {
    return "native_device_storage";
  }
  if (matrix.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return matrix.backend == RNS8_BACKEND_HIP_DIRECT ? "device_byte_limb_storage" : "host_byte_limb_storage";
  }
  return hip_resident_rns_backend(matrix.backend) ? "device_resident_storage" : "host_resident_storage";
}

const char* storage_detail_for_matrix(const rns8_matrix& matrix) {
  if (native_vector_backend(matrix.backend)) {
    return "Bounded vector-ALU matrix owns compact native integer storage on the selected HIP device.";
  }
  if (matrix.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return matrix.backend == RNS8_BACKEND_HIP_DIRECT
               ? "Strict wrap64 matrix owns compact byte-limb storage on the selected HIP device."
               : "Strict wrap64 matrix owns compact host byte-limb storage.";
  }
  if (uses_finite_storage(matrix.desc.semantics)) {
    return hip_resident_rns_backend(matrix.backend)
               ? "Finite-u8 matrix owns one centered residue plane on the selected HIP device."
               : "Finite-u8 matrix owns one centered host residue plane.";
  }
  return hip_resident_rns_backend(matrix.backend)
             ? "RNS matrix owns centered residue planes on the selected HIP device."
             : "RNS matrix owns centered host residue planes.";
}

bool hipblaslt_scratch_bytes_for_plan(const rns8_plan& plan, std::size_t& bytes) {
  bytes = 0;
  if (plan.backend != RNS8_BACKEND_HIPBLASLT) {
    return false;
  }
  std::size_t workspace_bytes = 0;
  return rns8::detail::hipblaslt_baseline_workspace_requirements(
      plan.desc.m, plan.desc.n, plan.desc.k, bytes, workspace_bytes);
}

bool hipblaslt_workspace_bytes_for_plan(const rns8_plan& plan, std::size_t& bytes) {
  bytes = 0;
  if (plan.backend != RNS8_BACKEND_HIPBLASLT) {
    return false;
  }
  std::size_t scratch_bytes = 0;
  return rns8::detail::hipblaslt_baseline_workspace_requirements(
      plan.desc.m, plan.desc.n, plan.desc.k, scratch_bytes, bytes);
}

std::string build_autotune_key(const rns8_plan& plan) {
  std::string key = "backend=";
  key += backend_name(plan.backend);
  key += ";semantics=";
  key += semantics_name_for_key(plan.desc.semantics);
  key += ";m=" + std::to_string(plan.desc.m);
  key += ";n=" + std::to_string(plan.desc.n);
  key += ";k=" + std::to_string(plan.desc.k);
  if (uses_finite_storage(plan.desc.semantics)) {
    key += ";finite_modulus=" + std::to_string(plan.desc.finite_modulus);
  }
  key += ";prefix=" + std::to_string(plan.prefix);
  key += ";tile_m=" + std::to_string(plan.desc.tile_m);
  key += ";tile_n=" + std::to_string(plan.desc.tile_n);
  key += ";groups=" + std::to_string(plan.schedule_prefix_group_count);
  key += ";adaptive_prefix=" + std::to_string(plan.schedule_adaptive_prefix_active);
  key += ";adaptive_skip=" + std::to_string(plan.schedule_adaptive_skip_active);
  key += ";kernel=" + plan.backend_selected_kernel;
  key += ";epilogue=" + plan.backend_epilogue_mode;
  return key;
}

bool backend_library_version_matches_plan(const rns8_plan& plan, const rns8_backend_capability_info& capability) {
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return plan.backend_library_version.rfind("hipBLASLt ", 0) == 0 ||
           plan.backend_library_version == capability.library_version;
  }
  return plan.backend_library_version == capability.library_version;
}

rns8::detail::AutotuneRuntimeIdentity autotune_runtime_identity_for_plan(
    const rns8_context& ctx,
    const rns8_plan& plan) {
  rns8::detail::AutotuneRuntimeIdentity runtime{};
  const std::string target = ctx.device_info.gcn_arch;
  if (!target.empty() && target != "none") {
    runtime.target_id = target;
  } else if (!ctx.device_info.hip_available) {
    runtime.target_id = "cpu";
  }
  runtime.hip_sdk_or_library_version = plan.backend_library_version;
  return runtime;
}

void configure_plan_backend_metadata(rns8_plan& plan, const rns8_context* ctx = nullptr) {
  rns8_backend_capability_info capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = RNS8_ABI_VERSION;
  fill_backend_capability_info(plan.backend, capability);
  plan.backend_selected_kernel = selected_kernel_for_plan(plan);
  plan.backend_library = capability.library_name;
  plan.backend_library_version = capability.library_version;
  if (plan.backend == RNS8_BACKEND_HIPBLASLT && ctx && !ctx->hipblaslt_library_version.empty()) {
    plan.backend_library_version = ctx->hipblaslt_library_version;
  }
  plan.backend_capability_status = capability.status;
  plan.backend_epilogue_mode = epilogue_mode_for_plan(plan);
  plan.backend_workspace_mode = workspace_mode_for_plan(plan);
  plan.backend_isa_evidence = isa_evidence_for_plan(plan);
  plan.backend_workspace_required_bytes = workspace_required_bytes_for_plan(plan);
  plan.backend_performance_validated = capability.performance_validated;
  plan.backend_autotune_key = build_autotune_key(plan);
}

bool prepare_auto_candidate_backend(rns8_context& ctx, rns8_backend_kind backend) {
  static_cast<void>(ctx);
  rns8_backend_capability_info capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = RNS8_ABI_VERSION;
  fill_backend_capability_info(backend, capability);
  if (!capability.is_available || !capability.compiled_kernel_available ||
      !capability.exact_differential_validated) {
    return false;
  }

  rns8_device_info probe_info{};
  probe_info.struct_size = sizeof(probe_info);
  probe_info.abi_version = RNS8_ABI_VERSION;
  switch (backend) {
    case RNS8_BACKEND_HIPBLASLT:
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      if (ctx.hipblaslt_handle) {
        return true;
      }
      return rns8::detail::hipblaslt_create_context(
                 ctx.device_id, probe_info, &ctx.hipblaslt_handle, ctx.hipblaslt_library_version) == RNS8_SUCCESS;
#else
      return false;
#endif
    case RNS8_BACKEND_CK:
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      return rns8::detail::ck_probe(ctx.device_id, probe_info) == RNS8_SUCCESS;
#else
      return false;
#endif
    case RNS8_BACKEND_WMMA:
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      return rns8::detail::wmma_probe(ctx.device_id, probe_info) == RNS8_SUCCESS;
#else
      return false;
#endif
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      return rns8::detail::vector_alu_probe(ctx.device_id, probe_info) == RNS8_SUCCESS;
#else
      return false;
#endif
    default:
      return false;
  }
}

void select_auto_backend_from_reviewed_cache(rns8_context& ctx, rns8_plan& plan) {
  if (!ctx.auto_backend_selection || ctx.backend != RNS8_BACKEND_HIP_DIRECT ||
      !backend_supports_semantics(ctx.backend, plan.desc.semantics)) {
    return;
  }
  const rns8_backend_kind candidates[] = {
      RNS8_BACKEND_HIP_VECTOR_ALU_INT64,
      RNS8_BACKEND_HIPBLASLT,
      RNS8_BACKEND_CK,
      RNS8_BACKEND_WMMA,
  };
  const auto snapshot = rns8::detail::read_autotune_cache();
  const rns8::detail::AutotuneCacheEntry* best_hit = nullptr;
  rns8_plan best_plan{};

  for (const rns8_backend_kind candidate_backend : candidates) {
    if (!backend_supports_semantics(candidate_backend, plan.desc.semantics)) {
      continue;
    }
    if (candidate_backend == RNS8_BACKEND_HIPBLASLT && is_per_tile_bound_kind(plan.desc.bound_kind)) {
      continue;
    }
    if (!prepare_auto_candidate_backend(ctx, candidate_backend)) {
      continue;
    }
    rns8_plan candidate = plan;
    candidate.backend = candidate_backend;
    configure_plan_backend_metadata(candidate, &ctx);
    const auto runtime = autotune_runtime_identity_for_plan(ctx, candidate);
    const auto* hit =
        rns8::detail::find_validated_autotune_entry_for_runtime(snapshot, candidate.backend_autotune_key, runtime);
    if (!hit || hit->workspace_bytes != candidate.backend_workspace_required_bytes) {
      continue;
    }
    if (!best_hit || hit->measured_median_end_to_end_us < best_hit->measured_median_end_to_end_us) {
      best_hit = hit;
      best_plan = std::move(candidate);
    }
  }

  if (best_hit) {
    plan = std::move(best_plan);
    plan.backend_performance_validated = 1;
  }
}

std::vector<int8_t> gather_cell_residues(const rns8_matrix& matrix, int64_t row, int64_t col, uint32_t prefix) {
  std::vector<int8_t> residues(prefix);
  for (uint32_t p = 0; p < prefix; ++p) {
    residues[p] = matrix.residues[rns8::detail::residue_index(matrix, p, row, col)];
  }
  return residues;
}

uint64_t tile_index_for_cell(const rns8_plan& plan, int64_t row, int64_t col) {
  const auto tile_row = static_cast<uint64_t>(row) / static_cast<uint64_t>(plan.desc.tile_m);
  const auto tile_col = static_cast<uint64_t>(col) / static_cast<uint64_t>(plan.desc.tile_n);
  return tile_row * plan.schedule_tile_cols + tile_col;
}

const rns8_plan_tile_schedule_entry* tile_schedule_entry_for_cell(const rns8_plan& plan, int64_t row, int64_t col) {
  if (plan.tile_schedule.empty()) {
    return nullptr;
  }
  const uint64_t index = tile_index_for_cell(plan, row, col);
  if (index >= static_cast<uint64_t>(plan.tile_schedule.size())) {
    return nullptr;
  }
  return &plan.tile_schedule[static_cast<std::size_t>(index)];
}

uint32_t selected_prefix_for_cell(const rns8_plan& plan, int64_t row, int64_t col) {
  const rns8_plan_tile_schedule_entry* entry = tile_schedule_entry_for_cell(plan, row, col);
  return entry ? entry->selected_prefix : plan.prefix;
}

uint64_t bound_for_cell(const rns8_plan& plan, int64_t row, int64_t col) {
  if (plan.tile_bounds.empty()) {
    return plan.desc.bound;
  }
  const uint64_t index = tile_index_for_cell(plan, row, col);
  if (index >= static_cast<uint64_t>(plan.tile_bounds.size())) {
    return plan.desc.bound;
  }
  return plan.tile_bounds[static_cast<std::size_t>(index)];
}

uint64_t workspace_fingerprint_mix(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= 1099511628211ull;
  return hash;
}

uint64_t workspace_fingerprint_mix_string(uint64_t hash, const std::string& value) {
  for (const unsigned char c : value) {
    hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(c));
  }
  return workspace_fingerprint_mix(hash, static_cast<uint64_t>(value.size()));
}

uint64_t signed_to_fingerprint(int64_t value) {
  return static_cast<uint64_t>(value);
}

uint64_t gemm_output_source_version_values(uint64_t a_source_version, uint64_t b_source_version) {
  uint64_t hash = 1469598103934665603ull;
  hash = workspace_fingerprint_mix(hash, a_source_version);
  hash = workspace_fingerprint_mix(hash, b_source_version);
  return hash == 0 ? 1 : hash;
}

uint64_t gemm_output_source_version(const rns8_matrix& A, const rns8_matrix& B) {
  return gemm_output_source_version_values(A.source_version, B.source_version);
}

uint64_t plan_workspace_fingerprint(const rns8_plan& plan) {
  uint64_t hash = 1469598103934665603ull;
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.desc.semantics));
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.desc.bound_kind));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(plan.desc.m));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(plan.desc.n));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(plan.desc.k));
  hash = workspace_fingerprint_mix(hash, plan.desc.bound);
  hash = workspace_fingerprint_mix(hash, plan.desc.finite_modulus);
  hash = workspace_fingerprint_mix(hash, plan.desc.tile_m);
  hash = workspace_fingerprint_mix(hash, plan.desc.tile_n);
  hash = workspace_fingerprint_mix(hash, plan.prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_tile_rows);
  hash = workspace_fingerprint_mix(hash, plan.schedule_tile_cols);
  hash = workspace_fingerprint_mix(hash, plan.schedule_tile_count);
  hash = workspace_fingerprint_mix(hash, plan.schedule_min_required_prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_max_required_prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_min_selected_prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_max_selected_prefix);
  hash = workspace_fingerprint_mix(hash, plan.schedule_prefix_group_count);
  hash = workspace_fingerprint_mix(hash, plan.schedule_range_bit_length);
  hash = workspace_fingerprint_mix(hash, plan.schedule_adaptive_prefix_active);
  hash = workspace_fingerprint_mix(hash, plan.schedule_adaptive_skip_active);
  hash = workspace_fingerprint_mix(hash, plan.schedule_flags);
  hash = workspace_fingerprint_mix(hash, plan.backend_workspace_required_bytes);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_selected_kernel);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_library);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_library_version);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_capability_status);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_epilogue_mode);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_workspace_mode);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_isa_evidence);
  hash = workspace_fingerprint_mix_string(hash, plan.backend_autotune_key);
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.tile_bounds.size()));
  for (const uint64_t bound : plan.tile_bounds) {
    hash = workspace_fingerprint_mix(hash, bound);
  }
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(plan.tile_schedule.size()));
  for (const auto& entry : plan.tile_schedule) {
    hash = workspace_fingerprint_mix(hash, entry.struct_size);
    hash = workspace_fingerprint_mix(hash, entry.abi_version);
    hash = workspace_fingerprint_mix(hash, entry.flags);
    hash = workspace_fingerprint_mix(hash, entry.tile_row);
    hash = workspace_fingerprint_mix(hash, entry.tile_col);
    hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(entry.row_offset));
    hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(entry.col_offset));
    hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(entry.row_extent));
    hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(entry.col_extent));
    hash = workspace_fingerprint_mix(hash, entry.required_prefix);
    hash = workspace_fingerprint_mix(hash, entry.selected_prefix);
    hash = workspace_fingerprint_mix(hash, entry.group_index);
    hash = workspace_fingerprint_mix(hash, entry.range_bit_length);
  }
  return hash == 0 ? 1 : hash;
}

bool matrix_descriptor_matches(
    const rns8_matrix& matrix,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint32_t tile_m,
    uint32_t tile_n) {
  if (matrix.desc.semantics != semantics || matrix.desc.bound_kind != bound_kind || matrix.desc.rows != rows ||
      matrix.desc.cols != cols || matrix.desc.logical_layout != RNS8_LAYOUT_ROW_MAJOR ||
      matrix.desc.logical_ld < matrix.desc.cols || matrix.desc.flags != 0 || matrix.prefix != prefix ||
      matrix.desc.max_prefix != prefix) {
    return false;
  }
  if (semantics == RNS8_WRAP_U64_MOD_2_64 || is_per_tile_bound_kind(bound_kind)) {
    return matrix.desc.tile_m == tile_m && matrix.desc.tile_n == tile_n;
  }
  return true;
}

bool configured_tile_size_valid(uint32_t value) {
  return value >= 64 && value <= 512 && (value & (value - 1u)) == 0;
}

bool wrap_byte_limb_bytes(int64_t rows, int64_t cols, std::size_t& bytes) {
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const auto u_rows = static_cast<uint64_t>(rows);
  const auto u_cols = static_cast<uint64_t>(cols);
  constexpr uint64_t limbs_per_cell = 8;
  const uint64_t max_bytes = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
  if (u_cols != 0 && u_rows > max_bytes / u_cols / limbs_per_cell) {
    return false;
  }
  bytes = static_cast<std::size_t>(u_rows * u_cols * limbs_per_cell);
  return true;
}

bool matrix_cell_count(int64_t rows, int64_t cols, std::size_t& cells) {
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const auto u_rows = static_cast<uint64_t>(rows);
  const auto u_cols = static_cast<uint64_t>(cols);
  const uint64_t max_cells = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
  if (u_cols != 0 && u_rows > max_cells / u_cols) {
    return false;
  }
  cells = static_cast<std::size_t>(u_rows * u_cols);
  return true;
}

bool native_matrix_bytes(int64_t rows, int64_t cols, std::size_t element_size, std::size_t& bytes) {
  std::size_t cells = 0;
  if (element_size == 0 || !matrix_cell_count(rows, cols, cells)) {
    return false;
  }
  if (cells > std::numeric_limits<std::size_t>::max() / element_size) {
    return false;
  }
  bytes = cells * element_size;
  return true;
}

bool bounded_native_storage_matches(
    const rns8_matrix& matrix,
    rns8_semantics semantics,
    int64_t rows,
    int64_t cols) {
  if (semantics != RNS8_BOUNDED_I64 && semantics != RNS8_BOUNDED_U64) {
    return false;
  }
  std::size_t expected_bytes = 0;
  if (!native_matrix_bytes(rows, cols, sizeof(uint64_t), expected_bytes)) {
    return false;
  }
  if (semantics == RNS8_BOUNDED_I64) {
    return matrix.hip_native_i64 != nullptr && matrix.hip_native_i64_bytes == expected_bytes &&
           matrix.hip_native_u64 == nullptr && matrix.hip_native_u64_bytes == 0 && matrix.native_u64.empty();
  }
  return matrix.hip_native_u64 != nullptr && matrix.hip_native_u64_bytes == expected_bytes &&
         matrix.hip_native_i64 == nullptr && matrix.hip_native_i64_bytes == 0 && matrix.native_i64.empty();
}

bool bounded_native_state_current(const rns8_matrix& matrix) {
  return matrix.device_native_current && !matrix.host_native_current;
}

bool matrix_has_native_storage(const rns8_matrix& matrix) {
  return !matrix.native_i64.empty() || !matrix.native_u64.empty() || matrix.hip_native_i64 ||
         matrix.hip_native_u64 || matrix.hip_native_i64_bytes != 0 || matrix.hip_native_u64_bytes != 0;
}

void clear_native_current(rns8_matrix& matrix) {
  matrix.host_native_current = false;
  matrix.device_native_current = false;
}

bool rns_residue_count(int64_t rows, int64_t cols, uint32_t prefix, std::size_t& residues) {
  std::size_t cells = 0;
  if (prefix == 0 || !matrix_cell_count(rows, cols, cells)) {
    return false;
  }
  if (cells > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(prefix)) {
    return false;
  }
  residues = cells * static_cast<std::size_t>(prefix);
  return true;
}

bool rns_matrix_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows, int64_t cols, uint32_t prefix) {
  std::size_t expected_residues = 0;
  if (!rns_residue_count(rows, cols, prefix, expected_residues)) {
    return false;
  }
  if (matrix.residues.size() != expected_residues || !matrix.byte_limbs.empty() || matrix.hip_byte_limbs ||
      matrix.hip_byte_limb_bytes != 0 || matrix.host_byte_limbs_current || matrix.device_byte_limbs_current) {
    return false;
  }
  const std::size_t expected_bytes = expected_residues * sizeof(int8_t);
  if (hip_resident_rns_backend(backend)) {
    return matrix.hip_residues != nullptr && matrix.hip_residue_bytes == expected_bytes;
  }
  return matrix.hip_residues == nullptr && matrix.hip_residue_bytes == 0 && !matrix.device_residues_current;
}

bool finite_matrix_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows, int64_t cols) {
  std::size_t expected_cells = 0;
  if (!matrix_cell_count(rows, cols, expected_cells)) {
    return false;
  }
  if (matrix.prefix != 0 || matrix.residues.size() != expected_cells || !matrix.byte_limbs.empty() ||
      matrix.hip_byte_limbs || matrix.hip_byte_limb_bytes != 0 || matrix.host_byte_limbs_current ||
      matrix.device_byte_limbs_current) {
    return false;
  }
  const std::size_t expected_bytes = expected_cells * sizeof(int8_t);
  if (hip_resident_rns_backend(backend)) {
    return matrix.hip_residues != nullptr && matrix.hip_residue_bytes == expected_bytes;
  }
  return matrix.hip_residues == nullptr && matrix.hip_residue_bytes == 0 && !matrix.device_residues_current;
}

bool rns_residue_state_current_for_backend(const rns8_matrix& matrix, rns8_backend_kind backend) {
  if (hip_resident_rns_backend(backend)) {
    return matrix.device_residues_current;
  }
  return matrix.host_residues_current && !matrix.device_residues_current;
}

bool plan_schedule_contract_matches(const rns8_plan& plan) {
  if (!backend_supports_semantics(plan.backend, plan.desc.semantics) ||
      !configured_tile_size_valid(plan.desc.tile_m) || !configured_tile_size_valid(plan.desc.tile_n) ||
      plan.desc.flags != 0 || plan.prefix != plan.desc.max_prefix) {
    return false;
  }
  rns8_backend_capability_info capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = RNS8_ABI_VERSION;
  fill_backend_capability_info(plan.backend, capability);
  if (plan.backend_selected_kernel != selected_kernel_for_plan(plan) ||
      plan.backend_library != capability.library_name ||
      !backend_library_version_matches_plan(plan, capability) ||
      plan.backend_capability_status != capability.status ||
      plan.backend_epilogue_mode != epilogue_mode_for_plan(plan) ||
      plan.backend_workspace_mode != workspace_mode_for_plan(plan) ||
      plan.backend_isa_evidence != isa_evidence_for_plan(plan) ||
      plan.backend_workspace_required_bytes != workspace_required_bytes_for_plan(plan) ||
      plan.backend_autotune_key != build_autotune_key(plan)) {
    return false;
  }
  const uint64_t expected_tile_rows = ceil_div_i64_u32(plan.desc.m, plan.desc.tile_m);
  const uint64_t expected_tile_cols = ceil_div_i64_u32(plan.desc.n, plan.desc.tile_n);
  if (plan.schedule_tile_rows != expected_tile_rows || plan.schedule_tile_cols != expected_tile_cols) {
    return false;
  }
  if (expected_tile_cols != 0 && expected_tile_rows > std::numeric_limits<uint64_t>::max() / expected_tile_cols) {
    return false;
  }
  if (plan.schedule_tile_count != expected_tile_rows * expected_tile_cols || plan.schedule_tile_count == 0) {
    return false;
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return plan.desc.bound_kind == RNS8_BOUND_NONE && plan.desc.bound == 0 && plan.prefix == 0 &&
           plan.modulus_product == 0 && plan.tile_bounds.empty() && plan.tile_schedule.empty() &&
           plan.desc.tile_bounds == nullptr && plan.desc.tile_bounds_count == 0 &&
           plan.schedule_min_required_prefix == 0 && plan.schedule_max_required_prefix == 0 &&
           plan.schedule_min_selected_prefix == 0 && plan.schedule_max_selected_prefix == 0 &&
           plan.schedule_prefix_group_count == 0 && plan.schedule_range_bit_length == 0 &&
           plan.schedule_adaptive_prefix_active == 0 && plan.schedule_adaptive_skip_active == 0 &&
           plan.schedule_flags == 0;
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return plan.desc.bound_kind == RNS8_BOUND_NONE && plan.desc.bound == 0 && plan.prefix == 0 &&
           plan.desc.max_prefix == 0 && plan.modulus_product == 0 && plan.tile_bounds.empty() &&
           plan.tile_schedule.empty() && plan.desc.tile_bounds == nullptr && plan.desc.tile_bounds_count == 0 &&
           plan.schedule_min_required_prefix == 0 && plan.schedule_max_required_prefix == 0 &&
           plan.schedule_min_selected_prefix == 0 && plan.schedule_max_selected_prefix == 0 &&
           plan.schedule_prefix_group_count == 0 && plan.schedule_range_bit_length == 0 &&
           plan.schedule_adaptive_prefix_active == 0 && plan.schedule_adaptive_skip_active == 0 &&
           plan.schedule_flags == 0;
  }
  if (!uses_rns_storage(plan.desc.semantics) || plan.prefix == 0 || plan.prefix > RNS8_MAX_SUPPORTED_PREFIX) {
    return false;
  }
  if (plan.modulus_product != rns8::detail::modulus_product(plan.prefix)) {
    return false;
  }
  if ((plan.desc.semantics == RNS8_EXACT_WIDE_SIGNED || plan.desc.semantics == RNS8_EXACT_WIDE_UNSIGNED) &&
      (plan.desc.bound_kind != RNS8_BOUND_NONE || plan.desc.bound != 0 || plan.desc.tile_bounds != nullptr ||
       plan.desc.tile_bounds_count != 0)) {
    return false;
  }
  if (plan.desc.semantics == RNS8_BOUNDED_I64 &&
      plan.desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_ABS && plan.desc.bound_kind != RNS8_BOUND_PER_TILE_MAX_ABS) {
    return false;
  }
  if (plan.desc.semantics == RNS8_BOUNDED_U64 &&
      plan.desc.bound_kind != RNS8_BOUND_GLOBAL_MAX_UNSIGNED &&
      plan.desc.bound_kind != RNS8_BOUND_PER_TILE_MAX_UNSIGNED) {
    return false;
  }
  const bool per_tile = is_per_tile_bound_kind(plan.desc.bound_kind);
  if (!per_tile) {
    return plan.desc.tile_bounds == nullptr && plan.desc.tile_bounds_count == 0 && plan.tile_bounds.empty() &&
           plan.tile_schedule.empty() && plan.schedule_min_required_prefix > 0 &&
           plan.schedule_max_required_prefix == plan.schedule_min_required_prefix &&
           plan.schedule_min_selected_prefix == plan.prefix && plan.schedule_max_selected_prefix == plan.prefix &&
           plan.schedule_prefix_group_count == 1 && plan.schedule_adaptive_prefix_active == 0 &&
           plan.schedule_adaptive_skip_active == 0 && plan.schedule_flags == 0;
  }
  if (plan.desc.bound != 0 || plan.desc.tile_bounds_count != static_cast<uint64_t>(plan.tile_bounds.size()) ||
      plan.desc.tile_bounds_count != plan.schedule_tile_count || plan.desc.tile_bounds != nullptr ||
      plan.tile_schedule.size() != plan.tile_bounds.size() || plan.schedule_prefix_group_count == 0 ||
      plan.schedule_min_required_prefix == 0 || plan.schedule_min_selected_prefix == 0 ||
      plan.schedule_max_required_prefix > plan.prefix || plan.schedule_max_selected_prefix > plan.prefix ||
      plan.schedule_min_required_prefix > plan.schedule_max_required_prefix ||
      plan.schedule_min_selected_prefix > plan.schedule_max_selected_prefix || plan.schedule_flags != 0) {
    return false;
  }
  for (uint64_t index = 0; index < plan.schedule_tile_count; ++index) {
    const auto& entry = plan.tile_schedule[static_cast<std::size_t>(index)];
    if (!rns8::detail::valid_abi(entry.struct_size, entry.abi_version, sizeof(entry)) || entry.flags != 0 ||
        entry.tile_row != index / plan.schedule_tile_cols || entry.tile_col != index % plan.schedule_tile_cols ||
        entry.row_offset < 0 || entry.col_offset < 0 || entry.row_extent <= 0 || entry.col_extent <= 0 ||
        entry.row_offset >= plan.desc.m || entry.col_offset >= plan.desc.n ||
        entry.row_extent > plan.desc.m - entry.row_offset || entry.col_extent > plan.desc.n - entry.col_offset ||
        entry.required_prefix == 0 || entry.selected_prefix == 0 || entry.required_prefix > entry.selected_prefix ||
        entry.selected_prefix > plan.prefix || entry.group_index >= plan.schedule_prefix_group_count) {
      return false;
    }
  }
  return true;
}

bool wrap_matrix_storage_matches(const rns8_matrix& matrix, rns8_backend_kind backend, int64_t rows, int64_t cols) {
  std::size_t expected_bytes = 0;
  if (!wrap_byte_limb_bytes(rows, cols, expected_bytes)) {
    return false;
  }
  if (!matrix.residues.empty() || matrix.hip_residues || matrix.hip_residue_bytes != 0 ||
      matrix.byte_limbs.size() != expected_bytes || matrix.host_residues_current ||
      matrix.device_residues_current) {
    return false;
  }
  if (backend == RNS8_BACKEND_HIP_DIRECT) {
    return !matrix.host_byte_limbs_current && matrix.hip_byte_limbs != nullptr &&
           matrix.hip_byte_limb_bytes == expected_bytes;
  }
  return matrix.hip_byte_limbs == nullptr && matrix.hip_byte_limb_bytes == 0 &&
         matrix.hip_upload_buffer == nullptr && matrix.hip_upload_bytes == 0 &&
         matrix.hip_export_buffer == nullptr && matrix.hip_export_bytes == 0 &&
         matrix.hip_status_buffer == nullptr && matrix.hip_status_bytes == 0 &&
         !matrix.device_byte_limbs_current;
}

bool wrap_byte_limb_state_current_for_backend(const rns8_matrix& matrix, rns8_backend_kind backend) {
  if (matrix.host_residues_current || matrix.device_residues_current) {
    return false;
  }
  if (backend == RNS8_BACKEND_HIP_DIRECT) {
    return matrix.device_byte_limbs_current;
  }
  return matrix.host_byte_limbs_current && !matrix.device_byte_limbs_current;
}

bool valid_prepack_operand_role(rns8_operand_role operand_role) {
  return operand_role == RNS8_OPERAND_A || operand_role == RNS8_OPERAND_B;
}

const char* operand_role_name(rns8_operand_role operand_role) {
  switch (operand_role) {
    case RNS8_OPERAND_A:
      return "A";
    case RNS8_OPERAND_B:
      return "B";
  }
  return "unknown";
}

bool prepack_operand_shape_for_plan(
    const rns8_plan& plan,
    rns8_operand_role operand_role,
    int64_t& rows,
    int64_t& cols) {
  if (operand_role == RNS8_OPERAND_A) {
    rows = plan.desc.m;
    cols = plan.desc.k;
    return true;
  }
  if (operand_role == RNS8_OPERAND_B) {
    rows = plan.desc.k;
    cols = plan.desc.n;
    return true;
  }
  rows = 0;
  cols = 0;
  return false;
}

bool matrix_backend_can_feed_plan(const rns8_matrix& matrix, const rns8_plan& plan) {
  if (matrix.backend == plan.backend) {
    return true;
  }
  return matrix.backend == RNS8_BACKEND_HIP_DIRECT && hip_resident_rns_backend(plan.backend);
}

const char* prepack_operand_layout_version_for_plan(const rns8_plan& plan, rns8_operand_role operand_role) {
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    return operand_role == RNS8_OPERAND_A ? "hipblaslt_a_transposed_centered_i8_mk16_v1"
                                          : "hipblaslt_b_transposed_centered_i8_nk16_v1";
  }
  if (plan.backend == RNS8_BACKEND_CK) {
    return operand_role == RNS8_OPERAND_A ? "ck_a_canonical_rowmajor_i8_m64_kblock32768_v1"
                                          : "ck_b_canonical_colmajor_i8_n64_kblock32768_v1";
  }
  if (plan.backend == RNS8_BACKEND_WMMA) {
    return operand_role == RNS8_OPERAND_A ? "rocwmma_a_rowmajor_i8_m16_kblock65536_v1"
                                          : "rocwmma_b_colmajor_i8_n16_kblock65536_v1";
  }
  return persistent_layout_version_for_plan(plan);
}

bool prepack_operand_matrix_compatible(
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_operand_role operand_role) {
  if (!valid_prepack_operand_role(operand_role) || !plan_schedule_contract_matches(plan) ||
      !matrix_backend_can_feed_plan(matrix, plan)) {
    return false;
  }
  int64_t rows = 0;
  int64_t cols = 0;
  if (!prepack_operand_shape_for_plan(plan, operand_role, rows, cols)) {
    return false;
  }
  if (!matrix_descriptor_matches(
          matrix,
          plan.desc.semantics,
          plan.desc.bound_kind,
          rows,
          cols,
          plan.prefix,
          plan.desc.tile_m,
          plan.desc.tile_n)) {
    return false;
  }
  if (plan.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    return wrap_matrix_storage_matches(matrix, plan.backend, rows, cols) &&
           wrap_byte_limb_state_current_for_backend(matrix, plan.backend);
  }
  if (uses_finite_storage(plan.desc.semantics)) {
    return finite_matrix_storage_matches(matrix, plan.backend, rows, cols) &&
           matrix.finite_modulus == plan.desc.finite_modulus &&
           rns_residue_state_current_for_backend(matrix, plan.backend);
  }
  return rns_matrix_storage_matches(matrix, plan.backend, rows, cols, plan.prefix) &&
         rns_residue_state_current_for_backend(matrix, plan.backend);
}

uint64_t prepack_cache_key_hash(
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_operand_role operand_role,
    const std::string& matrix_layout_version,
    const std::string& operand_layout_version) {
  uint64_t hash = plan_workspace_fingerprint(plan);
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(operand_role));
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(matrix.backend));
  hash = workspace_fingerprint_mix(hash, static_cast<uint64_t>(matrix.desc.semantics));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(matrix.desc.rows));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(matrix.desc.cols));
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(matrix.desc.logical_ld));
  hash = workspace_fingerprint_mix(hash, matrix.prefix);
  hash = workspace_fingerprint_mix(hash, matrix.finite_modulus);
  hash = workspace_fingerprint_mix(hash, matrix.source_version);
  hash = workspace_fingerprint_mix(hash, signed_to_fingerprint(matrix.hip_device_id));
  hash = workspace_fingerprint_mix(hash, matrix.host_residues_current ? 1u : 0u);
  hash = workspace_fingerprint_mix(hash, matrix.device_residues_current ? 1u : 0u);
  hash = workspace_fingerprint_mix(hash, matrix.host_byte_limbs_current ? 1u : 0u);
  hash = workspace_fingerprint_mix(hash, matrix.device_byte_limbs_current ? 1u : 0u);
  hash = workspace_fingerprint_mix_string(hash, matrix_layout_version);
  hash = workspace_fingerprint_mix_string(hash, operand_layout_version);
  return hash == 0 ? 1 : hash;
}

std::string build_prepack_cache_key(
    const rns8_plan& plan,
    const rns8_matrix& matrix,
    rns8_operand_role operand_role,
    uint64_t plan_fingerprint,
    uint64_t cache_key_hash,
    const std::string& matrix_layout_version,
    const std::string& operand_layout_version) {
  std::string key = "prepack-v1";
  key += ";backend=";
  key += backend_name(plan.backend);
  key += ";semantics=";
  key += semantics_name_for_key(plan.desc.semantics);
  key += ";operand=";
  key += operand_role_name(operand_role);
  key += ";m=" + std::to_string(plan.desc.m);
  key += ";n=" + std::to_string(plan.desc.n);
  key += ";k=" + std::to_string(plan.desc.k);
  key += ";matrix_rows=" + std::to_string(matrix.desc.rows);
  key += ";matrix_cols=" + std::to_string(matrix.desc.cols);
  key += ";prefix=" + std::to_string(matrix.prefix);
  key += ";finite_modulus=" + std::to_string(matrix.finite_modulus);
  key += ";source_version=" + std::to_string(matrix.source_version);
  key += ";hip_device_id=" + std::to_string(matrix.hip_device_id);
  key += ";matrix_layout=" + matrix_layout_version;
  key += ";operand_layout=" + operand_layout_version;
  key += ";plan_fingerprint=" + std::to_string(plan_fingerprint);
  key += ";hash=" + std::to_string(cache_key_hash);
  return key;
}

bool wmma_b_prepack_cache_supported(const rns8_plan& plan) {
  return plan.backend == RNS8_BACKEND_WMMA && !uses_finite_storage(plan.desc.semantics) &&
         plan.desc.semantics != RNS8_WRAP_U64_MOD_2_64 && plan.tile_schedule.empty() && plan.desc.k > 0 &&
         plan.desc.k <= static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) && plan_schedule_contract_matches(plan);
}

bool wmma_b_prepack_bytes_for_plan(const rns8_plan& plan, std::size_t& b_pack_bytes, std::size_t& total_cache_bytes) {
  b_pack_bytes = 0;
  total_cache_bytes = 0;
  if (!wmma_b_prepack_cache_supported(plan)) {
    return false;
  }
  std::size_t a_bytes = 0;
  std::size_t total_workspace = 0;
  if (!rns8::detail::wmma_workspace_requirements(
          plan.desc.m, plan.desc.n, plan.desc.k, a_bytes, b_pack_bytes, total_workspace)) {
    return false;
  }
  if (plan.prefix != 0 && b_pack_bytes > std::numeric_limits<std::size_t>::max() / plan.prefix) {
    return false;
  }
  total_cache_bytes = b_pack_bytes * static_cast<std::size_t>(plan.prefix);
  return total_cache_bytes != 0;
}

bool prepack_cache_matches_plan(const rns8_prepack_cache& cache, const rns8_plan& plan) {
  std::size_t b_pack_bytes = 0;
  std::size_t total_cache_bytes = 0;
  return wmma_b_prepack_bytes_for_plan(plan, b_pack_bytes, total_cache_bytes) &&
         cache.backend == plan.backend && cache.semantics == plan.desc.semantics &&
         cache.operand_role == RNS8_OPERAND_B && cache.rows == plan.desc.k && cache.cols == plan.desc.n &&
         cache.k == plan.desc.k && cache.prefix == plan.prefix &&
         cache.finite_modulus == plan.desc.finite_modulus &&
         cache.plan_fingerprint == plan_workspace_fingerprint(plan) &&
         cache.matrix_layout_version == persistent_layout_version_for_semantics(plan.desc.semantics) &&
         cache.operand_layout_version == prepack_operand_layout_version_for_plan(plan, RNS8_OPERAND_B) &&
         cache.device_data != nullptr && cache.device_bytes == total_cache_bytes &&
         cache.operand_pack_bytes == b_pack_bytes;
}

rns8_status validate_rns_gemm_prepacked_b_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_prepack_cache& B,
    const rns8_matrix& C) {
  if (!context_accepts_backend(ctx, plan.backend) || !wmma_b_prepack_cache_supported(plan) ||
      !prepack_cache_matches_plan(B, plan) || B.hip_device_id != ctx.device_id ||
      !matrix_backend_compatible_with_plan(ctx, A, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) && (A.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          A,
          plan.desc.semantics,
          plan.desc.bound_kind,
          plan.desc.m,
          plan.desc.k,
          plan.prefix,
          plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C,
          plan.desc.semantics,
          plan.desc.bound_kind,
          plan.desc.m,
          plan.desc.n,
          plan.prefix,
          plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k, plan.prefix) ||
      !rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, plan.prefix) ||
      !rns_residue_state_current_for_backend(A, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_plan_context_workspace(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_workspace& workspace) {
  if (!plan_schedule_contract_matches(plan)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!context_accepts_backend(ctx, plan.backend) || workspace.backend != plan.backend) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (workspace.semantics != plan.desc.semantics || workspace.bound_kind != plan.desc.bound_kind) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (workspace.m != plan.desc.m || workspace.n != plan.desc.n || workspace.k != plan.desc.k ||
      workspace.prefix != plan.prefix) {
    return RNS8_WORKSPACE_TOO_SMALL;
  }
  if (workspace.bound != plan.desc.bound || workspace.finite_modulus != plan.desc.finite_modulus ||
      workspace.tile_m != plan.desc.tile_m || workspace.tile_n != plan.desc.tile_n ||
      workspace.schedule_tile_rows != plan.schedule_tile_rows ||
      workspace.schedule_tile_cols != plan.schedule_tile_cols ||
      workspace.schedule_tile_count != plan.schedule_tile_count ||
      workspace.schedule_min_required_prefix != plan.schedule_min_required_prefix ||
      workspace.schedule_max_required_prefix != plan.schedule_max_required_prefix ||
      workspace.schedule_min_selected_prefix != plan.schedule_min_selected_prefix ||
      workspace.schedule_max_selected_prefix != plan.schedule_max_selected_prefix ||
      workspace.schedule_prefix_group_count != plan.schedule_prefix_group_count ||
      workspace.schedule_range_bit_length != plan.schedule_range_bit_length ||
      workspace.schedule_adaptive_prefix_active != plan.schedule_adaptive_prefix_active ||
      workspace.schedule_adaptive_skip_active != plan.schedule_adaptive_skip_active ||
      workspace.schedule_flags != plan.schedule_flags ||
      workspace.schedule_fingerprint != plan_workspace_fingerprint(plan) ||
      workspace.backend_workspace_required_bytes != plan.backend_workspace_required_bytes ||
      workspace.backend_selected_kernel != plan.backend_selected_kernel ||
      workspace.backend_library != plan.backend_library ||
      workspace.backend_library_version != plan.backend_library_version ||
      workspace.backend_capability_status != plan.backend_capability_status ||
      workspace.backend_epilogue_mode != plan.backend_epilogue_mode ||
      workspace.backend_workspace_mode != plan.backend_workspace_mode ||
      workspace.backend_isa_evidence != plan.backend_isa_evidence ||
      workspace.backend_autotune_key != plan.backend_autotune_key ||
      workspace.backend_performance_validated != plan.backend_performance_validated) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_device_backend(plan.backend) && workspace.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIP_DIRECT) {
    const bool scheduled = !plan.tile_schedule.empty();
    if (scheduled && plan.tile_schedule.size() >
                         std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const std::size_t expected_schedule_bytes =
        scheduled ? plan.tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry) : 0;
    if (scheduled) {
      if (!workspace.hip_tile_schedule ||
          workspace.hip_tile_schedule_bytes != expected_schedule_bytes ||
          workspace.hip_tile_schedule_count != static_cast<uint64_t>(plan.tile_schedule.size())) {
        return RNS8_INVALID_ARGUMENT;
      }
    } else if (workspace.hip_tile_schedule || workspace.hip_tile_schedule_bytes != 0 ||
               workspace.hip_tile_schedule_count != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
  } else if (workspace.hip_tile_schedule || workspace.hip_tile_schedule_bytes != 0 ||
             workspace.hip_tile_schedule_count != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.backend == RNS8_BACKEND_HIPBLASLT) {
    std::size_t expected_scratch_bytes = 0;
    std::size_t expected_workspace_bytes = 0;
    if (!hipblaslt_scratch_bytes_for_plan(plan, expected_scratch_bytes)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!hipblaslt_workspace_bytes_for_plan(plan, expected_workspace_bytes)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!workspace.hipblaslt_int32_scratch ||
        workspace.hipblaslt_int32_scratch_bytes != expected_scratch_bytes ||
        !workspace.hipblaslt_workspace ||
        workspace.hipblaslt_workspace_bytes != expected_workspace_bytes) {
      return RNS8_INVALID_ARGUMENT;
    }
  } else if (workspace.hipblaslt_int32_scratch || workspace.hipblaslt_int32_scratch_bytes != 0 ||
             workspace.hipblaslt_workspace || workspace.hipblaslt_workspace_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (rns8::detail::accelerator_backend_kind(plan.backend)) {
    if (!rns8::detail::accelerator_backend_compiled(plan.backend)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const std::size_t expected_workspace_bytes =
        plan.backend_workspace_required_bytes > std::numeric_limits<std::size_t>::max()
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(plan.backend_workspace_required_bytes);
    if (expected_workspace_bytes != 0) {
      if (!workspace.accelerator_workspace ||
          workspace.accelerator_workspace_bytes != expected_workspace_bytes) {
        return RNS8_INVALID_ARGUMENT;
      }
    } else if (workspace.accelerator_workspace || workspace.accelerator_workspace_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (workspace.accelerator_auxiliary || workspace.accelerator_auxiliary_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
  } else if (workspace.accelerator_workspace || workspace.accelerator_workspace_bytes != 0 ||
             workspace.accelerator_auxiliary || workspace.accelerator_auxiliary_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_rns_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (!uses_rns_storage(plan.desc.semantics) || plan.prefix == 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_backend_compatible_with_plan(ctx, A, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, B, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_device_backend(plan.backend) &&
      (A.hip_device_id != ctx.device_id || B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          A, plan.desc.semantics, plan.desc.bound_kind, plan.desc.m, plan.desc.k, plan.prefix, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          B, plan.desc.semantics, plan.desc.bound_kind, plan.desc.k, plan.desc.n, plan.prefix, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C, plan.desc.semantics, plan.desc.bound_kind, plan.desc.m, plan.desc.n, plan.prefix, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (native_vector_backend(plan.backend)) {
    if (plan.desc.semantics != RNS8_BOUNDED_I64 && plan.desc.semantics != RNS8_BOUNDED_U64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!bounded_native_storage_matches(A, plan.desc.semantics, plan.desc.m, plan.desc.k) ||
        !bounded_native_storage_matches(B, plan.desc.semantics, plan.desc.k, plan.desc.n) ||
        !bounded_native_storage_matches(C, plan.desc.semantics, plan.desc.m, plan.desc.n) ||
        !bounded_native_state_current(A) || !bounded_native_state_current(B)) {
      return RNS8_INVALID_ARGUMENT;
    }
    return RNS8_SUCCESS;
  }
  if (!rns_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k, plan.prefix) ||
      !rns_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n, plan.prefix) ||
      !rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, plan.prefix)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_residue_state_current_for_backend(A, plan.backend) ||
      !rns_residue_state_current_for_backend(B, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_finite_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (!uses_finite_storage(plan.desc.semantics) ||
      !rns8::detail::valid_finite_modulus_for_semantics(plan.desc.semantics, modulus) ||
      plan.desc.finite_modulus != modulus) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_backend_compatible_with_plan(ctx, A, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, B, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) &&
      (A.hip_device_id != ctx.device_id || B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          A, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.m, plan.desc.k, 0, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          B, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.k, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.m, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!finite_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k) ||
      !finite_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n) ||
      !finite_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!rns_residue_state_current_for_backend(A, plan.backend) ||
      !rns_residue_state_current_for_backend(B, plan.backend) || A.finite_modulus != modulus ||
      B.finite_modulus != modulus) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_wrap_gemm_operands(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& A,
    const rns8_matrix& B,
    const rns8_matrix& C) {
  if (plan.desc.semantics != RNS8_WRAP_U64_MOD_2_64 || plan.prefix != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_backend_compatible_with_plan(ctx, A, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, B, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) &&
      (A.hip_device_id != ctx.device_id || B.hip_device_id != ctx.device_id || C.hip_device_id != ctx.device_id)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          A, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, plan.desc.m, plan.desc.k, 0, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          B, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, plan.desc.k, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n) ||
      !matrix_descriptor_matches(
          C, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, plan.desc.m, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (A.hip_residues || B.hip_residues || C.hip_residues) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!wrap_matrix_storage_matches(A, plan.backend, plan.desc.m, plan.desc.k) ||
      !wrap_matrix_storage_matches(B, plan.backend, plan.desc.k, plan.desc.n) ||
      !wrap_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!wrap_byte_limb_state_current_for_backend(A, plan.backend) ||
      !wrap_byte_limb_state_current_for_backend(B, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_export_matrix(
    const rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& C,
    rns8_semantics semantics,
    rns8_bound_kind bound_kind,
    uint32_t prefix) {
  if (!plan_schedule_contract_matches(plan)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!context_accepts_backend(ctx, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend) ||
      plan.desc.semantics != semantics) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_device_backend(plan.backend) && C.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          C, semantics, bound_kind, plan.desc.m, plan.desc.n, prefix, plan.desc.tile_m, plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (native_vector_backend(plan.backend)) {
    return bounded_native_storage_matches(C, semantics, plan.desc.m, plan.desc.n) &&
                   bounded_native_state_current(C)
               ? RNS8_SUCCESS
               : RNS8_INVALID_ARGUMENT;
  }
  if (uses_rns_storage(semantics) &&
      (!rns_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n, prefix) ||
       !rns_residue_state_current_for_backend(C, plan.backend))) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (semantics == RNS8_WRAP_U64_MOD_2_64 &&
      (!wrap_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n) ||
       !wrap_byte_limb_state_current_for_backend(C, plan.backend))) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status validate_finite_export_matrix(
    const rns8_context& ctx,
    const rns8_plan& plan,
    uint16_t modulus,
    const rns8_matrix& C) {
  if (!plan_schedule_contract_matches(plan) || !uses_finite_storage(plan.desc.semantics) ||
      !rns8::detail::valid_finite_modulus_for_semantics(plan.desc.semantics, modulus) ||
      plan.desc.finite_modulus != modulus) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!context_accepts_backend(ctx, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, C, plan.backend)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (hip_resident_rns_backend(plan.backend) && C.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!matrix_descriptor_matches(
          C, plan.desc.semantics, RNS8_BOUND_NONE, plan.desc.m, plan.desc.n, 0, plan.desc.tile_m,
          plan.desc.tile_n)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (!finite_matrix_storage_matches(C, plan.backend, plan.desc.m, plan.desc.n) ||
      !rns_residue_state_current_for_backend(C, plan.backend) || C.finite_modulus != modulus) {
    return RNS8_INVALID_ARGUMENT;
  }
  return RNS8_SUCCESS;
}

rns8_status free_hip_matrix_storage(rns8_matrix& matrix) {
  rns8_status status = RNS8_SUCCESS;
  if (matrix.hip_upload_buffer) {
    status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_upload_buffer);
    matrix.hip_upload_buffer = nullptr;
    matrix.hip_upload_bytes = 0;
  }
  if (matrix.hip_status_buffer) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_status_buffer);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_status_buffer = nullptr;
    matrix.hip_status_bytes = 0;
  }
  if (matrix.hip_export_buffer) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_export_buffer);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_export_buffer = nullptr;
    matrix.hip_export_bytes = 0;
  }
  if (matrix.hip_export_tile_bounds) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_export_tile_bounds);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_export_tile_bounds = nullptr;
    matrix.hip_export_tile_bounds_bytes = 0;
    matrix.hip_export_tile_bounds_count = 0;
  }
  if (matrix.hip_export_tile_schedule) {
    const rns8_status free_status =
        rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_export_tile_schedule);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_export_tile_schedule = nullptr;
    matrix.hip_export_tile_schedule_bytes = 0;
    matrix.hip_export_tile_schedule_count = 0;
  }
  if (matrix.hip_residues) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_residues);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_residues = nullptr;
    matrix.hip_residue_bytes = 0;
  }
  if (matrix.hip_byte_limbs) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_byte_limbs);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_byte_limbs = nullptr;
    matrix.hip_byte_limb_bytes = 0;
  }
  if (matrix.hip_native_i64) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_native_i64);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_native_i64 = nullptr;
    matrix.hip_native_i64_bytes = 0;
  }
  if (matrix.hip_native_u64) {
    const rns8_status free_status = rns8::detail::hip_direct_free(matrix.hip_device_id, matrix.hip_native_u64);
    if (status == RNS8_SUCCESS) {
      status = free_status;
    }
    matrix.hip_native_u64 = nullptr;
    matrix.hip_native_u64_bytes = 0;
  }
  matrix.hip_export_schedule_fingerprint = 0;
  matrix.hip_export_tile_max_elements = 0;
  matrix.finite_modulus = 0;
  matrix.device_residues_current = false;
  matrix.device_byte_limbs_current = false;
  matrix.host_native_current = false;
  matrix.device_native_current = false;
  return status;
}

rns8_status allocate_hip_matrix_storage(rns8_context& ctx, rns8_matrix& matrix) {
  matrix.hip_device_id = ctx.device_id;
  if (matrix.desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
    if (matrix.byte_limbs.empty()) {
      return RNS8_INVALID_ARGUMENT;
    }
    matrix.hip_byte_limb_bytes = matrix.byte_limbs.size() * sizeof(uint8_t);
    rns8_status status =
        rns8::detail::hip_direct_allocate(ctx.device_id, matrix.hip_byte_limb_bytes, &matrix.hip_byte_limbs);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_byte_limbs, matrix.hip_byte_limb_bytes);
    if (status != RNS8_SUCCESS) {
      (void)free_hip_matrix_storage(matrix);
      return status;
    }
    matrix.host_residues_current = false;
    matrix.device_residues_current = false;
    matrix.host_byte_limbs_current = false;
    matrix.device_byte_limbs_current = false;
    return RNS8_SUCCESS;
  }
  if (matrix.residues.empty()) {
    return RNS8_INVALID_ARGUMENT;
  }
  matrix.hip_residue_bytes = matrix.residues.size() * sizeof(int8_t);
  rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, matrix.hip_residue_bytes, &matrix.hip_residues);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_residues, matrix.hip_residue_bytes);
  if (status != RNS8_SUCCESS) {
    (void)free_hip_matrix_storage(matrix);
    return status;
  }
  matrix.host_residues_current = false;
  matrix.device_residues_current = false;
  matrix.host_byte_limbs_current = false;
  matrix.device_byte_limbs_current = false;
  return RNS8_SUCCESS;
}

rns8_status ensure_hip_native_storage(rns8_context& ctx, rns8_matrix& matrix) {
  if (matrix.desc.semantics != RNS8_BOUNDED_I64 && matrix.desc.semantics != RNS8_BOUNDED_U64) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (matrix.hip_device_id != -1 && matrix.hip_device_id != ctx.device_id) {
    return RNS8_INVALID_ARGUMENT;
  }
  matrix.hip_device_id = ctx.device_id;
  std::size_t expected_bytes = 0;
  if (!native_matrix_bytes(matrix.desc.rows, matrix.desc.cols, sizeof(uint64_t), expected_bytes)) {
    return RNS8_RANGE_ERROR;
  }
  if (matrix.desc.semantics == RNS8_BOUNDED_I64) {
    if (matrix.hip_native_i64 && matrix.hip_native_i64_bytes == expected_bytes) {
      return RNS8_SUCCESS;
    }
    if (matrix.hip_native_i64 || matrix.hip_native_i64_bytes != 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    matrix.hip_native_i64_bytes = expected_bytes;
    rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, expected_bytes, &matrix.hip_native_i64);
    if (status != RNS8_SUCCESS) {
      matrix.hip_native_i64 = nullptr;
      matrix.hip_native_i64_bytes = 0;
      return status;
    }
    status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_native_i64, matrix.hip_native_i64_bytes);
    if (status != RNS8_SUCCESS) {
      (void)free_hip_matrix_storage(matrix);
      return status;
    }
    return RNS8_SUCCESS;
  }
  if (matrix.hip_native_u64 && matrix.hip_native_u64_bytes == expected_bytes) {
    return RNS8_SUCCESS;
  }
  if (matrix.hip_native_u64 || matrix.hip_native_u64_bytes != 0) {
    return RNS8_INVALID_ARGUMENT;
  }
  matrix.hip_native_u64_bytes = expected_bytes;
  rns8_status status = rns8::detail::hip_direct_allocate(ctx.device_id, expected_bytes, &matrix.hip_native_u64);
  if (status != RNS8_SUCCESS) {
    matrix.hip_native_u64 = nullptr;
    matrix.hip_native_u64_bytes = 0;
    return status;
  }
  status = rns8::detail::hip_direct_zero(ctx.device_id, matrix.hip_native_u64, matrix.hip_native_u64_bytes);
  if (status != RNS8_SUCCESS) {
    (void)free_hip_matrix_storage(matrix);
    return status;
  }
  return RNS8_SUCCESS;
}

rns8_status upload_native_i64(rns8_context& ctx, rns8_matrix& matrix, const int64_t* src, int64_t ld) {
  if (!src || matrix.desc.semantics != RNS8_BOUNDED_I64 || !valid_matrix_access(matrix.desc.rows, matrix.desc.cols, ld)) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = ensure_hip_native_storage(ctx, matrix);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::vector<int64_t> staged(static_cast<std::size_t>(matrix.desc.rows) * static_cast<std::size_t>(matrix.desc.cols));
  for (int64_t row = 0; row < matrix.desc.rows; ++row) {
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      staged[static_cast<std::size_t>(row * matrix.desc.cols + col)] = src[row * ld + col];
    }
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_native_i64, staged.data(), staged.size() * sizeof(int64_t));
  if (status != RNS8_SUCCESS) {
    matrix.device_native_current = false;
    return status;
  }
  matrix.host_native_current = false;
  matrix.device_native_current = true;
  return RNS8_SUCCESS;
}

rns8_status upload_native_u64(rns8_context& ctx, rns8_matrix& matrix, const uint64_t* src, int64_t ld) {
  if (!src || matrix.desc.semantics != RNS8_BOUNDED_U64 || !valid_matrix_access(matrix.desc.rows, matrix.desc.cols, ld)) {
    return RNS8_INVALID_ARGUMENT;
  }
  rns8_status status = ensure_hip_native_storage(ctx, matrix);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  std::vector<uint64_t> staged(static_cast<std::size_t>(matrix.desc.rows) * static_cast<std::size_t>(matrix.desc.cols));
  for (int64_t row = 0; row < matrix.desc.rows; ++row) {
    for (int64_t col = 0; col < matrix.desc.cols; ++col) {
      staged[static_cast<std::size_t>(row * matrix.desc.cols + col)] = src[row * ld + col];
    }
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_native_u64, staged.data(), staged.size() * sizeof(uint64_t));
  if (status != RNS8_SUCCESS) {
    matrix.device_native_current = false;
    return status;
  }
  matrix.host_native_current = false;
  matrix.device_native_current = true;
  return RNS8_SUCCESS;
}

bool should_populate_native_on_pack(const rns8_context& ctx, const rns8_matrix& matrix) {
  if (matrix.desc.semantics != RNS8_BOUNDED_I64 && matrix.desc.semantics != RNS8_BOUNDED_U64) {
    return false;
  }
  return native_vector_backend(ctx.backend) ||
         (ctx.auto_backend_selection && ctx.backend == RNS8_BACKEND_HIP_DIRECT && matrix.backend == RNS8_BACKEND_HIP_DIRECT);
}

bool signed_value_within_bound(int64_t value, uint64_t bound) {
  if (value < 0) {
    const uint64_t magnitude = value == std::numeric_limits<int64_t>::min()
                                   ? (uint64_t{1} << 63u)
                                   : static_cast<uint64_t>(-value);
    return magnitude <= bound;
  }
  return static_cast<uint64_t>(value) <= bound;
}

rns8_status export_native_i64(
    rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& C,
    int64_t* dst,
    int64_t ld) {
  std::vector<int64_t> staged(static_cast<std::size_t>(plan.desc.m) * static_cast<std::size_t>(plan.desc.n), 0);
  rns8_status status = run_timed_api_status("vector_alu_output_d2h", [&]() {
    return rns8::detail::hip_direct_copy_device_to_host(
        ctx.device_id, staged.data(), C.hip_native_i64, staged.size() * sizeof(int64_t));
  });
  if (status != RNS8_SUCCESS) {
    return status;
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      const int64_t value = staged[static_cast<std::size_t>(row * plan.desc.n + col)];
      if (!signed_value_within_bound(value, bound_for_cell(plan, row, col))) {
        return RNS8_RANGE_ERROR;
      }
    }
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      const int64_t value = staged[static_cast<std::size_t>(row * plan.desc.n + col)];
      dst[row * ld + col] = value;
    }
  }
  return RNS8_SUCCESS;
}

rns8_status export_native_u64(
    rns8_context& ctx,
    const rns8_plan& plan,
    const rns8_matrix& C,
    uint64_t* dst,
    int64_t ld) {
  std::vector<uint64_t> staged(static_cast<std::size_t>(plan.desc.m) * static_cast<std::size_t>(plan.desc.n), 0);
  rns8_status status = run_timed_api_status("vector_alu_output_d2h", [&]() {
    return rns8::detail::hip_direct_copy_device_to_host(
        ctx.device_id, staged.data(), C.hip_native_u64, staged.size() * sizeof(uint64_t));
  });
  if (status != RNS8_SUCCESS) {
    return status;
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      const uint64_t value = staged[static_cast<std::size_t>(row * plan.desc.n + col)];
      if (value > bound_for_cell(plan, row, col)) {
        return RNS8_RANGE_ERROR;
      }
    }
  }
  for (int64_t row = 0; row < plan.desc.m; ++row) {
    for (int64_t col = 0; col < plan.desc.n; ++col) {
      const uint64_t value = staged[static_cast<std::size_t>(row * plan.desc.n + col)];
      dst[row * ld + col] = value;
    }
  }
  return RNS8_SUCCESS;
}

rns8_status ensure_hip_export_tile_metadata(rns8_context& ctx, const rns8_plan& plan, rns8_matrix& matrix) {
  if (!context_accepts_backend(ctx, plan.backend) ||
      !matrix_backend_compatible_with_plan(ctx, matrix, plan.backend) ||
      !hip_resident_rns_backend(plan.backend) || plan.tile_schedule.empty()) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (plan.tile_bounds.size() != plan.tile_schedule.size() ||
      plan.tile_schedule.size() > std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry) ||
      plan.tile_bounds.size() > std::numeric_limits<std::size_t>::max() / sizeof(uint64_t)) {
    return RNS8_INTERNAL_ERROR;
  }
  uint64_t max_tile_elements = 0;
  for (const auto& entry : plan.tile_schedule) {
    if (entry.row_extent <= 0 || entry.col_extent <= 0) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t row_extent = static_cast<uint64_t>(entry.row_extent);
    const uint64_t col_extent = static_cast<uint64_t>(entry.col_extent);
    if (col_extent != 0 && row_extent > std::numeric_limits<uint64_t>::max() / col_extent) {
      return RNS8_RANGE_ERROR;
    }
    max_tile_elements = std::max(max_tile_elements, row_extent * col_extent);
  }
  if (max_tile_elements == 0 || max_tile_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return RNS8_RANGE_ERROR;
  }
  const uint64_t fingerprint = plan_workspace_fingerprint(plan);
  const uint64_t entry_count = static_cast<uint64_t>(plan.tile_schedule.size());
  const std::size_t schedule_bytes = plan.tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry);
  const std::size_t bounds_bytes = plan.tile_bounds.size() * sizeof(uint64_t);
  const bool metadata_current =
      matrix.hip_export_schedule_fingerprint == fingerprint &&
      matrix.hip_export_tile_schedule != nullptr &&
      matrix.hip_export_tile_schedule_bytes >= schedule_bytes &&
      matrix.hip_export_tile_schedule_count == entry_count &&
      matrix.hip_export_tile_bounds != nullptr &&
      matrix.hip_export_tile_bounds_bytes >= bounds_bytes &&
      matrix.hip_export_tile_bounds_count == entry_count &&
      matrix.hip_export_tile_max_elements == max_tile_elements;
  if (metadata_current) {
    return RNS8_SUCCESS;
  }
  rns8_status status = rns8::detail::hip_direct_ensure_upload_buffer(
      ctx.device_id, schedule_bytes, &matrix.hip_export_tile_schedule, &matrix.hip_export_tile_schedule_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_ensure_upload_buffer(
      ctx.device_id, bounds_bytes, &matrix.hip_export_tile_bounds, &matrix.hip_export_tile_bounds_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_export_tile_schedule, plan.tile_schedule.data(), schedule_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx.device_id, matrix.hip_export_tile_bounds, plan.tile_bounds.data(), bounds_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  matrix.hip_export_tile_schedule_count = entry_count;
  matrix.hip_export_tile_bounds_count = entry_count;
  matrix.hip_export_schedule_fingerprint = fingerprint;
  matrix.hip_export_tile_max_elements = max_tile_elements;
  return RNS8_SUCCESS;
}

}  // namespace

rns8_status rns8_create_context(int device_id, const rns8_context_options* options, rns8_context** out) {
  return guard_api([&]() -> rns8_status {
    if (!out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;

    rns8_backend_kind requested = RNS8_BACKEND_CPU_REFERENCE;
    bool requested_auto = false;
    if (options) {
      if (!rns8::detail::valid_abi(options->struct_size, options->abi_version, sizeof(*options))) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (options->flags != 0) {
        return RNS8_INVALID_ARGUMENT;
      }
      requested_auto = options->requested_backend == RNS8_BACKEND_AUTO;
      requested = requested_auto ? RNS8_BACKEND_AUTO : options->requested_backend;
    }

    auto* ctx = new (std::nothrow) rns8_context();
    if (!ctx) {
      return RNS8_INTERNAL_ERROR;
    }

    if (requested_auto) {
      ctx->auto_backend_selection = true;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      ctx->backend = RNS8_BACKEND_HIP_DIRECT;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status hip_status = rns8::detail::hip_direct_probe(ctx->device_id, ctx->device_info);
      if (hip_status == RNS8_SUCCESS) {
        *out = ctx;
        return RNS8_SUCCESS;
      }
#endif
      ctx->backend = RNS8_BACKEND_CPU_REFERENCE;
      ctx->device_id = -1;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      rns8::detail::fill_cpu_device_info(ctx->device_info);
      *out = ctx;
      return RNS8_SUCCESS;
    }

    if (requested == RNS8_BACKEND_CPU_REFERENCE) {
      ctx->backend = RNS8_BACKEND_CPU_REFERENCE;
      ctx->device_id = -1;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      rns8::detail::fill_cpu_device_info(ctx->device_info);
      *out = ctx;
      return RNS8_SUCCESS;
    }

    if (requested == RNS8_BACKEND_HIP_DIRECT) {
      ctx->backend = RNS8_BACKEND_HIP_DIRECT;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::hip_direct_probe(ctx->device_id, ctx->device_info);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      *out = ctx;
      return RNS8_SUCCESS;
    }

    if (requested == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
      ctx->backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::vector_alu_probe(ctx->device_id, ctx->device_info);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      ctx->device_info.backend = RNS8_BACKEND_HIP_VECTOR_ALU_INT64;
      *out = ctx;
      return RNS8_SUCCESS;
#else
      delete ctx;
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }

    if (requested == RNS8_BACKEND_HIPBLASLT) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      ctx->backend = RNS8_BACKEND_HIPBLASLT;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::hipblaslt_create_context(
          ctx->device_id, ctx->device_info, &ctx->hipblaslt_handle, ctx->hipblaslt_library_version);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      *out = ctx;
      return RNS8_SUCCESS;
#else
      delete ctx;
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }

    if (requested == RNS8_BACKEND_CK) {
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      ctx->backend = RNS8_BACKEND_CK;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::ck_probe(ctx->device_id, ctx->device_info);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      *out = ctx;
      return RNS8_SUCCESS;
#else
      delete ctx;
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }

    if (requested == RNS8_BACKEND_WMMA) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      ctx->backend = RNS8_BACKEND_WMMA;
      ctx->device_id = device_id < 0 ? 0 : device_id;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      const rns8_status status = rns8::detail::wmma_probe(ctx->device_id, ctx->device_info);
      if (status != RNS8_SUCCESS) {
        delete ctx;
        return status;
      }
      *out = ctx;
      return RNS8_SUCCESS;
#else
      delete ctx;
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }

    if (requested == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      ctx->backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
      ctx->device_id = -1;
      ctx->device_info.struct_size = sizeof(ctx->device_info);
      ctx->device_info.abi_version = RNS8_ABI_VERSION;
      rns8::detail::fill_wrap64_device_info(ctx->device_info);
      *out = ctx;
      return RNS8_SUCCESS;
    }

    delete ctx;
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_destroy_context(rns8_context* ctx) {
  if (ctx && ctx->hipblaslt_handle) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
    const rns8_status status = rns8::detail::hipblaslt_destroy_context(ctx->device_id, ctx->hipblaslt_handle);
    ctx->hipblaslt_handle = nullptr;
    delete ctx;
    return status;
#else
    delete ctx;
    return RNS8_UNSUPPORTED_BACKEND;
#endif
  }
  delete ctx;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_device_info(rns8_context* ctx, rns8_device_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = ctx->device_info;
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_backend_capability_info(rns8_backend_kind backend, rns8_backend_capability_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out)) ||
        !known_backend_kind(backend)) {
      return RNS8_INVALID_ARGUMENT;
    }
    fill_backend_capability_info(backend, *out);
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_create_plan(rns8_context* ctx, const rns8_gemm_desc* desc, rns8_plan** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint32_t prefix =
        desc->max_prefix == 0 ? rns8::detail::default_prefix_for_semantics(desc->semantics) : desc->max_prefix;
    const rns8_status validation = rns8::detail::validate_gemm_desc(*desc, prefix);
    if (validation != RNS8_SUCCESS) {
      return validation;
    }

    const rns8_backend_kind requested = effective_backend(desc->requested_backend, ctx->backend);
    if (requested != ctx->backend) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (requested != RNS8_BACKEND_CPU_REFERENCE && requested != RNS8_BACKEND_HIP_DIRECT &&
        requested != RNS8_BACKEND_HIPBLASLT && requested != RNS8_BACKEND_CK &&
        requested != RNS8_BACKEND_WMMA && requested != RNS8_BACKEND_WRAP64_BYTE_LIMB &&
        requested != RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (!backend_supports_semantics(requested, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (requested == RNS8_BACKEND_HIPBLASLT && is_per_tile_bound_kind(desc->bound_kind)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    auto* plan = new (std::nothrow) rns8_plan();
    if (!plan) {
      return RNS8_INTERNAL_ERROR;
    }
    plan->desc = *desc;
    plan->desc.max_prefix = prefix;
    if (plan->desc.tile_m == 0) {
      plan->desc.tile_m = 128;
    }
    if (plan->desc.tile_n == 0) {
      plan->desc.tile_n = 128;
    }
    plan->prefix = prefix;
    plan->modulus_product = prefix == 0 ? 0 : rns8::detail::modulus_product(prefix);
    plan->backend = requested;
    const rns8_status schedule_status = configure_plan_schedule(*plan);
    if (schedule_status != RNS8_SUCCESS) {
      delete plan;
      return schedule_status;
    }
    configure_plan_backend_metadata(*plan, ctx);
    if (desc->requested_backend == RNS8_BACKEND_AUTO) {
      select_auto_backend_from_reviewed_cache(*ctx, *plan);
    }
    *out = plan;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_plan(rns8_plan* plan) {
  delete plan;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_plan_schedule_info(const rns8_plan* plan, rns8_plan_schedule_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->tile_m = plan->desc.tile_m;
    out->tile_n = plan->desc.tile_n;
    out->tile_rows = plan->schedule_tile_rows;
    out->tile_cols = plan->schedule_tile_cols;
    out->tile_count = plan->schedule_tile_count;
    out->min_required_prefix = plan->schedule_min_required_prefix;
    out->max_required_prefix = plan->schedule_max_required_prefix;
    out->min_selected_prefix = plan->schedule_min_selected_prefix;
    out->max_selected_prefix = plan->schedule_max_selected_prefix;
    out->prefix_group_count = plan->schedule_prefix_group_count;
    out->adaptive_prefix_active = plan->schedule_adaptive_prefix_active;
    out->adaptive_skip_active = plan->schedule_adaptive_skip_active;
    out->range_bit_length = plan->schedule_range_bit_length;
    out->flags = plan->schedule_flags;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_plan_tile_schedule(
    const rns8_plan* plan,
    rns8_plan_tile_schedule_entry* entries,
    uint64_t capacity,
    uint64_t* written) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !written) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }
    *written = plan->schedule_tile_count;
    if (!entries || capacity == 0) {
      return RNS8_SUCCESS;
    }
    if (capacity < plan->schedule_tile_count) {
      return RNS8_WORKSPACE_TOO_SMALL;
    }
    if (!plan->tile_schedule.empty()) {
      std::copy(plan->tile_schedule.begin(), plan->tile_schedule.end(), entries);
      return RNS8_SUCCESS;
    }
    for (uint64_t index = 0; index < plan->schedule_tile_count; ++index) {
      entries[index] = make_tile_schedule_entry(
          *plan,
          index,
          plan->schedule_min_required_prefix,
          plan->schedule_min_selected_prefix,
          0,
          plan->schedule_range_bit_length);
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_plan_backend_info(const rns8_plan* plan, rns8_plan_backend_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }

    rns8_backend_capability_info capability{};
    capability.struct_size = sizeof(capability);
    capability.abi_version = RNS8_ABI_VERSION;
    fill_backend_capability_info(plan->backend, capability);

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = plan->backend;
    out->is_accelerator = capability.is_accelerator;
    out->is_correctness_backend = capability.is_correctness_backend;
    out->is_matrix_engine_backend = capability.is_matrix_engine_backend;
    out->compiled_kernel_available = capability.compiled_kernel_available;
    out->exact_differential_validated = capability.exact_differential_validated;
    out->performance_validated = plan->backend_performance_validated;
    out->flags = capability.flags;
    out->workspace_required_bytes = plan->backend_workspace_required_bytes;
    set_text(out->selected_kernel, sizeof(out->selected_kernel), plan->backend_selected_kernel);
    set_text(out->accelerator_library, sizeof(out->accelerator_library), plan->backend_library);
    set_text(out->accelerator_version, sizeof(out->accelerator_version), plan->backend_library_version);
    set_text(out->capability_status, sizeof(out->capability_status), plan->backend_capability_status);
    set_text(out->epilogue_mode, sizeof(out->epilogue_mode), plan->backend_epilogue_mode);
    set_text(out->workspace_mode, sizeof(out->workspace_mode), plan->backend_workspace_mode);
    set_text(out->isa_evidence, sizeof(out->isa_evidence), plan->backend_isa_evidence);
    set_text(out->autotune_key, sizeof(out->autotune_key), plan->backend_autotune_key);
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_plan_packing_info(const rns8_plan* plan, rns8_plan_packing_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !out || !rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = plan->backend;
    out->semantics = plan->desc.semantics;
    out->uses_resident_matrix_inputs = 1;
    out->reusable_prepack_cache_available = 0;
    out->production_prepack_cache_available = 0;
    out->flags = 0;

    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
      out->uses_transient_pack_workspace = 1;
      out->uses_matrix_engine_pack_layout = 1;
      if (!hipblaslt_pack_workspace_breakdown(
              *plan,
              out->a_pack_workspace_bytes,
              out->b_pack_workspace_bytes,
              out->accumulator_workspace_bytes,
              out->library_workspace_bytes,
              out->total_transient_workspace_bytes)) {
        return RNS8_RANGE_ERROR;
      }
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "hipblaslt_a_transposed_centered_i8_mk16_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "hipblaslt_b_transposed_centered_i8_nk16_v1");
      set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "transient_per_dispatch_workspace");
      set_text(
          out->detail,
          sizeof(out->detail),
          "hipBLASLt packs A/B into transient aligned INT8 buffers and uses INT32 scratch; no reusable production prepack cache.");
      return RNS8_SUCCESS;
    }

    if (plan->backend == RNS8_BACKEND_CK) {
      out->uses_transient_pack_workspace = 1;
      out->uses_matrix_engine_pack_layout = 1;
      if (!ck_pack_workspace_breakdown(
              *plan,
              out->a_pack_workspace_bytes,
              out->b_pack_workspace_bytes,
              out->accumulator_workspace_bytes,
              out->total_transient_workspace_bytes)) {
        return RNS8_RANGE_ERROR;
      }
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "ck_a_canonical_rowmajor_i8_m64_kblock32768_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "ck_b_canonical_colmajor_i8_n64_kblock32768_v1");
      set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "transient_per_dispatch_workspace");
      set_text(
          out->detail,
          sizeof(out->detail),
          "CK packs A/B into transient canonical INT8 workspaces and uses temporary centered output storage; no reusable production prepack cache.");
      return RNS8_SUCCESS;
    }

    if (plan->backend == RNS8_BACKEND_WMMA) {
      out->uses_transient_pack_workspace = 1;
      out->uses_matrix_engine_pack_layout = 1;
      if (!wmma_pack_workspace_breakdown(
              *plan,
              out->a_pack_workspace_bytes,
              out->b_pack_workspace_bytes,
              out->total_transient_workspace_bytes)) {
        return RNS8_RANGE_ERROR;
      }
      set_text(out->a_layout_version, sizeof(out->a_layout_version), "rocwmma_a_rowmajor_i8_m16_kblock65536_v1");
      set_text(out->b_layout_version, sizeof(out->b_layout_version), "rocwmma_b_colmajor_i8_n16_kblock65536_v1");
      set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
      if (wmma_b_prepack_cache_supported(*plan)) {
        out->reusable_prepack_cache_available = 1;
        set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "reusable_b_prepack_cache");
        set_text(
            out->detail,
            sizeof(out->detail),
            "rocWMMA supports a reusable B prepack cache for this non-tiled RNS plan; A remains transient per dispatch and no production cache is reported.");
      } else {
        set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "transient_per_dispatch_workspace");
        set_text(
            out->detail,
            sizeof(out->detail),
            "rocWMMA packs A/B into transient INT8 matrix-engine workspaces; no reusable production prepack cache.");
      }
      return RNS8_SUCCESS;
    }

    if (plan->backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      set_text(out->a_layout_version, sizeof(out->a_layout_version), native_layout_version_for_semantics(plan->desc.semantics));
      set_text(out->b_layout_version, sizeof(out->b_layout_version), native_layout_version_for_semantics(plan->desc.semantics));
      set_text(
          out->output_layout_version,
          sizeof(out->output_layout_version),
          native_layout_version_for_semantics(plan->desc.semantics));
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "native_device_no_prepack_cache");
      set_text(
          out->detail,
          sizeof(out->detail),
          "Native vector-ALU backend consumes compact bounded i64/u64 device storage directly and exports native C without CRT reconstruction.");
      return RNS8_SUCCESS;
    }

    set_text(out->a_layout_version, sizeof(out->a_layout_version), persistent_layout_version_for_plan(*plan));
    set_text(out->b_layout_version, sizeof(out->b_layout_version), persistent_layout_version_for_plan(*plan));
    set_text(out->output_layout_version, sizeof(out->output_layout_version), persistent_layout_version_for_plan(*plan));
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "device_resident_no_prepack_cache");
      set_text(
          out->detail,
          sizeof(out->detail),
          "Direct HIP consumes persistent device-resident matrix storage directly; no transient matrix-engine pack workspace or reusable prepack cache.");
    } else if (plan->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "host_byte_limb_no_prepack_cache");
      set_text(
          out->detail,
          sizeof(out->detail),
          "Strict wrap64 byte-limb reference consumes persistent byte-limb storage directly; no matrix-engine prepack cache.");
    } else {
      set_text(out->prepack_cache_scope, sizeof(out->prepack_cache_scope), "host_resident_no_prepack_cache");
      set_text(
          out->detail,
          sizeof(out->detail),
          "CPU reference consumes persistent host-resident storage directly; no matrix-engine prepack cache.");
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_create_workspace(rns8_context* ctx, const rns8_plan* plan, rns8_workspace** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!plan_schedule_contract_matches(*plan)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!context_accepts_backend(*ctx, plan->backend)) {
      return RNS8_INVALID_ARGUMENT;
    }
    auto* workspace = new (std::nothrow) rns8_workspace();
    if (!workspace) {
      return RNS8_INTERNAL_ERROR;
    }
    workspace->semantics = plan->desc.semantics;
    workspace->bound_kind = plan->desc.bound_kind;
    workspace->m = plan->desc.m;
    workspace->n = plan->desc.n;
    workspace->k = plan->desc.k;
    workspace->bound = plan->desc.bound;
    workspace->finite_modulus = plan->desc.finite_modulus;
    workspace->tile_m = plan->desc.tile_m;
    workspace->tile_n = plan->desc.tile_n;
    workspace->prefix = plan->prefix;
    workspace->schedule_tile_rows = plan->schedule_tile_rows;
    workspace->schedule_tile_cols = plan->schedule_tile_cols;
    workspace->schedule_tile_count = plan->schedule_tile_count;
    workspace->schedule_min_required_prefix = plan->schedule_min_required_prefix;
    workspace->schedule_max_required_prefix = plan->schedule_max_required_prefix;
    workspace->schedule_min_selected_prefix = plan->schedule_min_selected_prefix;
    workspace->schedule_max_selected_prefix = plan->schedule_max_selected_prefix;
    workspace->schedule_prefix_group_count = plan->schedule_prefix_group_count;
    workspace->schedule_range_bit_length = plan->schedule_range_bit_length;
    workspace->schedule_adaptive_prefix_active = plan->schedule_adaptive_prefix_active;
    workspace->schedule_adaptive_skip_active = plan->schedule_adaptive_skip_active;
    workspace->schedule_flags = plan->schedule_flags;
    workspace->schedule_fingerprint = plan_workspace_fingerprint(*plan);
    workspace->backend_workspace_required_bytes = plan->backend_workspace_required_bytes;
    workspace->backend_selected_kernel = plan->backend_selected_kernel;
    workspace->backend_library = plan->backend_library;
    workspace->backend_library_version = plan->backend_library_version;
    workspace->backend_capability_status = plan->backend_capability_status;
    workspace->backend_epilogue_mode = plan->backend_epilogue_mode;
    workspace->backend_workspace_mode = plan->backend_workspace_mode;
    workspace->backend_isa_evidence = plan->backend_isa_evidence;
    workspace->backend_autotune_key = plan->backend_autotune_key;
    workspace->backend_performance_validated = plan->backend_performance_validated;
    workspace->backend = plan->backend;
    workspace->hip_device_id = hip_device_backend(plan->backend) ? ctx->device_id : -1;
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT && !plan->tile_schedule.empty()) {
      if (plan->tile_schedule.size() >
          std::numeric_limits<std::size_t>::max() / sizeof(rns8_plan_tile_schedule_entry)) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      const std::size_t schedule_bytes = plan->tile_schedule.size() * sizeof(rns8_plan_tile_schedule_entry);
      rns8_status status =
          rns8::detail::hip_direct_allocate(ctx->device_id, schedule_bytes, &workspace->hip_tile_schedule);
      if (status != RNS8_SUCCESS) {
        delete workspace;
        return status;
      }
      workspace->hip_tile_schedule_bytes = schedule_bytes;
      workspace->hip_tile_schedule_count = static_cast<uint64_t>(plan->tile_schedule.size());
      status = rns8::detail::hip_direct_copy_host_to_device(
          ctx->device_id, workspace->hip_tile_schedule, plan->tile_schedule.data(), schedule_bytes);
      if (status != RNS8_SUCCESS) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hip_tile_schedule);
        delete workspace;
        return status;
      }
    }
    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
      if (!ctx->hipblaslt_handle) {
        delete workspace;
        return RNS8_UNSUPPORTED_BACKEND;
      }
      std::size_t scratch_bytes = 0;
      std::size_t workspace_bytes = 0;
      if (!hipblaslt_scratch_bytes_for_plan(*plan, scratch_bytes) ||
          !hipblaslt_workspace_bytes_for_plan(*plan, workspace_bytes)) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      rns8_status status =
          rns8::detail::hip_direct_allocate(ctx->device_id, scratch_bytes, &workspace->hipblaslt_int32_scratch);
      if (status != RNS8_SUCCESS) {
        delete workspace;
        return status;
      }
      workspace->hipblaslt_int32_scratch_bytes = scratch_bytes;
      status = rns8::detail::hip_direct_allocate(ctx->device_id, workspace_bytes, &workspace->hipblaslt_workspace);
      if (status != RNS8_SUCCESS) {
        (void)rns8::detail::hip_direct_free(ctx->device_id, workspace->hipblaslt_int32_scratch);
        delete workspace;
        return status;
      }
      workspace->hipblaslt_workspace_bytes = workspace_bytes;
    }
    if (rns8::detail::accelerator_backend_kind(plan->backend)) {
      if (!rns8::detail::accelerator_backend_compiled(plan->backend)) {
        delete workspace;
        return RNS8_UNSUPPORTED_BACKEND;
      }
      if (plan->backend_workspace_required_bytes > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        delete workspace;
        return RNS8_RANGE_ERROR;
      }
      const std::size_t workspace_bytes = static_cast<std::size_t>(plan->backend_workspace_required_bytes);
      if (workspace_bytes != 0) {
        rns8_status status =
            rns8::detail::hip_direct_allocate(ctx->device_id, workspace_bytes, &workspace->accelerator_workspace);
        if (status != RNS8_SUCCESS) {
          delete workspace;
          return status;
        }
        workspace->accelerator_workspace_bytes = workspace_bytes;
      }
    }
    *out = workspace;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_workspace(rns8_workspace* workspace) {
  if (workspace) {
    rns8_status status = RNS8_SUCCESS;
    if (workspace->hip_tile_schedule) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->hip_tile_schedule);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->hip_tile_schedule = nullptr;
      workspace->hip_tile_schedule_bytes = 0;
      workspace->hip_tile_schedule_count = 0;
    }
    if (workspace->hipblaslt_int32_scratch) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->hipblaslt_int32_scratch);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->hipblaslt_int32_scratch = nullptr;
      workspace->hipblaslt_int32_scratch_bytes = 0;
    }
    if (workspace->hipblaslt_workspace) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->hipblaslt_workspace);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->hipblaslt_workspace = nullptr;
      workspace->hipblaslt_workspace_bytes = 0;
    }
    if (workspace->accelerator_auxiliary) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->accelerator_auxiliary);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->accelerator_auxiliary = nullptr;
      workspace->accelerator_auxiliary_bytes = 0;
    }
    if (workspace->accelerator_workspace) {
      const rns8_status free_status =
          rns8::detail::hip_direct_free(workspace->hip_device_id, workspace->accelerator_workspace);
      if (status == RNS8_SUCCESS) {
        status = free_status;
      }
      workspace->accelerator_workspace = nullptr;
      workspace->accelerator_workspace_bytes = 0;
    }
    delete workspace;
    return status;
  }
  delete workspace;
  return RNS8_SUCCESS;
}

rns8_status rns8_create_matrix(rns8_context* ctx, const rns8_matrix_desc* desc, rns8_matrix** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!rns8::detail::valid_abi(desc->struct_size, desc->abi_version, sizeof(*desc))) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint32_t prefix =
        desc->max_prefix == 0 ? rns8::detail::default_prefix_for_semantics(desc->semantics) : desc->max_prefix;
    const rns8_status validation = rns8::detail::validate_matrix_desc(*desc, prefix);
    if (validation != RNS8_SUCCESS) {
      return validation;
    }
    if (!backend_supports_semantics(ctx->backend, desc->semantics)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    auto* matrix = new (std::nothrow) rns8_matrix();
    if (!matrix) {
      return RNS8_INTERNAL_ERROR;
    }
    matrix->desc = *desc;
    matrix->backend = ctx->backend;
    matrix->desc.max_prefix = prefix;
    if (matrix->desc.logical_ld == 0) {
      matrix->desc.logical_ld = matrix->desc.cols;
    }
    if (matrix->desc.tile_m == 0) {
      matrix->desc.tile_m = 128;
    }
    if (matrix->desc.tile_n == 0) {
      matrix->desc.tile_n = 128;
    }
    matrix->prefix = prefix;
    if (native_vector_backend(ctx->backend)) {
      if (matrix->desc.semantics == RNS8_BOUNDED_I64) {
        matrix->native_i64.clear();
      } else if (matrix->desc.semantics == RNS8_BOUNDED_U64) {
        matrix->native_u64.clear();
      } else {
        delete matrix;
        return RNS8_UNSUPPORTED_BACKEND;
      }
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
      matrix->host_native_current = false;
      matrix->device_native_current = false;
      const rns8_status status = ensure_hip_native_storage(*ctx, *matrix);
      if (status != RNS8_SUCCESS) {
        (void)free_hip_matrix_storage(*matrix);
        delete matrix;
        return status;
      }
    } else if (matrix->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      std::size_t bytes = 0;
      if (!wrap_byte_limb_bytes(desc->rows, desc->cols, bytes)) {
        delete matrix;
        return RNS8_RANGE_ERROR;
      }
      matrix->byte_limbs.assign(bytes, 0);
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = true;
      matrix->device_byte_limbs_current = false;
    } else if (uses_finite_storage(matrix->desc.semantics)) {
      std::size_t cells = 0;
      if (!matrix_cell_count(desc->rows, desc->cols, cells)) {
        delete matrix;
        return RNS8_RANGE_ERROR;
      }
      matrix->residues.assign(cells, 0);
      matrix->finite_modulus = 0;
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
    } else {
      std::size_t elements = 0;
      if (!rns_residue_count(desc->rows, desc->cols, prefix, elements)) {
        delete matrix;
        return RNS8_RANGE_ERROR;
      }
      matrix->residues.assign(elements, 0);
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
    }
    if (hip_resident_rns_backend(ctx->backend)) {
      const rns8_status status = allocate_hip_matrix_storage(*ctx, *matrix);
      if (status != RNS8_SUCCESS) {
        delete matrix;
        return status;
      }
      if (ctx->auto_backend_selection &&
          (matrix->desc.semantics == RNS8_BOUNDED_I64 || matrix->desc.semantics == RNS8_BOUNDED_U64)) {
        const rns8_status native_status = ensure_hip_native_storage(*ctx, *matrix);
        if (native_status != RNS8_SUCCESS) {
          (void)free_hip_matrix_storage(*matrix);
          delete matrix;
          return native_status;
        }
      }
    }
    *out = matrix;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_matrix(rns8_matrix* matrix) {
  if (matrix) {
    const rns8_status status = free_hip_matrix_storage(*matrix);
    delete matrix;
    return status;
  }
  delete matrix;
  return RNS8_SUCCESS;
}

rns8_status rns8_get_matrix_storage_info(const rns8_matrix* matrix, rns8_matrix_storage_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!matrix || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = matrix->backend;
    out->semantics = matrix->desc.semantics;
    out->logical_layout = matrix->desc.logical_layout;
    out->bound_kind = matrix->desc.bound_kind;
    out->rows = matrix->desc.rows;
    out->cols = matrix->desc.cols;
    out->logical_ld = matrix->desc.logical_ld;
    out->max_prefix = matrix->prefix;
    out->finite_modulus = matrix->finite_modulus;
    out->source_version = matrix->source_version;
    out->host_residues_current = matrix->host_residues_current ? 1u : 0u;
    out->device_residues_current = matrix->device_residues_current ? 1u : 0u;
    out->host_byte_limbs_current = matrix->host_byte_limbs_current ? 1u : 0u;
    out->device_byte_limbs_current = matrix->device_byte_limbs_current ? 1u : 0u;
    out->host_native_current = matrix->host_native_current ? 1u : 0u;
    out->device_native_current = matrix->device_native_current ? 1u : 0u;
    out->uses_residue_storage = (!matrix->residues.empty() || matrix->hip_residues != nullptr) ? 1u : 0u;
    out->uses_byte_limb_storage = (!matrix->byte_limbs.empty() || matrix->hip_byte_limbs != nullptr) ? 1u : 0u;
    out->uses_native_storage = matrix_has_native_storage(*matrix) ? 1u : 0u;
    out->hip_device_id = static_cast<int32_t>(matrix->hip_device_id);
    out->flags = 0;
    out->host_residue_bytes = static_cast<uint64_t>(matrix->residues.size() * sizeof(int8_t));
    out->device_residue_bytes = static_cast<uint64_t>(matrix->hip_residue_bytes);
    out->host_byte_limb_bytes = static_cast<uint64_t>(matrix->byte_limbs.size() * sizeof(uint8_t));
    out->device_byte_limb_bytes = static_cast<uint64_t>(matrix->hip_byte_limb_bytes);
    out->host_native_bytes = static_cast<uint64_t>(
        matrix->native_i64.size() * sizeof(int64_t) + matrix->native_u64.size() * sizeof(uint64_t));
    out->device_native_bytes =
        static_cast<uint64_t>(matrix->hip_native_i64_bytes + matrix->hip_native_u64_bytes);
    set_text(
        out->layout_version,
        sizeof(out->layout_version),
        native_vector_backend(matrix->backend) ? native_layout_version_for_semantics(matrix->desc.semantics)
                                               : persistent_layout_version_for_semantics(matrix->desc.semantics));
    set_text(out->storage_scope, sizeof(out->storage_scope), storage_scope_for_matrix(*matrix));
    set_text(out->detail, sizeof(out->detail), storage_detail_for_matrix(*matrix));
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_prepack_cache_key_info(
    const rns8_plan* plan,
    const rns8_matrix* matrix,
    rns8_operand_role operand_role,
    rns8_prepack_cache_key_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !matrix || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!prepack_operand_matrix_compatible(*plan, *matrix, operand_role)) {
      return RNS8_INVALID_ARGUMENT;
    }

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    const std::string matrix_layout_version = persistent_layout_version_for_semantics(matrix->desc.semantics);
    const std::string operand_layout_version = prepack_operand_layout_version_for_plan(*plan, operand_role);
    const uint64_t plan_fingerprint = plan_workspace_fingerprint(*plan);
    const uint64_t cache_key_hash =
        prepack_cache_key_hash(*plan, *matrix, operand_role, matrix_layout_version, operand_layout_version);
    const std::string cache_key = build_prepack_cache_key(
        *plan,
        *matrix,
        operand_role,
        plan_fingerprint,
        cache_key_hash,
        matrix_layout_version,
        operand_layout_version);

    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = plan->backend;
    out->semantics = plan->desc.semantics;
    out->operand_role = operand_role;
    out->cache_key_valid = 1;
    out->reusable_prepack_cache_available =
        (operand_role == RNS8_OPERAND_B && wmma_b_prepack_cache_supported(*plan)) ? 1u : 0u;
    out->production_prepack_cache_available = 0;
    out->flags = 0;
    out->hip_device_id = matrix->hip_device_id;
    out->reserved0 = 0;
    out->matrix_rows = matrix->desc.rows;
    out->matrix_cols = matrix->desc.cols;
    out->max_prefix = matrix->prefix;
    out->finite_modulus = matrix->finite_modulus;
    out->source_version = matrix->source_version;
    out->plan_fingerprint = plan_fingerprint;
    out->cache_key_hash = cache_key_hash;
    set_text(out->matrix_layout_version, sizeof(out->matrix_layout_version), matrix_layout_version);
    set_text(out->operand_layout_version, sizeof(out->operand_layout_version), operand_layout_version);
    set_text(out->cache_scope, sizeof(out->cache_scope), "validated_key_no_production_cache");
    set_text(out->cache_key, sizeof(out->cache_key), cache_key);
    set_text(
        out->detail,
        sizeof(out->detail),
        "Plan and operand matrix are compatible for future prepack cache keying; no production cache is available.");
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_create_prepack_cache(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* matrix,
    rns8_operand_role operand_role,
    rns8_prepack_cache** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !matrix || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!valid_prepack_operand_role(operand_role)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (operand_role != RNS8_OPERAND_B || !wmma_b_prepack_cache_supported(*plan)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (!context_accepts_backend(*ctx, plan->backend) || !prepack_operand_matrix_compatible(*plan, *matrix, operand_role) ||
        matrix->hip_device_id != ctx->device_id || !matrix->device_residues_current) {
      return RNS8_INVALID_ARGUMENT;
    }

    std::size_t b_pack_bytes = 0;
    std::size_t total_cache_bytes = 0;
    if (!wmma_b_prepack_bytes_for_plan(*plan, b_pack_bytes, total_cache_bytes)) {
      return RNS8_RANGE_ERROR;
    }

    auto* cache = new (std::nothrow) rns8_prepack_cache();
    if (!cache) {
      return RNS8_INTERNAL_ERROR;
    }
    cache->backend = plan->backend;
    cache->semantics = plan->desc.semantics;
    cache->operand_role = operand_role;
    cache->rows = matrix->desc.rows;
    cache->cols = matrix->desc.cols;
    cache->k = plan->desc.k;
    cache->prefix = plan->prefix;
    cache->finite_modulus = matrix->finite_modulus;
    cache->source_version = matrix->source_version;
    cache->plan_fingerprint = plan_workspace_fingerprint(*plan);
    cache->matrix_layout_version = persistent_layout_version_for_semantics(matrix->desc.semantics);
    cache->operand_layout_version = prepack_operand_layout_version_for_plan(*plan, operand_role);
    cache->cache_key_hash =
        prepack_cache_key_hash(*plan, *matrix, operand_role, cache->matrix_layout_version, cache->operand_layout_version);
    cache->cache_key = build_prepack_cache_key(
        *plan,
        *matrix,
        operand_role,
        cache->plan_fingerprint,
        cache->cache_key_hash,
        cache->matrix_layout_version,
        cache->operand_layout_version);
    cache->hip_device_id = ctx->device_id;
    cache->device_bytes = total_cache_bytes;
    cache->operand_pack_bytes = b_pack_bytes;

    rns8_status status = rns8::detail::hip_direct_allocate(ctx->device_id, total_cache_bytes, &cache->device_data);
    if (status != RNS8_SUCCESS) {
      delete cache;
      return status;
    }
    status = rns8::detail::wmma_prepack_b_rns_device(
        ctx->device_id,
        matrix->hip_residues,
        cache->device_data,
        cache->device_bytes,
        plan->desc.k,
        plan->desc.n,
        matrix->desc.cols,
        plan->prefix);
    if (status != RNS8_SUCCESS) {
      (void)rns8::detail::hip_direct_free(ctx->device_id, cache->device_data);
      cache->device_data = nullptr;
      delete cache;
      return status;
    }
    *out = cache;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_get_prepack_cache_info(const rns8_prepack_cache* cache, rns8_prepack_cache_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!cache || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::valid_abi(out->struct_size, out->abi_version, sizeof(*out))) {
      return RNS8_INVALID_ARGUMENT;
    }

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = cache->backend;
    out->semantics = cache->semantics;
    out->operand_role = cache->operand_role;
    out->cache_key_valid = cache->cache_key_hash != 0 && !cache->cache_key.empty() ? 1u : 0u;
    out->reusable_prepack_cache_available = 1;
    out->production_prepack_cache_available = 0;
    out->flags = 0;
    out->hip_device_id = cache->hip_device_id;
    out->reserved0 = 0;
    out->matrix_rows = cache->rows;
    out->matrix_cols = cache->cols;
    out->k = cache->k;
    out->max_prefix = cache->prefix;
    out->finite_modulus = cache->finite_modulus;
    out->source_version = cache->source_version;
    out->plan_fingerprint = cache->plan_fingerprint;
    out->cache_key_hash = cache->cache_key_hash;
    out->device_bytes = static_cast<uint64_t>(cache->device_bytes);
    out->operand_pack_bytes = static_cast<uint64_t>(cache->operand_pack_bytes);
    set_text(out->matrix_layout_version, sizeof(out->matrix_layout_version), cache->matrix_layout_version);
    set_text(out->operand_layout_version, sizeof(out->operand_layout_version), cache->operand_layout_version);
    set_text(out->cache_scope, sizeof(out->cache_scope), "runtime_reusable_b_prepack_cache");
    set_text(out->cache_key, sizeof(out->cache_key), cache->cache_key);
    set_text(
        out->detail,
        sizeof(out->detail),
        "Created reusable accelerator prepack cache; no production cache availability is reported.");
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_prepack_cache(rns8_prepack_cache* cache) {
  if (!cache) {
    return RNS8_SUCCESS;
  }
  rns8_status status = RNS8_SUCCESS;
  if (cache->device_data) {
    status = rns8::detail::hip_direct_free(cache->hip_device_id, cache->device_data);
    cache->device_data = nullptr;
    cache->device_bytes = 0;
  }
  delete cache;
  return status;
}

rns8_status rns8_pack_i64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const int64_t* src,
    int64_t ld,
    uint64_t source_version) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || !src || !valid_matrix_access(matrix->desc.rows, matrix->desc.cols, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (matrix->desc.semantics != RNS8_BOUNDED_I64 && matrix->desc.semantics != RNS8_EXACT_WIDE_SIGNED) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (native_vector_backend(ctx->backend)) {
      if (matrix->desc.semantics != RNS8_BOUNDED_I64 ||
          !bounded_native_storage_matches(*matrix, RNS8_BOUNDED_I64, matrix->desc.rows, matrix->desc.cols)) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status status = upload_native_i64(*ctx, *matrix, src, ld);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
      matrix->source_version = source_version;
      return RNS8_SUCCESS;
    }
    if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols, matrix->prefix)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (hip_resident_rns_backend(ctx->backend) && matrix->hip_device_id != ctx->device_id) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (hip_resident_rns_backend(ctx->backend)) {
      const rns8_status status = rns8::detail::hip_direct_pack_i64_device(
          ctx->device_id,
          src,
          &matrix->hip_upload_buffer,
          &matrix->hip_upload_bytes,
          matrix->hip_residues,
          matrix->desc.rows,
          matrix->desc.cols,
          ld,
          matrix->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->device_residues_current = true;
      matrix->host_residues_current = false;
    } else {
      rns8::detail::pack_i64_matrix(*matrix, src, ld);
      matrix->host_residues_current = true;
      matrix->device_residues_current = false;
    }
    if (should_populate_native_on_pack(*ctx, *matrix)) {
      const rns8_status native_status = upload_native_i64(*ctx, *matrix, src, ld);
      if (native_status != RNS8_SUCCESS) {
        return native_status;
      }
    } else {
      matrix->host_native_current = false;
      matrix->device_native_current = false;
    }
    matrix->source_version = source_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_pack_u64(
    rns8_context* ctx,
    rns8_matrix* matrix,
    const uint64_t* src,
    int64_t ld,
    uint64_t source_version) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || !src || !valid_matrix_access(matrix->desc.rows, matrix->desc.cols, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (matrix->desc.semantics != RNS8_BOUNDED_U64 && matrix->desc.semantics != RNS8_EXACT_WIDE_UNSIGNED &&
        matrix->desc.semantics != RNS8_WRAP_U64_MOD_2_64) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (native_vector_backend(ctx->backend)) {
      if (matrix->desc.semantics != RNS8_BOUNDED_U64 ||
          !bounded_native_storage_matches(*matrix, RNS8_BOUNDED_U64, matrix->desc.rows, matrix->desc.cols)) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status status = upload_native_u64(*ctx, *matrix, src, ld);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->host_residues_current = false;
      matrix->device_residues_current = false;
      matrix->host_byte_limbs_current = false;
      matrix->device_byte_limbs_current = false;
      matrix->source_version = source_version;
      return RNS8_SUCCESS;
    }
    if (matrix->desc.semantics == RNS8_WRAP_U64_MOD_2_64) {
      if (!wrap_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols)) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (ctx->backend == RNS8_BACKEND_HIP_DIRECT && matrix->hip_device_id != ctx->device_id) {
        return RNS8_INVALID_ARGUMENT;
      }
      if (ctx->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
        rns8::detail::pack_wrap_u64_matrix(*matrix, src, ld);
        matrix->host_residues_current = false;
        matrix->device_residues_current = false;
        matrix->host_byte_limbs_current = true;
        matrix->device_byte_limbs_current = false;
      } else if (ctx->backend == RNS8_BACKEND_HIP_DIRECT) {
        const rns8_status status = rns8::detail::wrap64_hip_pack_u64_device(
            ctx->device_id,
            src,
            &matrix->hip_upload_buffer,
            &matrix->hip_upload_bytes,
            matrix->hip_byte_limbs,
            matrix->desc.rows,
            matrix->desc.cols,
            ld);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        matrix->host_residues_current = false;
        matrix->device_residues_current = false;
        matrix->host_byte_limbs_current = false;
        matrix->device_byte_limbs_current = true;
      } else {
        return RNS8_UNSUPPORTED_BACKEND;
      }
    } else if (!rns_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols, matrix->prefix)) {
      return RNS8_INVALID_ARGUMENT;
    } else if (hip_resident_rns_backend(ctx->backend)) {
      if (matrix->hip_device_id != ctx->device_id) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status status = rns8::detail::hip_direct_pack_u64_device(
          ctx->device_id,
          src,
          &matrix->hip_upload_buffer,
          &matrix->hip_upload_bytes,
          matrix->hip_residues,
          matrix->desc.rows,
          matrix->desc.cols,
          ld,
          matrix->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->device_residues_current = true;
      matrix->host_residues_current = false;
    } else {
      rns8::detail::pack_u64_matrix(*matrix, src, ld);
      matrix->host_residues_current = true;
      matrix->device_residues_current = false;
    }
    if (should_populate_native_on_pack(*ctx, *matrix)) {
      const rns8_status native_status = upload_native_u64(*ctx, *matrix, src, ld);
      if (native_status != RNS8_SUCCESS) {
        return native_status;
      }
    } else if (matrix->desc.semantics != RNS8_WRAP_U64_MOD_2_64) {
      matrix->host_native_current = false;
      matrix->device_native_current = false;
    }
    matrix->source_version = source_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_pack_finite_u8(
    rns8_context* ctx,
    rns8_matrix* matrix,
    uint16_t modulus,
    const uint8_t* src,
    int64_t ld,
    uint64_t source_version) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !matrix || !src || !valid_matrix_access(matrix->desc.rows, matrix->desc.cols, ld) ||
        !rns8::detail::valid_finite_modulus_for_semantics(matrix->desc.semantics, modulus)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != matrix->backend) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!uses_finite_storage(matrix->desc.semantics) ||
        !finite_matrix_storage_matches(*matrix, ctx->backend, matrix->desc.rows, matrix->desc.cols)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (hip_resident_rns_backend(ctx->backend)) {
      if (matrix->hip_device_id != ctx->device_id) {
        return RNS8_INVALID_ARGUMENT;
      }
      const rns8_status status = rns8::detail::hip_direct_pack_finite_u8_device(
          ctx->device_id,
          src,
          &matrix->hip_upload_buffer,
          &matrix->hip_upload_bytes,
          matrix->hip_residues,
          matrix->desc.rows,
          matrix->desc.cols,
          ld,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      matrix->host_residues_current = false;
      matrix->device_residues_current = true;
    } else if (ctx->backend == RNS8_BACKEND_CPU_REFERENCE) {
      rns8::detail::pack_finite_u8_matrix(*matrix, src, ld, modulus);
      matrix->host_residues_current = true;
      matrix->device_residues_current = false;
    } else {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    matrix->host_byte_limbs_current = false;
    matrix->device_byte_limbs_current = false;
    matrix->finite_modulus = modulus;
    matrix->source_version = source_version;
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_gemm_rns(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    const rns8_status operand_status = validate_rns_gemm_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_VECTOR_ALU_INT64) {
      auto* mutable_c = C;
      rns8_status status = rns8::detail::hip_direct_ensure_upload_buffer(
          ctx->device_id, sizeof(uint32_t), &mutable_c->hip_status_buffer, &mutable_c->hip_status_bytes);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      status = run_timed_api_status("vector_alu_status_memset", [&]() {
        return rns8::detail::hip_direct_zero(ctx->device_id, mutable_c->hip_status_buffer, sizeof(uint32_t));
      });
      if (status != RNS8_SUCCESS) {
        return status;
      }
      if (plan->desc.semantics == RNS8_BOUNDED_I64) {
        status = rns8::detail::vector_alu_gemm_i64_device(
            ctx->device_id,
            A->hip_native_i64,
            B->hip_native_i64,
            mutable_c->hip_native_i64,
            mutable_c->hip_status_buffer,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k);
      } else {
        status = rns8::detail::vector_alu_gemm_u64_device(
            ctx->device_id,
            A->hip_native_u64,
            B->hip_native_u64,
            mutable_c->hip_native_u64,
            mutable_c->hip_status_buffer,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k);
      }
      if (status != RNS8_SUCCESS) {
        return status;
      }
      uint32_t host_status = 0;
      status = run_timed_api_status("vector_alu_status_d2h", [&]() {
        return rns8::detail::hip_direct_copy_device_to_host(
            ctx->device_id, &host_status, mutable_c->hip_status_buffer, sizeof(host_status));
      });
      if (status != RNS8_SUCCESS) {
        return status;
      }
      if (host_status != 0) {
        return RNS8_RANGE_ERROR;
      }
      mutable_c->host_residues_current = false;
      mutable_c->device_residues_current = false;
      mutable_c->host_byte_limbs_current = false;
      mutable_c->device_byte_limbs_current = false;
      mutable_c->host_native_current = false;
      mutable_c->device_native_current = true;
      mutable_c->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE) {
      const rns8_status status = rns8::detail::cpu_gemm_rns(*plan, *A, *B, *C);
      if (status == RNS8_SUCCESS) {
        C->host_residues_current = true;
        C->device_residues_current = false;
        C->host_byte_limbs_current = false;
        C->device_byte_limbs_current = false;
        clear_native_current(*C);
        if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
          C->source_version = gemm_output_source_version(*A, *B);
        }
      }
      return status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      rns8_status status = RNS8_SUCCESS;
      if (!plan->tile_schedule.empty()) {
        status = rns8::detail::hip_direct_gemm_rns_tiled_device_schedule(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->tile_schedule.data(),
            workspace->hip_tile_schedule,
            static_cast<uint64_t>(plan->tile_schedule.size()));
      } else {
        status = rns8::detail::hip_direct_gemm_rns_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->prefix);
      }
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      clear_native_current(*C);
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
    }
    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      if (!ctx->hipblaslt_handle || plan->schedule_adaptive_prefix_active || plan->schedule_adaptive_skip_active ||
          !plan->tile_schedule.empty()) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      const rns8_status status = rns8::detail::hipblaslt_gemm_rns_device(
          ctx->device_id,
          ctx->hipblaslt_handle,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          workspace->hipblaslt_int32_scratch,
          workspace->hipblaslt_int32_scratch_bytes,
          workspace->hipblaslt_workspace,
          workspace->hipblaslt_workspace_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          plan->prefix);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      clear_native_current(*C);
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_CK) {
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      rns8_status status = RNS8_SUCCESS;
      if (!plan->tile_schedule.empty()) {
        status = rns8::detail::ck_gemm_rns_tiled_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            workspace->accelerator_workspace,
            workspace->accelerator_workspace_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->tile_schedule.data(),
            static_cast<uint64_t>(plan->tile_schedule.size()));
      } else {
        status = rns8::detail::ck_gemm_rns_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            workspace->accelerator_workspace,
            workspace->accelerator_workspace_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->prefix);
      }
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      clear_native_current(*C);
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_WMMA) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      rns8_status status = RNS8_SUCCESS;
      if (!plan->tile_schedule.empty()) {
        status = rns8::detail::wmma_gemm_rns_tiled_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            workspace->accelerator_workspace,
            workspace->accelerator_workspace_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->tile_schedule.data(),
            static_cast<uint64_t>(plan->tile_schedule.size()));
      } else {
        status = rns8::detail::wmma_gemm_rns_device(
            ctx->device_id,
            A->hip_residues,
            B->hip_residues,
            C->hip_residues,
            workspace->accelerator_workspace,
            workspace->accelerator_workspace_bytes,
            plan->desc.m,
            plan->desc.n,
            plan->desc.k,
            A->desc.cols,
            B->desc.cols,
            C->desc.cols,
            plan->prefix);
      }
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->device_residues_current = true;
      C->host_residues_current = false;
      clear_native_current(*C);
      if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_gemm_rns_prepacked_b(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_prepack_cache* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    const rns8_status operand_status = validate_rns_gemm_prepacked_b_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend != RNS8_BACKEND_WMMA) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    const rns8_status status = rns8::detail::wmma_gemm_rns_prepacked_b_device(
        ctx->device_id,
        A->hip_residues,
        B->device_data,
        C->hip_residues,
        workspace->accelerator_workspace,
        workspace->accelerator_workspace_bytes,
        plan->desc.m,
        plan->desc.n,
        plan->desc.k,
        A->desc.cols,
        C->desc.cols,
        plan->prefix);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    C->device_residues_current = true;
    C->host_residues_current = false;
    C->host_byte_limbs_current = false;
    C->device_byte_limbs_current = false;
    clear_native_current(*C);
    if (plan->desc.semantics == RNS8_BOUNDED_I64 || plan->desc.semantics == RNS8_BOUNDED_U64) {
      C->source_version = gemm_output_source_version_values(A->source_version, B->source_version);
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_gemm_finite_u8(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    const rns8_status operand_status = validate_finite_gemm_operands(*ctx, *plan, modulus, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE) {
      const rns8_status status = rns8::detail::cpu_gemm_finite_u8(*plan, modulus, *A, *B, *C);
      if (status == RNS8_SUCCESS) {
        C->host_residues_current = true;
        C->device_residues_current = false;
        C->host_byte_limbs_current = false;
        C->device_byte_limbs_current = false;
        clear_native_current(*C);
        C->finite_modulus = modulus;
        C->source_version = gemm_output_source_version(*A, *B);
      }
      return status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = rns8::detail::hip_direct_gemm_finite_u8_resident_device(
          ctx->device_id,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = true;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = false;
      clear_native_current(*C);
      C->finite_modulus = modulus;
      C->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
    }
    if (plan->backend == RNS8_BACKEND_HIPBLASLT) {
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
      if (!ctx->hipblaslt_handle) {
        return RNS8_UNSUPPORTED_BACKEND;
      }
      const rns8_status status = rns8::detail::hipblaslt_gemm_finite_u8_device(
          ctx->device_id,
          ctx->hipblaslt_handle,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          workspace->hipblaslt_int32_scratch,
          workspace->hipblaslt_int32_scratch_bytes,
          workspace->hipblaslt_workspace,
          workspace->hipblaslt_workspace_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = true;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = false;
      clear_native_current(*C);
      C->finite_modulus = modulus;
      C->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_CK) {
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      const rns8_status status = rns8::detail::ck_gemm_finite_u8_device(
          ctx->device_id,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          workspace->accelerator_workspace,
          workspace->accelerator_workspace_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = true;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = false;
      clear_native_current(*C);
      C->finite_modulus = modulus;
      C->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    if (plan->backend == RNS8_BACKEND_WMMA) {
#if defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA
      const rns8_status status = rns8::detail::wmma_gemm_finite_u8_device(
          ctx->device_id,
          A->hip_residues,
          B->hip_residues,
          C->hip_residues,
          workspace->accelerator_workspace,
          workspace->accelerator_workspace_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k,
          A->desc.cols,
          B->desc.cols,
          C->desc.cols,
          modulus);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = true;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = false;
      clear_native_current(*C);
      C->finite_modulus = modulus;
      C->source_version = gemm_output_source_version(*A, *B);
      return RNS8_SUCCESS;
#else
      return RNS8_UNSUPPORTED_BACKEND;
#endif
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_gemm_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    rns8_workspace* workspace) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !A || !B || !C || !workspace) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status workspace_status = validate_plan_context_workspace(*ctx, *plan, *workspace);
    if (workspace_status != RNS8_SUCCESS) {
      return workspace_status;
    }
    const rns8_status operand_status = validate_wrap_gemm_operands(*ctx, *plan, *A, *B, *C);
    if (operand_status != RNS8_SUCCESS) {
      return operand_status;
    }
    if (plan->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      const rns8_status status = rns8::detail::cpu_gemm_wrap_u64(*plan, *A, *B, *C);
      if (status == RNS8_SUCCESS) {
        C->host_residues_current = false;
        C->device_residues_current = false;
        C->host_byte_limbs_current = true;
        C->device_byte_limbs_current = false;
        clear_native_current(*C);
      }
      return status;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      const rns8_status status = rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
          ctx->device_id,
          A->hip_byte_limbs,
          B->hip_byte_limbs,
          C->hip_byte_limbs,
          plan->desc.m,
          plan->desc.n,
          plan->desc.k);
      if (status != RNS8_SUCCESS) {
        return status;
      }
      C->host_residues_current = false;
      C->device_residues_current = false;
      C->host_byte_limbs_current = false;
      C->device_byte_limbs_current = true;
      clear_native_current(*C);
      return RNS8_SUCCESS;
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_export_i64(rns8_context* ctx, const rns8_plan* plan, const rns8_matrix* C, int64_t* dst, int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_matrix_access(plan->desc.m, plan->desc.n, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_BOUNDED_I64, plan->desc.bound_kind, plan->prefix);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (native_vector_backend(plan->backend)) {
      return export_native_i64(*ctx, *plan, *C, dst, ld);
    }
    if (hip_resident_rns_backend(plan->backend)) {
      if (!plan->tile_schedule.empty()) {
        if (plan->tile_bounds.size() != plan->tile_schedule.size()) {
          return RNS8_INTERNAL_ERROR;
        }
        auto* mutable_c = const_cast<rns8_matrix*>(C);
        const rns8_status metadata_status = ensure_hip_export_tile_metadata(*ctx, *plan, *mutable_c);
        if (metadata_status != RNS8_SUCCESS) {
          return metadata_status;
        }
        return rns8::detail::hip_direct_export_i64_tiled_device(
            ctx->device_id,
            C->hip_residues,
            &mutable_c->hip_export_buffer,
            &mutable_c->hip_export_bytes,
            &mutable_c->hip_status_buffer,
            &mutable_c->hip_status_bytes,
            plan->desc.m,
            plan->desc.n,
            mutable_c->hip_export_tile_schedule,
            mutable_c->hip_export_tile_bounds,
            mutable_c->hip_export_tile_schedule_count,
            mutable_c->hip_export_tile_max_elements,
            dst,
            ld);
      }
      return rns8::detail::hip_direct_export_i64_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          plan->desc.bound,
          dst,
          ld);
    }
    std::vector<int64_t> staged(static_cast<std::size_t>(plan->desc.m) * static_cast<std::size_t>(plan->desc.n), 0);
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        int64_t value = 0;
        const uint32_t prefix = selected_prefix_for_cell(*plan, row, col);
        const uint64_t bound = bound_for_cell(*plan, row, col);
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, prefix);
        const rns8_status status = rns8::detail::reconstruct_signed(residues, prefix, bound, value);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        staged[static_cast<std::size_t>(row * plan->desc.n + col)] = value;
      }
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        dst[row * ld + col] = staged[static_cast<std::size_t>(row * plan->desc.n + col)];
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_matrix_access(plan->desc.m, plan->desc.n, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_BOUNDED_U64, plan->desc.bound_kind, plan->prefix);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (native_vector_backend(plan->backend)) {
      return export_native_u64(*ctx, *plan, *C, dst, ld);
    }
    if (hip_resident_rns_backend(plan->backend)) {
      if (!plan->tile_schedule.empty()) {
        if (plan->tile_bounds.size() != plan->tile_schedule.size()) {
          return RNS8_INTERNAL_ERROR;
        }
        auto* mutable_c = const_cast<rns8_matrix*>(C);
        const rns8_status metadata_status = ensure_hip_export_tile_metadata(*ctx, *plan, *mutable_c);
        if (metadata_status != RNS8_SUCCESS) {
          return metadata_status;
        }
        return rns8::detail::hip_direct_export_u64_tiled_device(
            ctx->device_id,
            C->hip_residues,
            &mutable_c->hip_export_buffer,
            &mutable_c->hip_export_bytes,
            &mutable_c->hip_status_buffer,
            &mutable_c->hip_status_bytes,
            plan->desc.m,
            plan->desc.n,
            mutable_c->hip_export_tile_schedule,
            mutable_c->hip_export_tile_bounds,
            mutable_c->hip_export_tile_schedule_count,
            mutable_c->hip_export_tile_max_elements,
            dst,
            ld);
      }
      return rns8::detail::hip_direct_export_u64_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          plan->desc.bound,
          dst,
          ld);
    }
    std::vector<uint64_t> staged(static_cast<std::size_t>(plan->desc.m) * static_cast<std::size_t>(plan->desc.n), 0);
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        uint64_t value = 0;
        const uint32_t prefix = selected_prefix_for_cell(*plan, row, col);
        const uint64_t bound = bound_for_cell(*plan, row, col);
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, prefix);
        const rns8_status status = rns8::detail::reconstruct_unsigned(residues, prefix, bound, value);
        if (status != RNS8_SUCCESS) {
          return status;
        }
        staged[static_cast<std::size_t>(row * plan->desc.n + col)] = value;
      }
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        dst[row * ld + col] = staged[static_cast<std::size_t>(row * plan->desc.n + col)];
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_wrap_u64(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_matrix_access(plan->desc.m, plan->desc.n, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE, 0);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (!wrap_matrix_storage_matches(*C, plan->backend, plan->desc.m, plan->desc.n)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend == RNS8_BACKEND_WRAP64_BYTE_LIMB) {
      if (!C->host_byte_limbs_current) {
        return RNS8_INVALID_ARGUMENT;
      }
      for (int64_t row = 0; row < plan->desc.m; ++row) {
        for (int64_t col = 0; col < plan->desc.n; ++col) {
          dst[row * ld + col] = rns8::detail::wrap_u64_matrix_cell(*C, row, col);
        }
      }
      return RNS8_SUCCESS;
    }
    if (plan->backend == RNS8_BACKEND_HIP_DIRECT) {
      return rns8::detail::wrap64_hip_export_u64_device(
          ctx->device_id,
          C->hip_byte_limbs,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          plan->desc.m,
          plan->desc.n,
          dst,
          ld);
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_export_finite_u8(
    rns8_context* ctx,
    const rns8_plan* plan,
    uint16_t modulus,
    const rns8_matrix* C,
    uint8_t* dst,
    int64_t ld) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_matrix_access(plan->desc.m, plan->desc.n, ld)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status = validate_finite_export_matrix(*ctx, *plan, modulus, *C);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (hip_resident_rns_backend(plan->backend)) {
      auto* mutable_c = const_cast<rns8_matrix*>(C);
      return rns8::detail::hip_direct_export_finite_u8_device(
          ctx->device_id,
          C->hip_residues,
          &mutable_c->hip_export_buffer,
          &mutable_c->hip_export_bytes,
          plan->desc.m,
          plan->desc.n,
          modulus,
          dst,
          ld);
    }
    if (plan->backend == RNS8_BACKEND_CPU_REFERENCE) {
      rns8::detail::export_finite_u8_matrix(*C, dst, ld, modulus);
      return RNS8_SUCCESS;
    }
    return RNS8_UNSUPPORTED_BACKEND;
  });
}

rns8_status rns8_export_exact_wide_signed_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_limb_export_access(plan->desc.m, plan->desc.n, ld, limb_count)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE, plan->prefix);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (hip_resident_rns_backend(plan->backend)) {
      if (!C->device_residues_current) {
        return RNS8_INVALID_ARGUMENT;
      }
      return rns8::detail::hip_direct_export_exact_wide_signed_limbs_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          dst,
          ld,
          limb_count);
    }
    std::vector<uint64_t> staged(
        static_cast<std::size_t>(plan->desc.m) * static_cast<std::size_t>(plan->desc.n) *
            static_cast<std::size_t>(limb_count),
        0);
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, plan->prefix);
        uint64_t* cell_dst =
            staged.data() + static_cast<std::size_t>((row * plan->desc.n + col) * limb_count);
        const rns8_status status = rns8::detail::export_exact_wide_signed_limbs(
            residues, plan->prefix, cell_dst, limb_count);
        if (status != RNS8_SUCCESS) {
          return status;
        }
      }
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const uint64_t* cell_src =
            staged.data() + static_cast<std::size_t>((row * plan->desc.n + col) * limb_count);
        uint64_t* cell_dst = dst + static_cast<std::size_t>((row * ld + col) * limb_count);
        std::copy(cell_src, cell_src + limb_count, cell_dst);
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_export_exact_wide_unsigned_limbs(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* C,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !C || !dst || !valid_limb_export_access(plan->desc.m, plan->desc.n, ld, limb_count)) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status export_status =
        validate_export_matrix(*ctx, *plan, *C, RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE, plan->prefix);
    if (export_status != RNS8_SUCCESS) {
      return export_status;
    }
    if (hip_resident_rns_backend(plan->backend)) {
      if (!C->device_residues_current) {
        return RNS8_INVALID_ARGUMENT;
      }
      return rns8::detail::hip_direct_export_exact_wide_unsigned_limbs_device(
          ctx->device_id,
          C->hip_residues,
          &const_cast<rns8_matrix*>(C)->hip_export_buffer,
          &const_cast<rns8_matrix*>(C)->hip_export_bytes,
          &const_cast<rns8_matrix*>(C)->hip_status_buffer,
          &const_cast<rns8_matrix*>(C)->hip_status_bytes,
          plan->desc.m,
          plan->desc.n,
          plan->prefix,
          dst,
          ld,
          limb_count);
    }
    std::vector<uint64_t> staged(
        static_cast<std::size_t>(plan->desc.m) * static_cast<std::size_t>(plan->desc.n) *
            static_cast<std::size_t>(limb_count),
        0);
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const std::vector<int8_t> residues = gather_cell_residues(*C, row, col, plan->prefix);
        uint64_t* cell_dst =
            staged.data() + static_cast<std::size_t>((row * plan->desc.n + col) * limb_count);
        const rns8_status status = rns8::detail::export_exact_wide_unsigned_limbs(
            residues, plan->prefix, cell_dst, limb_count);
        if (status != RNS8_SUCCESS) {
          return status;
        }
      }
    }
    for (int64_t row = 0; row < plan->desc.m; ++row) {
      for (int64_t col = 0; col < plan->desc.n; ++col) {
        const uint64_t* cell_src =
            staged.data() + static_cast<std::size_t>((row * plan->desc.n + col) * limb_count);
        uint64_t* cell_dst = dst + static_cast<std::size_t>((row * ld + col) * limb_count);
        std::copy(cell_src, cell_src + limb_count, cell_dst);
      }
    }
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_gemm_i64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const int64_t* A,
    int64_t lda,
    const int64_t* B,
    int64_t ldb,
    int64_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_typed_oneshot_contract(*ctx, *desc, RNS8_BOUNDED_I64, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }

    resident_oneshot_state state;
    rns8_status status = create_resident_oneshot_state(ctx, *desc, state);
    if (status == RNS8_SUCCESS) status = rns8_pack_i64(ctx, state.A, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_i64(ctx, state.B, B, ldb, 1);
    if (status == RNS8_SUCCESS) status = rns8_gemm_rns(ctx, state.plan, state.A, state.B, state.C, state.workspace);
    if (status == RNS8_SUCCESS) status = rns8_export_i64(ctx, state.plan, state.C, C, ldc);
    return status;
  });
}

rns8_status rns8_gemm_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_typed_oneshot_contract(*ctx, *desc, RNS8_BOUNDED_U64, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }

    resident_oneshot_state state;
    rns8_status status = create_resident_oneshot_state(ctx, *desc, state);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, state.A, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, state.B, B, ldb, 1);
    if (status == RNS8_SUCCESS) status = rns8_gemm_rns(ctx, state.plan, state.A, state.B, state.C, state.workspace);
    if (status == RNS8_SUCCESS) status = rns8_export_u64(ctx, state.plan, state.C, C, ldc);
    return status;
  });
}

rns8_status rns8_gemm_wrap_u64_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    const uint64_t* A,
    int64_t lda,
    const uint64_t* B,
    int64_t ldb,
    uint64_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_typed_oneshot_contract(*ctx, *desc, RNS8_WRAP_U64_MOD_2_64, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }

    resident_oneshot_state state;
    rns8_status status = create_resident_oneshot_state(ctx, *desc, state);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, state.A, A, lda, 1);
    if (status == RNS8_SUCCESS) status = rns8_pack_u64(ctx, state.B, B, ldb, 1);
    if (status == RNS8_SUCCESS) {
      status = rns8_gemm_wrap_u64(ctx, state.plan, state.A, state.B, state.C, state.workspace);
    }
    if (status == RNS8_SUCCESS) status = rns8_export_wrap_u64(ctx, state.plan, state.C, C, ldc);
    return status;
  });
}

rns8_status rns8_gemm_finite_ring_u8_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_finite_u8_oneshot_contract(*ctx, *desc, RNS8_FINITE_RING_U8, modulus, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }
    return finite_u8_oneshot_resident(ctx, *desc, modulus, A, lda, B, ldb, C, ldc);
  });
}

rns8_status rns8_gemm_finite_field_u8_oneshot(
    rns8_context* ctx,
    const rns8_gemm_desc* desc,
    uint16_t modulus,
    const uint8_t* A,
    int64_t lda,
    const uint8_t* B,
    int64_t ldb,
    uint8_t* C,
    int64_t ldc) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !desc || !A || !B || !C) {
      return RNS8_INVALID_ARGUMENT;
    }
    const rns8_status preflight =
        validate_finite_u8_oneshot_contract(*ctx, *desc, RNS8_FINITE_FIELD_U8, modulus, lda, ldb, ldc);
    if (preflight != RNS8_SUCCESS) {
      return preflight;
    }
    return finite_u8_oneshot_resident(ctx, *desc, modulus, A, lda, B, ldb, C, ldc);
  });
}
