#include "core/api_internal.hpp"

namespace rns8::detail::api {

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
    case RNS8_BACKEND_ROCWMMA:
    case RNS8_BACKEND_AMDGPU_BUILTINS:
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
    case RNS8_BACKEND_ROCWMMA:
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
    case RNS8_BACKEND_AMDGPU_BUILTINS:
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
    case RNS8_BACKEND_ROCWMMA:
      return "rocwmma";
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return "wrap64-byte-limb";
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
      return "hip-vector-alu-int64";
    case RNS8_BACKEND_AMDGPU_BUILTINS:
      return "amdgpu-builtins";
  }
  return "unknown";
}

bool accelerator_backend(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_HIPBLASLT || backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_ROCWMMA ||
         backend == RNS8_BACKEND_AMDGPU_BUILTINS;
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
          hipblaslt_backend_compiled() ? "hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2"
                                       : "not_implemented");
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
          hipblaslt_backend_compiled() ? "hipblaslt_library_int8_matmul_specialized_reduce_251_255_256"
                                       : "not_validated");
      set_text(
          info.status,
          sizeof(info.status),
          hipblaslt_backend_compiled() ? "implemented_baseline_backend" : "not_implemented_evidence_only");
      set_text(
          info.detail,
          sizeof(info.detail),
          hipblaslt_backend_compiled()
              ? "hipBLASLt INT8->INT32 GEMM with separate HIP centered-residue reduction specialized for mod 256/255/251; no adaptive per-tile support."
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
      set_text(
          info.selected_kernel,
          sizeof(info.selected_kernel),
          "ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3");
      set_text(info.library_name, sizeof(info.library_name), "Composable Kernel");
      set_text(info.library_version, sizeof(info.library_version), "repo-local release/rocm-rel-7.1");
      set_text(info.enable_flag, sizeof(info.enable_flag), "RNS8_ENABLE_CK");
      set_text(info.epilogue_mode, sizeof(info.epilogue_mode), "ck_fused_i32_to_centered_residue");
      set_text(
          info.workspace_mode,
          sizeof(info.workspace_mode),
#if RNS8_CK_USE_XDL
          "resident_device_buffers_with_ck_centered_pack_workspace");
#else
          "resident_device_buffers_with_ck_canonical_pack_workspace");
#endif
      set_text(
          info.isa_evidence,
          sizeof(info.isa_evidence),
          "ck_cshuffle_int8_matrix_isa_gate_no_divide");
      set_text(info.status, sizeof(info.status), "implemented_opt_in_ck_backend");
      set_text(
          info.detail,
          sizeof(info.detail),
          "Opt-in CK backend using target-selected CShuffle int8 matrix GEMM with fused centered-residue epilogue dispatched through compile-time static reducers for every supported byte modulus in the default RNS ladder and finite-u8 allow-list; bounded-u64 is not exposed until an unsigned exactness contract is validated.");
#else
      rns8::detail::fill_disabled_accelerator_capability(backend, info);
#endif
      break;
    case RNS8_BACKEND_ROCWMMA:
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
      set_text(
          info.selected_kernel,
          sizeof(info.selected_kernel),
          "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2");
      set_text(info.library_name, sizeof(info.library_name), "rocWMMA");
      set_text(info.library_version, sizeof(info.library_version), "repo-local release/rocm-rel-7.1");
      set_text(info.enable_flag, sizeof(info.enable_flag), "RNS8_ENABLE_ROCWMMA");
      set_text(info.epilogue_mode, sizeof(info.epilogue_mode), "rocwmma_fused_i32_to_centered_residue");
      set_text(info.workspace_mode, sizeof(info.workspace_mode), "resident_device_buffers_with_rocwmma_pack_workspace");
      set_text(
          info.isa_evidence,
          sizeof(info.isa_evidence),
          "rocwmma_i8_matrix_isa_gate_no_divide");
      set_text(info.status, sizeof(info.status), "implemented_opt_in_rocwmma_backend");
      set_text(
          info.detail,
          sizeof(info.detail),
          "Opt-in rocWMMA backend using signed int8 matrix GEMM with fused centered-residue reduction specialized for mod 256/255/251.");
#else
      rns8::detail::fill_disabled_accelerator_capability(backend, info);
#endif
      break;
    case RNS8_BACKEND_AMDGPU_BUILTINS:
#if defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS && \
    defined(RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE) && RNS8_AMDGPU_BUILTIN_KERNELS_AVAILABLE
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
      set_text(
          info.selected_kernel,
          sizeof(info.selected_kernel),
          "amdgpu_builtin_matrix_core_target_family_runtime_dispatch_v1");
      set_text(info.library_name, sizeof(info.library_name), "AMDGPU builtins");
      set_text(info.library_version, sizeof(info.library_version), "compiled_target_specific");
      set_text(info.enable_flag, sizeof(info.enable_flag), "RNS8_ENABLE_AMDGPU_BUILTINS");
      set_text(
          info.epilogue_mode,
          sizeof(info.epilogue_mode),
          "amdgpu_builtin_fused_i32_to_centered_residue_then_crt_export");
      set_text(
          info.workspace_mode,
          sizeof(info.workspace_mode),
          "resident_device_buffers_direct_matrix_core_no_dense_pack_workspace");
      set_text(info.isa_evidence, sizeof(info.isa_evidence), "amdgpu_builtin_matrix_isa_gate_no_divide");
      set_text(info.status, sizeof(info.status), "implemented_opt_in_amdgpu_builtin_backend");
      set_text(
          info.detail,
          sizeof(info.detail),
          "Opt-in target-specific AMDGPU builtin backend using compiled dense MFMA/WMMA kernels and explicit sparse-A SMFMAC/SWMMAC kernels; promotion still requires exact CPU parity and measured timings.");
#else
      rns8::detail::fill_disabled_accelerator_capability(backend, info);
#endif
      break;
    case RNS8_BACKEND_AUTO:
      break;
  }
}

}  // namespace rns8::detail::api

using namespace rns8::detail::api;

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
