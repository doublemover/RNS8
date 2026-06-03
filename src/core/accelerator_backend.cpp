#include "core/accelerator_backend.hpp"

#include "core/internal.hpp"

namespace rns8::detail {

namespace {

constexpr accelerator_backend_descriptor kCkDescriptor{
    RNS8_BACKEND_CK,
    "ck",
    "Composable Kernel",
    "RNS8_ENABLE_CK",
    "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1_disabled",
    "ck_fused_i32_to_centered_residue_disabled",
    "resident_device_buffers_with_ck_pack_workspace_disabled",
    "not_validated",
    "not_enabled_in_this_build",
    "CK has an opt-in Windows gfx1100 correctness backend with exact differentials and ISA/schema evidence; this build "
    "did not enable RNS8_ENABLE_CK. Discovery remains evidence only and does not select the backend.",
    1,
    1,
    1,
    0,
    1,
};

constexpr accelerator_backend_descriptor kRocwmmaDescriptor{
    RNS8_BACKEND_ROCWMMA,
    "rocwmma",
    "rocWMMA/AMDGPU builtins",
    "RNS8_ENABLE_ROCWMMA/RNS8_ENABLE_AMDGPU_BUILTINS",
    "rocwmma_i8_i32_signed_hot_residue_v1_disabled",
    "rocwmma_fused_i32_to_centered_residue_disabled",
    "resident_device_buffers_with_rocwmma_pack_workspace_disabled",
    "not_validated",
    "not_enabled_or_builtin_not_implemented",
    "rocWMMA has an opt-in Windows gfx1100 correctness backend with exact differentials and ISA/schema evidence; this "
    "build did not enable RNS8_ENABLE_ROCWMMA. AMDGPU builtins remain fail-fast until target-specific exact kernels "
    "exist.",
    1,
    1,
    1,
    0,
    1,
};

void set_text(char* dst, std::size_t dst_size, const char* text) {
  copy_c_string(dst, dst_size, text ? text : "");
}

}  // namespace

bool accelerator_backend_kind(rns8_backend_kind backend) {
  return backend == RNS8_BACKEND_CK || backend == RNS8_BACKEND_ROCWMMA;
}

bool accelerator_backend_compiled(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_CK:
#if defined(RNS8_ENABLE_CK) && RNS8_ENABLE_CK
      return true;
#else
      return false;
#endif
    case RNS8_BACKEND_ROCWMMA:
#if (defined(RNS8_ENABLE_ROCWMMA) && RNS8_ENABLE_ROCWMMA) || \
    (defined(RNS8_ENABLE_AMDGPU_BUILTINS) && RNS8_ENABLE_AMDGPU_BUILTINS)
      return true;
#else
      return false;
#endif
    default:
      return false;
  }
}

bool accelerator_backend_supports_semantics(rns8_backend_kind backend, rns8_semantics semantics) {
  if (!accelerator_backend_compiled(backend)) {
    return false;
  }
  const accelerator_backend_descriptor* descriptor = accelerator_backend_descriptor_for(backend);
  if (!descriptor) {
    return false;
  }
  switch (semantics) {
    case RNS8_BOUNDED_I64:
    case RNS8_BOUNDED_U64:
      return descriptor->supports_bounded_rns != 0;
    case RNS8_EXACT_WIDE_SIGNED:
    case RNS8_EXACT_WIDE_UNSIGNED:
      return descriptor->supports_exact_wide_rns != 0;
    case RNS8_FINITE_RING_U8:
    case RNS8_FINITE_FIELD_U8:
      return descriptor->supports_finite_u8 != 0;
    case RNS8_WRAP_U64_MOD_2_64:
      return descriptor->supports_wrap64 != 0;
  }
  return false;
}

const accelerator_backend_descriptor* accelerator_backend_descriptor_for(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_CK:
      return &kCkDescriptor;
    case RNS8_BACKEND_ROCWMMA:
      return &kRocwmmaDescriptor;
    default:
      return nullptr;
  }
}

void fill_disabled_accelerator_capability(rns8_backend_kind backend, rns8_backend_capability_info& info) {
  const accelerator_backend_descriptor* descriptor = accelerator_backend_descriptor_for(backend);
  if (!descriptor) {
    return;
  }
  info.requires_feature_detection = 1;
  info.enable_flag_fail_fast = 1;
  info.candidate_evidence_only = 1;
  info.supports_bounded_rns = descriptor->supports_bounded_rns;
  info.supports_exact_wide_rns = descriptor->supports_exact_wide_rns;
  info.supports_finite_u8 = descriptor->supports_finite_u8;
  info.supports_wrap64 = descriptor->supports_wrap64;
  info.is_matrix_engine_backend = 0;
  set_text(info.selected_kernel, sizeof(info.selected_kernel), descriptor->disabled_selected_kernel);
  set_text(info.library_name, sizeof(info.library_name), descriptor->library_name);
  set_text(info.enable_flag, sizeof(info.enable_flag), descriptor->enable_flag);
  set_text(info.epilogue_mode, sizeof(info.epilogue_mode), descriptor->disabled_epilogue_mode);
  set_text(info.workspace_mode, sizeof(info.workspace_mode), descriptor->disabled_workspace_mode);
  set_text(info.isa_evidence, sizeof(info.isa_evidence), descriptor->disabled_isa_evidence);
  set_text(info.status, sizeof(info.status), descriptor->disabled_status);
  set_text(info.detail, sizeof(info.detail), descriptor->disabled_detail);
}

}  // namespace rns8::detail
