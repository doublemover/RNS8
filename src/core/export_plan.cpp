#include "core/api_internal.hpp"

namespace rns8::detail::api {

const char* export_output_layout_name(export_output_layout layout) {
  switch (layout) {
    case export_output_layout::scalar_i64:
      return "scalar_i64";
    case export_output_layout::scalar_u64:
      return "scalar_u64";
    case export_output_layout::finite_u8:
      return "finite_u8";
    case export_output_layout::fixed_u64_limbs:
      return "fixed_u64_limbs";
  }
  return "unknown";
}

const char* export_status_policy_name(export_status_policy policy) {
  switch (policy) {
    case export_status_policy::none:
      return "none";
    case export_status_policy::range_checked_status_buffer:
      return "range_checked_status_buffer";
  }
  return "unknown";
}

const char* export_d2h_policy_name(export_d2h_policy policy) {
  switch (policy) {
    case export_d2h_policy::host_ld_padded:
      return "host_ld_padded";
  }
  return "unknown";
}

namespace {

bool export_backend_is_hip_rns(rns8_backend_kind backend) {
  return hip_resident_rns_backend(backend);
}

const char* bounded_i64_export_kernel(const rns8_plan& plan) {
  if (export_backend_is_hip_rns(plan.backend)) {
    return !plan.tile_schedule.empty() ? "hip_direct_export_i64_tiled_device" : "hip_direct_export_i64_device";
  }
  if (native_vector_backend(plan.backend)) {
    return "vector_alu_output_d2h";
  }
  return "cpu_reference_export_i64";
}

const char* bounded_u64_export_kernel(const rns8_plan& plan) {
  if (export_backend_is_hip_rns(plan.backend)) {
    return !plan.tile_schedule.empty() ? "hip_direct_export_u64_tiled_device" : "hip_direct_export_u64_device";
  }
  if (native_vector_backend(plan.backend)) {
    return "vector_alu_output_d2h";
  }
  return "cpu_reference_export_u64";
}

const char* finite_export_kernel(const rns8_plan& plan) {
  return export_backend_is_hip_rns(plan.backend) ? "hip_direct_export_finite_u8_device" : "finite_reference_export_u8";
}

const char* exact_wide_export_kernel(const rns8_plan& plan, rns8_semantics semantics) {
  if (semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    return export_backend_is_hip_rns(plan.backend) ? "hip_direct_export_exact_wide_unsigned_limbs_device"
                                                   : "cpu_reference_export_exact_wide_unsigned_limbs";
  }
  return export_backend_is_hip_rns(plan.backend) ? "hip_direct_export_exact_wide_signed_limbs_device"
                                                 : "cpu_reference_export_exact_wide_signed_limbs";
}

}  // namespace

export_reconstruction_plan make_export_plan(
    const rns8_plan& plan,
    rns8_semantics semantics,
    uint32_t limb_count) {
  export_reconstruction_plan export_plan{};
  export_plan.limb_count = limb_count;
  export_plan.all_zero_tiled_output = plan_all_zero_output_tiles(plan);
  export_plan.requires_hip_tile_metadata =
      export_backend_is_hip_rns(plan.backend) && !plan.tile_schedule.empty() && !export_plan.all_zero_tiled_output;
  if (semantics == RNS8_BOUNDED_I64) {
    export_plan.output_layout = export_output_layout::scalar_i64;
    export_plan.status_policy = export_status_policy::range_checked_status_buffer;
    export_plan.selected_export_kernel = bounded_i64_export_kernel(plan);
  } else if (semantics == RNS8_BOUNDED_U64) {
    export_plan.output_layout = export_output_layout::scalar_u64;
    export_plan.status_policy = export_status_policy::range_checked_status_buffer;
    export_plan.selected_export_kernel = bounded_u64_export_kernel(plan);
  } else if (semantics == RNS8_WRAP_U64_MOD_2_64) {
    export_plan.output_layout = export_output_layout::scalar_u64;
    export_plan.selected_export_kernel = plan.backend == RNS8_BACKEND_HIP_DIRECT ? "wrap64_hip_export_u64_device"
                                                                                 : "wrap64_reference_export_u64";
  } else if (uses_finite_storage(semantics)) {
    export_plan.output_layout = export_output_layout::finite_u8;
    export_plan.selected_export_kernel = finite_export_kernel(plan);
  } else if (semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    export_plan.output_layout = export_output_layout::fixed_u64_limbs;
    export_plan.status_policy = export_status_policy::range_checked_status_buffer;
    export_plan.selected_export_kernel = exact_wide_export_kernel(plan, semantics);
  }
  return export_plan;
}

}  // namespace rns8::detail::api
