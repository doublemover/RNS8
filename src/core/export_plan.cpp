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
    case export_d2h_policy::compact_contiguous:
      return "compact_contiguous";
    case export_d2h_policy::device_residue_current:
      return "device_residue_current";
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

std::string export_backend_name(rns8_backend_kind backend) {
  switch (backend) {
    case RNS8_BACKEND_CPU_REFERENCE:
      return "cpu";
    case RNS8_BACKEND_HIP_DIRECT:
      return "hip-direct";
    case RNS8_BACKEND_HIP_VECTOR_ALU_INT64:
      return "hip-vector-alu-int64";
    case RNS8_BACKEND_HIPBLASLT:
      return "hipblaslt";
    case RNS8_BACKEND_CK:
      return "ck";
    case RNS8_BACKEND_ROCWMMA:
      return "rocwmma";
    case RNS8_BACKEND_WRAP64_BYTE_LIMB:
      return "wrap64-byte-limb";
    case RNS8_BACKEND_AUTO:
      return "auto";
  }
  return "unknown";
}

std::string export_signedness_for_semantics(rns8_semantics semantics) {
  switch (semantics) {
    case RNS8_BOUNDED_I64:
    case RNS8_EXACT_WIDE_SIGNED:
      return "signed";
    case RNS8_BOUNDED_U64:
    case RNS8_WRAP_U64_MOD_2_64:
    case RNS8_EXACT_WIDE_UNSIGNED:
      return "unsigned";
    case RNS8_FINITE_FIELD_U8:
    case RNS8_FINITE_RING_U8:
      return "finite";
  }
  return "unknown";
}

std::string prefix_contract_for_plan(const rns8_plan& plan) {
  return "prefix=" + std::to_string(plan.prefix) +
         ";min_selected=" + std::to_string(plan.schedule_min_selected_prefix) +
         ";max_selected=" + std::to_string(plan.schedule_max_selected_prefix) +
         ";groups=" + std::to_string(plan.schedule_prefix_group_count);
}

std::string selector_key_for_plan(const rns8_plan& plan, const export_reconstruction_plan& export_plan) {
  std::string key = "semantics=" + export_plan.semantic_contract;
  key += ";backend=" + export_plan.backend;
  key += ";target_id=" + export_plan.target_id;
  key += ";prefix=" + std::to_string(plan.prefix);
  key += ";limb_count=" + std::to_string(export_plan.limb_count);
  key += ";signedness=" + export_plan.signedness;
  key += ";output_layout=" + std::string(export_output_layout_name(export_plan.output_layout));
  key += ";status_policy=" + std::string(export_status_policy_name(export_plan.status_policy));
  key += ";d2h_policy=" + std::string(export_d2h_policy_name(export_plan.d2h_policy));
  key += ";final_output_mode=" + export_plan.final_output_mode;
  key += ";selected_kernel=" + std::string(export_plan.selected_export_kernel ? export_plan.selected_export_kernel : "unknown");
  return key;
}

}  // namespace

export_reconstruction_plan make_export_plan(
    const rns8_plan& plan,
    rns8_semantics semantics,
    uint32_t limb_count) {
  export_reconstruction_plan export_plan{};
  export_plan.limb_count = limb_count;
  export_plan.semantic_contract = semantics_name_for_key(semantics);
  export_plan.backend = export_backend_name(plan.backend);
  export_plan.target_id = plan.backend_target_id.empty() ? (hip_resident_rns_backend(plan.backend) ? "unknown" : "cpu")
                                                         : plan.backend_target_id;
  export_plan.prefix_contract = prefix_contract_for_plan(plan);
  export_plan.signedness = export_signedness_for_semantics(semantics);
  export_plan.selector_policy = "semantic_prefix_limb_layout_status_d2h_backend_target";
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
    export_plan.status_elision_reason = "wraparound_mod_2_64_has_no_range_status";
  } else if (uses_finite_storage(semantics)) {
    export_plan.output_layout = export_output_layout::finite_u8;
    export_plan.selected_export_kernel = finite_export_kernel(plan);
    export_plan.status_elision_reason = "finite_u8_canonical_output_has_no_range_status";
  } else if (semantics == RNS8_EXACT_WIDE_SIGNED || semantics == RNS8_EXACT_WIDE_UNSIGNED) {
    export_plan.output_layout = export_output_layout::fixed_u64_limbs;
    export_plan.status_policy = export_status_policy::range_checked_status_buffer;
    export_plan.selected_export_kernel = exact_wide_export_kernel(plan, semantics);
  }
  if (export_plan.status_policy == export_status_policy::none && export_plan.status_elision_reason.empty()) {
    export_plan.status_elision_reason = "status_policy_none_for_selected_semantics";
  }
  if (export_plan.all_zero_tiled_output) {
    export_plan.cache_visibility = "selector_visible_all_zero_tiled_output";
  } else if (export_plan.requires_hip_tile_metadata) {
    export_plan.cache_visibility = "selector_visible_tile_metadata_required";
  }
  export_plan.stale_entry_reason =
      "selector_key_mismatch_rejects_semantic_prefix_limb_layout_status_d2h_backend_target";
  export_plan.selector_key = selector_key_for_plan(plan, export_plan);
  return export_plan;
}

}  // namespace rns8::detail::api
