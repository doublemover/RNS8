using namespace rns8::detail::api;

namespace rns8::detail {

rns8_status materialize_native_matrix_as_direct_rns(
    rns8_context* ctx,
    const rns8_plan* plan,
    const rns8_matrix* source,
    rns8_matrix* target) {
  return api::guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !source || !target) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (ctx->backend != RNS8_BACKEND_HIP_DIRECT) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (plan->backend != RNS8_BACKEND_HIP_DIRECT ||
        (plan->desc.semantics != RNS8_BOUNDED_I64 && plan->desc.semantics != RNS8_BOUNDED_U64) ||
        !api::context_accepts_backend(*ctx, plan->backend) || source->hip_device_id != ctx->device_id ||
        target->hip_device_id != ctx->device_id) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (source->backend != RNS8_BACKEND_HIP_VECTOR_ALU_INT64 && source->backend != RNS8_BACKEND_HIP_DIRECT) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (target->backend != RNS8_BACKEND_HIP_DIRECT || source->desc.semantics != plan->desc.semantics ||
        target->desc.semantics != plan->desc.semantics || source->desc.rows != target->desc.rows ||
        source->desc.cols != target->desc.cols) {
      return RNS8_INVALID_ARGUMENT;
    }
    const uint32_t storage_prefix = api::rns_storage_prefix_for_plan(*plan);
    const rns8_bound_kind storage_bound_kind = api::storage_bound_kind_for_plan(*plan);
    if (!api::matrix_descriptor_matches(
            *source,
            plan->desc.semantics,
            storage_bound_kind,
            source->desc.rows,
            source->desc.cols,
            storage_prefix,
            plan->desc.tile_m,
            plan->desc.tile_n) ||
        !api::matrix_descriptor_matches(
            *target,
            plan->desc.semantics,
            storage_bound_kind,
            target->desc.rows,
            target->desc.cols,
            storage_prefix,
            plan->desc.tile_m,
            plan->desc.tile_n) ||
        !api::bounded_native_storage_matches(*source, plan->desc.semantics, source->desc.rows, source->desc.cols) ||
        !api::bounded_native_state_current(*source) ||
        !api::rns_matrix_storage_matches(
            *target, RNS8_BACKEND_HIP_DIRECT, target->desc.rows, target->desc.cols, storage_prefix)) {
      return RNS8_INVALID_ARGUMENT;
    }

    rns8_status status = RNS8_SUCCESS;
    if (plan->desc.semantics == RNS8_BOUNDED_I64) {
      status = rns8::detail::hip_direct_native_i64_to_rns_device(
          ctx->device_id, source->hip_native_i64, target->hip_residues, target->desc.rows, target->desc.cols,
          storage_prefix);
    } else {
      status = rns8::detail::hip_direct_native_u64_to_rns_device(
          ctx->device_id, source->hip_native_u64, target->hip_residues, target->desc.rows, target->desc.cols,
          storage_prefix);
    }
    if (status != RNS8_SUCCESS) {
      api::clear_device_residues_current(*target);
      return status;
    }
    api::mark_output_device_residues_current(*target);
    target->source_version = source->source_version;
    return RNS8_SUCCESS;
  });
}

rns8_status force_native_to_rns_bridge_inputs(rns8_matrix* a_matrix, rns8_matrix* b_matrix) {
  return api::guard_api([&]() -> rns8_status {
    if (!a_matrix || !b_matrix) {
      return RNS8_INVALID_ARGUMENT;
    }
    const auto force_one = [](rns8_matrix& matrix) -> bool {
      if (matrix.backend != RNS8_BACKEND_HIP_DIRECT ||
          (matrix.desc.semantics != RNS8_BOUNDED_I64 && matrix.desc.semantics != RNS8_BOUNDED_U64) ||
          !api::bounded_native_storage_matches(matrix, matrix.desc.semantics, matrix.desc.rows, matrix.desc.cols) ||
          !api::bounded_native_state_current(matrix)) {
        return false;
      }
      api::clear_residue_current(matrix);
      api::clear_byte_limb_current(matrix);
      return true;
    };
    if (a_matrix->desc.semantics != b_matrix->desc.semantics ||
        !force_one(*a_matrix) || !force_one(*b_matrix)) {
      return RNS8_INVALID_ARGUMENT;
    }
    return RNS8_SUCCESS;
  });
}

}  // namespace rns8::detail
