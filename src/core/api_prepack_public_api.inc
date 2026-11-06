namespace rns8::detail::api {
void retire_unpublished_prepack_cache(rns8_context& ctx, rns8_prepack_cache* cache) noexcept {
  if (!cache || cache->failed_cleanup_retired)
    return;
  if (rns8_destroy_prepack_cache(cache) == RNS8_SUCCESS)
    return;
  ctx.hip_failure_poisoned = true;
  cache->failed_cleanup_retired = true;
  cache->failed_cleanup_next = ctx.failed_prepack_cleanup;
  ctx.failed_prepack_cleanup = cache;
}
rns8_status drain_failed_prepack_cleanup(rns8_context& ctx) noexcept {
  while (ctx.failed_prepack_cleanup) {
    auto* cache = ctx.failed_prepack_cleanup;
    auto* next = cache->failed_cleanup_next;
    const auto status = rns8_destroy_prepack_cache(cache);
    if (status != RNS8_SUCCESS)
      return status;
    ctx.failed_prepack_cleanup = next;
  }
  return RNS8_SUCCESS;
}
} // namespace rns8::detail::api
namespace {
struct UnpublishedPrepackDeleter {
  rns8_context* ctx;
  void operator()(rns8_prepack_cache* cache) const noexcept {
    rns8::detail::api::retire_unpublished_prepack_cache(*ctx, cache);
  }
};
using UnpublishedPrepack = std::unique_ptr<rns8_prepack_cache, UnpublishedPrepackDeleter>;
rns8_status hold_prepack_construction(rns8_context& ctx, rns8_prepack_cache& cache,
                                      rns8::detail::api::SynchronousLease& lease, size_t token) {
  auto status = rns8::detail::api::add_synchronous_owner_hold(ctx, token, cache.in_flight_readers);
  if (status == RNS8_SUCCESS)
    status = lease.hold_buffer(cache.data_owner);
  if (status == RNS8_SUCCESS)
    status = lease.hold_buffer(cache.native_upload_owner);
  return status;
}
} // namespace

rns8_status rns8_get_prepack_cache_key_info(const rns8_plan* plan, const rns8_matrix* matrix,
                                            rns8_operand_role operand_role,
                                            rns8_prepack_cache_key_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!plan || !matrix || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::minimum_abi_header(out)) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!prepack_operand_matrix_compatible(*plan, *matrix, operand_role)) {
      return RNS8_INVALID_ARGUMENT;
    }

    const uint64_t struct_size = out->struct_size;
    const uint32_t abi_version = out->abi_version;
    const std::string matrix_layout_version =
        persistent_layout_version_for_semantics(matrix->desc.semantics);
    const std::string operand_layout_version =
        prepack_operand_layout_version_for_plan(*plan, operand_role);
    const uint64_t plan_fingerprint = plan_workspace_fingerprint(*plan);
    const uint64_t cache_key_hash = prepack_cache_key_hash(
        *plan, *matrix, operand_role, matrix_layout_version, operand_layout_version);
    const std::string cache_key =
        build_prepack_cache_key(*plan, *matrix, operand_role, plan_fingerprint, cache_key_hash,
                                matrix_layout_version, operand_layout_version);

    *out = {};
    out->struct_size = struct_size;
    out->abi_version = abi_version;
    out->backend = plan->backend;
    out->semantics = plan->desc.semantics;
    out->operand_role = operand_role;
    out->cache_key_valid = 1;
    const bool stable_source_version = matrix->source_version != 0 && !matrix->generation_exhausted;
    const bool reusable_cache = stable_source_version && operand_role == RNS8_OPERAND_B &&
                                rocwmma_b_prepack_cache_supported(*plan);
    const bool production_cache = stable_source_version && operand_role == RNS8_OPERAND_B &&
                                  rocwmma_b_prepack_cache_production_supported(*plan);
    out->reusable_prepack_cache_available = reusable_cache ? 1u : 0u;
    out->production_prepack_cache_available = production_cache ? 1u : 0u;
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
    set_text(out->operand_layout_version, sizeof(out->operand_layout_version),
             operand_layout_version);
    set_text(out->cache_scope, sizeof(out->cache_scope),
             !stable_source_version ? "validated_key_unstable_source_version_no_runtime_cache"
             : production_cache     ? "validated_production_b_prepack_cache_key"
             : reusable_cache       ? "validated_reusable_b_prepack_cache_key_evidence_only"
                                    : "validated_key_no_production_cache");
    set_text(out->cache_key, sizeof(out->cache_key), cache_key);
    set_text(
        out->detail, sizeof(out->detail),
        !stable_source_version
            ? "Plan and operand matrix are compatible for prepack cache keying, but source_version=0 cannot identify a reusable runtime cache."
        : production_cache
            ? "Plan and B operand matrix are compatible with the production rocWMMA reusable B prepack cache."
        : reusable_cache
            ? "Plan and B operand matrix are compatible with a reusable rocWMMA B prepack cache, but this semantic contract is not production-promotable yet."
            : "Plan and operand matrix are compatible for prepack cache keying, but no reusable production cache is available for this role.");
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_create_prepack_cache(rns8_context* ctx, const rns8_plan* plan,
                                      const rns8_matrix* matrix, rns8_operand_role operand_role,
                                      rns8_prepack_cache** out) {
  return guard_api([&]() -> rns8_status {
    if (!ctx || !plan || !matrix || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!valid_prepack_operand_role(operand_role)) {
      return RNS8_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (ctx->hip_failure_poisoned)
      return RNS8_BACKEND_FAILURE;
    if (operand_role != RNS8_OPERAND_B || !rocwmma_b_prepack_cache_supported(*plan)) {
      return RNS8_UNSUPPORTED_BACKEND;
    }
    if (matrix->in_flight_writer)
      return RNS8_BUSY;
    if (!context_accepts_backend(*ctx, plan->backend) ||
        !prepack_operand_matrix_compatible(*plan, *matrix, operand_role) ||
        matrix->hip_device_id != ctx->device_id || !matrix->device_residues_current ||
        matrix->source_version == 0 || matrix->generation_exhausted) {
      return RNS8_INVALID_ARGUMENT;
    }

    if (rns8::detail::exact_wide_semantics(plan->desc.semantics) &&
        matrix->exact_range_prefix < plan->prefix)
      return RNS8_RANGE_ERROR;

    std::size_t b_pack_bytes = 0;
    std::size_t total_cache_bytes = 0;
    if (!rocwmma_b_prepack_bytes_for_plan(*plan, b_pack_bytes, total_cache_bytes)) {
      return RNS8_RANGE_ERROR;
    }

    UnpublishedPrepack cache_owner(new (std::nothrow) rns8_prepack_cache(),
                                   UnpublishedPrepackDeleter{ctx});
    auto* cache = cache_owner.get();
    if (!cache) {
      return RNS8_INTERNAL_ERROR;
    }
    cache->backend = plan->backend;
    cache->semantics = plan->desc.semantics;
    cache->operand_role = operand_role;
    cache->rows = matrix->desc.rows;
    cache->cols = matrix->desc.cols;
    cache->k = plan->desc.k;
    cache->tile_m = plan->desc.tile_m;
    cache->tile_n = plan->desc.tile_n;
    cache->prefix = plan->prefix;
    cache->finite_modulus = matrix->finite_modulus;
    cache->prefix_schedule_fingerprint = prepack_prefix_schedule_fingerprint(*plan);
    cache->k_block_size = rocwmma_b_prepack_k_block_size(*plan);
    cache->k_block_cap = rocwmma_b_prepack_k_block_cap();
    cache->source_version = matrix->source_version;
    cache->exact_max_magnitude = matrix->exact_max_magnitude;
    cache->exact_range_prefix = matrix->exact_range_prefix;
    cache->plan_fingerprint = plan_workspace_fingerprint(*plan);
    cache->target_id = prepack_target_id_for_device(matrix->hip_device_id);
    cache->selected_kernel = plan->backend_selected_kernel;
    cache->prepack_kernel_variant = rocwmma_b_prepack_kernel_variant();
    cache->matrix_layout_version = persistent_layout_version_for_semantics(matrix->desc.semantics);
    cache->operand_layout_version = prepack_operand_layout_version_for_plan(*plan, operand_role);
    cache->cache_key_hash = prepack_cache_key_hash(
        *plan, *matrix, operand_role, cache->matrix_layout_version, cache->operand_layout_version);
    cache->cache_key = build_prepack_cache_key(
        *plan, *matrix, operand_role, cache->plan_fingerprint, cache->cache_key_hash,
        cache->matrix_layout_version, cache->operand_layout_version);
    cache->hip_device_id = ctx->device_id;
    cache->device_bytes = total_cache_bytes;
    cache->operand_pack_bytes = b_pack_bytes;

    rns8_status status =
        cache->data_owner.allocate_alias(ctx->device_id, total_cache_bytes, cache->device_data);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    size_t token = 0;
    status = begin_synchronous_read_lease(*ctx, *matrix, token);
    if (status != RNS8_SUCCESS)
      return status;
    SynchronousLease lease(*ctx, token);
    status = hold_prepack_construction(*ctx, *cache, lease, token);
    if (status != RNS8_SUCCESS)
      return lease.finish(status);
    status = synchronous_dispatch_checkpoint(*ctx, "prepack_rns_create");
    if (status != RNS8_SUCCESS)
      return lease.finish(status);
    status = rns8::detail::rocwmma_prepack_b_rns_device(
        ctx->device_id, matrix->hip_residues, cache->device_data, cache->device_bytes, plan->desc.k,
        plan->desc.n, matrix->desc.cols, plan->prefix);
    if (status == RNS8_SUCCESS)
      status = synchronous_dispatch_checkpoint(*ctx, "prepack_rns_publish");
    status = lease.finish(status);
    if (status != RNS8_SUCCESS) {
      return status;
    }
    *out = cache_owner.release();
    return RNS8_SUCCESS;
  });
}

namespace {

template <typename NativeT>
rns8_status
create_b_prepack_cache_from_native(rns8_context* ctx, const rns8_plan* plan, const NativeT* values,
                                   int64_t ld, uint64_t source_version,
                                   rns8_semantics expected_semantics,
                                   const char* native_layout_version, rns8_prepack_cache** out) {
  if (!out) {
    return RNS8_INVALID_ARGUMENT;
  }
  *out = nullptr;
  if (!ctx || !plan || !values || source_version == 0 ||
      !valid_matrix_access(plan->desc.k, plan->desc.n, ld)) {
    return RNS8_INVALID_ARGUMENT;
  }
  if (ctx->hip_failure_poisoned)
    return RNS8_BACKEND_FAILURE;
  if (plan->desc.semantics != expected_semantics ||
      !rocwmma_b_prepack_cache_production_supported(*plan)) {
    return RNS8_UNSUPPORTED_BACKEND;
  }
  if (!context_accepts_backend(*ctx, plan->backend) || ctx->device_id < 0) {
    return RNS8_INVALID_ARGUMENT;
  }

  std::size_t b_pack_bytes = 0;
  std::size_t total_cache_bytes = 0;
  if (!rocwmma_b_prepack_bytes_for_plan(*plan, b_pack_bytes, total_cache_bytes)) {
    return RNS8_RANGE_ERROR;
  }
  std::size_t native_bytes = 0;
  if (!native_matrix_bytes(plan->desc.k, plan->desc.n, sizeof(NativeT), native_bytes)) {
    return RNS8_RANGE_ERROR;
  }

  rns8_matrix synthetic_source{};
  synthetic_source.backend = plan->backend;
  synthetic_source.desc.struct_size = sizeof(synthetic_source.desc);
  synthetic_source.desc.abi_version = RNS8_ABI_VERSION;
  synthetic_source.desc.semantics = plan->desc.semantics;
  synthetic_source.desc.bound_kind = plan->desc.bound_kind;
  synthetic_source.desc.rows = plan->desc.k;
  synthetic_source.desc.cols = plan->desc.n;
  synthetic_source.desc.logical_ld = ld;
  synthetic_source.desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  synthetic_source.desc.max_prefix = plan->prefix;
  synthetic_source.desc.tile_m = plan->desc.tile_m;
  synthetic_source.desc.tile_n = plan->desc.tile_n;
  synthetic_source.prefix = plan->prefix;
  synthetic_source.source_version = source_version;
  synthetic_source.hip_device_id = ctx->device_id;
  synthetic_source.device_native_current = true;

  UnpublishedPrepack cache_owner(new (std::nothrow) rns8_prepack_cache(),
                                 UnpublishedPrepackDeleter{ctx});
  auto* cache = cache_owner.get();
  if (!cache) {
    return RNS8_INTERNAL_ERROR;
  }
  cache->backend = plan->backend;
  cache->semantics = plan->desc.semantics;
  cache->operand_role = RNS8_OPERAND_B;
  cache->rows = plan->desc.k;
  cache->cols = plan->desc.n;
  cache->k = plan->desc.k;
  cache->tile_m = plan->desc.tile_m;
  cache->tile_n = plan->desc.tile_n;
  cache->prefix = plan->prefix;
  cache->finite_modulus = 0;
  cache->prefix_schedule_fingerprint = prepack_prefix_schedule_fingerprint(*plan);
  cache->k_block_size = rocwmma_b_prepack_k_block_size(*plan);
  cache->k_block_cap = rocwmma_b_prepack_k_block_cap();
  cache->source_version = source_version;
  cache->plan_fingerprint = plan_workspace_fingerprint(*plan);
  cache->target_id = prepack_target_id_for_context(*ctx);
  cache->selected_kernel = plan->backend_selected_kernel;
  cache->prepack_kernel_variant = rocwmma_b_prepack_kernel_variant();
  cache->matrix_layout_version = native_layout_version;
  cache->operand_layout_version = prepack_operand_layout_version_for_plan(*plan, RNS8_OPERAND_B);
  cache->cache_key_hash =
      prepack_cache_key_hash(*plan, synthetic_source, RNS8_OPERAND_B, cache->matrix_layout_version,
                             cache->operand_layout_version);
  cache->cache_key = build_prepack_cache_key(
      *plan, synthetic_source, RNS8_OPERAND_B, cache->plan_fingerprint, cache->cache_key_hash,
      cache->matrix_layout_version, cache->operand_layout_version);
  cache->hip_device_id = ctx->device_id;
  cache->device_bytes = total_cache_bytes;
  cache->operand_pack_bytes = b_pack_bytes;

  rns8_status status =
      cache->data_owner.allocate_alias(ctx->device_id, total_cache_bytes, cache->device_data);
  if (status != RNS8_SUCCESS) {
    return status;
  }

  status = cache->native_upload_owner.allocate(ctx->device_id, native_bytes);
  if (status != RNS8_SUCCESS) {
    return status;
  }

  static_assert(sizeof(NativeT) == sizeof(uint64_t));
  const auto rows = static_cast<std::size_t>(plan->desc.k),
             cols = static_cast<std::size_t>(plan->desc.n);
  cache->native_staging_words.resize(native_bytes / sizeof(uint64_t));
  // The DMA source always belongs to the unpublished owner, including the
  // contiguous-input case. Raw signed/unsigned native word bits are preserved.
  for (std::size_t row = 0; row < rows; ++row) {
    std::memcpy(cache->native_staging_words.data() + row * cols,
                values + static_cast<int64_t>(row) * ld, cols * sizeof(NativeT));
  }
  size_t token = 0;
  status = begin_synchronous_owner_lease(*ctx, token);
  if (status != RNS8_SUCCESS)
    return status;
  SynchronousLease lease(*ctx, token);
  status = hold_prepack_construction(*ctx, *cache, lease, token);
  if (status != RNS8_SUCCESS)
    return lease.finish(status);
  status = synchronous_dispatch_checkpoint(*ctx, "prepack_native_create");
  if (status != RNS8_SUCCESS)
    return lease.finish(status);
  void* native_device = cache->native_upload_owner.get();
  status = rns8::detail::hip_direct_copy_host_to_device(
      ctx->device_id, native_device, cache->native_staging_words.data(), native_bytes);
  if (status == RNS8_SUCCESS) {
    if constexpr (std::is_same_v<NativeT, int64_t>) {
      status = rns8::detail::rocwmma_prepack_b_native_i64_device(
          ctx->device_id, native_device, cache->device_data, cache->device_bytes, plan->desc.k,
          plan->desc.n, plan->desc.n, plan->prefix);
    } else {
      status = rns8::detail::rocwmma_prepack_b_native_u64_device(
          ctx->device_id, native_device, cache->device_data, cache->device_bytes, plan->desc.k,
          plan->desc.n, plan->desc.n, plan->prefix);
    }
  }

  if (status == RNS8_SUCCESS)
    status = synchronous_dispatch_checkpoint(*ctx, "prepack_native_publish");
  status = lease.finish(status);
  if (status != RNS8_SUCCESS) {
    return status;
  }
  const auto free_status = cache->native_upload_owner.reset();
  if (free_status != RNS8_SUCCESS) {
    return free_status;
  }
  std::vector<uint64_t>().swap(cache->native_staging_words);

  *out = cache_owner.release();
  return RNS8_SUCCESS;
}

} // namespace

rns8_status rns8_create_b_prepack_cache_i64(rns8_context* ctx, const rns8_plan* plan,
                                            const int64_t* values, int64_t ld,
                                            uint64_t source_version, rns8_prepack_cache** out) {
  return guard_api([&]() -> rns8_status {
    return create_b_prepack_cache_from_native(ctx, plan, values, ld, source_version,
                                              RNS8_BOUNDED_I64, "native_i64_rowmajor_v1", out);
  });
}

rns8_status rns8_create_b_prepack_cache_u64(rns8_context* ctx, const rns8_plan* plan,
                                            const uint64_t* values, int64_t ld,
                                            uint64_t source_version, rns8_prepack_cache** out) {
  return guard_api([&]() -> rns8_status {
    return create_b_prepack_cache_from_native(ctx, plan, values, ld, source_version,
                                              RNS8_BOUNDED_U64, "native_u64_rowmajor_v1", out);
  });
}

rns8_status rns8_get_prepack_cache_info(const rns8_prepack_cache* cache,
                                        rns8_prepack_cache_info* out) {
  return guard_api([&]() -> rns8_status {
    if (!cache || !out) {
      return RNS8_INVALID_ARGUMENT;
    }
    if (!rns8::detail::minimum_abi_header(out)) {
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
    const bool stable_source_version = cache->source_version != 0;
    out->reusable_prepack_cache_available = stable_source_version ? 1u : 0u;
    out->production_prepack_cache_available =
        stable_source_version && rocwmma_b_prepack_cache_production_supported(*cache) ? 1u : 0u;
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
    set_text(out->matrix_layout_version, sizeof(out->matrix_layout_version),
             cache->matrix_layout_version);
    set_text(out->operand_layout_version, sizeof(out->operand_layout_version),
             cache->operand_layout_version);
    set_text(out->cache_scope, sizeof(out->cache_scope),
             !stable_source_version ? "runtime_unstable_source_prepack_cache_rejected"
             : out->production_prepack_cache_available ? "runtime_production_b_prepack_cache"
                                                       : "runtime_reusable_b_prepack_cache");
    set_text(out->cache_key, sizeof(out->cache_key), cache->cache_key);
    set_text(
        out->detail, sizeof(out->detail),
        !stable_source_version
            ? "Runtime prepack cache metadata has source_version=0 and is not eligible for reusable execution or promotion."
        : out->production_prepack_cache_available
            ? "Created production reusable rocWMMA B prepack cache with source-version, device, layout, and plan-fingerprint key material."
            : "Created reusable accelerator prepack cache outside the production cache eligibility surface.");
    return RNS8_SUCCESS;
  });
}

rns8_status rns8_destroy_prepack_cache(rns8_prepack_cache* cache) {
  if (!cache) {
    return RNS8_SUCCESS;
  }
  if (cache->in_flight_readers)
    return RNS8_BUSY;
  const auto upload_status = cache->native_upload_owner.reset();
  if (upload_status != RNS8_SUCCESS)
    return upload_status;
  rns8_status status = RNS8_SUCCESS;
  if (cache->device_data) {
    status = cache->data_owner.reset_alias(cache->device_data, cache->device_bytes);
    if (status != RNS8_SUCCESS)
      return status;
    cache->device_data = nullptr;
    cache->device_bytes = 0;
  }
  delete cache;
  return status;
}
