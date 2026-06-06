rns8_status pack_transpose_centered(
    const int8_t* src,
    int8_t* dst,
    int src_rows,
    int src_cols,
    int src_ld,
    int dst_rows,
    int dst_cols,
    int dst_ld) {
  const hipError_t err = timed_hip_operation("hipblaslt_pack_transpose_centered", [&]() {
    const int code = rns8_hipblaslt_pack_transpose_centered_device(
        src, dst, src_rows, src_cols, src_ld, dst_rows, dst_cols, dst_ld);
    return code == static_cast<int>(hipSuccess) ? hipSuccess : static_cast<hipError_t>(code);
  });
  return err == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
}

std::size_t packed_b_plane_bytes_for(const padded_block_shape& shape) {
  return static_cast<std::size_t>(shape.n) * static_cast<std::size_t>(shape.k);
}

std::size_t packed_a_plane_bytes_for(const padded_block_shape& shape) {
  return static_cast<std::size_t>(shape.m) * static_cast<std::size_t>(shape.k);
}

rns8_status clear_a_prepack_cache(rns8_workspace& state) {
  rns8_status status = RNS8_SUCCESS;
  if (state.hipblaslt_a_prepack_cache) {
    status = hip_direct_free(state.hipblaslt_a_prepack_device_id, state.hipblaslt_a_prepack_cache);
  }
  state.hipblaslt_a_prepack_cache = nullptr;
  state.hipblaslt_a_prepack_cache_bytes = 0;
  state.hipblaslt_a_prepack_source_version = 0;
  state.hipblaslt_a_prepack_m = 0;
  state.hipblaslt_a_prepack_k = 0;
  state.hipblaslt_a_prepack_lda = 0;
  state.hipblaslt_a_prepack_prefix = 0;
  state.hipblaslt_a_prepack_device_id = -1;
  state.hipblaslt_a_prepack_current = false;
  return status;
}

rns8_status clear_b_prepack_cache(rns8_workspace& state) {
  rns8_status status = RNS8_SUCCESS;
  if (state.hipblaslt_b_prepack_cache) {
    status = hip_direct_free(state.hipblaslt_b_prepack_device_id, state.hipblaslt_b_prepack_cache);
  }
  state.hipblaslt_b_prepack_cache = nullptr;
  state.hipblaslt_b_prepack_cache_bytes = 0;
  state.hipblaslt_b_prepack_source_version = 0;
  state.hipblaslt_b_prepack_k = 0;
  state.hipblaslt_b_prepack_n = 0;
  state.hipblaslt_b_prepack_ldb = 0;
  state.hipblaslt_b_prepack_prefix = 0;
  state.hipblaslt_b_prepack_device_id = -1;
  state.hipblaslt_b_prepack_current = false;
  return status;
}

bool a_prepack_cache_matches(
    const rns8_workspace& state,
    int device_id,
    uint64_t source_version,
    int64_t m,
    int64_t k,
    int64_t lda,
    uint32_t prefix,
    std::size_t total_bytes) {
  return state.hipblaslt_a_prepack_current &&
         state.hipblaslt_a_prepack_cache != nullptr &&
         state.hipblaslt_a_prepack_cache_bytes == total_bytes &&
         state.hipblaslt_a_prepack_source_version == source_version &&
         state.hipblaslt_a_prepack_m == m &&
         state.hipblaslt_a_prepack_k == k &&
         state.hipblaslt_a_prepack_lda == lda &&
         state.hipblaslt_a_prepack_prefix == prefix &&
         state.hipblaslt_a_prepack_device_id == device_id;
}

bool b_prepack_cache_matches(
    const rns8_workspace& state,
    int device_id,
    uint64_t source_version,
    int64_t k,
    int64_t n,
    int64_t ldb,
    uint32_t prefix,
    std::size_t total_bytes) {
  return state.hipblaslt_b_prepack_current &&
         state.hipblaslt_b_prepack_cache != nullptr &&
         state.hipblaslt_b_prepack_cache_bytes == total_bytes &&
         state.hipblaslt_b_prepack_source_version == source_version &&
         state.hipblaslt_b_prepack_k == k &&
         state.hipblaslt_b_prepack_n == n &&
         state.hipblaslt_b_prepack_ldb == ldb &&
         state.hipblaslt_b_prepack_prefix == prefix &&
         state.hipblaslt_b_prepack_device_id == device_id;
}

rns8_status ensure_a_prepack_cache(
    int device_id,
    const int8_t* a_base,
    rns8_workspace& state,
    uint64_t source_version,
    int64_t m,
    int64_t k,
    int64_t lda,
    uint32_t prefix,
    const padded_block_shape& shape) {
  if (!a_base || prefix == 0 || k > static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const std::size_t plane_bytes = packed_a_plane_bytes_for(shape);
  if (plane_bytes == 0 || prefix > std::numeric_limits<std::size_t>::max() / plane_bytes) {
    return RNS8_RANGE_ERROR;
  }
  const std::size_t total_bytes = plane_bytes * static_cast<std::size_t>(prefix);
  if (a_prepack_cache_matches(state, device_id, source_version, m, k, lda, prefix, total_bytes)) {
    return RNS8_SUCCESS;
  }
  state.hipblaslt_a_prepack_current = false;
  if (state.hipblaslt_a_prepack_cache &&
      (state.hipblaslt_a_prepack_cache_bytes != total_bytes ||
       state.hipblaslt_a_prepack_device_id != device_id)) {
    const rns8_status clear_status = clear_a_prepack_cache(state);
    if (clear_status != RNS8_SUCCESS) {
      return clear_status;
    }
  }
  if (!state.hipblaslt_a_prepack_cache) {
    rns8_status alloc_status = hip_direct_allocate(device_id, total_bytes, &state.hipblaslt_a_prepack_cache);
    if (alloc_status != RNS8_SUCCESS) {
      state.hipblaslt_a_prepack_cache = nullptr;
      state.hipblaslt_a_prepack_cache_bytes = 0;
      return alloc_status;
    }
    state.hipblaslt_a_prepack_cache_bytes = total_bytes;
    state.hipblaslt_a_prepack_device_id = device_id;
  }

  auto* cache_base = static_cast<int8_t*>(state.hipblaslt_a_prepack_cache);
  for (uint32_t p = 0; p < prefix; ++p) {
    const std::size_t a_offset =
        static_cast<std::size_t>(p) * static_cast<std::size_t>(m) * static_cast<std::size_t>(lda);
    const rns8_status pack_status = pack_transpose_centered(
        a_base + a_offset,
        cache_base + static_cast<std::size_t>(p) * plane_bytes,
        static_cast<int>(m),
        static_cast<int>(k),
        static_cast<int>(lda),
        shape.k,
        shape.m,
        shape.k);
    if (pack_status != RNS8_SUCCESS) {
      state.hipblaslt_a_prepack_current = false;
      return pack_status;
    }
  }

  state.hipblaslt_a_prepack_source_version = source_version;
  state.hipblaslt_a_prepack_m = m;
  state.hipblaslt_a_prepack_k = k;
  state.hipblaslt_a_prepack_lda = lda;
  state.hipblaslt_a_prepack_prefix = prefix;
  state.hipblaslt_a_prepack_device_id = device_id;
  state.hipblaslt_a_prepack_current = true;
  return RNS8_SUCCESS;
}

rns8_status ensure_b_prepack_cache(
    int device_id,
    const int8_t* b_base,
    rns8_workspace& state,
    uint64_t source_version,
    int64_t k,
    int64_t n,
    int64_t ldb,
    uint32_t prefix,
    const padded_block_shape& shape) {
  if (!b_base || prefix == 0 || k > static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK)) {
    return RNS8_INVALID_ARGUMENT;
  }
  const std::size_t plane_bytes = packed_b_plane_bytes_for(shape);
  if (plane_bytes == 0 || prefix > std::numeric_limits<std::size_t>::max() / plane_bytes) {
    return RNS8_RANGE_ERROR;
  }
  const std::size_t total_bytes = plane_bytes * static_cast<std::size_t>(prefix);
  if (b_prepack_cache_matches(state, device_id, source_version, k, n, ldb, prefix, total_bytes)) {
    return RNS8_SUCCESS;
  }
  state.hipblaslt_b_prepack_current = false;
  if (state.hipblaslt_b_prepack_cache &&
      (state.hipblaslt_b_prepack_cache_bytes != total_bytes ||
       state.hipblaslt_b_prepack_device_id != device_id)) {
    const rns8_status clear_status = clear_b_prepack_cache(state);
    if (clear_status != RNS8_SUCCESS) {
      return clear_status;
    }
  }
  if (!state.hipblaslt_b_prepack_cache) {
    rns8_status alloc_status = hip_direct_allocate(device_id, total_bytes, &state.hipblaslt_b_prepack_cache);
    if (alloc_status != RNS8_SUCCESS) {
      state.hipblaslt_b_prepack_cache = nullptr;
      state.hipblaslt_b_prepack_cache_bytes = 0;
      return alloc_status;
    }
    state.hipblaslt_b_prepack_cache_bytes = total_bytes;
    state.hipblaslt_b_prepack_device_id = device_id;
  }

  auto* cache_base = static_cast<int8_t*>(state.hipblaslt_b_prepack_cache);
  for (uint32_t p = 0; p < prefix; ++p) {
    const std::size_t b_offset =
        static_cast<std::size_t>(p) * static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb);
    const rns8_status pack_status = pack_transpose_centered(
        b_base + b_offset,
        cache_base + static_cast<std::size_t>(p) * plane_bytes,
        static_cast<int>(k),
        static_cast<int>(n),
        static_cast<int>(ldb),
        shape.n,
        shape.k,
        shape.n);
    if (pack_status != RNS8_SUCCESS) {
      state.hipblaslt_b_prepack_current = false;
      return pack_status;
    }
  }

  state.hipblaslt_b_prepack_source_version = source_version;
  state.hipblaslt_b_prepack_k = k;
  state.hipblaslt_b_prepack_n = n;
  state.hipblaslt_b_prepack_ldb = ldb;
  state.hipblaslt_b_prepack_prefix = prefix;
  state.hipblaslt_b_prepack_device_id = device_id;
  state.hipblaslt_b_prepack_current = true;
  return RNS8_SUCCESS;
}

