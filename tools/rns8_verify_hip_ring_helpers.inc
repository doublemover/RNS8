rns8_status run_hip_resident_ring_gemm(
    const std::vector<int8_t>& A,
    const std::vector<int8_t>& B,
    std::vector<int8_t>& C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus,
    uint32_t modulus_index,
    uint32_t selected_prefix) {
  const std::size_t a_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(lda) * sizeof(int8_t);
  const std::size_t b_bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(ldb) * sizeof(int8_t);
  const std::size_t c_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(ldc) * sizeof(int8_t);
  if (A.size() * sizeof(int8_t) < a_bytes || B.size() * sizeof(int8_t) < b_bytes ||
      C.size() * sizeof(int8_t) < c_bytes) {
    return RNS8_INVALID_ARGUMENT;
  }

  void* device_a = nullptr;
  void* device_b = nullptr;
  void* device_c = nullptr;
  rns8_status status = rns8::detail::hip_direct_allocate(0, a_bytes, &device_a);
  if (status == RNS8_SUCCESS) {
    status = rns8::detail::hip_direct_allocate(0, b_bytes, &device_b);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8::detail::hip_direct_allocate(0, c_bytes, &device_c);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8::detail::hip_direct_copy_host_to_device(0, device_a, A.data(), a_bytes);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8::detail::hip_direct_copy_host_to_device(0, device_b, B.data(), b_bytes);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8::detail::hip_direct_copy_host_to_device(0, device_c, C.data(), c_bytes);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8::detail::hip_direct_ring_gemm_i8_device(
        0,
        device_a,
        device_b,
        device_c,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        modulus,
        modulus_index,
        selected_prefix);
  }
  if (status == RNS8_SUCCESS) {
    status = rns8::detail::hip_direct_copy_device_to_host(0, C.data(), device_c, c_bytes);
  }
  const rns8_status free_c = rns8::detail::hip_direct_free(0, device_c);
  const rns8_status free_b = rns8::detail::hip_direct_free(0, device_b);
  const rns8_status free_a = rns8::detail::hip_direct_free(0, device_a);
  if (status == RNS8_SUCCESS && free_c != RNS8_SUCCESS) status = free_c;
  if (status == RNS8_SUCCESS && free_b != RNS8_SUCCESS) status = free_b;
  if (status == RNS8_SUCCESS && free_a != RNS8_SUCCESS) status = free_a;
  return status;
}

rns8_gemm_desc signed_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc signed_desc_for_backend(int64_t m, int64_t n, int64_t k, uint64_t bound, rns8_backend_kind backend) {
  auto desc = signed_desc(m, n, k, bound);
  desc.requested_backend = backend;
  return desc;
}

rns8_gemm_desc unsigned_desc(int64_t m, int64_t n, int64_t k, uint64_t bound) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_U64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  desc.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.bound = bound;
  return desc;
}

rns8_gemm_desc unsigned_desc_for_backend(
    int64_t m,
    int64_t n,
    int64_t k,
    uint64_t bound,
    rns8_backend_kind backend) {
  auto desc = unsigned_desc(m, n, k, bound);
  desc.requested_backend = backend;
  return desc;
}

rns8_gemm_desc wrap_desc_for_backend(int64_t m, int64_t n, int64_t k, rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  return desc;
}

rns8_gemm_desc finite_desc_for_backend(
    int64_t m,
    int64_t n,
    int64_t k,
    rns8_semantics semantics,
    rns8_backend_kind backend) {
  rns8_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = semantics;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.requested_backend = backend;
  desc.m = m;
  desc.n = n;
  desc.k = k;
  desc.max_prefix = 0;
  desc.finite_modulus = 255;
  return desc;
}

rns8_matrix_desc matrix_desc(int64_t rows, int64_t cols, rns8_semantics semantics, rns8_bound_kind bound_kind) {
  rns8_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.rows = rows;
  desc.cols = cols;
  desc.logical_ld = cols;
  desc.semantics = semantics;
  desc.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  desc.bound_kind = bound_kind;
  desc.tile_m = 128;
  desc.tile_n = 128;
  desc.max_prefix = semantics == RNS8_WRAP_U64_MOD_2_64 || semantics == RNS8_FINITE_RING_U8 ||
                            semantics == RNS8_FINITE_FIELD_U8
                        ? 0
                        : RNS8_DEFAULT_BOUNDED_PREFIX;
  return desc;
}

using rns8::detail::cpp_int;

cpp_int abs_cpp(cpp_int value) {
  return value < 0 ? -value : value;
}

cpp_int i64_min_cpp() {
  return -cpp_int(std::numeric_limits<int64_t>::max()) - 1;
}

