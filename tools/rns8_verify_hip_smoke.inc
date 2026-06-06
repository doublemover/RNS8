bool verify_hip_smoke() {
  if (!rns8::detail::hip_direct_compiled()) {
    std::cerr << "direct HIP backend was not compiled\n";
    return false;
  }

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  const rns8_status probe = rns8::detail::hip_direct_probe(0, info);
  if (probe != RNS8_SUCCESS) {
    std::cerr << "direct HIP probe failed: " << rns8_status_string(probe) << "\n";
    return false;
  }

  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 4;
  const uint16_t modulus = 251;
  const std::vector<int8_t> A = {1, -2, 3, -4, -5, 6, -7, 8};
  const std::vector<int8_t> B = {9, -10, 11, -12, -13, 14, 15, -16};
  std::vector<int8_t> cpu(4, 0);
  std::vector<int8_t> gpu(4, 0);
  rns8::detail::ring_gemm_modulus(A.data(), B.data(), cpu.data(), m, n, k, k, n, n, modulus);
  const rns8_status status = run_hip_resident_ring_gemm(A, B, gpu, m, n, k, k, n, n, modulus, 3, 4);
  if (status != RNS8_SUCCESS || cpu != gpu) {
    std::cerr << "direct HIP ring GEMM smoke failed: " << rns8_status_string(status) << "\n";
    return false;
  }

  const int64_t split_k = static_cast<int64_t>(RNS8_SAFE_INT32_K_BLOCK) + 1;
  const uint64_t split_bound = static_cast<uint64_t>(split_k) * 127u * 127u;
  std::vector<int64_t> split_a(static_cast<std::size_t>(split_k), 127);
  std::vector<int64_t> split_b(static_cast<std::size_t>(split_k), 127);

  rns8_context* cpu_ctx = create_cpu_context();
  rns8_context* hip_ctx = create_hip_context();
  rns8_context* wrap_ctx = create_wrap64_context();
  if (!cpu_ctx || !hip_ctx || !wrap_ctx) {
    std::cerr << "failed to create CPU, wrap64, or direct HIP context for smoke\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  {
    const int64_t rows = 2;
    const int64_t cols = 4;
    const int64_t ld = 5;
    const std::vector<int64_t> src = {
        0,
        1,
        -1,
        std::numeric_limits<int64_t>::max(),
        999,
        -std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min(),
        251,
        -251,
        999};
    auto desc = matrix_desc(rows, cols, RNS8_BOUNDED_I64, RNS8_BOUND_GLOBAL_MAX_ABS);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    rns8_status cpu_status = rns8_create_matrix(cpu_ctx, &desc, &cpu_matrix);
    rns8_status hip_status = rns8_create_matrix(hip_ctx, &desc, &hip_matrix);
    if (cpu_status == RNS8_SUCCESS) {
      cpu_status = rns8_pack_i64(cpu_ctx, cpu_matrix, src.data(), ld, 1);
    }
    if (hip_status == RNS8_SUCCESS) {
      hip_status = rns8_pack_i64(hip_ctx, hip_matrix, src.data(), ld, 1);
    }
    if (hip_status == RNS8_SUCCESS && hip_matrix) {
      hip_status = rns8::detail::hip_direct_copy_device_to_host(
          hip_matrix->hip_device_id,
          hip_matrix->residues.data(),
          hip_matrix->hip_residues,
          hip_matrix->hip_residue_bytes);
    }
    const bool equal = cpu_matrix && hip_matrix && cpu_matrix->residues == hip_matrix->residues;
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
    if (cpu_status != RNS8_SUCCESS || hip_status != RNS8_SUCCESS || !equal) {
      std::cerr << "direct HIP i64 residue pack smoke failed: CPU=" << rns8_status_string(cpu_status)
                << " HIP=" << rns8_status_string(hip_status) << "\n";
      rns8_destroy_context(wrap_ctx);
      rns8_destroy_context(hip_ctx);
      rns8_destroy_context(cpu_ctx);
      return false;
    }
  }

  {
    const int64_t rows = 2;
    const int64_t cols = 4;
    const int64_t ld = 5;
    const std::vector<uint64_t> src = {
        0,
        1,
        127,
        128,
        999,
        255,
        256,
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max() - 1,
        999};
    auto desc = matrix_desc(rows, cols, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
    rns8_matrix* cpu_matrix = nullptr;
    rns8_matrix* hip_matrix = nullptr;
    rns8_status cpu_status = rns8_create_matrix(cpu_ctx, &desc, &cpu_matrix);
    rns8_status hip_status = rns8_create_matrix(hip_ctx, &desc, &hip_matrix);
    if (cpu_status == RNS8_SUCCESS) {
      cpu_status = rns8_pack_u64(cpu_ctx, cpu_matrix, src.data(), ld, 1);
    }
    if (hip_status == RNS8_SUCCESS) {
      hip_status = rns8_pack_u64(hip_ctx, hip_matrix, src.data(), ld, 1);
    }
    if (hip_status == RNS8_SUCCESS && hip_matrix) {
      hip_status = rns8::detail::hip_direct_copy_device_to_host(
          hip_matrix->hip_device_id,
          hip_matrix->residues.data(),
          hip_matrix->hip_residues,
          hip_matrix->hip_residue_bytes);
    }
    const bool equal = cpu_matrix && hip_matrix && cpu_matrix->residues == hip_matrix->residues;
    rns8_destroy_matrix(hip_matrix);
    rns8_destroy_matrix(cpu_matrix);
    if (cpu_status != RNS8_SUCCESS || hip_status != RNS8_SUCCESS || !equal) {
      std::cerr << "direct HIP u64 residue pack smoke failed: CPU=" << rns8_status_string(cpu_status)
                << " HIP=" << rns8_status_string(hip_status) << "\n";
      rns8_destroy_context(wrap_ctx);
      rns8_destroy_context(hip_ctx);
      rns8_destroy_context(cpu_ctx);
      return false;
    }
  }

  int64_t cpu_split_c[1] = {};
  int64_t hip_split_c[1] = {};
  auto cpu_split_desc = signed_desc_for_backend(1, 1, split_k, split_bound, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_split_desc = signed_desc_for_backend(1, 1, split_k, split_bound, RNS8_BACKEND_HIP_DIRECT);
  const rns8_status cpu_split_status =
      rns8_gemm_i64_oneshot(cpu_ctx, &cpu_split_desc, split_a.data(), split_k, split_b.data(), 1, cpu_split_c, 1);
  const rns8_status hip_split_status =
      rns8_gemm_i64_oneshot(hip_ctx, &hip_split_desc, split_a.data(), split_k, split_b.data(), 1, hip_split_c, 1);
  if (cpu_split_status != RNS8_SUCCESS || hip_split_status != RNS8_SUCCESS || cpu_split_c[0] != hip_split_c[0] ||
      hip_split_c[0] != static_cast<int64_t>(split_bound)) {
    std::cerr << "direct HIP bounded i64 split smoke failed: CPU=" << rns8_status_string(cpu_split_status)
              << " HIP=" << rns8_status_string(hip_split_status) << "\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  const uint64_t u_a[] = {17, 3, 255, 9, 41, 5};
  const uint64_t u_b[] = {11, 7, 13, 19, 23, 29};
  uint64_t cpu_u_c[4] = {};
  uint64_t hip_u_c[4] = {};
  auto cpu_u_desc = unsigned_desc_for_backend(2, 2, 3, 20000, RNS8_BACKEND_CPU_REFERENCE);
  auto hip_u_desc = unsigned_desc_for_backend(2, 2, 3, 20000, RNS8_BACKEND_HIP_DIRECT);
  const rns8_status cpu_u_status = rns8_gemm_u64_oneshot(cpu_ctx, &cpu_u_desc, u_a, 3, u_b, 2, cpu_u_c, 2);
  const rns8_status hip_u_status = rns8_gemm_u64_oneshot(hip_ctx, &hip_u_desc, u_a, 3, u_b, 2, hip_u_c, 2);
  if (cpu_u_status != RNS8_SUCCESS || hip_u_status != RNS8_SUCCESS ||
      std::vector<uint64_t>(std::begin(cpu_u_c), std::end(cpu_u_c)) !=
          std::vector<uint64_t>(std::begin(hip_u_c), std::end(hip_u_c))) {
    std::cerr << "direct HIP bounded u64 smoke failed: CPU=" << rns8_status_string(cpu_u_status)
              << " HIP=" << rns8_status_string(hip_u_status) << "\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  const int64_t adaptive_m = 65;
  const int64_t adaptive_n = 65;
  const int64_t adaptive_k = 1;
  std::vector<uint64_t> adaptive_a(static_cast<std::size_t>(adaptive_m * adaptive_k));
  std::vector<uint64_t> adaptive_b(static_cast<std::size_t>(adaptive_k * adaptive_n));
  std::vector<uint64_t> cpu_adaptive_c(static_cast<std::size_t>(adaptive_m * adaptive_n), 0);
  std::vector<uint64_t> hip_adaptive_c(static_cast<std::size_t>(adaptive_m * adaptive_n), 0);
  for (int64_t row = 0; row < adaptive_m; ++row) {
    adaptive_a[static_cast<std::size_t>(row)] = row < 64 ? 1 : 1000000;
  }
  for (int64_t col = 0; col < adaptive_n; ++col) {
    adaptive_b[static_cast<std::size_t>(col)] = col < 64 ? 7 : 1000;
  }
  const std::vector<uint64_t> adaptive_bounds = {7, 1000, 7000000, 1000000000};
  auto cpu_adaptive_desc =
      unsigned_desc_for_backend(adaptive_m, adaptive_n, adaptive_k, 0, RNS8_BACKEND_CPU_REFERENCE);
  cpu_adaptive_desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  cpu_adaptive_desc.tile_m = 64;
  cpu_adaptive_desc.tile_n = 64;
  cpu_adaptive_desc.tile_bounds = adaptive_bounds.data();
  cpu_adaptive_desc.tile_bounds_count = adaptive_bounds.size();
  auto hip_adaptive_desc = cpu_adaptive_desc;
  hip_adaptive_desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
  const rns8_status cpu_adaptive_status = rns8_gemm_u64_oneshot(
      cpu_ctx,
      &cpu_adaptive_desc,
      adaptive_a.data(),
      adaptive_k,
      adaptive_b.data(),
      adaptive_n,
      cpu_adaptive_c.data(),
      adaptive_n);
  const rns8_status hip_adaptive_status = rns8_gemm_u64_oneshot(
      hip_ctx,
      &hip_adaptive_desc,
      adaptive_a.data(),
      adaptive_k,
      adaptive_b.data(),
      adaptive_n,
      hip_adaptive_c.data(),
      adaptive_n);
  if (cpu_adaptive_status != RNS8_SUCCESS || hip_adaptive_status != RNS8_SUCCESS ||
      cpu_adaptive_c != hip_adaptive_c) {
    std::cerr << "direct HIP per-tile bounded u64 smoke failed: CPU="
              << rns8_status_string(cpu_adaptive_status) << " HIP=" << rns8_status_string(hip_adaptive_status)
              << "\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  {
    const int64_t finite_m = 2;
    const int64_t finite_n = 3;
    const int64_t finite_k = 4;
    const int64_t finite_ldc = 4;
    const std::vector<uint8_t> finite_a = {254, 128, 7, 3, 5, 250, 251, 1};
    const std::vector<uint8_t> finite_b = {2, 3, 4, 250, 11, 9, 13, 17, 19, 23, 29, 31};
    std::vector<uint8_t> cpu_finite_c(static_cast<std::size_t>(finite_m * finite_ldc), 0xcc);
    std::vector<uint8_t> hip_finite_c(static_cast<std::size_t>(finite_m * finite_ldc), 0xcc);
    auto cpu_finite_desc =
        finite_desc_for_backend(finite_m, finite_n, finite_k, RNS8_FINITE_RING_U8, RNS8_BACKEND_CPU_REFERENCE);
    auto hip_finite_desc =
        finite_desc_for_backend(finite_m, finite_n, finite_k, RNS8_FINITE_RING_U8, RNS8_BACKEND_HIP_DIRECT);
    const rns8_status cpu_finite_status = rns8_gemm_finite_ring_u8_oneshot(
        cpu_ctx,
        &cpu_finite_desc,
        255,
        finite_a.data(),
        finite_k,
        finite_b.data(),
        finite_n,
        cpu_finite_c.data(),
        finite_ldc);
    const rns8_status hip_finite_status = rns8_gemm_finite_ring_u8_oneshot(
        hip_ctx,
        &hip_finite_desc,
        255,
        finite_a.data(),
        finite_k,
        finite_b.data(),
        finite_n,
        hip_finite_c.data(),
        finite_ldc);
    if (cpu_finite_status != RNS8_SUCCESS || hip_finite_status != RNS8_SUCCESS ||
        cpu_finite_c != hip_finite_c) {
      std::cerr << "direct HIP finite ring u8 smoke failed: CPU=" << rns8_status_string(cpu_finite_status)
                << " HIP=" << rns8_status_string(hip_finite_status) << "\n";
      rns8_destroy_context(wrap_ctx);
      rns8_destroy_context(hip_ctx);
      rns8_destroy_context(cpu_ctx);
      return false;
    }

    rns8_plan* cpu_finite_plan = nullptr;
    rns8_plan* hip_finite_plan = nullptr;
    rns8_workspace* cpu_finite_workspace = nullptr;
    rns8_workspace* hip_finite_workspace = nullptr;
    rns8_matrix* cpu_finite_a = nullptr;
    rns8_matrix* cpu_finite_b = nullptr;
    rns8_matrix* cpu_finite_out = nullptr;
    rns8_matrix* hip_finite_a = nullptr;
    rns8_matrix* hip_finite_b = nullptr;
    rns8_matrix* hip_finite_out = nullptr;
    auto cleanup_finite_resident = [&]() {
      rns8_destroy_matrix(hip_finite_out);
      rns8_destroy_matrix(hip_finite_b);
      rns8_destroy_matrix(hip_finite_a);
      rns8_destroy_matrix(cpu_finite_out);
      rns8_destroy_matrix(cpu_finite_b);
      rns8_destroy_matrix(cpu_finite_a);
      rns8_destroy_workspace(hip_finite_workspace);
      rns8_destroy_workspace(cpu_finite_workspace);
      rns8_destroy_plan(hip_finite_plan);
      rns8_destroy_plan(cpu_finite_plan);
    };
    const rns8_matrix_desc finite_a_desc =
        matrix_desc(finite_m, finite_k, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
    const rns8_matrix_desc finite_b_desc =
        matrix_desc(finite_k, finite_n, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
    const rns8_matrix_desc finite_c_desc =
        matrix_desc(finite_m, finite_n, RNS8_FINITE_RING_U8, RNS8_BOUND_NONE);
    std::vector<uint8_t> cpu_finite_resident_c(static_cast<std::size_t>(finite_m * finite_ldc), 0xdd);
    std::vector<uint8_t> hip_finite_resident_c(static_cast<std::size_t>(finite_m * finite_ldc), 0xdd);
    rns8_status finite_resident_status = rns8_create_plan(cpu_ctx, &cpu_finite_desc, &cpu_finite_plan);
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_create_plan(hip_ctx, &hip_finite_desc, &hip_finite_plan);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_create_workspace(cpu_ctx, cpu_finite_plan, &cpu_finite_workspace);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_create_workspace(hip_ctx, hip_finite_plan, &hip_finite_workspace);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_create_matrix(cpu_ctx, &finite_a_desc, &cpu_finite_a);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_create_matrix(cpu_ctx, &finite_b_desc, &cpu_finite_b);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_create_matrix(cpu_ctx, &finite_c_desc, &cpu_finite_out);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_create_matrix(hip_ctx, &finite_a_desc, &hip_finite_a);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_create_matrix(hip_ctx, &finite_b_desc, &hip_finite_b);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_create_matrix(hip_ctx, &finite_c_desc, &hip_finite_out);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_pack_finite_u8(cpu_ctx, cpu_finite_a, 255, finite_a.data(), finite_k, 1);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_pack_finite_u8(cpu_ctx, cpu_finite_b, 255, finite_b.data(), finite_n, 2);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status =
          rns8_gemm_finite_u8(cpu_ctx, cpu_finite_plan, 255, cpu_finite_a, cpu_finite_b, cpu_finite_out,
                              cpu_finite_workspace);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status =
          rns8_export_finite_u8(cpu_ctx, cpu_finite_plan, 255, cpu_finite_out, cpu_finite_resident_c.data(),
                                finite_ldc);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_pack_finite_u8(hip_ctx, hip_finite_a, 255, finite_a.data(), finite_k, 1);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status = rns8_pack_finite_u8(hip_ctx, hip_finite_b, 255, finite_b.data(), finite_n, 2);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status =
          rns8_gemm_finite_u8(hip_ctx, hip_finite_plan, 255, hip_finite_a, hip_finite_b, hip_finite_out,
                              hip_finite_workspace);
    }
    if (finite_resident_status == RNS8_SUCCESS) {
      finite_resident_status =
          rns8_export_finite_u8(hip_ctx, hip_finite_plan, 255, hip_finite_out, hip_finite_resident_c.data(),
                                finite_ldc);
    }
    if (finite_resident_status != RNS8_SUCCESS || cpu_finite_resident_c != hip_finite_resident_c) {
      std::cerr << "direct HIP resident finite ring u8 smoke failed: "
                << rns8_status_string(finite_resident_status) << "\n";
      cleanup_finite_resident();
      rns8_destroy_context(wrap_ctx);
      rns8_destroy_context(hip_ctx);
      rns8_destroy_context(cpu_ctx);
      return false;
    }
    cleanup_finite_resident();
  }

  const int64_t wrap_m = 2;
  const int64_t wrap_n = 3;
  const int64_t wrap_k = 4;
  const int64_t wrap_ldc = 4;
  const std::vector<uint64_t> wrap_a = {
      0,
      1,
      std::numeric_limits<uint64_t>::max(),
      0x8080808080808080ull,
      255,
      256,
      std::numeric_limits<uint64_t>::max() - 1,
      0x0102030405060708ull};
  const std::vector<uint64_t> wrap_b = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      29,
      0x8080808080808080ull,
      31,
      37,
      41,
      43,
      47,
      53,
      59};
  std::vector<uint64_t> cpu_wrap_c(static_cast<std::size_t>(wrap_m * wrap_ldc), 0xfeedfacefeedfaceull);
  std::vector<uint64_t> hip_wrap_c(static_cast<std::size_t>(wrap_m * wrap_ldc), 0xfeedfacefeedfaceull);
  auto cpu_wrap_desc = wrap_desc_for_backend(wrap_m, wrap_n, wrap_k, RNS8_BACKEND_WRAP64_BYTE_LIMB);
  auto hip_wrap_desc = wrap_desc_for_backend(wrap_m, wrap_n, wrap_k, RNS8_BACKEND_HIP_DIRECT);
  const rns8_status cpu_wrap_status = rns8_gemm_wrap_u64_oneshot(
      wrap_ctx, &cpu_wrap_desc, wrap_a.data(), wrap_k, wrap_b.data(), wrap_n, cpu_wrap_c.data(), wrap_ldc);
  const rns8_status hip_wrap_status = rns8_gemm_wrap_u64_oneshot(
      hip_ctx, &hip_wrap_desc, wrap_a.data(), wrap_k, wrap_b.data(), wrap_n, hip_wrap_c.data(), wrap_ldc);
  if (cpu_wrap_status != RNS8_SUCCESS || hip_wrap_status != RNS8_SUCCESS || cpu_wrap_c != hip_wrap_c) {
    std::cerr << "direct HIP wrap64 byte-limb smoke failed: CPU=" << rns8_status_string(cpu_wrap_status)
              << " HIP=" << rns8_status_string(hip_wrap_status) << "\n";
    rns8_destroy_context(wrap_ctx);
    rns8_destroy_context(hip_ctx);
    rns8_destroy_context(cpu_ctx);
    return false;
  }

  rns8_destroy_context(wrap_ctx);
  rns8_destroy_context(hip_ctx);
  rns8_destroy_context(cpu_ctx);
  return true;
}

}  // namespace
