TEST_CASE("sparse A 4:2 contract packs and expands canonical K groups") {
  rns8_sparse_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_FINITE_RING_U8;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8;
  desc.rows = 2;
  desc.expanded_k = 8;
  desc.group_size = 4;
  desc.nonzeros_per_group = 2;
  desc.finite_modulus = 251;

  uint64_t group_count = 0;
  uint64_t packed_value_count = 0;
  REQUIRE(rns8_sparse_a_4_to_2_layout_counts(&desc, &group_count, &packed_value_count) == RNS8_SUCCESS);
  CHECK(group_count == 4);
  CHECK(packed_value_count == 8);

  const std::vector<uint8_t> dense = {
      5, 0, 7, 0, 0, 9, 0, 11,
      0, 13, 17, 0, 19, 0, 0, 23,
  };
  std::vector<uint8_t> packed_values(static_cast<std::size_t>(packed_value_count), 0);
  std::vector<uint8_t> packed_indices(static_cast<std::size_t>(group_count), 0);
  REQUIRE(rns8_pack_sparse_a_4_to_2_u8(
              &desc, dense.data(), desc.expanded_k, packed_values.data(), packed_indices.data()) == RNS8_SUCCESS);
  CHECK(packed_values == std::vector<uint8_t>{5, 7, 9, 11, 13, 17, 19, 23});
  CHECK(packed_indices == std::vector<uint8_t>{8, 13, 9, 12});

  std::vector<uint8_t> expanded(dense.size(), 0xff);
  REQUIRE(rns8_expand_sparse_a_4_to_2_u8(
              &desc, packed_values.data(), packed_indices.data(), expanded.data(), desc.expanded_k) == RNS8_SUCCESS);
  CHECK(expanded == dense);
}

TEST_CASE("sparse A 4:2 contract rejects malformed descriptors and groups") {
  rns8_sparse_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_FINITE_RING_U8;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8;
  desc.rows = 1;
  desc.expanded_k = 4;
  desc.group_size = 4;
  desc.nonzeros_per_group = 2;
  desc.finite_modulus = 251;

  uint64_t group_count = 0;
  uint64_t packed_value_count = 0;
  auto bad_k = desc;
  bad_k.expanded_k = 6;
  CHECK(rns8_sparse_a_4_to_2_layout_counts(&bad_k, &group_count, &packed_value_count) == RNS8_INVALID_ARGUMENT);
  auto bad_signedness = desc;
  bad_signedness.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSPECIFIED;
  CHECK(
      rns8_sparse_a_4_to_2_layout_counts(&bad_signedness, &group_count, &packed_value_count) ==
      RNS8_INVALID_ARGUMENT);

  const std::vector<uint8_t> too_many = {1, 2, 3, 0};
  std::vector<uint8_t> packed_values(2, 0);
  std::vector<uint8_t> packed_indices(1, 0);
  CHECK(rns8_pack_sparse_a_4_to_2_u8(
            &desc, too_many.data(), desc.expanded_k, packed_values.data(), packed_indices.data()) ==
        RNS8_INVALID_ARGUMENT);

  const std::vector<uint8_t> too_few = {1, 0, 0, 0};
  CHECK(rns8_pack_sparse_a_4_to_2_u8(
            &desc, too_few.data(), desc.expanded_k, packed_values.data(), packed_indices.data()) ==
        RNS8_INVALID_ARGUMENT);

  packed_values = {7, 9};
  packed_indices = {0xf1};
  std::vector<uint8_t> dense(4, 0);
  CHECK(rns8_expand_sparse_a_4_to_2_u8(
            &desc, packed_values.data(), packed_indices.data(), dense.data(), desc.expanded_k) ==
        RNS8_INVALID_ARGUMENT);
}

TEST_CASE("resident sparse A storage stamps source version and cache identity") {
  rns8_context* ctx = create_cpu_context();
  rns8_sparse_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_FINITE_RING_U8;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8;
  desc.rows = 2;
  desc.expanded_k = 8;
  desc.group_size = 4;
  desc.nonzeros_per_group = 2;
  desc.finite_modulus = 251;

  rns8_sparse_matrix* sparse = nullptr;
  REQUIRE(rns8_create_sparse_matrix(ctx, &desc, &sparse) == RNS8_SUCCESS);
  REQUIRE(sparse != nullptr);

  const std::vector<uint8_t> dense = {
      5, 0, 7, 0, 0, 9, 0, 11,
      0, 13, 17, 0, 19, 0, 0, 23,
  };
  REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, sparse, dense.data(), desc.expanded_k, 42) == RNS8_SUCCESS);

  rns8_sparse_matrix_storage_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_sparse_matrix_storage_info(sparse, &info) == RNS8_SUCCESS);
  CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
  CHECK(info.semantics == RNS8_FINITE_RING_U8);
  CHECK(info.source_version == 42);
  CHECK(info.host_current == 1);
  CHECK(info.device_current == 0);
  CHECK(info.group_count == 4);
  CHECK(info.packed_value_count == 8);
  CHECK(std::string(info.layout_version) == "sparse_a_4_to_2_canonical_2bit_k_groups_v1");
  CHECK(std::string(info.cache_key).find("contract=sparse_a_4_to_2_structured_k_v1") != std::string::npos);
  CHECK(std::string(info.cache_key).find("source_version=42") != std::string::npos);

  std::vector<uint8_t> expanded(dense.size(), 0);
  REQUIRE(rns8_expand_sparse_a_4_to_2_matrix_u8(sparse, expanded.data(), desc.expanded_k) == RNS8_SUCCESS);
  CHECK(expanded == dense);

  rns8_destroy_sparse_matrix(sparse);
  rns8_destroy_context(ctx);
}

TEST_CASE("resident sparse finite-u8 CPU GEMM matches dense finite reference") {
  constexpr uint16_t modulus = 251;
  rns8_context* ctx = create_cpu_context();
  auto gemm = finite_gemm_desc(RNS8_FINITE_RING_U8, 2, 3, 8, modulus);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &gemm, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);

  auto a_desc = finite_matrix_desc(2, 8, RNS8_FINITE_RING_U8);
  auto b_desc = finite_matrix_desc(8, 3, RNS8_FINITE_RING_U8);
  auto c_desc = finite_matrix_desc(2, 3, RNS8_FINITE_RING_U8);
  rns8_matrix* dense_a = nullptr;
  rns8_matrix* dense_b = nullptr;
  rns8_matrix* dense_c = nullptr;
  rns8_matrix* sparse_c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &dense_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &dense_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &dense_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &sparse_c) == RNS8_SUCCESS);

  const std::vector<uint8_t> dense_a_values = {
      5, 0, 7, 0, 0, 9, 0, 11,
      0, 13, 17, 0, 19, 0, 0, 23,
  };
  const std::vector<uint8_t> dense_b_values = {
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12,
      13, 14, 15,
      16, 17, 18,
      19, 20, 21,
      22, 23, 24,
  };
  REQUIRE(rns8_pack_finite_u8(ctx, dense_a, modulus, dense_a_values.data(), 8, 11) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(ctx, dense_b, modulus, dense_b_values.data(), 3, 12) == RNS8_SUCCESS);

  rns8_sparse_matrix_desc sparse_desc{};
  sparse_desc.struct_size = sizeof(sparse_desc);
  sparse_desc.abi_version = RNS8_ABI_VERSION;
  sparse_desc.semantics = RNS8_FINITE_RING_U8;
  sparse_desc.bound_kind = RNS8_BOUND_NONE;
  sparse_desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  sparse_desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  sparse_desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  sparse_desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8;
  sparse_desc.rows = 2;
  sparse_desc.expanded_k = 8;
  sparse_desc.group_size = 4;
  sparse_desc.nonzeros_per_group = 2;
  sparse_desc.finite_modulus = modulus;
  rns8_sparse_matrix* sparse_a = nullptr;
  REQUIRE(rns8_create_sparse_matrix(ctx, &sparse_desc, &sparse_a) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, sparse_a, dense_a_values.data(), 8, 11) == RNS8_SUCCESS);

  REQUIRE(rns8_gemm_finite_u8(ctx, plan, modulus, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_finite_u8_sparse_a(ctx, plan, modulus, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
  std::vector<uint8_t> dense_out(6, 0);
  std::vector<uint8_t> sparse_out(6, 0);
  REQUIRE(rns8_export_finite_u8(ctx, plan, modulus, dense_c, dense_out.data(), 3) == RNS8_SUCCESS);
  REQUIRE(rns8_export_finite_u8(ctx, plan, modulus, sparse_c, sparse_out.data(), 3) == RNS8_SUCCESS);
  CHECK(sparse_out == dense_out);

  rns8_destroy_sparse_matrix(sparse_a);
  rns8_destroy_matrix(sparse_c);
  rns8_destroy_matrix(dense_c);
  rns8_destroy_matrix(dense_b);
  rns8_destroy_matrix(dense_a);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("resident sparse RNS CPU GEMM matches dense bounded reference") {
  rns8_context* ctx = create_cpu_context();
  auto gemm = bounded_gemm_desc(RNS8_BOUNDED_I64, 2, 2, 4);
  gemm.bound = 128;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &gemm, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);

  auto a_desc = bounded_matrix_desc(2, 4, RNS8_BOUNDED_I64);
  auto b_desc = bounded_matrix_desc(4, 2, RNS8_BOUNDED_I64);
  auto c_desc = bounded_matrix_desc(2, 2, RNS8_BOUNDED_I64);
  rns8_matrix* dense_a = nullptr;
  rns8_matrix* dense_b = nullptr;
  rns8_matrix* dense_c = nullptr;
  rns8_matrix* sparse_c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &dense_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &dense_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &dense_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &sparse_c) == RNS8_SUCCESS);

  const std::vector<int64_t> a_values = {
      1, 0, -2, 0,
      0, 3, 0, -4,
  };
  const std::vector<int64_t> b_values = {
      2, 5,
      -1, 7,
      3, -2,
      4, 1,
  };
  REQUIRE(rns8_pack_i64(ctx, dense_a, a_values.data(), 4, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, dense_b, b_values.data(), 2, 202) == RNS8_SUCCESS);

  rns8_sparse_matrix_desc sparse_desc{};
  sparse_desc.struct_size = sizeof(sparse_desc);
  sparse_desc.abi_version = RNS8_ABI_VERSION;
  sparse_desc.semantics = RNS8_BOUNDED_I64;
  sparse_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  sparse_desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  sparse_desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  sparse_desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  sparse_desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
  sparse_desc.rows = 2;
  sparse_desc.expanded_k = 4;
  sparse_desc.group_size = 4;
  sparse_desc.nonzeros_per_group = 2;
  sparse_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  rns8_sparse_matrix* sparse_a = nullptr;
  REQUIRE(rns8_create_sparse_matrix(ctx, &sparse_desc, &sparse_a) == RNS8_SUCCESS);
  REQUIRE(
      rns8_pack_sparse_a_4_to_2_matrix_u8(
          ctx,
          sparse_a,
          reinterpret_cast<const uint8_t*>(dense_a->residues.data()),
          dense_a->desc.cols,
          dense_a->source_version) == RNS8_SUCCESS);

  REQUIRE(rns8_gemm_rns(ctx, plan, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns_sparse_a(ctx, plan, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
  std::vector<int64_t> dense_out(4, 0);
  std::vector<int64_t> sparse_out(4, 0);
  REQUIRE(rns8_export_i64(ctx, plan, dense_c, dense_out.data(), 2) == RNS8_SUCCESS);
  REQUIRE(rns8_export_i64(ctx, plan, sparse_c, sparse_out.data(), 2) == RNS8_SUCCESS);
  CHECK(sparse_out == dense_out);
  CHECK(sparse_c->source_version == rns8::detail::api::gemm_output_source_version_values(101, 202));

  rns8_destroy_sparse_matrix(sparse_a);
  rns8_destroy_matrix(sparse_c);
  rns8_destroy_matrix(dense_c);
  rns8_destroy_matrix(dense_b);
  rns8_destroy_matrix(dense_a);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("resident sparse RNS CPU GEMM matches dense bounded unsigned reference") {
  rns8_context* ctx = create_cpu_context();
  auto gemm = bounded_gemm_desc(RNS8_BOUNDED_U64, 2, 2, 4);
  gemm.bound = 256;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &gemm, &plan) == RNS8_SUCCESS);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);

  auto a_desc = bounded_matrix_desc(2, 4, RNS8_BOUNDED_U64);
  auto b_desc = bounded_matrix_desc(4, 2, RNS8_BOUNDED_U64);
  auto c_desc = bounded_matrix_desc(2, 2, RNS8_BOUNDED_U64);
  rns8_matrix* dense_a = nullptr;
  rns8_matrix* dense_b = nullptr;
  rns8_matrix* dense_c = nullptr;
  rns8_matrix* sparse_c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &dense_a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &dense_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &dense_c) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &sparse_c) == RNS8_SUCCESS);

  const std::vector<uint64_t> a_values = {
      5, 0, 7, 0,
      0, 11, 0, 13,
  };
  const std::vector<uint64_t> b_values = {
      2, 5,
      1, 7,
      3, 2,
      4, 1,
  };
  REQUIRE(rns8_pack_u64(ctx, dense_a, a_values.data(), 4, 301) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, dense_b, b_values.data(), 2, 302) == RNS8_SUCCESS);

  rns8_sparse_matrix_desc sparse_desc{};
  sparse_desc.struct_size = sizeof(sparse_desc);
  sparse_desc.abi_version = RNS8_ABI_VERSION;
  sparse_desc.semantics = RNS8_BOUNDED_U64;
  sparse_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_UNSIGNED;
  sparse_desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  sparse_desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  sparse_desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  sparse_desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
  sparse_desc.rows = 2;
  sparse_desc.expanded_k = 4;
  sparse_desc.group_size = 4;
  sparse_desc.nonzeros_per_group = 2;
  sparse_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  rns8_sparse_matrix* sparse_a = nullptr;
  REQUIRE(rns8_create_sparse_matrix(ctx, &sparse_desc, &sparse_a) == RNS8_SUCCESS);
  REQUIRE(
      rns8_pack_sparse_a_4_to_2_matrix_u8(
          ctx,
          sparse_a,
          reinterpret_cast<const uint8_t*>(dense_a->residues.data()),
          dense_a->desc.cols,
          dense_a->source_version) == RNS8_SUCCESS);

  REQUIRE(rns8_gemm_rns(ctx, plan, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns_sparse_a(ctx, plan, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
  std::vector<uint64_t> dense_out(4, 0);
  std::vector<uint64_t> sparse_out(4, 0);
  REQUIRE(rns8_export_u64(ctx, plan, dense_c, dense_out.data(), 2) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(ctx, plan, sparse_c, sparse_out.data(), 2) == RNS8_SUCCESS);
  CHECK(sparse_out == dense_out);
  CHECK(sparse_c->source_version == rns8::detail::api::gemm_output_source_version_values(301, 302));

  rns8_destroy_sparse_matrix(sparse_a);
  rns8_destroy_matrix(sparse_c);
  rns8_destroy_matrix(dense_c);
  rns8_destroy_matrix(dense_b);
  rns8_destroy_matrix(dense_a);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}

TEST_CASE("resident sparse RNS CPU GEMM matches dense exact-wide reference") {
  rns8_context* ctx = create_cpu_context();
  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 4;
  constexpr uint32_t limb_count = 2;

  auto run_signed = [&]() {
    auto gemm = exact_desc(RNS8_EXACT_WIDE_SIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(ctx, &gemm, &plan) == RNS8_SUCCESS);
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);

    auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_SIGNED);
    auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_SIGNED);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_SIGNED);
    rns8_matrix* dense_a = nullptr;
    rns8_matrix* dense_b = nullptr;
    rns8_matrix* dense_c = nullptr;
    rns8_matrix* sparse_c = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &dense_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &dense_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &dense_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &sparse_c) == RNS8_SUCCESS);

    const std::vector<int64_t> a_values = {
        3, 0, -5, 0,
        0, 7, 0, -9,
    };
    const std::vector<int64_t> b_values = {
        2, -1,
        5, 3,
        -4, 6,
        7, -2,
    };
    REQUIRE(rns8_pack_i64(ctx, dense_a, a_values.data(), k, 401) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, dense_b, b_values.data(), n, 402) == RNS8_SUCCESS);

    rns8_sparse_matrix_desc sparse_desc{};
    sparse_desc.struct_size = sizeof(sparse_desc);
    sparse_desc.abi_version = RNS8_ABI_VERSION;
    sparse_desc.semantics = RNS8_EXACT_WIDE_SIGNED;
    sparse_desc.bound_kind = RNS8_BOUND_NONE;
    sparse_desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
    sparse_desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
    sparse_desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
    sparse_desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
    sparse_desc.rows = m;
    sparse_desc.expanded_k = k;
    sparse_desc.group_size = 4;
    sparse_desc.nonzeros_per_group = 2;
    sparse_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_sparse_matrix* sparse_a = nullptr;
    REQUIRE(rns8_create_sparse_matrix(ctx, &sparse_desc, &sparse_a) == RNS8_SUCCESS);
    REQUIRE(
        rns8_pack_sparse_a_4_to_2_matrix_u8(
            ctx,
            sparse_a,
            reinterpret_cast<const uint8_t*>(dense_a->residues.data()),
            dense_a->desc.cols,
            dense_a->source_version) == RNS8_SUCCESS);

    REQUIRE(rns8_gemm_rns(ctx, plan, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_sparse_a(ctx, plan, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
    std::vector<uint64_t> dense_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    std::vector<uint64_t> sparse_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, dense_c, dense_limbs.data(), n, limb_count) == RNS8_SUCCESS);
    REQUIRE(
        rns8_export_exact_wide_signed_limbs(ctx, plan, sparse_c, sparse_limbs.data(), n, limb_count) == RNS8_SUCCESS);
    CHECK(sparse_limbs == dense_limbs);
    CHECK(sparse_c->source_version == rns8::detail::api::gemm_output_source_version_values(401, 402));

    rns8_destroy_sparse_matrix(sparse_a);
    rns8_destroy_matrix(sparse_c);
    rns8_destroy_matrix(dense_c);
    rns8_destroy_matrix(dense_b);
    rns8_destroy_matrix(dense_a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  };

  auto run_unsigned = [&]() {
    auto gemm = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, m, n, k);
    rns8_plan* plan = nullptr;
    REQUIRE(rns8_create_plan(ctx, &gemm, &plan) == RNS8_SUCCESS);
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);

    auto a_desc = exact_matrix_desc(m, k, RNS8_EXACT_WIDE_UNSIGNED);
    auto b_desc = exact_matrix_desc(k, n, RNS8_EXACT_WIDE_UNSIGNED);
    auto c_desc = exact_matrix_desc(m, n, RNS8_EXACT_WIDE_UNSIGNED);
    rns8_matrix* dense_a = nullptr;
    rns8_matrix* dense_b = nullptr;
    rns8_matrix* dense_c = nullptr;
    rns8_matrix* sparse_c = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &a_desc, &dense_a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &b_desc, &dense_b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &dense_c) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &c_desc, &sparse_c) == RNS8_SUCCESS);

    const std::vector<uint64_t> a_values = {
        3, 0, 5, 0,
        0, 7, 0, 9,
    };
    const std::vector<uint64_t> b_values = {
        2, 1,
        5, 3,
        4, 6,
        7, 2,
    };
    REQUIRE(rns8_pack_u64(ctx, dense_a, a_values.data(), k, 501) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_u64(ctx, dense_b, b_values.data(), n, 502) == RNS8_SUCCESS);

    rns8_sparse_matrix_desc sparse_desc{};
    sparse_desc.struct_size = sizeof(sparse_desc);
    sparse_desc.abi_version = RNS8_ABI_VERSION;
    sparse_desc.semantics = RNS8_EXACT_WIDE_UNSIGNED;
    sparse_desc.bound_kind = RNS8_BOUND_NONE;
    sparse_desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
    sparse_desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
    sparse_desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
    sparse_desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
    sparse_desc.rows = m;
    sparse_desc.expanded_k = k;
    sparse_desc.group_size = 4;
    sparse_desc.nonzeros_per_group = 2;
    sparse_desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_sparse_matrix* sparse_a = nullptr;
    REQUIRE(rns8_create_sparse_matrix(ctx, &sparse_desc, &sparse_a) == RNS8_SUCCESS);
    REQUIRE(
        rns8_pack_sparse_a_4_to_2_matrix_u8(
            ctx,
            sparse_a,
            reinterpret_cast<const uint8_t*>(dense_a->residues.data()),
            dense_a->desc.cols,
            dense_a->source_version) == RNS8_SUCCESS);

    REQUIRE(rns8_gemm_rns(ctx, plan, dense_a, dense_b, dense_c, workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns_sparse_a(ctx, plan, sparse_a, dense_b, sparse_c, workspace) == RNS8_SUCCESS);
    std::vector<uint64_t> dense_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    std::vector<uint64_t> sparse_limbs(static_cast<std::size_t>(m * n * limb_count), 0);
    REQUIRE(
        rns8_export_exact_wide_unsigned_limbs(ctx, plan, dense_c, dense_limbs.data(), n, limb_count) == RNS8_SUCCESS);
    REQUIRE(
        rns8_export_exact_wide_unsigned_limbs(ctx, plan, sparse_c, sparse_limbs.data(), n, limb_count) == RNS8_SUCCESS);
    CHECK(sparse_limbs == dense_limbs);
    CHECK(sparse_c->source_version == rns8::detail::api::gemm_output_source_version_values(501, 502));

    rns8_destroy_sparse_matrix(sparse_a);
    rns8_destroy_matrix(sparse_c);
    rns8_destroy_matrix(dense_c);
    rns8_destroy_matrix(dense_b);
    rns8_destroy_matrix(dense_a);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
  };

  run_signed();
  run_unsigned();
  rns8_destroy_context(ctx);
}
