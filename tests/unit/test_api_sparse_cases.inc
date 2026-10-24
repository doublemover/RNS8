TEST_CASE("sparse operation predicate is target and full-address specific without device work", "[api][sparse]") {
  using rns8::detail::amdgpu_builtins_sparse_a_contract;
  for (const char* target : {"gfx942", "gfx942:sramecc+:xnack-", "gfx1200", "gfx1201"}) {
    CHECK(amdgpu_builtins_sparse_a_contract(target, 17, 19, 32, 23, 24, 20, false) == RNS8_SUCCESS);
    for (uint32_t modulus : {127u, 163u, 167u, 251u, 255u, 256u})
      CHECK(amdgpu_builtins_sparse_a_contract(target, 17, 19, 32, 23, 24, modulus, true) == RNS8_SUCCESS);
  }
  for (const char* target : {"", "gfx1100", "gfx1101", "gfx90a", "gfx950", "gfx942extra", "gfx12000"})
    CHECK(amdgpu_builtins_sparse_a_contract(target, 1, 1, 4, 1, 1, 1, false) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(amdgpu_builtins_sparse_a_contract(nullptr, 1, 1, 4, 1, 1, 1, false) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 1, 1, 3, 1, 1, 1, false) == RNS8_INVALID_ARGUMENT);
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 1, 2, 4, 1, 2, 1, false) == RNS8_INVALID_ARGUMENT);
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 1, 1, 4, 1, 1, 0, false) == RNS8_INVALID_ARGUMENT);
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 1, 1, 4, 1, 1, 21, false) == RNS8_INVALID_ARGUMENT);
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 1, 1, 4, 1, 1, 257, true) == RNS8_INVALID_ARGUMENT);
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 1, 1, RNS8_SAFE_INT32_K_BLOCK, 1, 1, 1, false) == RNS8_SUCCESS);
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 1, 1, RNS8_SAFE_INT32_K_BLOCK + 4, 1, 1, 1, false) == RNS8_RANGE_ERROR);
  // Each individual dimension fits int; the full packed/plane extent does not.
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 65536, 1, 65536, 1, 1, 2, false) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 1, 1, 4, INT32_MAX, 1, 1, false) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(amdgpu_builtins_sparse_a_contract("gfx942", 2, 1, 4, 1, INT32_MAX, 251, true) == RNS8_UNSUPPORTED_BACKEND);
}

TEST_CASE("sparse host planes reject unrepresentable strides before access") {
  using namespace rns8::detail;
  AddressDomain domain;
  constexpr uint64_t limit = std::numeric_limits<std::ptrdiff_t>::max();
  REQUIRE(checked_host_plane_address(1, 4, INT64_MAX, 1, 1, domain));
  REQUIRE(domain.byte_extent == 4);
  REQUIRE_FALSE(checked_host_plane_address(1, 4, INT64_MAX, 3, 1, domain));
  REQUIRE_FALSE(checked_host_plane_address(UINT64_MAX, 1, 1, 2, 1, domain));
  REQUIRE(checked_host_plane_address(1, 1, limit - 1, 2, 1, domain));
  REQUIRE(domain.byte_extent == limit);
  REQUIRE_FALSE(checked_host_plane_address(1, 1, limit, 2, 1, domain));
  REQUIRE(checked_host_plane_address(1, 1, limit / 8 - 1, 2, 8, domain));
  REQUIRE_FALSE(checked_host_plane_address(1, 1, limit / 8, 2, 8, domain));
  for (uint64_t planes : {1u, 3u, 20u})
    for (uint64_t rows : {1u, 2u, 7u}) {
      REQUIRE(checked_host_plane_address(rows, 4, 7, planes, 1, domain));
      REQUIRE(domain.byte_extent == (planes * rows - 1) * 7 + 4);
    }
  rns8_sparse_matrix_desc desc{};
  desc.struct_size = sizeof(desc); desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_EXACT_WIDE_UNSIGNED;
  desc.bound_kind = RNS8_BOUND_NONE;
  desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
  desc.rows = 1; desc.expanded_k = 4; desc.group_size = 4; desc.nonzeros_per_group = 2;
  desc.max_prefix = 3;
  uint8_t dense[]{1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0};
  std::vector<uint8_t> values(6, 123), indices(3, 123), output(12, 123);
  REQUIRE(rns8_pack_sparse_a_4_to_2_u8(&desc, dense, INT64_MAX, values.data(), indices.data()) == RNS8_INVALID_ARGUMENT);
  REQUIRE(values == std::vector<uint8_t>(6, 123));
  REQUIRE(indices == std::vector<uint8_t>(3, 123));
  REQUIRE(rns8_expand_sparse_a_4_to_2_u8(&desc, values.data(), indices.data(), output.data(), INT64_MAX) == RNS8_INVALID_ARGUMENT);
  REQUIRE(output == std::vector<uint8_t>(12, 123));
  auto* ctx = create_cpu_context();
  rns8_sparse_matrix* matrix = nullptr;
  REQUIRE(rns8_create_sparse_matrix(ctx, &desc, &matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, matrix, dense, 4, 7) == RNS8_SUCCESS);
  const auto saved_values = matrix->packed_values;
  const auto saved_bound = matrix->exact_max_magnitude;
  REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, matrix, dense, INT64_MAX, 8) == RNS8_INVALID_ARGUMENT);
  REQUIRE(matrix->packed_values == saved_values);
  REQUIRE(matrix->exact_max_magnitude == saved_bound);
  REQUIRE(matrix->source_version == 7);
  REQUIRE(matrix->host_current);
  REQUIRE(rns8_destroy_sparse_matrix(matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  desc.max_prefix = 1;
  REQUIRE(rns8_pack_sparse_a_4_to_2_u8(&desc, dense, INT64_MAX, values.data(), indices.data()) == RNS8_SUCCESS);
  REQUIRE(rns8_expand_sparse_a_4_to_2_u8(&desc, values.data(), indices.data(), output.data(), INT64_MAX) == RNS8_SUCCESS);
  REQUIRE(std::equal(output.begin(), output.begin() + 4, dense));
  desc.rows = 2; desc.max_prefix = 20;
  std::vector<uint8_t> padded(20 * 2 * 7, 123), expanded(padded.size(), 123);
  values.resize(20 * 2 * 2); indices.resize(20 * 2);
  for (std::size_t row = 0; row < 40; ++row)
    std::copy(dense, dense + 4, padded.begin() + row * 7);
  REQUIRE(rns8_pack_sparse_a_4_to_2_u8(&desc, padded.data(), 7, values.data(), indices.data()) == RNS8_SUCCESS);
  REQUIRE(rns8_expand_sparse_a_4_to_2_u8(&desc, values.data(), indices.data(), expanded.data(), 7) == RNS8_SUCCESS);
  REQUIRE(expanded == padded);
  desc.rows = INT64_MAX;
  uint64_t groups = 77, packed = 88;
  REQUIRE(rns8_sparse_a_4_to_2_layout_counts(&desc, &groups, &packed) == RNS8_INVALID_ARGUMENT);
  REQUIRE(groups == 77);
  REQUIRE(packed == 88);
}

TEST_CASE("unsupported sparse target preserves an existing public output", "[api][sparse][hip]") {
  if (!rns8::detail::amdgpu_builtins_compiled()) SKIP("AMDGPU builtins not compiled");
  rns8_context_options options{};
  options.struct_size = sizeof(options); options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_AMDGPU_BUILTINS;
  rns8_context* raw_ctx = nullptr;
  const auto created = rns8_create_context(0, &options, &raw_ctx);
  if (created == RNS8_UNSUPPORTED_BACKEND || created == RNS8_UNSUPPORTED_ARCH)
    SKIP("no applicable AMDGPU builtin device");
  REQUIRE(created == RNS8_SUCCESS);
  std::unique_ptr<rns8_context, decltype(&rns8_destroy_context)> ctx(raw_ctx, rns8_destroy_context);
  const auto target_support = rns8::detail::amdgpu_builtins_sparse_a_contract(
      ctx->device_info.gcn_arch, 1, 1, 4, 1, 1, 1, false);
  if (target_support == RNS8_SUCCESS) SKIP("this device supports the sparse instruction; negative target case is inapplicable");
  REQUIRE(target_support == RNS8_UNSUPPORTED_BACKEND);
  for (bool finite : {false, true}) {
    auto desc = finite ? finite_gemm_desc(RNS8_FINITE_RING_U8, 1, 1, 4, 251)
                       : bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 4);
    desc.requested_backend = RNS8_BACKEND_AMDGPU_BUILTINS;
    if (!finite) desc.bound = 1024;
    rns8_plan* raw_plan = nullptr;
    REQUIRE(rns8_create_plan(ctx.get(), &desc, &raw_plan) == RNS8_SUCCESS);
    std::unique_ptr<rns8_plan, decltype(&rns8_destroy_plan)> plan(raw_plan, rns8_destroy_plan);
    rns8_workspace* raw_workspace = nullptr;
    REQUIRE(rns8_create_workspace(ctx.get(), plan.get(), &raw_workspace) == RNS8_SUCCESS);
    std::unique_ptr<rns8_workspace, decltype(&rns8_destroy_workspace)> workspace(raw_workspace, rns8_destroy_workspace);
    auto bd = finite ? finite_matrix_desc(4, 1, RNS8_FINITE_RING_U8) : bounded_matrix_desc(4, 1, RNS8_BOUNDED_I64);
    auto cd = finite ? finite_matrix_desc(1, 1, RNS8_FINITE_RING_U8) : bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
    rns8_matrix *raw_b = nullptr, *raw_c = nullptr;
    REQUIRE(rns8_create_matrix(ctx.get(), &bd, &raw_b) == RNS8_SUCCESS);
    std::unique_ptr<rns8_matrix, decltype(&rns8_destroy_matrix)> b(raw_b, rns8_destroy_matrix);
    REQUIRE(rns8_create_matrix(ctx.get(), &cd, &raw_c) == RNS8_SUCCESS);
    std::unique_ptr<rns8_matrix, decltype(&rns8_destroy_matrix)> c(raw_c, rns8_destroy_matrix);
    const std::vector<uint8_t> bytes_b{3, 4, 5, 6}, bytes_c{42};
    const std::vector<int64_t> integers_b{3, 4, 5, 6}, integers_c{42};
    if (finite) {
      REQUIRE(rns8_pack_finite_u8(ctx.get(), b.get(), 251, bytes_b.data(), 1, 17) == RNS8_SUCCESS);
      REQUIRE(rns8_pack_finite_u8(ctx.get(), c.get(), 251, bytes_c.data(), 1, 99) == RNS8_SUCCESS);
    } else {
      REQUIRE(rns8_pack_i64(ctx.get(), b.get(), integers_b.data(), 1, 17) == RNS8_SUCCESS);
      REQUIRE(rns8_pack_i64(ctx.get(), c.get(), integers_c.data(), 1, 99) == RNS8_SUCCESS);
    }
    rns8_sparse_matrix_desc sd{};
    sd.struct_size = sizeof(sd); sd.abi_version = RNS8_ABI_VERSION;
    sd.semantics = desc.semantics; sd.bound_kind = desc.bound_kind;
    sd.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K; sd.sparse_operand = RNS8_SPARSE_OPERAND_A;
    sd.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
    sd.value_signedness = finite ? RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8 : RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
    sd.rows = 1; sd.expanded_k = 4; sd.group_size = 4; sd.nonzeros_per_group = 2;
    sd.max_prefix = finite ? 0 : desc.max_prefix; sd.finite_modulus = finite ? 251 : 0;
    rns8_sparse_matrix* raw_a = nullptr;
    REQUIRE(rns8_create_sparse_matrix(ctx.get(), &sd, &raw_a) == RNS8_SUCCESS);
    std::unique_ptr<rns8_sparse_matrix, decltype(&rns8_destroy_sparse_matrix)> a(raw_a, rns8_destroy_sparse_matrix);
    std::vector<uint8_t> planes((finite ? 1 : sd.max_prefix) * 4);
    for (std::size_t p = 0; p < planes.size() / 4; ++p) { planes[4*p] = 1; planes[4*p+2] = 2; }
    REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx.get(), a.get(), planes.data(), 4, 23) == RNS8_SUCCESS);
    const auto generation = c->content_generation, source = c->source_version;
    const auto host_current = c->host_residues_current, device_current = c->device_residues_current;
    const auto proof = c->exact_max_magnitude; const auto range_prefix = c->exact_range_prefix;
    const auto old_host = c->residues;
    const auto status = finite ? rns8_gemm_finite_u8_sparse_a(ctx.get(), plan.get(), 251, a.get(), b.get(), c.get(), workspace.get())
                               : rns8_gemm_rns_sparse_a(ctx.get(), plan.get(), a.get(), b.get(), c.get(), workspace.get());
    REQUIRE(status == RNS8_UNSUPPORTED_BACKEND);
    CHECK(c->content_generation == generation); CHECK(c->source_version == source);
    CHECK(c->host_residues_current == host_current); CHECK(c->device_residues_current == device_current);
    CHECK(c->exact_max_magnitude == proof); CHECK(c->exact_range_prefix == range_prefix);
    CHECK(c->residues == old_host);
    if (finite) {
      uint8_t observed = 0;
      REQUIRE(rns8_export_finite_u8(ctx.get(), plan.get(), 251, c.get(), &observed, 1) == RNS8_SUCCESS);
      CHECK(observed == 42);
    } else {
      int64_t observed = 0;
      REQUIRE(rns8_export_i64(ctx.get(), plan.get(), c.get(), &observed, 1) == RNS8_SUCCESS);
      CHECK(observed == 42);
    }
  }
}

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
  CHECK(sparse_c->source_version == sparse_c->content_generation);

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
  CHECK(sparse_c->source_version == sparse_c->content_generation);

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
    CHECK(sparse_c->source_version == sparse_c->content_generation);

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
    CHECK(sparse_c->source_version == sparse_c->content_generation);

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

TEST_CASE("sparse APIs reject mismatched sparse contracts without mutating output") {
  rns8_context* ctx = create_cpu_context();

  auto bounded = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 4);
  bounded.bound = 64;
  rns8_plan* bounded_plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &bounded, &bounded_plan) == RNS8_SUCCESS);
  rns8_workspace* bounded_workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, bounded_plan, &bounded_workspace) == RNS8_SUCCESS);

  auto bounded_b_desc = bounded_matrix_desc(4, 1, RNS8_BOUNDED_I64);
  auto bounded_c_desc = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
  rns8_matrix* bounded_b = nullptr;
  rns8_matrix* bounded_c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &bounded_b_desc, &bounded_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &bounded_c_desc, &bounded_c) == RNS8_SUCCESS);
  const std::vector<int64_t> b_values = {1, 2, 3, 4};
  REQUIRE(rns8_pack_i64(ctx, bounded_b, b_values.data(), 1, 22) == RNS8_SUCCESS);
  bounded_c->source_version = 7001;
  for (auto& residue : bounded_c->residues) {
    residue = static_cast<int8_t>(-77);
  }

  rns8_sparse_matrix_desc finite_sparse_desc{};
  finite_sparse_desc.struct_size = sizeof(finite_sparse_desc);
  finite_sparse_desc.abi_version = RNS8_ABI_VERSION;
  finite_sparse_desc.semantics = RNS8_FINITE_RING_U8;
  finite_sparse_desc.bound_kind = RNS8_BOUND_NONE;
  finite_sparse_desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  finite_sparse_desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  finite_sparse_desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  finite_sparse_desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8;
  finite_sparse_desc.rows = 1;
  finite_sparse_desc.expanded_k = 4;
  finite_sparse_desc.group_size = 4;
  finite_sparse_desc.nonzeros_per_group = 2;
  finite_sparse_desc.finite_modulus = 251;
  rns8_sparse_matrix* finite_sparse_a = nullptr;
  REQUIRE(rns8_create_sparse_matrix(ctx, &finite_sparse_desc, &finite_sparse_a) == RNS8_SUCCESS);
  const std::vector<uint8_t> finite_sparse_values = {5, 0, 7, 0};
  REQUIRE(
      rns8_pack_sparse_a_4_to_2_matrix_u8(
          ctx, finite_sparse_a, finite_sparse_values.data(), finite_sparse_desc.expanded_k, 11) ==
      RNS8_SUCCESS);

  CHECK(
      rns8_gemm_rns_sparse_a(ctx, bounded_plan, finite_sparse_a, bounded_b, bounded_c, bounded_workspace) ==
      RNS8_INVALID_ARGUMENT);
  CHECK(bounded_c->source_version == 7001);
  for (const auto residue : bounded_c->residues) {
    CHECK(residue == static_cast<int8_t>(-77));
  }
  CHECK_FALSE(bounded_c->device_residues_current);

  auto finite = finite_gemm_desc(RNS8_FINITE_RING_U8, 1, 1, 4, 251);
  rns8_plan* finite_plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &finite, &finite_plan) == RNS8_SUCCESS);
  rns8_workspace* finite_workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, finite_plan, &finite_workspace) == RNS8_SUCCESS);

  auto finite_b_desc = finite_matrix_desc(4, 1, RNS8_FINITE_RING_U8);
  auto finite_c_desc = finite_matrix_desc(1, 1, RNS8_FINITE_RING_U8);
  rns8_matrix* finite_b = nullptr;
  rns8_matrix* finite_c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &finite_b_desc, &finite_b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &finite_c_desc, &finite_c) == RNS8_SUCCESS);
  const std::vector<uint8_t> finite_b_values = {1, 2, 3, 4};
  REQUIRE(rns8_pack_finite_u8(ctx, finite_b, 251, finite_b_values.data(), 1, 33) == RNS8_SUCCESS);
  finite_c->source_version = 8001;
  for (auto& residue : finite_c->residues) {
    residue = static_cast<int8_t>(-66);
  }

  rns8_sparse_matrix_desc bounded_sparse_desc{};
  bounded_sparse_desc.struct_size = sizeof(bounded_sparse_desc);
  bounded_sparse_desc.abi_version = RNS8_ABI_VERSION;
  bounded_sparse_desc.semantics = RNS8_BOUNDED_I64;
  bounded_sparse_desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  bounded_sparse_desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  bounded_sparse_desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  bounded_sparse_desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  bounded_sparse_desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
  bounded_sparse_desc.rows = 1;
  bounded_sparse_desc.expanded_k = 4;
  bounded_sparse_desc.group_size = 4;
  bounded_sparse_desc.nonzeros_per_group = 2;
  bounded_sparse_desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  rns8_sparse_matrix* bounded_sparse_a = nullptr;
  REQUIRE(rns8_create_sparse_matrix(ctx, &bounded_sparse_desc, &bounded_sparse_a) == RNS8_SUCCESS);
  const std::vector<int64_t> bounded_a_native = {3, 0, -5, 0};
  rns8_matrix* dense_bounded_a = nullptr;
  auto bounded_a_desc = bounded_matrix_desc(1, 4, RNS8_BOUNDED_I64);
  REQUIRE(rns8_create_matrix(ctx, &bounded_a_desc, &dense_bounded_a) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, dense_bounded_a, bounded_a_native.data(), 4, 44) == RNS8_SUCCESS);
  REQUIRE(
      rns8_pack_sparse_a_4_to_2_matrix_u8(
          ctx,
          bounded_sparse_a,
          reinterpret_cast<const uint8_t*>(dense_bounded_a->residues.data()),
          dense_bounded_a->desc.cols,
          dense_bounded_a->source_version) == RNS8_SUCCESS);

  CHECK(
      rns8_gemm_finite_u8_sparse_a(ctx, finite_plan, 251, bounded_sparse_a, finite_b, finite_c, finite_workspace) ==
      RNS8_INVALID_ARGUMENT);
  CHECK(
      rns8_gemm_finite_u8_sparse_a(ctx, finite_plan, 257, finite_sparse_a, finite_b, finite_c, finite_workspace) ==
      RNS8_INVALID_ARGUMENT);
  CHECK(finite_c->source_version == 8001);
  for (const auto residue : finite_c->residues) {
    CHECK(residue == static_cast<int8_t>(-66));
  }
  CHECK_FALSE(finite_c->device_residues_current);

  // Reservation admission is shared by CPU and device sparse entry points.
  // These are deterministic state fixtures, not simulated asynchronous GPU work.
  auto check_reservations = [&](rns8_matrix* b, rns8_matrix* c, auto invoke) {
    const auto residues = c->residues;
    const auto generation = c->content_generation;
    const auto source_version = c->source_version;
    const auto host_current = c->host_residues_current;
    const auto device_current = c->device_residues_current;
    const auto exact_magnitude = c->exact_max_magnitude;
    const auto exact_prefix = c->exact_range_prefix;
    auto unchanged = [&]() {
      CHECK(c->residues == residues);
      CHECK(c->content_generation == generation);
      CHECK(c->source_version == source_version);
      CHECK(c->host_residues_current == host_current);
      CHECK(c->device_residues_current == device_current);
      CHECK(c->exact_max_magnitude == exact_magnitude);
      CHECK(c->exact_range_prefix == exact_prefix);
    };
    b->in_flight_writer = true;
    CHECK(invoke() == RNS8_BUSY);
    CHECK(b->in_flight_writer);
    b->in_flight_writer = false;
    unchanged();
    c->in_flight_writer = true;
    CHECK(invoke() == RNS8_BUSY);
    CHECK(c->in_flight_writer);
    c->in_flight_writer = false;
    unchanged();
    c->in_flight_readers = 2;
    CHECK(invoke() == RNS8_BUSY);
    CHECK(c->in_flight_readers == 2);
    c->in_flight_readers = 0;
    unchanged();
    b->in_flight_readers = 2;
    CHECK(invoke() == RNS8_SUCCESS);
    CHECK(b->in_flight_readers == 2);
    b->in_flight_readers = 0;
  };
  check_reservations(bounded_b, bounded_c, [&]() {
    return rns8_gemm_rns_sparse_a(ctx, bounded_plan, bounded_sparse_a, bounded_b, bounded_c, bounded_workspace);
  });
  check_reservations(finite_b, finite_c, [&]() {
    return rns8_gemm_finite_u8_sparse_a(ctx, finite_plan, 251, finite_sparse_a, finite_b, finite_c, finite_workspace);
  });

  rns8_destroy_matrix(dense_bounded_a);
  rns8_destroy_sparse_matrix(bounded_sparse_a);
  rns8_destroy_matrix(finite_c);
  rns8_destroy_matrix(finite_b);
  rns8_destroy_workspace(finite_workspace);
  rns8_destroy_plan(finite_plan);
  rns8_destroy_sparse_matrix(finite_sparse_a);
  rns8_destroy_matrix(bounded_c);
  rns8_destroy_matrix(bounded_b);
  rns8_destroy_workspace(bounded_workspace);
  rns8_destroy_plan(bounded_plan);
  rns8_destroy_context(ctx);
}
