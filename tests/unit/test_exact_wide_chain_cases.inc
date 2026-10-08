namespace {
struct ExactChainFixture {
  rns8_context* ctx = create_cpu();
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  std::vector<rns8_matrix*> matrices;
  explicit ExactChainFixture(rns8_semantics semantics, int64_t size = 1, uint32_t prefix = 20)
      : ExactChainFixture(semantics, size, size, size, prefix) {}
  ExactChainFixture(rns8_semantics semantics, int64_t m, int64_t n, int64_t k, uint32_t prefix) {
    auto desc = exact_desc(semantics, m, n, k);
    desc.max_prefix = prefix;
    desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    const int64_t shapes[4][2] = {{m, k}, {k, n}, {m, n}, {m, n}};
    for (unsigned i = 0; i < 4; ++i) {
      auto md = exact_matrix_desc(shapes[i][0], shapes[i][1], semantics);
      rns8_matrix* matrix = nullptr;
      REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
      matrices.push_back(matrix);
    }
  }
  ~ExactChainFixture() {
    for (auto* matrix : matrices) rns8_destroy_matrix(matrix);
    rns8_destroy_workspace(workspace);
    rns8_destroy_plan(plan);
    rns8_destroy_context(ctx);
  }
  rns8_status multiply(unsigned a, unsigned b, unsigned c) {
    return rns8_gemm_rns(ctx, plan, matrices[a], matrices[b], matrices[c], workspace);
  }
};
}

TEST_CASE("exact-wide chains preserve safe signed and unsigned repeated products", "[exact-chain]") {
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactChainFixture f(semantics);
    const int64_t signed_value = -7;
    const uint64_t unsigned_value = 7;
    for (unsigned i = 0; i < 2; ++i) {
      REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                   ? rns8_pack_i64(f.ctx, f.matrices[i], &signed_value, 1, 1)
                   : rns8_pack_u64(f.ctx, f.matrices[i], &unsigned_value, 1, 1)) == RNS8_SUCCESS);
    }
    boost::multiprecision::cpp_int expected = semantics == RNS8_EXACT_WIDE_SIGNED ? -7 : 7;
    unsigned input = 0, output = 2;
    for (unsigned step = 0; step < 40; ++step) {
      INFO("step=" << step << " semantics=" << semantics);
      REQUIRE(f.multiply(input, 1, output) == RNS8_SUCCESS);
      expected *= semantics == RNS8_EXACT_WIDE_SIGNED ? -7 : 7;
      std::vector<uint64_t> limbs(3);
      REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                   ? rns8_export_exact_wide_signed_limbs(f.ctx, f.plan, f.matrices[output], limbs.data(), 1, 3)
                   : rns8_export_exact_wide_unsigned_limbs(f.ctx, f.plan, f.matrices[output], limbs.data(), 1, 3)) == RNS8_SUCCESS);
      CHECK(limbs == (semantics == RNS8_EXACT_WIDE_SIGNED
                          ? signed_twos_complement_limbs(expected, 3) : unsigned_limbs(expected, 3)));
      CHECK(f.matrices[output]->exact_max_magnitude == boost::multiprecision::abs(expected));
      std::swap(input, output);
    }
  }
}

TEST_CASE("exact-wide overflowing chains reject before changing destination", "[exact-chain]") {
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    ExactChainFixture f(semantics);
    const int64_t signed_value = std::numeric_limits<int64_t>::min();
    const uint64_t unsigned_value = std::numeric_limits<uint64_t>::max();
    for (unsigned i = 0; i < 2; ++i) {
      REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                   ? rns8_pack_i64(f.ctx, f.matrices[i], &signed_value, 1, 11 + i)
                   : rns8_pack_u64(f.ctx, f.matrices[i], &unsigned_value, 1, 11 + i)) == RNS8_SUCCESS);
    }
    REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
    const auto saved_residues = f.matrices[2]->residues;
    const auto saved_bound = f.matrices[2]->exact_max_magnitude;
    const auto saved_version = f.matrices[2]->source_version;
    const auto destination_before = f.matrices[3]->residues;
    REQUIRE(f.multiply(2, 1, 3) == RNS8_RANGE_ERROR);
    CHECK(f.matrices[3]->residues == destination_before);
    CHECK(f.matrices[3]->exact_max_magnitude == 0);
    CHECK(f.matrices[3]->source_version == 0);
    CHECK(f.matrices[2]->residues == saved_residues);
    CHECK(f.matrices[2]->exact_max_magnitude == saved_bound);
    CHECK(f.matrices[2]->source_version == saved_version);
    const uint64_t zero = 0;
    const int64_t signed_zero = 0;
    REQUIRE((semantics == RNS8_EXACT_WIDE_SIGNED
                 ? rns8_pack_i64(f.ctx, f.matrices[1], &signed_zero, 1, 13)
                 : rns8_pack_u64(f.ctx, f.matrices[1], &zero, 1, 13)) == RNS8_SUCCESS);
    REQUIRE(f.multiply(2, 1, 3) == RNS8_SUCCESS);
    CHECK(f.matrices[3]->exact_max_magnitude == 0);
  }
}

TEST_CASE("exact-wide range proof includes K and ignores host padding", "[exact-chain]") {
  ExactChainFixture f(RNS8_EXACT_WIDE_SIGNED, 2);
  const int64_t a[] = {2, -3, INT64_MIN, -1, 4, INT64_MIN};
  const int64_t b[] = {5, -6, INT64_MIN, 7, -8, INT64_MIN};
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[0], a, 3, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(f.ctx, f.matrices[1], b, 3, 1) == RNS8_SUCCESS);
  CHECK(f.matrices[0]->exact_max_magnitude == 4);
  CHECK(f.matrices[1]->exact_max_magnitude == 8);
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  CHECK(f.matrices[2]->exact_max_magnitude <= 64);
  CHECK(f.matrices[2]->exact_max_magnitude >= 26);
  REQUIRE(f.multiply(2, 1, 3) == RNS8_SUCCESS);
  CHECK(f.matrices[3]->exact_max_magnitude <= 1024);
  CHECK(f.matrices[3]->exact_max_magnitude >= 70);
  std::vector<uint64_t> limbs(8);
  REQUIRE(rns8_export_exact_wide_signed_limbs(f.ctx, f.plan, f.matrices[3], limbs.data(), 2, 2) == RNS8_SUCCESS);
  const int64_t expected[] = {29, -30, -67, 70};
  for (unsigned cell = 0; cell < 4; ++cell) {
    CHECK(limbs[2 * cell] == static_cast<uint64_t>(expected[cell]));
    CHECK(limbs[2 * cell + 1] == (expected[cell] < 0 ? UINT64_MAX : 0));
  }
}

TEST_CASE("exact-wide chains reject aliases and unwritten higher prefix planes", "[exact-chain]") {
  ExactChainFixture f(RNS8_EXACT_WIDE_UNSIGNED, 1, 18);
  const uint64_t value = 17;
  REQUIRE(rns8_pack_u64(f.ctx, f.matrices[0], &value, 1, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(f.ctx, f.matrices[1], &value, 1, 2) == RNS8_SUCCESS);
  const auto saved = f.matrices[0]->residues;
  CHECK(f.multiply(0, 1, 0) == RNS8_INVALID_ARGUMENT);
  CHECK(f.multiply(0, 1, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(f.matrices[0]->residues == saved);
  REQUIRE(f.multiply(0, 1, 2) == RNS8_SUCCESS);
  CHECK(f.matrices[2]->prefix == 20);
  CHECK(f.matrices[2]->exact_range_prefix == 18);
  auto larger_desc = exact_desc(RNS8_EXACT_WIDE_UNSIGNED, 1, 1, 1);
  larger_desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* larger = nullptr;
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_plan(f.ctx, &larger_desc, &larger) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(f.ctx, larger, &workspace) == RNS8_SUCCESS);
  CHECK(rns8_gemm_rns(f.ctx, larger, f.matrices[2], f.matrices[1], f.matrices[3], workspace) == RNS8_RANGE_ERROR);
  uint64_t output[3] = {777, 777, 777};
  CHECK(rns8_export_exact_wide_unsigned_limbs(f.ctx, larger, f.matrices[2], output, 1, 3) == RNS8_RANGE_ERROR);
  CHECK(output[0] == 777);
  CHECK(output[1] == 777);
  CHECK(output[2] == 777);
  REQUIRE(rns8_pack_u64(f.ctx, f.matrices[2], &value, 1, 3) == RNS8_SUCCESS);
  CHECK(f.matrices[2]->exact_range_prefix == 20);
  CHECK(rns8_gemm_rns(f.ctx, larger, f.matrices[2], f.matrices[1], f.matrices[3], workspace) == RNS8_SUCCESS);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(larger);
}

TEST_CASE("exact-wide pack rejects aliases outside its selected CRT range", "[exact-chain]") {
  auto* ctx = create_cpu();
  for (auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto md = exact_matrix_desc(1, 1, semantics);
    md.max_prefix = 1;
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
    const auto before = matrix->residues;
    const int64_t signed_value = 128;
    const uint64_t unsigned_value = 256;
    CHECK((semantics == RNS8_EXACT_WIDE_SIGNED
               ? rns8_pack_i64(ctx, matrix, &signed_value, 1, 1)
               : rns8_pack_u64(ctx, matrix, &unsigned_value, 1, 1)) == RNS8_RANGE_ERROR);
    CHECK(matrix->residues == before);
    CHECK(matrix->source_version == 0);
    rns8_destroy_matrix(matrix);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide sparse representatives carry range into a dense continuation", "[exact-chain]") {
  auto* ctx = create_cpu();
  auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 4);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  rns8_sparse_matrix_desc sd{};
  sd.struct_size = sizeof(sd);
  sd.abi_version = RNS8_ABI_VERSION;
  sd.semantics = RNS8_EXACT_WIDE_SIGNED;
  sd.bound_kind = RNS8_BOUND_NONE;
  sd.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  sd.sparse_operand = RNS8_SPARSE_OPERAND_A;
  sd.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  sd.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
  sd.rows = 1;
  sd.expanded_k = 4;
  sd.group_size = 4;
  sd.nonzeros_per_group = 2;
  sd.max_prefix = 20;
  rns8_sparse_matrix* a = nullptr;
  REQUIRE(rns8_create_sparse_matrix(ctx, &sd, &a) == RNS8_SUCCESS);
  std::vector<uint8_t> planes(20 * 5, 99);
  for (unsigned p = 0; p < 20; ++p) {
    for (unsigned c = 0; c < 4; ++c) {
      const int value = c == 0 ? -3 : c == 2 ? 5 : 0;
      planes[p * 5 + c] = static_cast<uint8_t>(rns8::detail::centered_residue(
          boost::multiprecision::cpp_int(value), rns8::detail::kDefaultModuli[p]));
    }
  }
  REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, a, planes.data(), 5, 1) == RNS8_SUCCESS);
  CHECK(a->exact_max_magnitude == 5);
  const auto saved_values = a->packed_values;
  const auto saved_indices = a->packed_indices;
  auto malformed = planes;
  malformed[19 * 5 + 1] = 7;  // Late final-plane failure, after earlier groups.
  CHECK(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, a, malformed.data(), 5, 2) == RNS8_INVALID_ARGUMENT);
  CHECK(a->packed_values == saved_values);
  CHECK(a->packed_indices == saved_indices);
  CHECK(a->exact_max_magnitude == 5);
  CHECK(a->source_version == 1);

  rns8_matrix *b = nullptr, *c = nullptr;
  auto bd = exact_matrix_desc(4, 1, sd.semantics);
  auto cd = exact_matrix_desc(1, 1, sd.semantics);
  REQUIRE(rns8_create_matrix(ctx, &bd, &b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &cd, &c) == RNS8_SUCCESS);
  const int64_t values[] = {7, 11, -13, 17};
  REQUIRE(rns8_pack_i64(ctx, b, values, 1, 1) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns_sparse_a(ctx, plan, a, b, c, workspace) == RNS8_SUCCESS);
  CHECK(c->exact_max_magnitude == 4 * 5 * 17);
  uint64_t limbs[2] = {};
  REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c, limbs, 1, 2) == RNS8_SUCCESS);
  CHECK(limbs[0] == static_cast<uint64_t>(-86));
  CHECK(limbs[1] == UINT64_MAX);
  rns8_destroy_matrix(c);
  rns8_destroy_matrix(b);
  rns8_destroy_sparse_matrix(a);
  rns8_destroy_workspace(workspace);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
