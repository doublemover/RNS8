#include <cstring>
TEST_CASE("Gram physical method and query identities keep MFMA distinct from other recipes") {
  using namespace rns8::detail;
  for (uint32_t method = 0; method < 8; ++method) {
    CHECK(gram_method_valid(method) == (method <= 3 || method >= 6));
    if (!gram_method_valid(method))
      continue;
    CHECK(gram_matrix_kind(method) == (method >= 6 ? 2u : method & 1));
    CHECK(gram_separate_mirror(method) == (method >= 6 ? bool(method & 1) : bool(method & 2)));
  }
  for (uint32_t policy = 0; policy < 3; ++policy)
    for (uint32_t method : {0u, 1u, 2u, 3u, 6u, 7u}) {
      const auto algorithm = method >= 6 ? 12 + 2 * policy + (method & 1) : 4 * policy + method;
      CHECK(gram_query_method(algorithm) == method);
      CHECK(gram_query_policy(algorithm) == policy);
    }
  CHECK_FALSE(gram_method_valid(UINT32_MAX));
  auto* ctx = create_cpu_context();
  auto md = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
  rns8_matrix *x = nullptr, *c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &x) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
  int64_t value = 3;
  REQUIRE(rns8_pack_i64(ctx, x, &value, 1, 0) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, c, &value, 1, 0) == RNS8_SUCCESS);
  const auto before = c->residues;
  const auto generation = c->content_generation;
  for (auto method : {RNS8_GRAM_MFMA_GFX942, RNS8_GRAM_MFMA_GFX942_SEPARATE_MIRROR})
    CHECK(rns8_gram_residues(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, method, UINT64_MAX) ==
          RNS8_UNSUPPORTED_BACKEND);
  CHECK(c->residues == before);
  CHECK(c->content_generation == generation);
  REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("single-source Gram CPU products preserve full signed and finite semantics") {
  check_public_gram(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("Gram triangular jobs own every symmetric output exactly once") {
  using namespace rns8::detail;
  for (int n : {1, 15, 16, 17, 31, 32, 33, 129}) {
    GramProblem p;
    p.rows = 19;
    p.cols = n;
    p.source_ld = n + 3;
    p.output_ld = n + 2;
    p.source_instance = 1;
    p.prefix = 20;
    const auto support = query_gram_support(p, UINT64_MAX, false, false);
    REQUIRE(support.status == RNS8_SUCCESS);
    const auto jobs = gram_tiles(support);
    const uint64_t q = (n + 15) / 16;
    REQUIRE(jobs.size() == q * (q + 1) / 2);
    std::vector<int> owners(n * n, 0);
    for (auto job : jobs)
      for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
          const int row = job.row * 16 + r, col = job.col * 16 + c;
          if (row >= n || col >= n || row > col)
            continue;
          ++owners[row * n + col];
          if (row != col)
            ++owners[col * n + row];
        }
    for (auto count : owners)
      REQUIRE(count == 1);
    CHECK(query_gram_support(p, support.job_bytes - 1, false, false).status ==
          RNS8_WORKSPACE_TOO_SMALL);
    p.orientation = static_cast<GramOrientation>(2);
    CHECK(query_gram_support(p, UINT64_MAX, false, false).status == RNS8_INVALID_ARGUMENT);
  }
}

TEST_CASE("bounded Gram proves the accumulation instead of trusting a small result claim") {
  auto* ctx = create_cpu_context();
  auto md = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
  rns8_matrix *x = nullptr, *c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &x) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
  int64_t value = INT64_MAX;
  REQUIRE(rns8_pack_i64(ctx, x, &value, 1, 0) == RNS8_SUCCESS);
  const auto before = c->residues;
  CHECK(rns8_gram_residues(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, RNS8_GRAM_SCALAR,
                           UINT64_MAX) == RNS8_RANGE_ERROR);
  CHECK(c->residues == before);
  value = 7;
  REQUIRE(rns8_pack_i64(ctx, x, &value, 1, 0) == RNS8_SUCCESS);
  REQUIRE(rns8_gram_residues(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, RNS8_GRAM_SCALAR,
                             UINT64_MAX) == RNS8_SUCCESS);
  CHECK(c->exact_max_magnitude == 49);
  x->exact_range_prefix = 0;
  CHECK(rns8_gram_residues(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, RNS8_GRAM_SCALAR,
                           UINT64_MAX) == RNS8_RANGE_ERROR);
  REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("prepared CPU Gram retains jobs and refreshes source generations") {
  check_public_gram(RNS8_BACKEND_CPU_REFERENCE, true);
}
TEST_CASE("prepared Gram rejects larger bounds without changing its binding") {
  auto* ctx = create_cpu_context();
  auto md = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
  rns8_matrix *x = nullptr, *c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &x) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
  int64_t value = 7;
  REQUIRE(rns8_pack_i64(ctx, x, &value, 1, 0) == RNS8_SUCCESS);
  rns8_prepared_gram* owner = nullptr;
  CHECK(rns8_prepare_gram(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, RNS8_GRAM_SCALAR, 0, &owner) ==
        RNS8_WORKSPACE_TOO_SMALL);
  CHECK(owner == nullptr);
  REQUIRE(rns8_prepare_gram(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, RNS8_GRAM_SCALAR, UINT64_MAX,
                            &owner) == RNS8_SUCCESS);
  REQUIRE(rns8_execute_prepared_gram(owner) == RNS8_SUCCESS);
  const auto output = c->residues;
  const auto generation = c->content_generation;
  value = 8;
  REQUIRE(rns8_pack_i64(ctx, x, &value, 1, 0) == RNS8_SUCCESS);
  CHECK(rns8_refresh_prepared_gram(owner) == RNS8_RANGE_ERROR);
  CHECK(rns8_execute_prepared_gram(owner) == RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation == generation);
  CHECK(c->residues == output);
  value = -3;
  REQUIRE(rns8_pack_i64(ctx, x, &value, 1, 0) == RNS8_SUCCESS);
  REQUIRE(rns8_refresh_prepared_gram(owner) == RNS8_SUCCESS);
  REQUIRE(rns8_execute_prepared_gram(owner) == RNS8_SUCCESS);
  CHECK(c->exact_max_magnitude == 49);
  for (uint32_t i = 0; i < c->prefix; ++i)
    CHECK(c->residues[i] == rns8::detail::centered_residue(boost::multiprecision::cpp_int(9),
                                                           rns8::detail::kDefaultModuli[i]));
  REQUIRE(rns8_destroy_prepared_gram(owner) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("prepared finite Gram refresh cannot change its modulus") {
  auto* ctx = create_cpu_context();
  rns8_matrix_desc md{};
  md.struct_size = sizeof(md);
  md.abi_version = RNS8_ABI_VERSION;
  md.rows = md.cols = 1;
  md.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
  md.semantics = RNS8_FINITE_RING_U8;
  md.bound_kind = RNS8_BOUND_NONE;
  rns8_matrix *x = nullptr, *c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &x) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
  uint8_t value = 7;
  REQUIRE(rns8_pack_finite_u8(ctx, x, 250, &value, 1, 0) == RNS8_SUCCESS);
  rns8_prepared_gram* owner = nullptr;
  REQUIRE(rns8_prepare_gram(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, RNS8_GRAM_SCALAR, UINT64_MAX,
                            &owner) == RNS8_SUCCESS);
  REQUIRE(rns8_execute_prepared_gram(owner) == RNS8_SUCCESS);
  const auto generation = c->content_generation;
  REQUIRE(rns8_pack_finite_u8(ctx, x, 251, &value, 1, 0) == RNS8_SUCCESS);
  CHECK(rns8_refresh_prepared_gram(owner) == RNS8_RANGE_ERROR);
  CHECK(rns8_execute_prepared_gram(owner) == RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation == generation);
  CHECK(c->finite_modulus == 250);
  REQUIRE(rns8_pack_finite_u8(ctx, x, 250, &value, 1, 0) == RNS8_SUCCESS);
  REQUIRE(rns8_refresh_prepared_gram(owner) == RNS8_SUCCESS);
  REQUIRE(rns8_execute_prepared_gram(owner) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_prepared_gram(owner) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE(
    "Gram native, retained transpose and reconstruction-once compose without stale publication") {
  check_gram_prepared_ablations(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("Gram prepared payload aligns limb staging after odd transpose extents") {
  using namespace rns8::detail;
  for (int rows : {1, 3, 17})
    for (int cols : {1, 5, 19})
      for (uint32_t prefix : {1u, 3u, 20u})
        for (bool wmma : {false, true}) {
          GramProblem p;
          p.rows = rows;
          p.cols = cols;
          p.source_ld = cols;
          p.output_ld = cols;
          p.prefix = prefix;
          p.source_instance = 1;
          const auto r = query_gram_prepared_storage(p, true, wmma, 2, 3, UINT64_MAX);
          REQUIRE(r.status == RNS8_SUCCESS);
          CHECK(r.export_offset % 8 == 0);
          CHECK(r.status_offset % 4 == 0);
          CHECK(r.export_offset >= r.adapter_offset + r.adapter_bytes);
          CHECK(r.device_bytes == r.status_offset + 4);
          CHECK(query_gram_prepared_storage(p, true, wmma, 2, 3, r.host_bytes + r.device_bytes - 1)
                    .status == RNS8_WORKSPACE_TOO_SMALL);
          CHECK(query_gram_prepared_storage(p, true, wmma, 3, 3, UINT64_MAX).status ==
                RNS8_INVALID_ARGUMENT);
        }
}

TEST_CASE("raw Gram enum values reject through real descriptors without changing output") {
  auto* ctx = create_cpu_context();
  auto md = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
  rns8_matrix *x = nullptr, *c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &x) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
  int64_t input = 3, old = 11;
  REQUIRE(rns8_pack_i64(ctx, x, &input, 1, 0) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx, c, &old, 1, 0) == RNS8_SUCCESS);
  const auto residues = c->residues;
  const auto generation = c->content_generation;
  const auto references = ctx->prepared_references;
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  for (uint32_t raw : {UINT32_MAX, UINT32_C(0x80000000), 4u, 8u}) {
    rns8_gram_method method;
    std::memcpy(&method, &raw, sizeof(raw));
    CHECK(rns8_gram_residues(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, method, UINT64_MAX) ==
          RNS8_INVALID_ARGUMENT);
    auto* owner = reinterpret_cast<rns8_prepared_gram*>(&old);
    CHECK(rns8_prepare_gram_ex(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, method,
                               RNS8_GRAM_RESIDENT_RESIDUES, 0, UINT64_MAX,
                               &owner) == RNS8_INVALID_ARGUMENT);
    CHECK(owner == reinterpret_cast<rns8_prepared_gram*>(&old));
    CHECK(rns8_prepare_gram(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, method, UINT64_MAX,
                            &owner) == RNS8_INVALID_ARGUMENT);
    CHECK(owner == reinterpret_cast<rns8_prepared_gram*>(&old));
  }
  for (uint32_t raw : {UINT32_MAX, UINT32_C(0x80000000), 2u}) {
    rns8_gram_orientation orientation;
    std::memcpy(&orientation, &raw, sizeof(raw));
    CHECK(rns8_gram_residues(ctx, x, c, orientation, RNS8_GRAM_SCALAR, UINT64_MAX) ==
          RNS8_INVALID_ARGUMENT);
    auto* owner = reinterpret_cast<rns8_prepared_gram*>(&old);
    CHECK(rns8_prepare_gram_ex(ctx, x, c, orientation, RNS8_GRAM_SCALAR,
                               RNS8_GRAM_RESIDENT_RESIDUES, 0, UINT64_MAX,
                               &owner) == RNS8_INVALID_ARGUMENT);
    CHECK(owner == reinterpret_cast<rns8_prepared_gram*>(&old));
    for (bool poisoned : {false, true}) {
      ctx->hip_failure_poisoned = poisoned;
      for (uint32_t matrix : {0u, 1u}) {
        CHECK(rns8_prepare_gram_five_product(ctx, x, c, orientation, matrix, UINT64_MAX, &owner) ==
              RNS8_INVALID_ARGUMENT);
        CHECK(owner == reinterpret_cast<rns8_prepared_gram*>(&old));
      }
      ctx->hip_failure_poisoned = false;
    }
  }
  for (uint32_t matrix : {UINT32_MAX, UINT32_C(0x80000000), 2u}) {
    auto* owner = reinterpret_cast<rns8_prepared_gram*>(&old);
    CHECK(rns8_prepare_gram_five_product(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, matrix,
                                         UINT64_MAX, &owner) == RNS8_INVALID_ARGUMENT);
    CHECK(owner == reinterpret_cast<rns8_prepared_gram*>(&old));
  }
  for (uint32_t raw : {UINT32_MAX, UINT32_C(0x80000000), 3u}) {
    rns8_gram_input_policy policy;
    std::memcpy(&policy, &raw, sizeof(raw));
    auto* owner = reinterpret_cast<rns8_prepared_gram*>(&old);
    CHECK(rns8_prepare_gram_ex(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, RNS8_GRAM_SCALAR, policy,
                               0, UINT64_MAX, &owner) == RNS8_INVALID_ARGUMENT);
    CHECK(owner == reinterpret_cast<rns8_prepared_gram*>(&old));
  }
  CHECK(c->residues == residues);
  CHECK(c->content_generation == generation);
  CHECK(ctx->prepared_references == references);
  const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(after.allocate_calls == allocations.allocate_calls);
  CHECK(after.h2d_calls == allocations.h2d_calls);
  CHECK(after.d2h_calls == allocations.d2h_calls);
  REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
