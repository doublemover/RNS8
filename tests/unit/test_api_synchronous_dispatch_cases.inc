namespace {
rns8_context* create_dispatch_context(rns8_backend_kind backend) {
  if (backend == RNS8_BACKEND_CPU_REFERENCE)
    return create_cpu_context();
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = backend;
  rns8_context* ctx = nullptr;
  REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
  return ctx;
}
struct SynchronousDispatchFixture {
  std::vector<const rns8_matrix*> readers;
  std::vector<const rns8_matrix*> writers;
  const rns8_workspace* workspace = nullptr;
  std::vector<const uint64_t*> private_owners;
  const rns8_context* owner_context = nullptr;
  bool expect_owner_only = false;
  const char* target_operation = nullptr;
  const rns8_context* workspace_context = nullptr;
  unsigned calls = 0, drains = 0;
  bool fail = false, throw_after_issue = false, drain_fails = false, reservations_observed = false,
       write_started_on_failure = true;
  static rns8_status dispatch(void* opaque, const char* operation, bool* started) {
    auto& s = *static_cast<SynchronousDispatchFixture*>(opaque);
    ++s.calls;
    const bool trigger = !s.target_operation || std::string(operation) == s.target_operation;
    if (!trigger)
      return RNS8_SUCCESS;
    s.reservations_observed = true;
    for (auto* matrix : s.readers)
      s.reservations_observed &= matrix->in_flight_readers > 0;
    for (auto* matrix : s.writers)
      s.reservations_observed &= matrix->in_flight_writer;
    if (s.workspace)
      s.reservations_observed &= s.workspace->in_flight;
    if (s.workspace_context) {
      const auto* workspace = s.workspace_context->failed_work_leases.back().workspace;
      s.reservations_observed &= workspace && workspace->in_flight;
    }
    for (auto* counter : s.private_owners)
      s.reservations_observed &= *counter > 0;
    if (s.owner_context) {
      const auto& record = s.owner_context->failed_work_leases.back();
      s.reservations_observed &= record.owner_only == s.expect_owner_only &&
                                 record.owner_holds[0] && *record.owner_holds[0] > 0;
    }
    if (s.fail && started && s.write_started_on_failure)
      *started = true; // simulated issue; no faulty device work
    if (s.throw_after_issue)
      throw std::bad_alloc{};
    return s.fail ? RNS8_BACKEND_FAILURE : RNS8_SUCCESS;
  }
  static rns8_status drain(void* opaque) noexcept {
    auto& s = *static_cast<SynchronousDispatchFixture*>(opaque);
    ++s.drains;
    return s.drain_fails ? RNS8_BACKEND_FAILURE : RNS8_SUCCESS;
  }
};
} // namespace
TEST_CASE("prepared Gram preserves read-only publication and retains uncertain owner reservations",
          "[api][gram][lifetime]") {
  for (bool read_only : {false, true})
    for (bool late : {false, true})
      for (bool throws : {false, true})
        for (bool uncertain : {false, true}) {
          CAPTURE(read_only, late, throws, uncertain);
          auto* ctx = create_cpu_context();
          auto md = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
          rns8_matrix *x = nullptr, *c = nullptr;
          REQUIRE(rns8_create_matrix(ctx, &md, &x) == RNS8_SUCCESS);
          REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
          const int64_t value = 7;
          REQUIRE(rns8_pack_i64(ctx, x, &value, 1, 0) == RNS8_SUCCESS);
          rns8_prepared_gram* owner = nullptr;
          REQUIRE(rns8_prepare_gram_ex(ctx, x, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, RNS8_GRAM_SCALAR,
                                       RNS8_GRAM_RESIDENT_RESIDUES, 1, UINT64_MAX,
                                       &owner) == RNS8_SUCCESS);
          REQUIRE(rns8_execute_prepared_gram(owner) == RNS8_SUCCESS);
          const auto generation = c->content_generation;
          const auto residues = c->residues;
          uint64_t output = 91;
          SynchronousDispatchFixture state;
          state.readers = {x};
          state.writers = {c};
          state.fail = true;
          state.write_started_on_failure = late;
          state.throw_after_issue = throws;
          state.drain_fails = uncertain;
          const rns8_synchronous_test_hooks hooks{&state, &SynchronousDispatchFixture::dispatch,
                                                  &SynchronousDispatchFixture::drain};
          ctx->synchronous_test_hooks = &hooks;
          CHECK((read_only ? rns8_export_prepared_gram_limbs(owner, &output, 1)
                           : rns8_execute_prepared_gram(owner)) ==
                (throws ? RNS8_INTERNAL_ERROR : RNS8_BACKEND_FAILURE));
          CHECK(state.reservations_observed);
          CHECK(c->content_generation == generation);
          CHECK(output == 91);
          CHECK(c->residues == residues);
          CHECK(c->host_residues_current == (read_only || !late));
          if (uncertain) {
            CHECK(ctx->hip_failure_poisoned);
            CHECK(x->in_flight_readers == 1);
            CHECK(c->in_flight_writer);
            CHECK(rns8_destroy_prepared_gram(owner) == RNS8_BACKEND_FAILURE);
            CHECK(rns8_destroy_matrix(c) == RNS8_BUSY);
            state.drain_fails = false;
          } else {
            CHECK(x->in_flight_readers == 0);
            CHECK_FALSE(c->in_flight_writer);
          }
          REQUIRE(rns8_destroy_prepared_gram(owner) == RNS8_SUCCESS);
          CHECK(x->in_flight_readers == 0);
          CHECK_FALSE(c->in_flight_writer);
          ctx->synchronous_test_hooks = nullptr;
          REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
          REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
          REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
        }
}

TEST_CASE("public exports retain read ownership without invalidating resident values",
          "[api][export][lifetime]") {
  for (const auto semantics :
       {RNS8_BOUNDED_I64, RNS8_BOUNDED_U64, RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED,
        RNS8_FINITE_RING_U8, RNS8_WRAP_U64_MOD_2_64})
    for (unsigned failure = 0; failure < 4; ++failure) {
      CAPTURE(semantics, failure);
      const bool wrap = semantics == RNS8_WRAP_U64_MOD_2_64,
                 finite = semantics == RNS8_FINITE_RING_U8;
      const bool bounded = semantics == RNS8_BOUNDED_I64 || semantics == RNS8_BOUNDED_U64;
      const bool sign = semantics == RNS8_BOUNDED_I64 || semantics == RNS8_EXACT_WIDE_SIGNED;
      auto* ctx = wrap ? create_wrap_context() : create_cpu_context();
      auto desc = wrap      ? wrap_gemm_desc(1, 1, 1)
                  : finite  ? finite_gemm_desc(semantics, 1, 1, 1, 251)
                  : bounded ? bounded_gemm_desc(semantics, 1, 1, 1)
                            : exact_desc(semantics, 1, 1, 1);
      auto md = wrap      ? wrap_matrix_desc(1, 1)
                : finite  ? finite_matrix_desc(1, 1, semantics)
                : bounded ? bounded_matrix_desc(1, 1, semantics)
                          : exact_matrix_desc(1, 1, semantics);
      rns8_plan* plan = nullptr;
      rns8_matrix* matrix = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
      const int64_t one = 1;
      const uint64_t uone = 1;
      const uint8_t byte = 1;
      REQUIRE((finite ? rns8_pack_finite_u8(ctx, matrix, 251, &byte, 1, 1)
               : sign ? rns8_pack_i64(ctx, matrix, &one, 1, 1)
                      : rns8_pack_u64(ctx, matrix, &uone, 1, 1)) == RNS8_SUCCESS);
      const auto generation = matrix->content_generation;
      const auto residues = matrix->residues;
      std::array<int64_t, 5> signed_output;
      std::array<uint64_t, 5> unsigned_output;
      std::array<uint8_t, 5> byte_output;
      signed_output.fill(77);
      unsigned_output.fill(77);
      byte_output.fill(77);
      SynchronousDispatchFixture fixture;
      fixture.readers = {matrix};
      fixture.fail = failure == 0 || failure == 2;
      fixture.throw_after_issue = failure == 1;
      fixture.drain_fails = failure < 2;
      const rns8_synchronous_test_hooks hooks{&fixture, &SynchronousDispatchFixture::dispatch,
                                              &SynchronousDispatchFixture::drain};
      ctx->synchronous_test_hooks = &hooks;
      const auto status =
          finite    ? rns8_export_finite_u8(ctx, plan, 251, matrix, byte_output.data(), 3)
          : wrap    ? rns8_export_wrap_u64(ctx, plan, matrix, unsigned_output.data(), 3)
          : bounded ? (sign ? rns8_export_i64(ctx, plan, matrix, signed_output.data(), 3)
                            : rns8_export_u64(ctx, plan, matrix, unsigned_output.data(), 3))
          : sign
              ? rns8_export_exact_wide_signed_limbs(ctx, plan, matrix, unsigned_output.data(), 3, 3)
              : rns8_export_exact_wide_unsigned_limbs(ctx, plan, matrix, unsigned_output.data(), 3,
                                                      3);
      CHECK(status == (failure == 3   ? RNS8_SUCCESS
                       : failure == 1 ? RNS8_INTERNAL_ERROR
                                      : RNS8_BACKEND_FAILURE));
      CHECK(fixture.calls == 1);
      CHECK(fixture.reservations_observed);
      CHECK(matrix->content_generation == generation);
      CHECK(matrix->residues == residues);
      CHECK_FALSE(matrix->in_flight_writer);
      CHECK(signed_output[4] == 77);
      CHECK(unsigned_output[4] == 77);
      CHECK(byte_output[4] == 77);
      if (failure != 3) {
        CHECK(signed_output[0] == 77);
        CHECK(unsigned_output[0] == 77);
        CHECK(byte_output[0] == 77);
      }
      if (failure < 2) {
        CHECK(matrix->in_flight_readers == 1);
        CHECK(rns8_destroy_matrix(matrix) == RNS8_BUSY);
        fixture.drain_fails = false;
        REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
        CHECK(ctx->hip_failure_poisoned);
      }
      CHECK(matrix->in_flight_readers == 0);
      CHECK(ctx->failed_work_leases.empty());
      ctx->synchronous_test_hooks = nullptr;
      REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
}

TEST_CASE("public synchronous dispatch retains every dependency after uncertain completion",
          "[api][lifetime]") {
  for (const std::string operation :
       {"rns", "finite", "wrap64", "gram", "axpby", "complex", "verify"})
    for (unsigned failure = 0; failure < 9; ++failure) {
      CAPTURE(operation, failure);
      const bool wrap = operation == "wrap64",
                 finite = operation == "finite" || operation == "verify";
      auto* ctx = wrap ? create_wrap_context() : create_cpu_context();
      auto desc = wrap     ? wrap_gemm_desc(1, 1, 1)
                  : finite ? finite_gemm_desc(RNS8_FINITE_RING_U8, 1, 1, 1, 251)
                           : exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
      auto md = wrap     ? wrap_matrix_desc(1, 1)
                : finite ? finite_matrix_desc(1, 1, RNS8_FINITE_RING_U8)
                         : exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
      rns8_plan* plan = nullptr;
      rns8_workspace* work = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx, plan, &work) == RNS8_SUCCESS);
      rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *ai = nullptr,
                  *ci = nullptr;
      for (auto** value : {&a, &b, &c, &d, &ai, &ci}) {
        REQUIRE(rns8_create_matrix(ctx, &md, value) == RNS8_SUCCESS);
        const int64_t one = 1;
        const uint64_t uone = 1;
        const uint8_t byte = 1;
        REQUIRE((wrap     ? rns8_pack_u64(ctx, *value, &uone, 1, 1)
                 : finite ? rns8_pack_finite_u8(ctx, *value, 251, &byte, 1, 1)
                          : rns8_pack_i64(ctx, *value, &one, 1, 1)) == RNS8_SUCCESS);
      }
      rns8_finite_verify_options options{};
      options.struct_size = sizeof(options);
      options.abi_version = RNS8_ABI_VERSION;
      options.repetition_count = 2;
      options.max_workspace_bytes = UINT64_MAX;
      options.sampler = RNS8_VERIFY_SPLITMIX64_BINARY_V1;
      rns8_finite_verify_report report{};
      report.struct_size = sizeof(report);
      report.abi_version = RNS8_ABI_VERSION;
      auto execute = [&]() {
        if (operation == "finite")
          return rns8_gemm_finite_u8(ctx, plan, 251, a, b, c, work);
        if (operation == "verify")
          return rns8_verify_finite_product(ctx, a, b, c, &options, &report);
        if (operation == "wrap64")
          return rns8_gemm_wrap_u64(ctx, plan, a, b, c, work);
        if (operation == "gram")
          return rns8_gram_residues(ctx, a, c, RNS8_GRAM_TRANSPOSE_TIMES_SELF, RNS8_GRAM_SCALAR,
                                    UINT64_MAX);
        if (operation == "axpby")
          return rns8_gemm_exact_axpby(ctx, plan, a, b, d, c, 1, 1, work);
        if (operation == "complex")
          return rns8_gemm_complex_exact_signed(ctx, plan, a, ai, b, d, c, ci,
                                                RNS8_COMPLEX_THREE_REAL_PRODUCTS, UINT64_MAX, work);
        return rns8_gemm_rns(ctx, plan, a, b, c, work);
      };
      REQUIRE(execute() == RNS8_SUCCESS);
      const auto generation = c->content_generation, imag_generation = ci->content_generation;
      SynchronousDispatchFixture state;
      state.readers = operation == "complex"  ? std::vector<const rns8_matrix*>{a, ai, b, d}
                      : operation == "axpby"  ? std::vector<const rns8_matrix*>{a, b, d}
                      : operation == "verify" ? std::vector<const rns8_matrix*>{a, b, c}
                      : operation == "gram"   ? std::vector<const rns8_matrix*>{a}
                                              : std::vector<const rns8_matrix*>{a, b};
      if (operation != "verify")
        state.writers = {c};
      if (operation == "complex")
        state.writers.push_back(ci);
      if (operation != "verify" && operation != "gram")
        state.workspace = work;
      const bool early = failure >= 5;
      state.fail = failure != 0;
      state.write_started_on_failure = !early;
      state.throw_after_issue = failure == 3 || failure == 4 || failure == 6 || failure == 8;
      state.drain_fails = failure == 2 || failure == 3 || failure == 7 || failure == 8;
      const rns8_synchronous_test_hooks hooks{&state, &SynchronousDispatchFixture::dispatch,
                                              &SynchronousDispatchFixture::drain};
      ctx->synchronous_test_hooks = &hooks;
      auto* saved = ctx;
      ctx = nullptr;
      CHECK(execute() == RNS8_INVALID_ARGUMENT);
      ctx = saved;
      CHECK(state.calls == 0);
      CHECK(c->content_generation == generation);
      report.passed = 99;
      const auto expected = failure == 0              ? RNS8_SUCCESS
                            : state.throw_after_issue ? RNS8_INTERNAL_ERROR
                                                      : RNS8_BACKEND_FAILURE;
      CHECK(execute() == expected);
      CHECK(state.calls == 1);
      CHECK(state.reservations_observed);
      CHECK(c->content_generation == generation + (failure == 0 && operation != "verify"));
      if (operation == "complex")
        CHECK(ci->content_generation == imag_generation + (failure == 0));
      if (operation == "verify")
        CHECK(report.passed == (failure == 0 ? 1 : 99));
      if (early) {
        CHECK((c->host_residues_current || c->host_byte_limbs_current));
        if (operation == "complex")
          CHECK(ci->host_residues_current);
      }
      if (failure == 0)
        CHECK(state.drains == 0); // no added success-path device wait
      if (state.drain_fails) {
        CHECK(ctx->hip_failure_poisoned);
        CHECK_FALSE(ctx->failed_work_leases.empty());
        for (auto* matrix : state.readers) {
          CHECK(matrix->in_flight_readers > 0);
          CHECK(rns8_destroy_matrix(const_cast<rns8_matrix*>(matrix)) == RNS8_BUSY);
        }
        for (auto* matrix : state.writers)
          CHECK(matrix->in_flight_writer);
        state.drain_fails = false;
        REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
        CHECK(ctx->hip_failure_poisoned); // draining memory does not revive authority
      }
      for (auto* matrix : {a, b, c, d, ai, ci}) {
        CHECK(matrix->in_flight_readers == 0);
        CHECK_FALSE(matrix->in_flight_writer);
      }
      CHECK_FALSE(work->in_flight);
      CHECK(ctx->failed_work_leases.empty());
      ctx->synchronous_test_hooks = nullptr;
      for (auto* matrix : {a, b, c, d, ai, ci})
        REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(work) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
}

TEST_CASE("failed synchronous work retains private owner counters until one successful drain",
          "[api][lifetime]") {
  using namespace rns8::detail::api;
  auto* ctx = create_cpu_context();
  rns8_matrix a{}, b{}, c{};
  rns8_workspace work{};
  SynchronousDispatchFixture state;
  state.drain_fails = true;
  const rns8_synchronous_test_hooks hooks{&state, nullptr, &SynchronousDispatchFixture::drain};
  ctx->synchronous_test_hooks = &hooks;
  uint64_t first = 0, second = 0;
  size_t token = 0;
  rns8::detail::hip_direct_device_buffer first_buffer, second_buffer, third_buffer, fourth_buffer,
      extra_buffer;
  REQUIRE(begin_synchronous_lease(*ctx, a, b, c, work, token) == RNS8_SUCCESS);
  SynchronousLease lease(*ctx, token);
  REQUIRE(add_synchronous_owner_hold(*ctx, token, first) == RNS8_SUCCESS);
  REQUIRE(add_synchronous_owner_hold(*ctx, token, first) == RNS8_SUCCESS);
  REQUIRE(add_synchronous_owner_hold(*ctx, token, second) == RNS8_SUCCESS);
  REQUIRE(lease.hold_buffer(first_buffer) == RNS8_SUCCESS);
  REQUIRE(lease.hold_buffer(first_buffer) == RNS8_SUCCESS);
  REQUIRE(lease.hold_buffer(second_buffer) == RNS8_SUCCESS);
  REQUIRE(lease.hold_buffer(third_buffer) == RNS8_SUCCESS);
  REQUIRE(lease.hold_buffer(fourth_buffer) == RNS8_SUCCESS);
  CHECK(lease.hold_buffer(extra_buffer) == RNS8_INTERNAL_ERROR);
  CHECK(first == 1);
  CHECK(second == 1);
  CHECK(lease.finish(RNS8_BACKEND_FAILURE) == RNS8_BACKEND_FAILURE);
  CHECK(first == 1);
  CHECK(second == 1);
  CHECK(ctx->hip_failure_poisoned);
  CHECK(first_buffer.pending());
  CHECK(second_buffer.pending());
  CHECK_FALSE(extra_buffer.pending());
  CHECK(third_buffer.pending());
  CHECK(fourth_buffer.pending());
  state.drain_fails = false;
  REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
  CHECK(first == 0);
  CHECK(second == 0);
  CHECK(ctx->failed_work_leases.empty());
  CHECK_FALSE(first_buffer.pending());
  CHECK_FALSE(second_buffer.pending());
  CHECK_FALSE(third_buffer.pending());
  CHECK_FALSE(fourth_buffer.pending());
  REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
  CHECK(first == 0);
  CHECK(second == 0);
  ctx->synchronous_test_hooks = nullptr;
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("owner-only completion records retain uploads without a fake matrix writer",
          "[api][lifetime]") {
  using namespace rns8::detail::api;
  auto* ctx = create_cpu_context();
  uint64_t owner = 0;
  SynchronousDispatchFixture state;
  state.drain_fails = true;
  const rns8_synchronous_test_hooks hooks{&state, nullptr, &SynchronousDispatchFixture::drain};
  ctx->synchronous_test_hooks = &hooks;
  size_t token = 0;
  REQUIRE(begin_synchronous_owner_lease(*ctx, token) == RNS8_SUCCESS);
  SynchronousLease custody(*ctx, token);
  REQUIRE(add_synchronous_owner_hold(*ctx, token, owner) == RNS8_SUCCESS);
  REQUIRE(add_synchronous_owner_hold(*ctx, token, owner) == RNS8_SUCCESS);
  CHECK(owner == 1);
  CHECK(custody.begin_write() == RNS8_INVALID_ARGUMENT);
  CHECK(ctx->failed_work_leases[token].a == nullptr);
  CHECK(ctx->failed_work_leases[token].c == nullptr);
  // Releasing a newer ordinary record must not prune the live private record.
  rns8_matrix a{}, b{}, c{};
  rns8_workspace work{};
  size_t later = 0;
  REQUIRE(begin_synchronous_lease(*ctx, a, b, c, work, later) == RNS8_SUCCESS);
  {
    SynchronousLease ordinary(*ctx, later);
    REQUIRE(ordinary.finish(RNS8_SUCCESS) == RNS8_SUCCESS);
  }
  REQUIRE(ctx->failed_work_leases.size() == 1);
  CHECK(owner == 1);
  CHECK(custody.finish(RNS8_BACKEND_FAILURE) == RNS8_BACKEND_FAILURE);
  CHECK(ctx->hip_failure_poisoned);
  CHECK(owner == 1);
  state.drain_fails = false;
  REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
  CHECK(owner == 0);
  CHECK(ctx->failed_work_leases.empty());
  REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
  CHECK(owner == 0);
  ctx->synchronous_test_hooks = nullptr;
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("public sparse writers retain packed owners and distinguish prewrite from issued failure",
          "[api][sparse][lifetime]") {
  for (auto semantics : {RNS8_BOUNDED_I64, RNS8_EXACT_WIDE_SIGNED, RNS8_FINITE_RING_U8})
    for (unsigned failure = 0; failure < 6; ++failure) {
      CAPTURE(semantics, failure);
      const bool finite = semantics == RNS8_FINITE_RING_U8, bounded = semantics == RNS8_BOUNDED_I64;
      auto* ctx = create_cpu_context();
      auto desc = finite    ? finite_gemm_desc(semantics, 1, 1, 4, 251)
                  : bounded ? bounded_gemm_desc(semantics, 1, 1, 4)
                            : exact_desc(semantics, 1, 1, 4);
      auto bmd = finite    ? finite_matrix_desc(4, 1, semantics)
                 : bounded ? bounded_matrix_desc(4, 1, semantics)
                           : exact_matrix_desc(4, 1, semantics);
      auto cmd = bmd;
      cmd.rows = 1;
      rns8_plan* plan = nullptr;
      rns8_workspace* work = nullptr;
      rns8_matrix *b = nullptr, *c = nullptr;
      rns8_sparse_matrix* a = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx, plan, &work) == RNS8_SUCCESS);
      REQUIRE(rns8_create_matrix(ctx, &bmd, &b) == RNS8_SUCCESS);
      REQUIRE(rns8_create_matrix(ctx, &cmd, &c) == RNS8_SUCCESS);
      const int64_t bv[]{3, 4, 5, 6};
      const uint8_t fb[]{3, 4, 5, 6};
      REQUIRE((finite ? rns8_pack_finite_u8(ctx, b, 251, fb, 1, 1)
                      : rns8_pack_i64(ctx, b, bv, 1, 1)) == RNS8_SUCCESS);
      rns8_sparse_matrix_desc sd{};
      sd.struct_size = sizeof(sd);
      sd.abi_version = RNS8_ABI_VERSION;
      sd.semantics = semantics;
      sd.bound_kind = desc.bound_kind;
      sd.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
      sd.sparse_operand = RNS8_SPARSE_OPERAND_A;
      sd.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
      sd.value_signedness = finite ? RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8
                                   : RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
      sd.rows = 1;
      sd.expanded_k = 4;
      sd.group_size = 4;
      sd.nonzeros_per_group = 2;
      sd.max_prefix = finite ? 0 : bmd.max_prefix;
      sd.finite_modulus = finite ? 251 : 0;
      REQUIRE(rns8_create_sparse_matrix(ctx, &sd, &a) == RNS8_SUCCESS);
      std::vector<uint8_t> dense(4 * (finite ? 1 : sd.max_prefix));
      for (std::size_t index = 0; index < dense.size(); index += 4) {
        dense[index] = 1;
        dense[index + 2] = 2;
      }
      REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, a, dense.data(), 4, 1) == RNS8_SUCCESS);
      auto execute = [&]() {
        return finite ? rns8_gemm_finite_u8_sparse_a(ctx, plan, 251, a, b, c, work)
                      : rns8_gemm_rns_sparse_a(ctx, plan, a, b, c, work);
      };
      REQUIRE(execute() == RNS8_SUCCESS);
      const auto generation = c->content_generation;
      const auto old = c->residues;
      SynchronousDispatchFixture state;
      state.readers = {b};
      state.writers = {c};
      state.workspace = work;
      state.private_owners = {&a->in_flight_readers};
      state.fail = failure != 0;
      state.write_started_on_failure = failure != 1 && failure != 5;
      state.throw_after_issue = failure == 4 || failure == 5;
      state.drain_fails = failure == 3 || failure == 4;
      const rns8_synchronous_test_hooks hooks{&state, &SynchronousDispatchFixture::dispatch,
                                              &SynchronousDispatchFixture::drain};
      ctx->synchronous_test_hooks = &hooks;
      CHECK(execute() == (failure == 0              ? RNS8_SUCCESS
                          : state.throw_after_issue ? RNS8_INTERNAL_ERROR
                                                    : RNS8_BACKEND_FAILURE));
      CHECK(state.calls == 1);
      CHECK(state.reservations_observed);
      CHECK(c->residues == old);
      CHECK(c->content_generation == generation + (failure == 0));
      CHECK(c->host_residues_current == (failure == 0 || !state.write_started_on_failure));
      if (state.drain_fails) {
        CHECK(a->in_flight_readers == 1);
        CHECK(ctx->hip_failure_poisoned);
        CHECK(b->in_flight_readers == 1);
        CHECK(c->in_flight_writer);
        CHECK(work->in_flight);
        CHECK(rns8_destroy_sparse_matrix(a) == RNS8_BUSY);
        CHECK(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, a, dense.data(), 4, 2) == RNS8_BUSY);
        CHECK(a->source_version == 1);
        CHECK(rns8_destroy_matrix(b) == RNS8_BUSY);
        CHECK(rns8_destroy_matrix(c) == RNS8_BUSY);
        state.drain_fails = false;
        REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
      }
      CHECK(a->in_flight_readers == 0);
      CHECK(a->in_flight_writers == 0);
      CHECK(b->in_flight_readers == 0);
      CHECK_FALSE(c->in_flight_writer);
      CHECK_FALSE(work->in_flight);
      CHECK(ctx->failed_work_leases.empty());
      ctx->synchronous_test_hooks = nullptr;
      REQUIRE(rns8_destroy_sparse_matrix(a) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(b) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(work) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
}

TEST_CASE("sparse upload failure seam retains real host and device owners without a dense writer",
          "[api][sparse][hip][lifetime]") {
  using namespace rns8::detail;
  if (!amdgpu_builtins_compiled())
    SKIP("AMDGPU builtin sparse storage provider not compiled");
  for (bool finite : {false, true})
    for (bool late : {false, true})
      for (bool throws : {false, true})
        for (bool uncertain : {false, true}) {
          CAPTURE(finite, late, throws, uncertain);
          rns8_context_options options{};
          options.struct_size = sizeof(options);
          options.abi_version = RNS8_ABI_VERSION;
          options.requested_backend = RNS8_BACKEND_AMDGPU_BUILTINS;
          rns8_context* ctx = nullptr;
          const auto created = rns8_create_context(0, &options, &ctx);
          if (created == RNS8_UNSUPPORTED_BACKEND || created == RNS8_UNSUPPORTED_ARCH)
            SKIP("no applicable AMDGPU storage device");
          REQUIRE(created == RNS8_SUCCESS);
          rns8_sparse_matrix_desc sd{};
          sd.struct_size = sizeof(sd);
          sd.abi_version = RNS8_ABI_VERSION;
          sd.semantics = finite ? RNS8_FINITE_RING_U8 : RNS8_EXACT_WIDE_SIGNED;
          sd.bound_kind = RNS8_BOUND_NONE;
          sd.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
          sd.sparse_operand = RNS8_SPARSE_OPERAND_A;
          sd.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
          sd.value_signedness = finite ? RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8
                                       : RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
          sd.rows = 1;
          sd.expanded_k = 4;
          sd.group_size = 4;
          sd.nonzeros_per_group = 2;
          sd.max_prefix = finite ? 0 : 3;
          sd.finite_modulus = finite ? 251 : 0;
          rns8_sparse_matrix* matrix = nullptr;
          REQUIRE(rns8_create_sparse_matrix(ctx, &sd, &matrix) == RNS8_SUCCESS);
          std::vector<uint8_t> values(4 * (finite ? 1 : 3));
          for (std::size_t index = 0; index < values.size(); index += 4) {
            values[index] = 1;
            values[index + 2] = 2;
          }
          REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, matrix, values.data(), 4, 1) ==
                  RNS8_SUCCESS);
          REQUIRE(matrix->device_current);
          const auto packed = matrix->packed_values, indices = matrix->packed_indices;
          const auto key = matrix->cache_key;
          const auto* device_values = matrix->hip_packed_values;
          const auto* device_indices = matrix->hip_packed_indices;
          SynchronousDispatchFixture state;
          state.fail = true;
          state.write_started_on_failure = late;
          state.throw_after_issue = throws;
          state.drain_fails = uncertain;
          state.private_owners = {&matrix->in_flight_writers};
          const rns8_synchronous_test_hooks hooks{&state, &SynchronousDispatchFixture::dispatch,
                                                  &SynchronousDispatchFixture::drain};
          ctx->synchronous_test_hooks = &hooks;
          values[0] = 3;
          // The checkpoint fails before actual H2D; late means simulated issue only.
          CHECK(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, matrix, values.data(), 4, 2) ==
                (throws ? RNS8_INTERNAL_ERROR : RNS8_BACKEND_FAILURE));
          CHECK(state.calls == 1);
          CHECK(state.reservations_observed);
          CHECK(matrix->packed_values == packed);
          CHECK(matrix->packed_indices == indices);
          CHECK(matrix->source_version == 1);
          CHECK(matrix->cache_key == key);
          CHECK(matrix->host_current);
          CHECK(matrix->device_current == !late);
          if (uncertain) {
            CHECK(ctx->hip_failure_poisoned);
            CHECK(matrix->in_flight_writers == 1);
            CHECK(matrix->packed_values_owner.pending());
            CHECK(matrix->packed_indices_owner.pending());
            CHECK(rns8_destroy_sparse_matrix(matrix) == RNS8_BUSY);
            CHECK(matrix->hip_packed_values == device_values);
            CHECK(matrix->hip_packed_indices == device_indices);
            std::vector<uint8_t> expanded(values.size(), 91);
            CHECK(rns8_expand_sparse_a_4_to_2_matrix_u8(matrix, expanded.data(), 4) == RNS8_BUSY);
            CHECK(expanded == std::vector<uint8_t>(values.size(), 91));
            state.drain_fails = false;
            REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
          }
          CHECK(matrix->in_flight_writers == 0);
          CHECK_FALSE(matrix->packed_values_owner.pending());
          CHECK_FALSE(matrix->packed_indices_owner.pending());
          CHECK(ctx->failed_work_leases.empty());
          ctx->synchronous_test_hooks = nullptr;
          if (!uncertain) {
            REQUIRE(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx, matrix, values.data(), 4, 3) ==
                    RNS8_SUCCESS);
            CHECK(matrix->source_version == 3);
            CHECK(matrix->cache_key.find("source_version=3") != std::string::npos);
            CHECK(matrix->device_current);
          }
          REQUIRE(rns8_destroy_sparse_matrix(matrix) == RNS8_SUCCESS);
          REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
        }
}

TEST_CASE("unpublished prepack retirement retains staging until completion custody releases",
          "[api][prepack][lifetime]") {
  using namespace rns8::detail::api;
  for (bool native : {false, true})
    for (bool throws : {false, true}) {
      CAPTURE(native, throws);
      auto* ctx = create_cpu_context();
      rns8_matrix source{};
      SynchronousDispatchFixture state;
      state.drain_fails = true;
      const rns8_synchronous_test_hooks hooks{&state, nullptr, &SynchronousDispatchFixture::drain};
      ctx->synchronous_test_hooks = &hooks;
      auto retire = [&](rns8_prepack_cache* cache) {
        retire_unpublished_prepack_cache(*ctx, cache);
      };
      auto* retained = new rns8_prepack_cache();
      const std::vector<uint64_t> words{0, UINT64_MAX, uint64_t{1} << 63};
      retained->native_staging_words = words;
      try {
        std::unique_ptr<rns8_prepack_cache, decltype(retire)> owner(retained, retire);
        size_t token = 0;
        REQUIRE((native ? begin_synchronous_owner_lease(*ctx, token)
                        : begin_synchronous_read_lease(*ctx, source, token)) == RNS8_SUCCESS);
        SynchronousLease custody(*ctx, token);
        REQUIRE(add_synchronous_owner_hold(*ctx, token, retained->in_flight_readers) ==
                RNS8_SUCCESS);
        REQUIRE(custody.hold_buffer(retained->data_owner) == RNS8_SUCCESS);
        REQUIRE(custody.hold_buffer(retained->native_upload_owner) == RNS8_SUCCESS);
        if (throws)
          throw std::bad_alloc{};
        CHECK(custody.finish(RNS8_BACKEND_FAILURE) == RNS8_BACKEND_FAILURE);
      } catch (const std::bad_alloc&) {
        CHECK(throws);
      }
      REQUIRE(ctx->failed_prepack_cleanup == retained);
      CHECK(ctx->hip_failure_poisoned);
      CHECK(retained->in_flight_readers == 1);
      CHECK(retained->native_staging_words == words);
      CHECK(retained->data_owner.pending());
      CHECK(retained->native_upload_owner.pending());
      CHECK(source.in_flight_readers == (native ? 0 : 1));
      CHECK(rns8_destroy_prepack_cache(retained) == RNS8_BUSY);
      retire_unpublished_prepack_cache(*ctx, retained);
      CHECK(retained->failed_cleanup_next == nullptr); // duplicate retirement cannot form a cycle
      CHECK(rns8_drain_failed_work(ctx) == RNS8_BACKEND_FAILURE);
      CHECK(ctx->failed_prepack_cleanup == retained);
      CHECK(retained->native_staging_words == words);
      state.drain_fails = false;
      REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
      CHECK(ctx->failed_prepack_cleanup == nullptr);
      CHECK(ctx->failed_work_leases.empty());
      CHECK(source.in_flight_readers == 0);
      CHECK(ctx->hip_failure_poisoned);
      REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
      ctx->synchronous_test_hooks = nullptr;
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
}

TEST_CASE("public prepack construction retains unpublished device and native staging owners",
          "[api][prepack][hip][lifetime]") {
  if (!rns8::detail::rocwmma_compiled())
    SKIP("rocWMMA prepack provider not compiled");
  for (unsigned kind = 0; kind < 3; ++kind)
    for (bool publish : {false, true})
      for (bool throws : {false, true})
        for (bool uncertain : {false, true}) {
          CAPTURE(kind, publish, throws, uncertain);
          const bool native = kind != 0, u = kind == 2;
          rns8_context_options options{};
          options.struct_size = sizeof(options);
          options.abi_version = RNS8_ABI_VERSION;
          options.requested_backend = RNS8_BACKEND_ROCWMMA;
          rns8_context* ctx = nullptr;
          const auto created = rns8_create_context(0, &options, &ctx);
          if (created == RNS8_UNSUPPORTED_BACKEND || created == RNS8_UNSUPPORTED_ARCH)
            SKIP("no applicable rocWMMA prepack device");
          REQUIRE(created == RNS8_SUCCESS);
          auto desc = bounded_gemm_desc(u ? RNS8_BOUNDED_U64 : RNS8_BOUNDED_I64, 1, 3, 2);
          desc.requested_backend = RNS8_BACKEND_ROCWMMA;
          rns8_plan* plan = nullptr;
          REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
          constexpr int64_t ld = 8; // minimal final-row extent, with padding on preceding rows
          std::vector<int64_t> signed_values{INT64_MIN, -1, INT64_MAX, 77, 77, 77,
                                             77,        77, 0,         1,  -7};
          std::vector<uint64_t> unsigned_values{
              0, UINT64_MAX, uint64_t{1} << 63, 77, 77, 77, 77, 77, 1, 7, UINT64_MAX - 1};
          std::vector<uint64_t> expected_words;
          for (size_t row = 0; row < 2; ++row)
            for (size_t col = 0; col < 3; ++col)
              expected_words.push_back(u ? unsigned_values[row * ld + col]
                                         : static_cast<uint64_t>(signed_values[row * ld + col]));
          rns8_matrix* source = nullptr;
          if (!native) {
            auto md = bounded_matrix_desc(2, 3, desc.semantics);
            REQUIRE(rns8_create_matrix(ctx, &md, &source) == RNS8_SUCCESS);
            REQUIRE(rns8_pack_i64(ctx, source, signed_values.data(), ld, 1) == RNS8_SUCCESS);
          }
          SynchronousDispatchFixture state;
          state.fail = true;
          state.throw_after_issue = throws;
          state.drain_fails = uncertain;
          state.owner_context = ctx;
          state.expect_owner_only = native;
          if (source)
            state.readers = {source};
          state.target_operation =
              native ? (publish ? "prepack_native_publish" : "prepack_native_create")
                     : (publish ? "prepack_rns_publish" : "prepack_rns_create");
          const rns8_synchronous_test_hooks hooks{&state, &SynchronousDispatchFixture::dispatch,
                                                  &SynchronousDispatchFixture::drain};
          ctx->synchronous_test_hooks = &hooks;
          rns8_prepack_cache* cache = nullptr;
          const auto status =
              !native ? rns8_create_prepack_cache(ctx, plan, source, RNS8_OPERAND_B, &cache)
              : u ? rns8_create_b_prepack_cache_u64(ctx, plan, unsigned_values.data(), ld, 1,
                                                    &cache)
                  : rns8_create_b_prepack_cache_i64(ctx, plan, signed_values.data(), ld, 1, &cache);
          CHECK(status == (throws ? RNS8_INTERNAL_ERROR : RNS8_BACKEND_FAILURE));
          CHECK(cache == nullptr);
          CHECK(state.calls == (publish ? 2 : 1));
          CHECK(state.reservations_observed);
          if (uncertain) {
            auto* retained = ctx->failed_prepack_cleanup;
            REQUIRE(retained != nullptr);
            CHECK(ctx->hip_failure_poisoned);
            CHECK(retained->in_flight_readers == 1);
            CHECK(retained->data_owner.pending());
            CHECK(retained->native_upload_owner.pending());
            CHECK(rns8_destroy_prepack_cache(retained) == RNS8_BUSY);
            if (native) {
              REQUIRE(retained->native_staging_words == expected_words);
              std::fill(signed_values.begin(), signed_values.end(), 0);
              std::fill(unsigned_values.begin(), unsigned_values.end(), 0);
              CHECK(retained->native_staging_words == expected_words);
              CHECK(retained->native_upload_owner.get() != nullptr);
            } else {
              CHECK(source->in_flight_readers == 1);
              CHECK(rns8_destroy_matrix(source) == RNS8_BUSY);
            }
            state.drain_fails = false;
            REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
          }
          CHECK(ctx->failed_prepack_cleanup == nullptr);
          CHECK(ctx->failed_work_leases.empty());
          ctx->synchronous_test_hooks = nullptr;
          REQUIRE(rns8_destroy_matrix(source) == RNS8_SUCCESS);
          REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
          REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
        }
}

#include "test_api_cache_dispatch_cases.inc"
#include "test_api_ragged_dispatch_cases.inc"
#include "test_api_basis_dispatch_cases.inc"
#include "test_api_block_sparse_dispatch_cases.inc"
