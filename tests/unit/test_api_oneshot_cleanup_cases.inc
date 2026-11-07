namespace {
struct OneShotCleanupFixture {
  std::array<const void*, 5> owners{};
  std::array<unsigned, 5> attempts{}, released{};
  uint32_t failures = 0;
  static rns8_status release(void* opaque, const char*, const void* pointer) noexcept {
    auto& self = *static_cast<OneShotCleanupFixture*>(opaque);
    for (unsigned index = 0; index < self.owners.size(); ++index)
      if (pointer == self.owners[index]) {
        ++self.attempts[index];
        if (self.failures & (1u << index))
          return RNS8_INTERNAL_ERROR;
        ++self.released[index];
        return RNS8_SUCCESS;
      }
    return RNS8_INTERNAL_ERROR;
  }
};
} // namespace
TEST_CASE("one shot aggregate cleanup reports failures and retires every whole owner",
          "[api][oneshot][lifetime]") {
  using namespace rns8::detail::api;
  for (unsigned position = 0; position < 5; ++position)
    for (bool original_failure : {false, true}) {
      CAPTURE(position, original_failure);
      auto* ctx = create_cpu_context();
      OneShotCleanupFixture fixture;
      const rns8_cleanup_test_hooks hooks{&fixture, &OneShotCleanupFixture::release};
      {
        resident_oneshot_state state(ctx);
        auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
        REQUIRE(create_resident_oneshot_state(ctx, desc, state) == RNS8_SUCCESS);
        fixture.owners = {state.workspace, state.C, state.B, state.A, state.plan};
        fixture.failures = 1u << position;
        ctx->cleanup_test_hooks = &hooks;
        const auto operation = original_failure ? RNS8_RANGE_ERROR : RNS8_SUCCESS;
        CHECK(state.finish(operation) ==
              (original_failure ? RNS8_RANGE_ERROR : RNS8_INTERNAL_ERROR));
        CHECK(state.cleanup_status == RNS8_INTERNAL_ERROR);
        CHECK(fixture.released[position] == 0);
      }
      CHECK(ctx->hip_failure_poisoned);
      REQUIRE(ctx->failed_plan_cleanup);
      CHECK((ctx->failed_workspace_cleanup != nullptr) == (position == 0));
      CHECK((ctx->failed_matrix_cleanup != nullptr) == (position >= 1 && position <= 3));
      CHECK(rns8_destroy_context(ctx) == RNS8_INTERNAL_ERROR);
      CHECK(fixture.released[position] == 0);
      fixture.failures = 0;
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
      for (auto released : fixture.released)
        CHECK(released == 1);
    }
}

TEST_CASE("one shot aggregate construction unwind uses existing intrusive cleanup owners",
          "[api][oneshot][lifetime]") {
  using namespace rns8::detail::api;
  auto* ctx = create_cpu_context();
  OneShotCleanupFixture fixture;
  const rns8_cleanup_test_hooks hooks{&fixture, &OneShotCleanupFixture::release};
  try {
    resident_oneshot_state state(ctx);
    auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
    REQUIRE(create_resident_oneshot_state(ctx, desc, state) == RNS8_SUCCESS);
    fixture.owners = {state.workspace, state.C, state.B, state.A, state.plan};
    fixture.failures = 31;
    ctx->cleanup_test_hooks = &hooks;
    throw std::bad_alloc{};
  } catch (const std::bad_alloc&) {
  }
  REQUIRE(ctx->failed_workspace_cleanup);
  REQUIRE(ctx->failed_matrix_cleanup);
  REQUIRE(ctx->failed_plan_cleanup);
  unsigned matrices = 0;
  for (auto* matrix = ctx->failed_matrix_cleanup; matrix; matrix = matrix->failed_cleanup_next)
    ++matrices;
  CHECK(matrices == 3);
  for (auto released : fixture.released)
    CHECK(released == 0);
  fixture.failures = 0;
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  for (auto released : fixture.released)
    CHECK(released == 1);
}

TEST_CASE("public one shot retains its failed import matrix until completion is established",
          "[api][oneshot][lifetime]") {
  auto* ctx = create_cpu_context();
  auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
  SynchronousDispatchFixture fixture;
  fixture.fail = fixture.drain_fails = true;
  const rns8_synchronous_test_hooks hooks{&fixture, &SynchronousDispatchFixture::dispatch,
                                          &SynchronousDispatchFixture::drain};
  ctx->synchronous_test_hooks = &hooks;
  const int64_t one = 1;
  int64_t output = 77;
  CHECK(rns8_gemm_i64_oneshot(ctx, &desc, &one, 1, &one, 1, &output, 1) == RNS8_BACKEND_FAILURE);
  CHECK(output == 77);
  REQUIRE(ctx->failed_matrix_cleanup);
  REQUIRE(ctx->failed_plan_cleanup);
  REQUIRE(ctx->failed_work_leases.size() == 1);
  CHECK(ctx->failed_work_leases[0].a == ctx->failed_matrix_cleanup);
  CHECK(ctx->failed_matrix_cleanup->in_flight_writer);
  CHECK(rns8_destroy_context(ctx) == RNS8_BACKEND_FAILURE);
  fixture.drain_fails = false;
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("native one shot retains owned output and temporary inputs after simulated failure",
          "[api][hip][oneshot][lifetime]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  struct Failure {
    const char* stage = nullptr;
    bool drain_fails = true;
    unsigned calls = 0;
    static rns8_status dispatch(void* opaque, const char* stage, bool*) {
      auto& self = *static_cast<Failure*>(opaque);
      if (std::string(stage) != self.stage)
        return RNS8_SUCCESS;
      ++self.calls;
      return RNS8_BACKEND_FAILURE;
    }
    static rns8_status drain(void* opaque) noexcept {
      return static_cast<Failure*>(opaque)->drain_fails ? RNS8_BACKEND_FAILURE : RNS8_SUCCESS;
    }
  };
  for (const auto* stage : {"native_oneshot_prepare", "native_oneshot_compute"}) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_context* ctx = nullptr;
    if (rns8_create_context(0, &options, &ctx) != RNS8_SUCCESS)
      SKIP("HIP device required");
    auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
    desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
    desc.max_prefix = 9;
    Failure failure;
    failure.stage = stage;
    const rns8_synchronous_test_hooks hooks{&failure, &Failure::dispatch, &Failure::drain};
    ctx->synchronous_test_hooks = &hooks;
    rns8::detail::hip_direct_allocation_counters_reset();
    const int64_t one = 1;
    int64_t output = 77;
    CHECK(rns8_gemm_i64_oneshot(ctx, &desc, &one, 1, &one, 1, &output, 1) == RNS8_BACKEND_FAILURE);
    CHECK(output == 77);
    CHECK(failure.calls == 1);
    REQUIRE(ctx->failed_matrix_cleanup);
    REQUIRE(ctx->failed_plan_cleanup);
    CHECK(ctx->failed_matrix_cleanup->in_flight_writer);
    CHECK(ctx->failed_work_leases.size() == 1);
    const auto before = rns8::detail::hip_direct_allocation_counters_snapshot();
    CHECK(before.free_calls == 0);
    CHECK(before.allocate_calls >= (std::string(stage) == "native_oneshot_compute" ? 3 : 1));
    CHECK(rns8_destroy_context(ctx) == RNS8_BACKEND_FAILURE);
    // No failed kernel was submitted: transfers, if any, completed before the
    // private checkpoint. Context teardown also drains the actual deferred queue.
    failure.drain_fails = false;
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    CHECK(rns8::detail::hip_direct_allocation_counters_snapshot().free_calls ==
          before.allocate_calls);
  }
#else
  SKIP("HIP not compiled");
#endif
}
