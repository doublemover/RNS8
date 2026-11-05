TEST_CASE("public block-sparse construction retains complete unpublished owners on failure",
          "[api][block-sparse][hip][lifetime]") {
  if (!rns8::detail::api::direct_hip_compiled())
    SKIP("direct HIP block-sparse provider not compiled");
  for (const char* phase :
       {"block_sparse_download", "block_sparse_upload", "block_sparse_prepare_publish"})
    for (bool throws : {false, true})
      for (bool uncertain : {false, true}) {
        CAPTURE(phase, throws, uncertain);
        auto* ctx = create_dispatch_context(RNS8_BACKEND_HIP_DIRECT);
        auto md = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
        md.flags |= RNS8_MATRIX_RETAIN_NATIVE_INPUT;
        rns8_matrix* source = nullptr;
        REQUIRE(rns8_create_matrix(ctx, &md, &source) == RNS8_SUCCESS);
        const int64_t value = -7;
        REQUIRE(rns8_pack_i64(ctx, source, &value, 1, 1) == RNS8_SUCCESS);
        const auto generation = source->content_generation;
        source->in_flight_writer = true;
        rns8_block_sparse_operand* owner = nullptr;
        CHECK(rns8_prepare_block_sparse_operand(ctx, source, RNS8_BLOCK_SPARSE_LHS, UINT64_MAX,
                                                &owner) == RNS8_BUSY);
        source->in_flight_writer = false;
        SynchronousDispatchFixture state;
        state.readers = {source};
        state.owner_context = ctx;
        state.target_operation = phase;
        state.fail = true;
        state.throw_after_issue = throws;
        state.drain_fails = uncertain;
        const rns8_synchronous_test_hooks hooks{&state, &SynchronousDispatchFixture::dispatch,
                                                &SynchronousDispatchFixture::drain};
        ctx->synchronous_test_hooks = &hooks;
        auto* prior = reinterpret_cast<rns8_block_sparse_operand*>(uintptr_t{7});
        owner = prior;
        CHECK(rns8_prepare_block_sparse_operand(ctx, source, RNS8_BLOCK_SPARSE_LHS, UINT64_MAX,
                                                &owner) ==
              (throws ? RNS8_INTERNAL_ERROR : RNS8_BACKEND_FAILURE));
        CHECK(owner == prior);
        CHECK(state.reservations_observed);
        CHECK(source->content_generation == generation);
        CHECK(source->device_native_current);
        if (uncertain) {
          CHECK(ctx->hip_failure_poisoned);
          REQUIRE(ctx->failed_block_sparse_cleanup != nullptr);
          CHECK(source->in_flight_readers == 1);
          CHECK(source->prepared_references == 1);
          CHECK(ctx->prepared_references == 0);
          REQUIRE(ctx->failed_work_leases.size() == 1);
          REQUIRE(ctx->failed_work_leases.front().buffer_holds[0] != nullptr);
          CHECK(ctx->failed_work_leases.front().buffer_holds[0]->pending());
          CHECK(rns8_destroy_matrix(source) == RNS8_BUSY);
          CHECK(rns8_destroy_context(ctx) == RNS8_BACKEND_FAILURE);
          state.drain_fails = false;
          REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
        }
        CHECK(ctx->failed_block_sparse_cleanup == nullptr);
        CHECK(ctx->failed_work_leases.empty());
        CHECK(source->in_flight_readers == 0);
        CHECK(source->prepared_references == 0);
        CHECK(ctx->prepared_references == 0);
        ctx->synchronous_test_hooks = nullptr;
        REQUIRE(rns8_destroy_matrix(source) == RNS8_SUCCESS);
        REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
      }
}
