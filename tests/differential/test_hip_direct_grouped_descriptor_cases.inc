TEST_CASE("Direct-HIP grouped descriptor validates task ownership and phase contracts") {
  if (!hip_available()) {
    return;
  }

  rns8_context* ctx = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto desc = unsigned_desc(2, 2, 2, 1000, RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);

  rns8_workspace* workspace0 = nullptr;
  rns8_workspace* workspace1 = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace0) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace1) == RNS8_SUCCESS);

  rns8_matrix_desc a_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  rns8_matrix_desc b_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  rns8_matrix_desc c_desc = matrix_desc(2, 2, RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  a_desc.max_prefix = plan->prefix;
  b_desc.max_prefix = plan->prefix;
  c_desc.max_prefix = plan->prefix;

  rns8_matrix* a0 = nullptr;
  rns8_matrix* b0 = nullptr;
  rns8_matrix* c0 = nullptr;
  rns8_matrix* a1 = nullptr;
  rns8_matrix* b1 = nullptr;
  rns8_matrix* c1 = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a0) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b0) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c0) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &a_desc, &a1) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &b_desc, &b1) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &c_desc, &c1) == RNS8_SUCCESS);

  rns8::detail::hip_direct_grouped_gemm_task tasks[2] = {
      {a0, b0, c0, workspace0},
      {a1, b1, c1, workspace1},
  };
  rns8::detail::hip_direct_grouped_gemm_descriptor descriptor{};
  descriptor.plan = plan;
  descriptor.tasks = tasks;
  descriptor.task_count = 2;
  descriptor.semantics = RNS8_BOUNDED_U64;
  descriptor.m = 2;
  descriptor.n = 2;
  descriptor.k = 2;
  descriptor.prefix = plan->prefix;

  int descriptor_device = -1;
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_setup(descriptor, &descriptor_device) ==
        RNS8_SUCCESS);
  CHECK(descriptor_device == 0);
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_pack(descriptor, 101) ==
        RNS8_INVALID_ARGUMENT);

  const uint64_t a_input0[] = {1, 2, 3, 4};
  const uint64_t b_input0[] = {5, 6, 7, 8};
  const uint64_t a_input1[] = {9, 10, 11, 12};
  const uint64_t b_input1[] = {13, 14, 15, 16};
  REQUIRE(rns8_pack_u64(ctx, a0, a_input0, 2, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b0, b_input0, 2, 101) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, a1, a_input1, 2, 102) == RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx, b1, b_input1, 2, 102) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_pack(descriptor, 101) ==
        RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_gemm(descriptor) ==
        RNS8_INVALID_ARGUMENT);

  REQUIRE(rns8_gemm_rns(ctx, plan, a0, b0, c0, workspace0) == RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx, plan, a1, b1, c1, workspace1) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_gemm(descriptor) ==
        RNS8_SUCCESS);

  ++c1->source_version;
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_gemm(descriptor) ==
        RNS8_INVALID_ARGUMENT);
  --c1->source_version;

  rns8::detail::hip_direct_grouped_gemm_task shared_workspace_tasks[2] = {
      {a0, b0, c0, workspace0},
      {a1, b1, c1, workspace0},
  };
  auto shared_workspace_descriptor = descriptor;
  shared_workspace_descriptor.tasks = shared_workspace_tasks;
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_setup(shared_workspace_descriptor, nullptr) ==
        RNS8_INVALID_ARGUMENT);

  rns8_destroy_matrix(c1);
  rns8_destroy_matrix(b1);
  rns8_destroy_matrix(a1);
  rns8_destroy_matrix(c0);
  rns8_destroy_matrix(b0);
  rns8_destroy_matrix(a0);
  rns8_destroy_workspace(workspace1);
  rns8_destroy_workspace(workspace0);
  rns8_destroy_plan(plan);
  rns8_destroy_context(ctx);
}
