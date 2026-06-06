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

  const uint64_t grouped_a_slab[] = {1, 2, 3, 4, 9, 10, 11, 12};
  const uint64_t grouped_b_slab[] = {5, 6, 7, 8, 13, 14, 15, 16};

  rns8::detail::hip_direct_grouped_device_resources grouped_resources;
  REQUIRE(rns8::detail::hip_direct_allocate_grouped_task_device_resources(
              descriptor,
              8 * sizeof(uint64_t),
              8 * sizeof(uint64_t),
              8 * sizeof(uint64_t),
              sizeof(int),
              &grouped_resources) == RNS8_SUCCESS);
  int pointer_device = -1;
  REQUIRE(rns8::detail::hip_direct_prepare_grouped_task_residue_pointers(
              descriptor,
              grouped_resources,
              &pointer_device) == RNS8_SUCCESS);
  CHECK(pointer_device == 0);
  CHECK(grouped_resources.device_id == 0);
  REQUIRE(grouped_resources.c_matrices.size() == 2);
  REQUIRE(rns8::detail::hip_direct_pack_grouped_u64_task_inputs(
              descriptor,
              grouped_resources,
              grouped_a_slab,
              grouped_b_slab,
              2,
              2,
              101) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_pack(descriptor, 101) ==
        RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_gemm(descriptor) ==
        RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8::detail::hip_direct_gemm_grouped_rns_task_outputs(
              descriptor,
              grouped_resources) == RNS8_SUCCESS);
  CHECK(rns8::detail::hip_direct_validate_grouped_gemm_descriptor_after_gemm(descriptor) ==
        RNS8_SUCCESS);

  std::vector<uint64_t> grouped_export(8, UINT64_C(0xdeadbeefdeadbeef));
  REQUIRE(rns8::detail::hip_direct_export_grouped_u64_task_outputs_to_host(
              descriptor,
              grouped_resources,
              static_cast<uint64_t>(desc.bound),
              grouped_export.data(),
              grouped_export.size()) == RNS8_SUCCESS);
  std::vector<uint64_t> expected0(4);
  std::vector<uint64_t> expected1(4);
  REQUIRE(rns8_export_u64(ctx, plan, c0, expected0.data(), 2) == RNS8_SUCCESS);
  REQUIRE(rns8_export_u64(ctx, plan, c1, expected1.data(), 2) == RNS8_SUCCESS);
  CHECK(std::equal(expected0.begin(), expected0.end(), grouped_export.begin()));
  CHECK(std::equal(expected1.begin(), expected1.end(), grouped_export.begin() + 4));
  CHECK(grouped_resources.reset() == RNS8_SUCCESS);

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
