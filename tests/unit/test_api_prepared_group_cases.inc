TEST_CASE("heterogeneous prepared groups preflight all tasks before executing", "[api][prepared]") {
  auto* ctx = create_cpu_context();
  rns8_plan* plans[2]{};
  rns8_workspace* work[2]{};
  rns8_matrix *a[2]{}, *b[2]{}, *c[2]{};
  rns8_prepared_execution* executions[2]{};
  for (int i = 0; i < 2; ++i) {
    const int size = i + 1;
    auto desc = i ? finite_gemm_desc(RNS8_FINITE_RING_U8, size, size, size, 251)
                  : bounded_gemm_desc(RNS8_BOUNDED_I64, size, size, size);
    if (!i) desc.bound = 10000;
    auto md = i ? finite_matrix_desc(size, size, RNS8_FINITE_RING_U8)
                : bounded_matrix_desc(size, size, RNS8_BOUNDED_I64);
    REQUIRE(rns8_create_plan(ctx, &desc, &plans[i]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plans[i], &work[i]) == RNS8_SUCCESS);
    for (auto** out : {&a[i],&b[i],&c[i]}) REQUIRE(rns8_create_matrix(ctx, &md, out) == RNS8_SUCCESS);
    const int64_t x = 7, y = -3;
    const uint8_t av[] = {1,2,3,4}, bv[] = {5,6,7,8};
    REQUIRE((i ? rns8_pack_finite_u8(ctx, a[i], 251, av, size, 0)
               : rns8_pack_i64(ctx, a[i], &x, size, 0)) == RNS8_SUCCESS);
    REQUIRE((i ? rns8_pack_finite_u8(ctx, b[i], 251, bv, size, 0)
               : rns8_pack_i64(ctx, b[i], &y, size, 0)) == RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx, plans[i], a[i], b[i], c[i], work[i], &executions[i]) == RNS8_SUCCESS);
  }
  rns8_prepared_group *group = nullptr, *bad = nullptr;
  rns8_prepared_execution* duplicate[] = {executions[0],executions[0]};
  REQUIRE(rns8_create_prepared_group(duplicate, 2, &bad) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_create_prepared_group(executions, 2, &group) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_prepared_execution(executions[0]) == RNS8_BUSY);
  // Native queue opt-in cannot relabel a CPU/finite route as matrix execution.
  rns8_native_group_desc native{sizeof(native),RNS8_ABI_VERSION,2,4,3,0,UINT64_MAX};
  rns8_native_group_binding native_bindings[2]{};
  for(auto& binding:native_bindings)binding={sizeof(binding),RNS8_ABI_VERSION,1,nullptr};
  CHECK(rns8_prepare_native_group(group,&native,native_bindings,2)==RNS8_UNSUPPORTED_BACKEND);
  native.claim_chunk=33;CHECK(rns8_prepare_native_group(group,&native,native_bindings,2)==RNS8_INVALID_ARGUMENT);
  native.claim_chunk=1;native.schedule_mode=3;CHECK(rns8_prepare_native_group(group,&native,native_bindings,2)==RNS8_INVALID_ARGUMENT);
  rns8_native_group_info native_info{sizeof(native_info),RNS8_ABI_VERSION};
  CHECK(rns8_get_native_group_info(group,&native_info)==RNS8_INVALID_ARGUMENT);
  CHECK(rns8_capture_native_group_hip_graph(group)==RNS8_INVALID_ARGUMENT);
  rns8_completion_token tokens[2]{{123,456},{123,456}};
  uint32_t executed = 999;
  const auto before = c[0]->content_generation;
  const uint8_t changed[] = {5,6,7,8};
  REQUIRE(rns8_pack_finite_u8(ctx, a[1], 251, changed, 2, 0) == RNS8_SUCCESS);
  REQUIRE(rns8_execute_prepared_group(group, tokens, 2, &executed) == RNS8_INVALID_ARGUMENT);
  REQUIRE(executed == 0);
  REQUIRE(c[0]->content_generation == before);
  REQUIRE(tokens[0].prepared_id == 123);
  REQUIRE(rns8_refresh_prepared_group_inputs(group) == RNS8_SUCCESS);
  REQUIRE(rns8_execute_prepared_group(group, tokens, 2, &executed) == RNS8_SUCCESS);
  REQUIRE(executed == 2);
  int64_t result1 = 0;
  uint8_t result2[4]{};
  REQUIRE(rns8_export_i64(ctx, plans[0], c[0], &result1, 1) == RNS8_SUCCESS);
  REQUIRE(result1 == -21);
  REQUIRE(rns8_export_finite_u8(ctx, plans[1], 251, c[1], result2, 2) == RNS8_SUCCESS);
  REQUIRE(std::vector<uint8_t>(result2, result2 + 4) == std::vector<uint8_t>{67,78,91,106});
  // c[1] is now a valid finite input, but it cannot be both a group output and
  // another task's input even when ordinary launches happen to be sequential.
  rns8_matrix* d = nullptr;
  auto md = finite_matrix_desc(2, 2, RNS8_FINITE_RING_U8);
  REQUIRE(rns8_create_matrix(ctx, &md, &d) == RNS8_SUCCESS);
  rns8_prepared_execution* hazard = nullptr;
  REQUIRE(rns8_prepare_gemm(ctx, plans[1], c[1], b[1], d, work[1], &hazard) == RNS8_SUCCESS);
  rns8_prepared_execution* conflict[] = {executions[1],hazard};
  REQUIRE(rns8_create_prepared_group(conflict, 2, &bad) == RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_prepared_execution(hazard) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(d) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_prepared_group(group) == RNS8_SUCCESS);
  for (int i = 1; i >= 0; --i) {
    REQUIRE(rns8_destroy_prepared_execution(executions[i]) == RNS8_SUCCESS);
    for (auto* matrix : {a[i],b[i],c[i]}) REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(work[i]) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plans[i]) == RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
