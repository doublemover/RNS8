TEST_CASE("immutable schedule validation is amortized without hash-only equivalence", "[api][schedule]") {
  using namespace rns8::detail::api;
  auto* ctx = create_cpu_context();
  auto desc = exact_desc(RNS8_BOUNDED_U64, 16384, 16384, 1);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  desc.tile_m = desc.tile_n = 128;
  desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS;
  std::vector<uint64_t> bounds(128 * 128, 100000);
  std::vector<uint8_t> zero_rows(16384), zero_cols(16384);
  desc.tile_bounds = bounds.data(); desc.tile_bounds_count = bounds.size();
  desc.zero_a_rows = zero_rows.data(); desc.zero_a_rows_count = zero_rows.size();
  desc.zero_b_cols = zero_cols.data(); desc.zero_b_cols_count = zero_cols.size();
  rns8_plan *plan = nullptr, *equivalent = nullptr, *different = nullptr;
  plan_validation_work = {};
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(plan_validation_work.exhaustive_validations == 1);
  REQUIRE(plan_validation_work.fingerprint_traversals == 1);
  REQUIRE(plan->validated_contract);
  REQUIRE(rns8_create_plan(ctx, &desc, &equivalent) == RNS8_SUCCESS);
  REQUIRE(plan->validated_contract != equivalent->validated_contract);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  plan_validation_work = {};
  for (int iteration = 0; iteration < 32; ++iteration) {
    REQUIRE(validate_plan_context_workspace(*ctx, *plan, *workspace) == RNS8_SUCCESS);
    REQUIRE(validate_plan_context_workspace(*ctx, *equivalent, *workspace) == RNS8_SUCCESS);
    REQUIRE(plan_workspace_fingerprint(*plan) == workspace->schedule_fingerprint);
  }
  REQUIRE(plan_validation_work.exhaustive_validations == 0);
  REQUIRE(plan_validation_work.fingerprint_traversals == 0);
  REQUIRE(plan_validation_work.equivalent_comparisons == 1);
  // Caller-owned arrays are copied. Mutation cannot alter an existing seal.
  bounds[0] = 99999;
  REQUIRE(rns8_create_plan(ctx, &desc, &different) == RNS8_SUCCESS);
  REQUIRE(validate_plan_context_workspace(*ctx, *different, *workspace) == RNS8_INVALID_ARGUMENT);
  // Force an artificial hash collision: structural equality must still reject.
  auto collision = std::make_shared<rns8_validated_plan_contract>(*different->validated_contract);
  collision->fingerprint = plan->validated_contract->fingerprint;
  different->validated_contract = collision;
  REQUIRE(validate_plan_context_workspace(*ctx, *different, *workspace) == RNS8_INVALID_ARGUMENT);
  const auto fingerprint = workspace->schedule_fingerprint;
  workspace->schedule_fingerprint ^= 1;
  REQUIRE(validate_plan_context_workspace(*ctx, *plan, *workspace) == RNS8_INVALID_ARGUMENT);
  workspace->schedule_fingerprint = fingerprint;
  // Diagnostics check coverage and proof fields even on a sealed plan. They
  // never silently reseal changed state or certify the previous fingerprint.
  const auto entry = plan->tile_schedule[0];
  plan->tile_schedule[0].row_extent -= 1;
  REQUIRE_FALSE(revalidate_plan_schedule_contract(*plan));
  plan->tile_schedule[0] = entry;
  plan->tile_schedule[0].required_prefix += 1;
  REQUIRE_FALSE(revalidate_plan_schedule_contract(*plan));
  plan->tile_schedule[0] = entry;
  plan->zero_a_rows[0] = 1;
  REQUIRE_FALSE(revalidate_plan_schedule_contract(*plan));
  plan->zero_a_rows[0] = 0;
  REQUIRE(revalidate_plan_schedule_contract(*plan));
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  // Workspace owns the immutable contract, not a dangling original-plan pointer.
  REQUIRE(validate_plan_context_workspace(*ctx, *equivalent, *workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(equivalent) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(different) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
