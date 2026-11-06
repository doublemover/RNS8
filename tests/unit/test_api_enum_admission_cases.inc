TEST_CASE("raw streaming schedule values preserve public query output", "[api][abi][enum][host]") {
  rns8_stream_gemm_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.m = desc.n = desc.k = desc.lda = desc.ldb = desc.ldc = 1;
  desc.panel_m = desc.panel_n = desc.panel_k = 1;
  desc.prefix = 1;
  desc.max_abs_a = desc.max_abs_b = 1;
  desc.device_budget_bytes = desc.host_budget_bytes = 1u << 20;
  rns8_stream_gemm_resources info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  info.device_bytes = 71;
  const auto before = info;
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 2u}) {
    desc.schedule = raw_ffi_enum<rns8_stream_gemm_schedule>(bits);
    CHECK(rns8_query_stream_gemm(&desc, &info) == RNS8_INVALID_ARGUMENT);
    CHECK(std::memcmp(&info, &before, sizeof(info)) == 0);
  }
  for (auto schedule : {RNS8_STREAM_GEMM_SERIAL, RNS8_STREAM_GEMM_OVERLAPPED}) {
    desc.schedule = schedule;
    REQUIRE(rns8_query_stream_gemm(&desc, &info) == RNS8_SUCCESS);
    CHECK(info.device_bytes > 0);
    CHECK(info.output_tiles == 1);
  }
  const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(after.allocate_calls == allocations.allocate_calls);
  CHECK(after.h2d_calls == allocations.h2d_calls);
  CHECK(after.d2h_calls == allocations.d2h_calls);
}

TEST_CASE("raw locality order values reject with live CPU context and plan",
          "[api][abi][enum][host]") {
  auto* ctx = create_cpu_context();
  auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_locality_desc locality{};
  locality.struct_size = sizeof(locality);
  locality.abi_version = RNS8_ABI_VERSION;
  locality.topology_version = locality.cache_domains = locality.concurrent_groups_per_domain = 1;
  const auto references = ctx->prepared_references;
  const auto plan_references = plan->prepared_references;
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 4u}) {
    locality.order = raw_ffi_enum<rns8_locality_order>(bits);
    rns8_workspace* workspace = nullptr;
    CHECK(rns8_create_locality_workspace(ctx, plan, &locality, &workspace) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(workspace == nullptr);
    CHECK(ctx->prepared_references == references);
    CHECK(plan->prepared_references == plan_references);
  }
  // Valid order tags reach backend qualification instead of malformed-input rejection.
  for (auto order : {RNS8_LOCALITY_ORDINARY, RNS8_LOCALITY_OUTPUT_NEIGHBOR, RNS8_LOCALITY_MODULUS,
                     RNS8_LOCALITY_SAME_OUTPUT}) {
    locality.order = order;
    rns8_workspace* workspace = nullptr;
    CHECK(rns8_create_locality_workspace(ctx, plan, &locality, &workspace) ==
          RNS8_UNSUPPORTED_BACKEND);
    CHECK(workspace == nullptr);
  }
  const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(after.allocate_calls == allocations.allocate_calls);
  CHECK(after.h2d_calls == allocations.h2d_calls);
  CHECK(after.d2h_calls == allocations.d2h_calls);
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("raw backend values reject before context or plan acquisition",
          "[api][abi][enum][host]") {
  auto* ctx = create_cpu_context();
  const auto references = ctx->prepared_references;
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 10u}) {
    const auto backend = raw_ffi_enum<rns8_backend_kind>(bits);
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = backend;
    auto* const context_canary = reinterpret_cast<rns8_context*>(uintptr_t(1));
    rns8_context* invalid_context = context_canary;
    CHECK(rns8_create_context(-1, &options, &invalid_context) == RNS8_INVALID_ARGUMENT);
    CHECK(invalid_context == context_canary);
    auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
    desc.requested_backend = backend;
    auto* const plan_canary = reinterpret_cast<rns8_plan*>(uintptr_t(1));
    rns8_plan* plan = plan_canary;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == plan_canary);
    rns8_backend_capability_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    const auto before = info;
    CHECK(rns8_get_backend_capability_info(backend, &info) == RNS8_INVALID_ARGUMENT);
    CHECK(std::memcmp(&info, &before, sizeof(info)) == 0);
    rns8_operation_query query{};
    query.struct_size = sizeof(query);
    query.abi_version = RNS8_ABI_VERSION;
    query.problem = desc;
    query.operation = RNS8_OP_GEMM;
    query.execution_modes = RNS8_EXEC_SYNC;
    rns8_operation_support result{};
    const auto result_before = result;
    uint32_t required = 71;
    CHECK(rns8_query_operation_support(ctx, &query, &result, 1, &required) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(required == 71);
    CHECK(std::memcmp(&result, &result_before, sizeof(result)) == 0);
    CHECK(ctx->prepared_references == references);
  }
  for (auto backend : {RNS8_BACKEND_AUTO, RNS8_BACKEND_CPU_REFERENCE, RNS8_BACKEND_HIP_DIRECT,
                       RNS8_BACKEND_HIPBLASLT, RNS8_BACKEND_CK, RNS8_BACKEND_ROCWMMA,
                       RNS8_BACKEND_WRAP64_BYTE_LIMB, RNS8_BACKEND_HIP_VECTOR_ALU_INT64,
                       RNS8_BACKEND_AMDGPU_BUILTINS, RNS8_BACKEND_CPU_AMX}) {
    rns8_backend_capability_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    CHECK(rns8_get_backend_capability_info(backend, &info) == RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("constructor descriptor rejection precedes poisoned context and output publication",
          "[api][abi][enum][host]") {
  rns8_context ctx{};
  ctx.backend = RNS8_BACKEND_CPU_REFERENCE;
  ctx.hip_failure_poisoned = true; // Admission control only; no device fault is induced.
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  auto gemm = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
  auto matrix = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
  auto reject = [&](const rns8_context_options& bad_options, const rns8_gemm_desc& bad_gemm,
                    const rns8_matrix_desc& bad_matrix) {
    auto* const context_canary = reinterpret_cast<rns8_context*>(uintptr_t(1));
    auto* const plan_canary = reinterpret_cast<rns8_plan*>(uintptr_t(1));
    auto* const matrix_canary = reinterpret_cast<rns8_matrix*>(uintptr_t(1));
    auto* context_out = context_canary;
    auto* plan_out = plan_canary;
    auto* matrix_out = matrix_canary;
    CHECK(rns8_create_context(-1, &bad_options, &context_out) == RNS8_INVALID_ARGUMENT);
    CHECK(context_out == context_canary);
    CHECK(rns8_create_plan(&ctx, &bad_gemm, &plan_out) == RNS8_INVALID_ARGUMENT);
    CHECK(plan_out == plan_canary);
    CHECK(rns8_create_matrix(&ctx, &bad_matrix, &matrix_out) == RNS8_INVALID_ARGUMENT);
    CHECK(matrix_out == matrix_canary);
    CHECK(ctx.hip_failure_poisoned);
    CHECK(ctx.prepared_references == 0);
  };
  for (unsigned malformed = 0; malformed < 5; ++malformed) {
    auto bad_options = options;
    auto bad_gemm = gemm;
    auto bad_matrix = matrix;
    if (malformed < 3) {
      bad_options.struct_size = malformed == 0   ? 0
                                : malformed == 1 ? sizeof(uint64_t)
                                                 : sizeof(options) - 1;
      bad_gemm.struct_size = malformed == 0   ? 0
                             : malformed == 1 ? sizeof(uint32_t)
                                              : sizeof(gemm) - 1;
      bad_matrix.struct_size = malformed == 0   ? 0
                               : malformed == 1 ? sizeof(uint32_t)
                                                : sizeof(matrix) - 1;
    } else {
      bad_options.abi_version = bad_gemm.abi_version = bad_matrix.abi_version =
          malformed == 3 ? 0 : RNS8_ABI_VERSION + 1;
    }
    reject(bad_options, bad_gemm, bad_matrix);
  }
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000)}) {
    auto bad_options = options;
    auto bad_gemm = gemm;
    auto bad_matrix = matrix;
    bad_options.flags = bad_gemm.flags = bad_matrix.flags = bits;
    reject(bad_options, bad_gemm, bad_matrix);
    bad_gemm = gemm;
    bad_gemm.flags = RNS8_PLAN_DIRECT_RNS_PARALLEL;
    bad_gemm.bound_kind = raw_ffi_enum<rns8_bound_kind>(bits);
    reject(bad_options, bad_gemm, bad_matrix);
    bad_gemm.bound_kind = gemm.bound_kind;
    bad_gemm.semantics = raw_ffi_enum<rns8_semantics>(bits);
    reject(bad_options, bad_gemm, bad_matrix);
  }
  // Admitted descriptors still report context failure and clear their output slots.
  auto* plan_out = reinterpret_cast<rns8_plan*>(uintptr_t(1));
  auto* matrix_out = reinterpret_cast<rns8_matrix*>(uintptr_t(1));
  CHECK(rns8_create_plan(&ctx, &gemm, &plan_out) == RNS8_BACKEND_FAILURE);
  CHECK(plan_out == nullptr);
  CHECK(rns8_create_matrix(&ctx, &matrix, &matrix_out) == RNS8_BACKEND_FAILURE);
  CHECK(matrix_out == nullptr);
  const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(after.allocate_calls == allocations.allocate_calls);
  CHECK(after.h2d_calls == allocations.h2d_calls);
  CHECK(after.d2h_calls == allocations.d2h_calls);
}

TEST_CASE("ordered basis descriptor admission preserves public outputs before routing",
          "[api][abi][enum][host]") {
  auto* ctx = create_cpu_context();
  rns8_basis_plan_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.backend = RNS8_BACKEND_CPU_REFERENCE;
  desc.semantics = RNS8_EXACT_WIDE_SIGNED;
  desc.m = desc.n = desc.k = 1;
  desc.prefix = RNS8_MAX_SUPPORTED_PREFIX;
  desc.max_host_bytes = desc.max_device_bytes = UINT64_MAX;
  rns8_basis_resources resources{};
  resources.struct_size = sizeof(resources);
  resources.abi_version = RNS8_ABI_VERSION;
  resources.required_host_bytes = 71;
  rns8_operation_support support{};
  support.struct_size = sizeof(support);
  support.abi_version = RNS8_ABI_VERSION;
  support.required_host_bytes = 73;
  rns8_basis_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  info.identity = 79;
  const auto resources_before = resources;
  const auto support_before = support;
  const auto info_before = info;
  const auto references = ctx->prepared_references;
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  auto reject = [&](const rns8_basis_plan_desc* bad, rns8_operation_kind operation, uint32_t modes,
                    bool bad_descriptor) {
    CHECK(rns8_query_basis_operation(ctx, nullptr, bad, operation, modes, &support, &info) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(std::memcmp(&support, &support_before, sizeof(support)) == 0);
    CHECK(std::memcmp(&info, &info_before, sizeof(info)) == 0);
    if (bad_descriptor) {
      auto* const canary = reinterpret_cast<rns8_basis_plan*>(uintptr_t(1));
      auto* out = canary;
      CHECK(rns8_create_basis_plan(ctx, nullptr, bad, &out) == RNS8_INVALID_ARGUMENT);
      CHECK(out == canary);
      CHECK(rns8_query_basis_plan(nullptr, bad, &resources) == RNS8_INVALID_ARGUMENT);
      CHECK(std::memcmp(&resources, &resources_before, sizeof(resources)) == 0);
    }
    CHECK(ctx->prepared_references == references);
  };
  for (unsigned field = 0; field < 3; ++field)
    for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000),
                          field == 0   ? 10u
                          : field == 1 ? 8u
                                       : 1u}) {
      auto bad = desc;
      if (field == 0)
        bad.backend = raw_ffi_enum<rns8_backend_kind>(bits);
      if (field == 1)
        bad.semantics = raw_ffi_enum<rns8_semantics>(bits);
      if (field == 2)
        bad.flags = bits;
      reject(&bad, RNS8_OP_GEMM, RNS8_EXEC_SYNC, true);
    }
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), uint32_t(RNS8_OP_COUNT)})
    reject(&desc, raw_ffi_enum<rns8_operation_kind>(bits), RNS8_EXEC_SYNC, false);
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 128u, 0u})
    reject(&desc, RNS8_OP_GEMM, bits, false);
  for (unsigned malformed = 0; malformed < 2; ++malformed) {
    auto bad = desc;
    if (malformed == 0)
      bad.struct_size = sizeof(bad) - 1;
    else
      bad.abi_version = RNS8_ABI_VERSION + 1;
    reject(&bad, RNS8_OP_GEMM, RNS8_EXEC_SYNC, true);
  }
  // Only the readable size prefix exists here; payload reads would be out of bounds.
  alignas(rns8_basis_plan_desc) uint64_t short_header = sizeof(uint64_t);
  reject(reinterpret_cast<const rns8_basis_plan_desc*>(&short_header), RNS8_OP_GEMM, RNS8_EXEC_SYNC,
         true);
  const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(after.allocate_calls == allocations.allocate_calls);
  CHECK(after.h2d_calls == allocations.h2d_calls);
  CHECK(after.d2h_calls == allocations.d2h_calls);
  // Known arithmetic/provider tags retain their genuine unsupported distinction.
  auto unsupported = desc;
  unsupported.semantics = RNS8_BOUNDED_I64;
  CHECK(rns8_query_basis_plan(nullptr, &unsupported, &resources) == RNS8_UNSUPPORTED_BACKEND);
  unsupported = desc;
  unsupported.backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
  CHECK(rns8_query_basis_operation(ctx, nullptr, &unsupported, RNS8_OP_GEMM, RNS8_EXEC_SYNC,
                                   &support, &info) == RNS8_UNSUPPORTED_BACKEND);
  REQUIRE(rns8_query_basis_plan(nullptr, &desc, &resources) == RNS8_SUCCESS);
  REQUIRE(rns8_query_basis_operation(ctx, nullptr, &desc, RNS8_OP_GEMM, RNS8_EXEC_SYNC, &support,
                                     &info) == RNS8_SUCCESS);
  CHECK(resources.required_host_bytes == support.required_host_bytes);
  rns8_basis_plan* owner = nullptr;
  REQUIRE(rns8_create_basis_plan(ctx, nullptr, &desc, &owner) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_basis_plan(owner) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("raw sparse descriptor fields reject before storage or packed output writes",
          "[api][abi][enum][host]") {
  auto* ctx = create_cpu_context();
  rns8_sparse_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.semantics = RNS8_BOUNDED_I64;
  desc.bound_kind = RNS8_BOUND_GLOBAL_MAX_ABS;
  desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
  desc.rows = 1;
  desc.expanded_k = desc.group_size = 4;
  desc.nonzeros_per_group = 2;
  desc.max_prefix = 1;
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  const uint8_t dense[]{1, 0, 2, 0};
  for (unsigned field = 0; field < 8; ++field) {
    const uint32_t adjacent = field == 0   ? 8u
                              : field == 1 ? 6u
                              : field < 5  ? 2u
                              : field == 5 ? 3u
                                           : 1u;
    for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), adjacent}) {
      auto bad = desc;
      if (field == 0)
        bad.semantics = raw_ffi_enum<rns8_semantics>(bits);
      if (field == 1)
        bad.bound_kind = raw_ffi_enum<rns8_bound_kind>(bits);
      if (field == 2)
        bad.contract = raw_ffi_enum<rns8_sparse_contract>(bits);
      if (field == 3)
        bad.sparse_operand = raw_ffi_enum<rns8_sparse_operand>(bits);
      if (field == 4)
        bad.index_layout = raw_ffi_enum<rns8_sparse_index_layout>(bits);
      if (field == 5)
        bad.value_signedness = raw_ffi_enum<rns8_sparse_value_signedness>(bits);
      if (field == 6)
        bad.flags = bits;
      if (field == 7)
        bad.reserved0 = bits;
      uint64_t groups = 71, count = 73;
      CHECK(rns8_sparse_a_4_to_2_layout_counts(&bad, &groups, &count) == RNS8_INVALID_ARGUMENT);
      CHECK(groups == 71);
      CHECK(count == 73);
      uint8_t values[]{79, 83}, indices[]{89}, expanded[]{97, 101, 103, 107};
      CHECK(rns8_pack_sparse_a_4_to_2_u8(&bad, dense, 4, values, indices) == RNS8_INVALID_ARGUMENT);
      CHECK(values[0] == 79);
      CHECK(values[1] == 83);
      CHECK(indices[0] == 89);
      CHECK(rns8_expand_sparse_a_4_to_2_u8(&bad, values, indices, expanded, 4) ==
            RNS8_INVALID_ARGUMENT);
      const uint8_t expected[]{97, 101, 103, 107};
      CHECK(std::memcmp(expanded, expected, sizeof(expanded)) == 0);
      for (bool poisoned : {false, true}) {
        ctx->hip_failure_poisoned = poisoned; // CPU admission only; no device fault.
        auto* const canary = reinterpret_cast<rns8_sparse_matrix*>(uintptr_t(1));
        auto* out = canary;
        CHECK(rns8_create_sparse_matrix(ctx, &bad, &out) == RNS8_INVALID_ARGUMENT);
        CHECK(out == canary);
        ctx->hip_failure_poisoned = false;
      }
    }
  }
  const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(after.allocate_calls == allocations.allocate_calls);
  CHECK(after.h2d_calls == allocations.h2d_calls);
  CHECK(after.d2h_calls == allocations.d2h_calls);
  uint64_t groups = 0, count = 0;
  REQUIRE(rns8_sparse_a_4_to_2_layout_counts(&desc, &groups, &count) == RNS8_SUCCESS);
  CHECK(groups == 1);
  CHECK(count == 2);
  rns8_sparse_matrix* matrix = nullptr;
  REQUIRE(rns8_create_sparse_matrix(ctx, &desc, &matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_sparse_matrix(matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("cache and FP8 malformed options preserve handles before CPU provider rejection",
          "[api][abi][enum][host]") {
  auto* ctx = create_cpu_context();
  const auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_result_cache_desc cache_desc{};
  cache_desc.struct_size = sizeof(cache_desc);
  cache_desc.abi_version = RNS8_ABI_VERSION;
  rns8_fp8_desc fp8{};
  fp8.struct_size = sizeof(fp8);
  fp8.abi_version = RNS8_ABI_VERSION;
  fp8.format = 1;
  const auto references = ctx->prepared_references;
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  for (unsigned field = 0; field < 4; ++field) {
    auto bad_cache = cache_desc;
    auto bad_fp8 = fp8;
    if (field == 0)
      bad_cache.flags = bad_fp8.flags = UINT32_C(0x80000000);
    if (field == 1) {
      bad_cache.reserved0 = UINT32_MAX;
      bad_fp8.method = 4;
    }
    if (field == 2) {
      bad_cache.struct_size = sizeof(cache_desc) - 1;
      bad_fp8.struct_size = sizeof(fp8) - 1;
    }
    if (field == 3)
      bad_cache.abi_version = bad_fp8.abi_version = RNS8_ABI_VERSION + 1;
    auto* const cache_canary = reinterpret_cast<rns8_result_cache*>(uintptr_t(1));
    auto* cache = cache_canary;
    CHECK(rns8_create_result_cache(ctx, plan, &bad_cache, &cache) == RNS8_INVALID_ARGUMENT);
    CHECK(cache == cache_canary);
    auto* const fp8_canary = reinterpret_cast<rns8_fp8_recipe*>(uintptr_t(1));
    auto* recipe = fp8_canary;
    CHECK(rns8_prepare_fp8_recipe(ctx, plan, &bad_fp8, &recipe) == RNS8_INVALID_ARGUMENT);
    CHECK(recipe == fp8_canary);
  }
  // Each partial set is known but fails the explicit complete-cache contract.
  for (uint32_t flag :
       {RNS8_RESULT_CACHE_CONTRACT_OUTPUT_RECTANGLES, RNS8_RESULT_CACHE_CONTRACT_DIRECT_HIP_ONLY,
        RNS8_RESULT_CACHE_CONTRACT_FULL_K_RECOMPUTE, RNS8_RESULT_CACHE_CONTRACT_EXPLICIT_OPT_IN}) {
    auto bad_cache = cache_desc;
    bad_cache.flags = flag;
    auto* const canary = reinterpret_cast<rns8_result_cache*>(uintptr_t(1));
    auto* cache = canary;
    CHECK(rns8_create_result_cache(ctx, plan, &bad_cache, &cache) == RNS8_INVALID_ARGUMENT);
    CHECK(cache == canary);
  }
  rns8_result_cache* cache = nullptr;
  CHECK(rns8_create_result_cache(ctx, plan, &cache_desc, &cache) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(cache == nullptr);
  rns8_fp8_recipe* recipe = nullptr;
  CHECK(rns8_prepare_fp8_recipe(ctx, plan, &fp8, &recipe) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(recipe == nullptr);
  CHECK(ctx->prepared_references == references);
  const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(after.allocate_calls == allocations.allocate_calls);
  CHECK(after.h2d_calls == allocations.h2d_calls);
  CHECK(after.d2h_calls == allocations.d2h_calls);
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("raw roles graph tags and arithmetic methods reject with live CPU owners",
          "[api][abi][enum][host]") {
  auto* ctx = create_cpu_context();
  auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 1, 1, 1);
  desc.flags = RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  const auto md = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
  rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr, *imag = nullptr;
  for (auto** out : {&a, &b, &c, &imag}) {
    REQUIRE(rns8_create_matrix(ctx, &md, out) == RNS8_SUCCESS);
    const int64_t one = 1;
    REQUIRE(rns8_pack_i64(ctx, *out, &one, 1, 1) == RNS8_SUCCESS);
  }
  const auto generation = c->content_generation, imag_generation = imag->content_generation;
  const auto c_residues = c->residues, imag_residues = imag->residues;
  const auto references = ctx->prepared_references;
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  rns8_prepack_cache_key_info key{};
  key.struct_size = sizeof(key);
  key.abi_version = RNS8_ABI_VERSION;
  key.cache_key_valid = 71;
  const auto key_before = key;
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 3u}) {
    const auto role = raw_ffi_enum<rns8_operand_role>(bits);
    CHECK(rns8_get_prepack_cache_key_info(plan, a, role, &key) == RNS8_INVALID_ARGUMENT);
    CHECK(std::memcmp(&key, &key_before, sizeof(key)) == 0);
    for (bool poisoned : {false, true}) {
      ctx->hip_failure_poisoned = poisoned;
      auto* const canary = reinterpret_cast<rns8_prepack_cache*>(uintptr_t(1));
      auto* cache = canary;
      CHECK(rns8_create_prepack_cache(ctx, plan, a, role, &cache) == RNS8_INVALID_ARGUMENT);
      CHECK(cache == canary);
      ctx->hip_failure_poisoned = false;
    }
  }
  rns8_resident_lifetime_info lifetime{};
  lifetime.struct_size = sizeof(lifetime);
  lifetime.abi_version = RNS8_ABI_VERSION;
  lifetime.rows = 73;
  const auto lifetime_before = lifetime;
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 4u}) {
    CHECK(rns8_get_resident_lifetime_info(a, plan, workspace,
                                          raw_ffi_enum<rns8_resident_matrix_role>(bits),
                                          &lifetime) == RNS8_INVALID_ARGUMENT);
    CHECK(std::memcmp(&lifetime, &lifetime_before, sizeof(lifetime)) == 0);
  }
  rns8_resident_node node{};
  node.struct_size = sizeof(node);
  node.abi_version = RNS8_ABI_VERSION;
  node.kind = RNS8_RESIDENT_MATMUL;
  node.plan = plan;
  node.a = a;
  node.b = b;
  node.c = c;
  node.workspace = workspace;
  for (unsigned field = 0; field < 2; ++field)
    for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), field == 0 ? 6u : 2u}) {
      rns8_resident_node nodes[]{node, node};
      if (field == 0)
        nodes[1].kind = raw_ffi_enum<rns8_resident_node_kind>(bits);
      else
        nodes[1].complex_method = raw_ffi_enum<rns8_complex_product_method>(bits);
      auto* const canary = reinterpret_cast<rns8_resident_graph*>(uintptr_t(1));
      auto* graph = canary;
      CHECK(rns8_create_resident_graph(ctx, nodes, 2, &graph) == RNS8_INVALID_ARGUMENT);
      CHECK(graph == canary);
    }
  check_short_descriptor<rns8_resident_node>([&](const auto* input) {
    auto* const canary = reinterpret_cast<rns8_resident_graph*>(uintptr_t(1));
    auto* graph = canary;
    const auto status = rns8_create_resident_graph(ctx, input, 1, &graph);
    CHECK(graph == canary);
    return status;
  });
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 5u})
    CHECK(rns8_gemm_narrow_exact(ctx, plan, a, b, c, raw_ffi_enum<rns8_narrow_product_method>(bits),
                                 UINT64_MAX, workspace) == RNS8_INVALID_ARGUMENT);
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 3u}) {
    const auto method = raw_ffi_enum<rns8_adaptive_bound_method>(bits);
    CHECK(rns8_gemm_adaptive_exact(ctx, plan, a, b, c, method, 1, 1, 1, UINT64_MAX, workspace) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(rns8_gemm_adaptive_exact_cached(ctx, plan, a, b, c, method, 1, 1, 1, UINT64_MAX,
                                          workspace) == RNS8_INVALID_ARGUMENT);
  }
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 20u}) {
    const auto method = raw_ffi_enum<rns8_bilinear_method>(bits);
    double ratio = 79;
    CHECK(rns8_prepare_bilinear_workspace(ctx, plan, method, UINT64_MAX, workspace) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(rns8_gemm_bilinear_exact(ctx, plan, a, b, c, method, UINT64_MAX, workspace, &ratio) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(ratio == 79);
  }
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 2u})
    CHECK(rns8_gemm_complex_exact_signed(ctx, plan, a, a, b, b, c, imag,
                                         raw_ffi_enum<rns8_complex_product_method>(bits),
                                         UINT64_MAX, workspace) == RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation == generation);
  CHECK(imag->content_generation == imag_generation);
  CHECK(c->residues == c_residues);
  CHECK(imag->residues == imag_residues);
  CHECK_FALSE(workspace->in_flight);
  CHECK(ctx->prepared_references == references);
  const auto after = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(after.allocate_calls == allocations.allocate_calls);
  CHECK(after.h2d_calls == allocations.h2d_calls);
  CHECK(after.d2h_calls == allocations.d2h_calls);
  for (auto role : {RNS8_OPERAND_A, RNS8_OPERAND_B})
    CHECK(rns8_get_prepack_cache_key_info(plan, a, role, &key) == RNS8_SUCCESS);
  for (auto role : {RNS8_RESIDENT_MATRIX_ROLE_UNKNOWN, RNS8_RESIDENT_MATRIX_ROLE_A,
                    RNS8_RESIDENT_MATRIX_ROLE_B, RNS8_RESIDENT_MATRIX_ROLE_C})
    CHECK(rns8_get_resident_lifetime_info(a, plan, workspace, role, &lifetime) == RNS8_SUCCESS);
  CHECK(rns8_gemm_narrow_exact(ctx, plan, a, b, c, RNS8_NARROW_SCALAR_I64, UINT64_MAX, workspace) ==
        RNS8_UNSUPPORTED_BACKEND);
  CHECK(rns8_gemm_adaptive_exact(ctx, plan, a, b, c, RNS8_ADAPTIVE_TILE_MAX, 1, 1, 1, UINT64_MAX,
                                 workspace) == RNS8_UNSUPPORTED_BACKEND);
  rns8_resident_graph* graph = nullptr;
  REQUIRE(rns8_create_resident_graph(ctx, &node, 1, &graph) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_resident_graph(graph) == RNS8_SUCCESS);
  for (auto* matrix : {imag, c, b, a})
    REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("raw semantics bounds layouts and flag bits reach real CPU descriptor validators",
          "[api][abi][enum][host]") {
  auto* ctx = create_cpu_context();
  for (unsigned field = 0; field < 3; ++field)
    for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), field == 0 ? 8u : 6u}) {
      auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
      if (field == 0)
        desc.semantics = raw_ffi_enum<rns8_semantics>(bits);
      if (field == 1)
        desc.bound_kind = raw_ffi_enum<rns8_bound_kind>(bits);
      if (field == 2)
        desc.flags = bits == 6 ? (UINT32_C(1) << 31) : bits;
      auto* const canary = reinterpret_cast<rns8_plan*>(uintptr_t(1));
      rns8_plan* plan = canary;
      CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
      CHECK(plan == canary);
    }
  for (unsigned field = 0; field < 4; ++field)
    for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000),
                          field == 0   ? 8u
                          : field == 1 ? 6u
                                       : 3u}) {
      auto desc = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
      if (field == 0)
        desc.semantics = raw_ffi_enum<rns8_semantics>(bits);
      if (field == 1)
        desc.bound_kind = raw_ffi_enum<rns8_bound_kind>(bits);
      if (field == 2)
        desc.logical_layout = raw_ffi_enum<rns8_layout>(bits);
      if (field == 3)
        desc.flags = bits;
      auto* const canary = reinterpret_cast<rns8_matrix*>(uintptr_t(1));
      rns8_matrix* matrix = canary;
      CHECK(rns8_create_matrix(ctx, &desc, &matrix) == RNS8_INVALID_ARGUMENT);
      CHECK(matrix == canary);
    }
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 1u}) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    options.flags = bits;
    auto* const canary = reinterpret_cast<rns8_context*>(uintptr_t(1));
    rns8_context* rejected = canary;
    CHECK(rns8_create_context(-1, &options, &rejected) == RNS8_INVALID_ARGUMENT);
    CHECK(rejected == canary);
  }
  // Valid descriptors still allocate and destroy through the same validators.
  auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  auto md = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
  rns8_matrix* matrix = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
