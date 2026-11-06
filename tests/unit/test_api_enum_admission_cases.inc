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
