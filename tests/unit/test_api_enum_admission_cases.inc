namespace {
template <class T> T raw_ffi_enum(uint32_t bits) {
  static_assert(std::is_enum_v<T> && sizeof(T) == sizeof(bits));
  T value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
} // namespace

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
    rns8_context* invalid_context = nullptr;
    CHECK(rns8_create_context(-1, &options, &invalid_context) == RNS8_INVALID_ARGUMENT);
    CHECK(invalid_context == nullptr);
    auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
    desc.requested_backend = backend;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
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
      rns8_plan* plan = nullptr;
      CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
      CHECK(plan == nullptr);
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
      rns8_matrix* matrix = nullptr;
      CHECK(rns8_create_matrix(ctx, &desc, &matrix) == RNS8_INVALID_ARGUMENT);
      CHECK(matrix == nullptr);
    }
  for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 1u}) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_CPU_REFERENCE;
    options.flags = bits;
    rns8_context* rejected = nullptr;
    CHECK(rns8_create_context(-1, &options, &rejected) == RNS8_INVALID_ARGUMENT);
    CHECK(rejected == nullptr);
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
