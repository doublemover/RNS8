TEST_CASE("exact-wide RNS output accepts only explicit unbounded semantics") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
    desc.bound = 0;
    desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_destroy_plan(plan);

    auto matrix = matrix_desc(semantics, RNS8_BOUND_NONE);
    matrix.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_SUCCESS);
    rns8_destroy_matrix(storage);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("unbounded semantic descriptors reject stale bound metadata") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = gemm_desc(semantics, RNS8_BOUND_NONE);
    desc.bound = 1;
    desc.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("global descriptors reject stray per-tile bound storage") {
  rns8_context* ctx = create_cpu();
  const uint64_t stale_bound = 100;
  auto desc = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  desc.tile_bounds = &stale_bound;
  desc.tile_bounds_count = 1;
  rns8_plan* plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);

  auto exact = gemm_desc(RNS8_EXACT_WIDE_UNSIGNED, RNS8_BOUND_NONE);
  exact.bound = 0;
  exact.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  exact.tile_bounds_count = 1;
  CHECK(rns8_create_plan(ctx, &exact, &plan) == RNS8_INVALID_ARGUMENT);
  CHECK(plan == nullptr);
  rns8_destroy_context(ctx);
}

TEST_CASE("exact-wide semantics reject bounded-looking CRT metadata") {
  rns8_context* ctx = create_cpu();
  for (const rns8_semantics semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    for (const rns8_bound_kind bound_kind :
         {RNS8_BOUND_GLOBAL_MAX_ABS,
          RNS8_BOUND_GLOBAL_MAX_UNSIGNED,
          RNS8_BOUND_PER_TILE_MAX_ABS,
          RNS8_BOUND_PER_TILE_MAX_UNSIGNED,
          RNS8_BOUND_INPUT_RANGE_AND_K}) {
      auto desc = bounded_looking_desc(semantics, bound_kind);
      rns8_plan* plan = nullptr;
      CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
      CHECK(plan == nullptr);

      auto matrix = bounded_looking_matrix_desc(semantics, bound_kind);
      rns8_matrix* storage = nullptr;
      CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
      CHECK(storage == nullptr);
    }
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("strict wraparound is not accepted as bounded odd-modulus CRT") {
  rns8_context* ctx = create_cpu();
  for (const rns8_bound_kind bound_kind :
       {RNS8_BOUND_NONE,
        RNS8_BOUND_GLOBAL_MAX_UNSIGNED,
        RNS8_BOUND_PER_TILE_MAX_ABS,
        RNS8_BOUND_PER_TILE_MAX_UNSIGNED,
        RNS8_BOUND_INPUT_RANGE_AND_K}) {
    auto desc = bounded_looking_desc(RNS8_WRAP_U64_MOD_2_64, bound_kind);
    rns8_plan* plan = nullptr;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    desc.requested_backend = RNS8_BACKEND_WRAP64_BYTE_LIMB;
    CHECK(rns8_create_plan(ctx, &desc, &plan) == RNS8_INVALID_ARGUMENT);
    CHECK(plan == nullptr);

    auto matrix = bounded_looking_matrix_desc(RNS8_WRAP_U64_MOD_2_64, bound_kind);
    rns8_matrix* storage = nullptr;
    CHECK(rns8_create_matrix(ctx, &matrix, &storage) == RNS8_INVALID_ARGUMENT);
    CHECK(storage == nullptr);
  }
  rns8_destroy_context(ctx);
}

TEST_CASE("wrap64 byte-limb context reports explicit CPU reference backend") {
  rns8_context* ctx = create_wrap64();
  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(ctx, &info) == RNS8_SUCCESS);
  CHECK(info.backend == RNS8_BACKEND_WRAP64_BYTE_LIMB);
  CHECK(info.device_id == -1);
  CHECK(info.hip_available == 0);
  rns8_destroy_context(ctx);
}

TEST_CASE("auto backend selection never routes across explicit semantic backends") {
  rns8_context_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.requested_backend = RNS8_BACKEND_AUTO;
  rns8_context* auto_ctx = nullptr;
  REQUIRE(rns8_create_context(-1, &options, &auto_ctx) == RNS8_SUCCESS);

  rns8_device_info info{};
  info.struct_size = sizeof(info);
  info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_get_device_info(auto_ctx, &info) == RNS8_SUCCESS);
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  CHECK(info.backend == RNS8_BACKEND_HIP_DIRECT);
#else
  CHECK(info.backend == RNS8_BACKEND_CPU_REFERENCE);
#endif

  auto wrap = gemm_desc(RNS8_WRAP_U64_MOD_2_64, RNS8_BOUND_NONE);
  wrap.bound = 0;
  wrap.requested_backend = RNS8_BACKEND_AUTO;
  rns8_plan* plan = nullptr;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  CHECK(rns8_create_plan(auto_ctx, &wrap, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);
  plan = nullptr;
#else
  CHECK(rns8_create_plan(auto_ctx, &wrap, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);
#endif

  auto exact = gemm_desc(RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  exact.bound = 0;
  exact.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  exact.requested_backend = RNS8_BACKEND_AUTO;
  REQUIRE(rns8_create_plan(auto_ctx, &exact, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);
  plan = nullptr;
  rns8_destroy_context(auto_ctx);

  rns8_context* wrap_ctx = create_wrap64();
  CHECK(rns8_create_plan(wrap_ctx, &wrap, &plan) == RNS8_SUCCESS);
  rns8_destroy_plan(plan);

  auto bounded = gemm_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  bounded.requested_backend = RNS8_BACKEND_AUTO;
  CHECK(rns8_create_plan(wrap_ctx, &bounded, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);
  CHECK(rns8_create_plan(wrap_ctx, &exact, &plan) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(plan == nullptr);

  auto exact_matrix = matrix_desc(RNS8_EXACT_WIDE_SIGNED, RNS8_BOUND_NONE);
  exact_matrix.max_prefix = RNS8_MAX_SUPPORTED_PREFIX;
  rns8_matrix* storage = nullptr;
  CHECK(rns8_create_matrix(wrap_ctx, &exact_matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(storage == nullptr);

  auto bounded_matrix = matrix_desc(RNS8_BOUNDED_U64, RNS8_BOUND_GLOBAL_MAX_UNSIGNED);
  CHECK(rns8_create_matrix(wrap_ctx, &bounded_matrix, &storage) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(storage == nullptr);
  rns8_destroy_context(wrap_ctx);
}

TEST_CASE("future backend context kinds report unsupported status") {
  std::vector<rns8_backend_kind> backends;
#if !defined(RNS8_ENABLE_ROCWMMA) || !RNS8_ENABLE_ROCWMMA
  backends.push_back(RNS8_BACKEND_ROCWMMA);
#endif
#if !defined(RNS8_ENABLE_CK) || !RNS8_ENABLE_CK
  backends.push_back(RNS8_BACKEND_CK);
#endif
#if !defined(RNS8_ENABLE_HIPBLASLT) || !RNS8_ENABLE_HIPBLASLT
  backends.push_back(RNS8_BACKEND_HIPBLASLT);
#endif
  for (const rns8_backend_kind backend : backends) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = backend;
    rns8_context* ctx = nullptr;
    CHECK(rns8_create_context(0, &options, &ctx) == RNS8_UNSUPPORTED_BACKEND);
    CHECK(ctx == nullptr);

    auto invalid_size = options;
    invalid_size.struct_size = 0;
    CHECK(rns8_create_context(0, &invalid_size, &ctx) == RNS8_INVALID_ARGUMENT);
    CHECK(ctx == nullptr);

    auto invalid_version = options;
    invalid_version.abi_version = 0;
    CHECK(rns8_create_context(0, &invalid_version, &ctx) == RNS8_INVALID_ARGUMENT);
    CHECK(ctx == nullptr);
  }
}

