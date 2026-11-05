namespace {
template <class T, class Query> void check_extensible_query_header(Query query) {
  T value{};
  value.struct_size = sizeof(value);
  value.abi_version = RNS8_ABI_VERSION;
  REQUIRE(query(&value) == RNS8_SUCCESS);
  for (unsigned mode = 0; mode < 2; ++mode) {
    T invalid = value;
    if (mode == 0)
      invalid.struct_size = sizeof(T) - 1;
    else
      ++invalid.abi_version;
    const auto before = invalid;
    CHECK(query(&invalid) == RNS8_INVALID_ARGUMENT);
    CHECK(std::memcmp(&invalid, &before, sizeof(invalid)) == 0);
  }
  struct Extended {
    T value;
    uint64_t canary;
  } extended{};
  extended.value.struct_size = sizeof(extended);
  extended.value.abi_version = RNS8_ABI_VERSION;
  extended.canary = UINT64_C(0xfedcba9876543210);
  REQUIRE(query(&extended.value) == RNS8_SUCCESS);
  CHECK(extended.canary == UINT64_C(0xfedcba9876543210));
  using Prefix = decltype(value.struct_size);
  alignas(T) Prefix short_header = sizeof(Prefix);
  CHECK(query(reinterpret_cast<T*>(&short_header)) == RNS8_INVALID_ARGUMENT);
  CHECK(short_header == sizeof(Prefix));
  alignas(T) unsigned char misaligned[sizeof(T) + 1]{};
  std::memcpy(misaligned + 1, &value, sizeof(value));
  CHECK(query(reinterpret_cast<T*>(misaligned + 1)) == RNS8_INVALID_ARGUMENT);
  CHECK(std::memcmp(misaligned + 1, &value, sizeof(value)) == 0);
}

template <class T, class Validate> void check_short_descriptor(Validate validate) {
  using Prefix = decltype(T{}.struct_size);
  // Exactly the documented readable prefix, with no version/payload allocation.
  alignas(T) Prefix header = sizeof(Prefix);
  CHECK(validate(reinterpret_cast<const T*>(&header)) == RNS8_INVALID_ARGUMENT);
  CHECK(header == sizeof(Prefix));
}
} // namespace

TEST_CASE("public descriptor admission reads the size prefix before version or payload",
          "[api][abi][host]") {
  auto* ctx = create_cpu_context();
  auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  auto md = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
  rns8_matrix* matrix = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
  check_extensible_query_header<rns8_device_info>(
      [&](auto* out) { return rns8_get_device_info(ctx, out); });
  check_extensible_query_header<rns8_matrix_storage_info>(
      [&](auto* out) { return rns8_get_matrix_storage_info(matrix, out); });
  check_extensible_query_header<rns8_plan_backend_info>(
      [&](auto* out) { return rns8_get_plan_backend_info(plan, out); });
  check_extensible_query_header<rns8_backend_capability_info>(
      [](auto* out) { return rns8_get_backend_capability_info(RNS8_BACKEND_CPU_REFERENCE, out); });
  check_short_descriptor<rns8_context_options>([](const auto* input) {
    rns8_context* out = nullptr;
    const auto status = rns8_create_context(-1, input, &out);
    CHECK(out == nullptr);
    return status;
  });
  check_short_descriptor<rns8_gemm_desc>([&](const auto* input) {
    rns8_plan* out = nullptr;
    const auto status = rns8_create_plan(ctx, input, &out);
    CHECK(out == nullptr);
    int64_t a = 2, b = 3, c = 17;
    CHECK(rns8_gemm_i64_oneshot(ctx, input, &a, 1, &b, 1, &c, 1) == RNS8_INVALID_ARGUMENT);
    CHECK(c == 17);
    return status;
  });
  check_short_descriptor<rns8_matrix_desc>([&](const auto* input) {
    rns8_matrix* out = nullptr;
    const auto status = rns8_create_matrix(ctx, input, &out);
    CHECK(out == nullptr);
    return status;
  });
  check_short_descriptor<rns8_operation_query>([&](const auto* input) {
    rns8_operation_support out{};
    const auto before = out;
    uint32_t required = 71;
    const auto status = rns8_query_operation_support(ctx, input, &out, 1, &required);
    CHECK(required == 71);
    CHECK(std::memcmp(&out, &before, sizeof(out)) == 0);
    return status;
  });
  check_short_descriptor<rns8_native_device_desc>([](const auto* input) {
    rns8_native_device_resources out{};
    out.struct_size = sizeof(out);
    out.abi_version = RNS8_ABI_VERSION;
    const auto before = out;
    const auto status = rns8_query_native_device(input, &out);
    CHECK(std::memcmp(&out, &before, sizeof(out)) == 0);
    return status;
  });
  check_short_descriptor<rns8_pinned_host_desc>([](const auto* input) {
    uint64_t bytes = 71;
    const auto status = rns8_query_pinned_host_bytes(input, &bytes);
    CHECK(bytes == 71);
    return status;
  });
  check_short_descriptor<rns8_integer_split_i64_desc>([](const auto* input) {
    rns8_integer_split_i64_info out{};
    out.struct_size = sizeof(out);
    out.abi_version = RNS8_ABI_VERSION;
    const auto before = out;
    const auto status = rns8_query_integer_split_i64(input, &out);
    CHECK(std::memcmp(&out, &before, sizeof(out)) == 0);
    return status;
  });
  check_short_descriptor<rns8_sparse_matrix_desc>([&](const auto* input) {
    uint64_t groups = 71, count = 73;
    const auto status = rns8_sparse_a_4_to_2_layout_counts(input, &groups, &count);
    CHECK(groups == 71);
    CHECK(count == 73);
    uint8_t dense = 2, packed = 3, indices = 5;
    CHECK(rns8_pack_sparse_a_4_to_2_u8(input, &dense, 1, &packed, &indices) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(rns8_expand_sparse_a_4_to_2_u8(input, &packed, &indices, &dense, 1) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(dense == 2);
    CHECK(packed == 3);
    CHECK(indices == 5);
    rns8_sparse_matrix* out = nullptr;
    CHECK(rns8_create_sparse_matrix(ctx, input, &out) == RNS8_INVALID_ARGUMENT);
    CHECK(out == nullptr);
    return status;
  });
  REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
