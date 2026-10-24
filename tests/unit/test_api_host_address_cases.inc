TEST_CASE("native and limb host address admission proves byte extent", "[api][address]") {
  using rns8::detail::api::valid_matrix_access;
  using rns8::detail::api::valid_limb_export_access;
  for (const auto limbs : {1u, 4u, 32u}) {
    const int64_t pitch = static_cast<int64_t>((uint64_t{1} << 61) / limbs);
    CHECK_FALSE(valid_limb_export_access(2, 1, pitch, limbs));
    CHECK(valid_limb_export_access(1, 1, pitch, limbs));
    const uint64_t limit = static_cast<uint64_t>(PTRDIFF_MAX) / (8 * limbs);
    CHECK(valid_limb_export_access(2, 1, static_cast<int64_t>(limit - 1), limbs));
    CHECK_FALSE(valid_limb_export_access(2, 1, static_cast<int64_t>(limit), limbs));
  }
  CHECK_FALSE(valid_matrix_access(2, 1, INT64_MAX));
  CHECK(valid_matrix_access(1, 1, INT64_MAX));
  CHECK(valid_matrix_access(1, 1, INT64_MAX, 1));
  CHECK_FALSE(valid_matrix_access(2, 1, INT64_MAX, 1));
}

TEST_CASE("limb import and export reject inaccessible padding without touching caller storage", "[api][address]") {
  auto* ctx = create_cpu_context();
  for (const auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = exact_desc(semantics, 2, 1, 1);
    auto md = exact_matrix_desc(2, 1, semantics);
    rns8_plan* plan = nullptr; rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
    for (const auto count : {1u, 4u, 32u}) {
      std::vector<uint64_t> input(6 * count, 0), output(6 * count, 0xfeed);
      input[0] = 17; input[3 * count] = 23;
      REQUIRE(rns8::detail::api::import_exact_wide_limbs(*ctx, *matrix, input.data(), 3, count,
                  semantics == RNS8_EXACT_WIDE_SIGNED, 42) == RNS8_SUCCESS);
      const auto generation = matrix->content_generation;
      const auto export_to = [&](int64_t pitch) {
        return semantics == RNS8_EXACT_WIDE_SIGNED
          ? rns8_export_exact_wide_signed_limbs(ctx, plan, matrix, output.data(), pitch, count)
          : rns8_export_exact_wide_unsigned_limbs(ctx, plan, matrix, output.data(), pitch, count);
      };
      const int64_t impossible = static_cast<int64_t>((uint64_t{1} << 61) / count);
      REQUIRE(export_to(impossible) == RNS8_INVALID_ARGUMENT);
      CHECK(output == std::vector<uint64_t>(6 * count, 0xfeed));
      REQUIRE(rns8::detail::api::import_exact_wide_limbs(*ctx, *matrix, input.data(), impossible, count,
                  semantics == RNS8_EXACT_WIDE_SIGNED, 43) == RNS8_INVALID_ARGUMENT);
      CHECK(matrix->content_generation == generation);
      REQUIRE(export_to(3) == RNS8_SUCCESS);
      for (unsigned limb = 0; limb < count; ++limb) {
        CHECK(output[limb] == input[limb]);
        CHECK(output[3 * count + limb] == input[3 * count + limb]);
      }
      CHECK(output[count] == 0xfeed);
    }
    REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
