TEST_CASE("direct HIP rejects unsupported pack strides before touching prior values") {
  if (!hip_available()) SKIP("no HIP device for source-stride qualification");
  auto* ctx = create_context(RNS8_BACKEND_HIP_DIRECT);
  const int64_t signed_values[] = {1, -2, 3, -4};
  const uint64_t unsigned_values[] = {1, 2, 3, 4};
  for (const auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = matrix_desc(2, 2, semantics, RNS8_BOUND_NONE);
    desc.max_prefix = 4;
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &desc, &matrix) == RNS8_SUCCESS);
    const bool is_signed = semantics == RNS8_EXACT_WIDE_SIGNED;
    REQUIRE((is_signed ? rns8_pack_i64(ctx, matrix, signed_values, 2, 1)
                       : rns8_pack_u64(ctx, matrix, unsigned_values, 2, 1)) == RNS8_SUCCESS);
    const auto generation = matrix->content_generation;
    const auto bound = matrix->exact_max_magnitude;
    CHECK((is_signed ? rns8_pack_i64(ctx, matrix, signed_values, INT32_MAX, 2)
                     : rns8_pack_u64(ctx, matrix, unsigned_values, INT32_MAX, 2)) == RNS8_RANGE_ERROR);
    CHECK(matrix->content_generation == generation);
    CHECK(matrix->source_version == 1);
    CHECK(matrix->exact_max_magnitude == bound);
    CHECK(matrix->device_residues_current);
    // One plane fits signed indexing, but two planes do not. Rejection must
    // precede any access to these deliberately tiny backing allocations.
    CHECK(rns8::detail::hip_direct_gemm_rns_device(
        0, matrix->hip_residues, matrix->hip_residues, matrix->hip_residues,
        32767, 1, 65536, 65536, 1, 1, 2) == RNS8_UNSUPPORTED_BACKEND);
    REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("direct HIP exact base extension preserves every source prefix and signed representative") {
  if (!hip_available()) SKIP("no HIP device for exact base-extension qualification");
  using boost::multiprecision::cpp_int;
  auto* ctx = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr std::size_t cells = 7;
  rns8::detail::hip_direct_device_buffer retained_lift;
  REQUIRE(retained_lift.allocate(0,cells*19+sizeof(int))==RNS8_SUCCESS);
  for (const auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = matrix_desc(1, cells, semantics, RNS8_BOUND_NONE);
    desc.max_prefix = 20;
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &desc, &matrix) == RNS8_SUCCESS);
    for (uint32_t source = 1; source < 20; ++source) {
      const cpp_int product = rns8::detail::modulus_product(source);
      const bool signed_value = semantics == RNS8_EXACT_WIDE_SIGNED;
      const cpp_int bound = signed_value ? cpp_int((product - 1) / 2) : cpp_int(product - 1);
      std::vector<cpp_int> values{0, 1, bound, bound - 1, bound / 2, 7, 17};
      if (signed_value) { values[3] = -bound; values[4] = -1; }
      std::vector<int8_t> expected(cells * 20);
      for (uint32_t p = 0; p < 20; ++p)
        for (std::size_t cell = 0; cell < cells; ++cell)
          expected[p * cells + cell] = rns8::detail::centered_residue(values[cell], rns8::detail::kDefaultModuli[p]);
      for (uint32_t target = source + 1; target <= 20; ++target) {
        uint64_t bound_limbs[3]{};
        for (unsigned limb = 0; limb < 3; ++limb)
          bound_limbs[limb] = static_cast<uint64_t>((bound >> (64 * limb)) & UINT64_MAX);
        for (bool mixed_radix : {false, true}) for(bool retained:{false,true}) {
          auto staged = expected;
          std::fill(staged.begin() + source * cells, staged.end(), int8_t{99});
          REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, matrix->hip_residues, staged.data(), staged.size()) == RNS8_SUCCESS);
          bool committed = false;
          const auto before=rns8::detail::hip_direct_allocation_counters_snapshot();
          const auto lift=[&](const uint64_t* words){return retained?
              rns8::detail::hip_direct_lift_exact_suffix_workspace(0,matrix->hip_residues,cells,source,target,signed_value,words,retained_lift.get(),cells*19+sizeof(int),committed,mixed_radix):
              rns8::detail::hip_direct_lift_exact_suffix(0,matrix->hip_residues,cells,source,target,signed_value,words,committed,mixed_radix);};
          REQUIRE(lift(bound_limbs)==RNS8_SUCCESS);
          const auto after=rns8::detail::hip_direct_allocation_counters_snapshot();
          if(retained){CHECK(after.allocate_calls==before.allocate_calls);CHECK(after.free_calls==before.free_calls);}
          REQUIRE(committed);
          REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, staged.data(), matrix->hip_residues, staged.size()) == RNS8_SUCCESS);
          for (std::size_t i = 0; i < staged.size(); ++i)
            REQUIRE(staged[i] == (i < target * cells ? expected[i] : int8_t{99}));
          // A dishonest magnitude proof must not commit even a partial suffix.
          auto rejected = expected;
          std::fill(rejected.begin() + source * cells, rejected.end(), int8_t{99});
          REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, matrix->hip_residues, rejected.data(), rejected.size()) == RNS8_SUCCESS);
          uint64_t zero_bound[3]{};
          committed = true;
          REQUIRE(lift(zero_bound)==RNS8_RANGE_ERROR);
          REQUIRE_FALSE(committed);
          std::vector<int8_t> unchanged(rejected.size());
          REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, unchanged.data(), matrix->hip_residues, unchanged.size()) == RNS8_SUCCESS);
          REQUIRE(unchanged == rejected);
          // Retained failure status is reset before the next successful use.
          REQUIRE(lift(bound_limbs)==RNS8_SUCCESS);REQUIRE(committed);
        }
        auto initial = expected;
        std::fill(initial.begin() + source * cells, initial.end(), int8_t{99});
        REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, matrix->hip_residues, initial.data(), initial.size()) == RNS8_SUCCESS);
        matrix->device_residues_current = true;
        matrix->host_residues_current = false;
        matrix->host_native_current = false;
        matrix->device_native_current = false;
        matrix->exact_range_prefix = source;
        matrix->exact_max_magnitude = bound;
        const auto generation = matrix->content_generation;
        const auto identity = matrix->matrix_instance_id;
        const auto version = matrix->source_version;
        const auto bytes = cells * (target - source);
        CHECK(rns8_lift_exact_wide_hip(ctx, matrix, target, bytes - 1) == RNS8_WORKSPACE_TOO_SMALL);
        CHECK(matrix->exact_range_prefix == source);
        REQUIRE(rns8_lift_exact_wide_hip(ctx, matrix, target, bytes) == RNS8_SUCCESS);
        CHECK(matrix->exact_range_prefix == target);
        CHECK(matrix->content_generation == generation);
        CHECK(matrix->matrix_instance_id == identity);
        CHECK(matrix->source_version == version);
        CHECK(matrix->device_residues_current);
        CHECK_FALSE(matrix->host_residues_current);
        std::vector<int8_t> actual(initial.size());
        REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, actual.data(), matrix->hip_residues, actual.size()) == RNS8_SUCCESS);
        for (std::size_t i = 0; i < actual.size(); ++i)
          REQUIRE(actual[i] == (i < target * cells ? expected[i] : int8_t{99}));
        CHECK(rns8_lift_exact_wide_hip(ctx, matrix, target, 0) == RNS8_SUCCESS);
      }
      matrix->exact_range_prefix = source;
      matrix->exact_max_magnitude = product; // Cannot recover an aliased integer.
      CHECK(rns8_lift_exact_wide_hip(ctx, matrix, 20, UINT64_MAX) == RNS8_RANGE_ERROR);
    }
    REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
