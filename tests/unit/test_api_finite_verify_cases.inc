#include "../support/finite_verify_fixture.hpp"
TEST_CASE("finite verifier raw selectors and exact headers preserve live CPU state",
          "[api][abi][enum][host][finite]") {
  auto* ctx = create_cpu_context();
  auto desc = finite_matrix_desc(1, 1, RNS8_FINITE_RING_U8);
  rns8_matrix* value = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &desc, &value) == RNS8_SUCCESS);
  const uint8_t zero = 0;
  REQUIRE(rns8_pack_finite_u8(ctx, value, 6, &zero, 1, 1) == RNS8_SUCCESS);
  rns8_finite_verify_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = RNS8_ABI_VERSION;
  options.repetition_count = 1;
  options.sampler = RNS8_VERIFY_SPLITMIX64_BINARY_V1;
  options.max_workspace_bytes = UINT64_MAX;
  rns8_finite_verify_report report{};
  report.struct_size = sizeof(report);
  report.abi_version = RNS8_ABI_VERSION;
  report.passed = 71;
  const auto saved = report;
  const auto generation = value->content_generation;
  const auto residues = value->residues;
  const auto references = ctx->prepared_references;
  const auto allocations = rns8::detail::hip_direct_allocation_counters_snapshot();
  auto reject = [&](const auto* invalid) {
    CHECK(rns8_verify_finite_product(ctx, value, value, value, invalid, &report) ==
          RNS8_INVALID_ARGUMENT);
    CHECK(std::memcmp(&report, &saved, sizeof(report)) == 0);
    CHECK(value->content_generation == generation);
    CHECK(value->residues == residues);
    CHECK(value->in_flight_readers == 0);
    CHECK_FALSE(value->in_flight_writer);
    CHECK(ctx->prepared_references == references);
  };
  for (bool poisoned : {false, true}) {
    ctx->hip_failure_poisoned = poisoned;
    for (unsigned field = 0; field != 2; ++field)
      for (uint32_t bits : {UINT32_MAX, UINT32_C(0x80000000), 2u}) {
        CAPTURE(poisoned, field, bits);
        auto invalid = options;
        if (field == 0)
          invalid.sampler = bits;
        else
          invalid.flags = bits;
        reject(&invalid);
      }
    for (unsigned mode = 0; mode != 3; ++mode) {
      auto invalid = options;
      if (mode == 0)
        invalid.struct_size = sizeof(invalid) - 1;
      else if (mode == 1)
        invalid.struct_size = sizeof(invalid) + 8;
      else
        invalid.abi_version = UINT32_MAX;
      reject(&invalid);
    }
    check_short_descriptor<rns8_finite_verify_options>([&](const auto* invalid) {
      reject(invalid);
      return RNS8_INVALID_ARGUMENT;
    });
    ctx->hip_failure_poisoned = false;
  }
  struct Extended {
    rns8_finite_verify_options options;
    uint64_t canary;
  } extended{options, 73};
  extended.options.struct_size = sizeof(extended);
  reject(&extended.options);
  CHECK(extended.canary == 73);
  const auto rejected = rns8::detail::hip_direct_allocation_counters_snapshot();
  CHECK(rejected.allocate_calls == allocations.allocate_calls);
  CHECK(rejected.free_calls == allocations.free_calls);
  CHECK(rejected.h2d_calls == allocations.h2d_calls);
  CHECK(rejected.d2h_calls == allocations.d2h_calls);
  for (uint32_t flags : {0u, uint32_t(RNS8_VERIFY_CHECK_GENERATIONS)}) {
    options.flags = flags;
    std::fill_n(options.expected_generations, 3, generation);
    REQUIRE(rns8_verify_finite_product(ctx, value, value, value, &options, &report) ==
            RNS8_SUCCESS);
    CHECK(report.passed == 1);
    CHECK(value->content_generation == generation);
    CHECK(value->residues == residues);
  }
  REQUIRE(rns8_destroy_matrix(value) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
TEST_CASE("finite Freivalds public replay checks ring products without publication") {
  auto* ctx = create_cpu_context();
  for (uint16_t q : {2, 4, 6, 251, 256})
    finite_verify_fixture::public_roundtrip(ctx, q, 65);
  finite_verify_fixture::public_roundtrip(ctx, 256, 65537);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
TEST_CASE("binary Freivalds exhaustive ring errors obey half bound including zero divisors") {
  using namespace rns8::detail::finite_verify;
  const int8_t zero[2] = {0, 0};
  uint32_t br[1];
  for (uint32_t q : {2, 4, 6, 7})
    for (uint32_t x = 0; x < q; ++x)
      for (uint32_t y = 0; y < q; ++y) {
        if (!x && !y)
          continue;
        const auto center = [&](uint32_t z) {
          return int8_t(z >= (q + 1) / 2 ? int(z) - int(q) : int(z));
        };
        const int8_t error[2] = {center(x), center(y)};
        unsigned accepted = 0;
        for (unsigned bits = 0; bits < 4; ++bits) {
          const uint8_t r[2] = {uint8_t(bits & 1), uint8_t((bits >> 1) & 1)};
          const auto flags = cpu_trial(zero, zero, error, 1, 2, 1, q, r, br);
          CHECK((flags & 2) == 0);
          accepted += flags == 0;
          CHECK((flags == 0) == ((x * r[0] + y * r[1]) % q == 0));
        }
        CHECK(accepted <= 2);
      }
  const uint8_t ones[2] = {1, 1};
  const int8_t cancellation[2] = {1, -1};
  CHECK(cpu_trial(zero, zero, cancellation, 1, 2, 1, 251, ones, br) == 0);
  // An adversary with the replay seed constructs a nonzero nullspace error.
  uint64_t state = 3135;
  const uint8_t known[2] = {next_bit(state), next_bit(state)};
  const int8_t crafted[2] = {int8_t(known[1] || !known[0] ? 1 : 0), int8_t(known[0] ? -1 : 0)};
  CHECK(cpu_trial(zero, zero, crafted, 1, 2, 1, 251, known, br) == 0);
  // Integer error 6 disappears modulo 6, so this verifier cannot certify Z.
  const int8_t invisible[2] = {0, 0};
  CHECK(cpu_trial(zero, zero, invisible, 1, 2, 1, 6, ones, br) == 0);
}
TEST_CASE(
    "finite verifier preserves metadata on invalid options and never underflows to certainty") {
  auto* ctx = create_cpu_context();
  auto desc = finite_matrix_desc(1, 1, RNS8_FINITE_RING_U8);
  rns8_matrix* a = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &desc, &a) == RNS8_SUCCESS);
  const uint8_t zero = 0;
  REQUIRE(rns8_pack_finite_u8(ctx, a, 6, &zero, 1, 1) == RNS8_SUCCESS);
  rns8_finite_verify_options o{};
  o.struct_size = sizeof(o);
  o.abi_version = RNS8_ABI_VERSION;
  o.repetition_count = 1100;
  o.max_workspace_bytes = 5;
  o.sampler = RNS8_VERIFY_SPLITMIX64_BINARY_V1;
  rns8_finite_verify_report r{};
  r.struct_size = sizeof(r);
  r.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_verify_finite_product(ctx, a, a, a, &o, &r) == RNS8_SUCCESS);
  CHECK(r.ideal_failure_bound_exponent == 1100);
  CHECK(r.ideal_failure_probability == std::numeric_limits<double>::min());
  for (uint32_t count : {0u, 1048577u}) {
    o.repetition_count = count;
    r.passed = 99;
    CHECK(rns8_verify_finite_product(ctx, a, a, a, &o, &r) == RNS8_INVALID_ARGUMENT);
    CHECK(r.passed == 99);
  }
  o.repetition_count = 1;
  o.sampler = 999;
  CHECK(rns8_verify_finite_product(ctx, a, a, a, &o, &r) == RNS8_INVALID_ARGUMENT);
  o.sampler = RNS8_VERIFY_SPLITMIX64_BINARY_V1;
  a->desc.semantics = RNS8_WRAP_U64_MOD_2_64;
  CHECK(rns8_verify_finite_product(ctx, a, a, a, &o, &r) == RNS8_UNSUPPORTED_BACKEND);
  a->desc.semantics = RNS8_FINITE_RING_U8;
  REQUIRE(rns8_destroy_matrix(a) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
