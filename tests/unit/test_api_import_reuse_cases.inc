#include "backend_wrap64/canonical_host_transfer.hpp"
TEST_CASE("stable import admission requires every requested materialization",
          "[api][import][cpu-contract]") {
  using namespace rns8::detail::api;
  for (bool wide : {false, true})
    for (bool sign : {false, true}) {
      rns8_context ctx;
      ctx.backend = RNS8_BACKEND_HIP_DIRECT;
      ctx.device_id = 0;
      ctx.auto_backend_selection = !wide;
      rns8_matrix matrix;
      matrix.backend = ctx.backend;
      matrix.hip_device_id = 0;
      const auto semantics = wide ? (sign ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED)
                                  : (sign ? RNS8_BOUNDED_I64 : RNS8_BOUNDED_U64);
      matrix.desc =
          wide ? exact_matrix_desc(2, 3, semantics) : bounded_matrix_desc(2, 3, semantics);
      matrix.desc.max_prefix = 9;
      matrix.desc.flags = wide ? RNS8_MATRIX_RETAIN_NATIVE_INPUT : 0;
      matrix.prefix = 9;
      matrix.materialized_rns_prefix = 9;
      matrix.matrix_instance_id = 13;
      matrix.content_generation = 7;
      matrix.source_version = 77;
      matrix.imported_source = true;
      // Sentinels are inspected as identities/capacities only, never dereferenced
      // or passed to HIP. This is admission coverage, not fabricated GPU execution.
      matrix.hip_residues = reinterpret_cast<void*>(uintptr_t(0x1000));
      matrix.hip_residue_bytes = 54;
      if (sign) {
        matrix.hip_native_i64 = reinterpret_cast<void*>(uintptr_t(0x2000));
        matrix.hip_native_i64_bytes = 48;
      } else {
        matrix.hip_native_u64 = reinterpret_cast<void*>(uintptr_t(0x2000));
        matrix.hip_native_u64_bytes = 48;
      }
      matrix.device_residues_current = matrix.device_native_current = true;
      matrix.host_residues_current = false;
      matrix.exact_range_prefix = 9;
      matrix.exact_max_magnitude = 17;
      auto hit = [&]() { return same_source_device_rns_import_current(ctx, matrix, 77); };
      REQUIRE(hit());
      auto rejects = [&](auto& field, auto bad) {
        const auto saved = field;
        field = bad;
        CHECK_FALSE(hit());
        field = saved;
        REQUIRE(hit());
      };
      rejects(matrix.imported_source, false);
      rejects(matrix.generation_exhausted, true);
      rejects(matrix.source_version, uint64_t(78));
      rejects(matrix.matrix_instance_id, uint64_t(0));
      rejects(matrix.content_generation, uint64_t(0));
      rejects(matrix.in_flight_readers, uint64_t(1));
      rejects(matrix.in_flight_writer, true);
      rejects(ctx.hip_failure_poisoned, true);
      rejects(matrix.hip_device_id, 1);
      rejects(matrix.device_native_current, false);
      rejects(matrix.device_residues_current, false);
      rejects(matrix.materialized_rns_prefix, uint32_t(8));
      rejects(matrix.hip_residue_bytes, size_t(53));
      rejects(matrix.desc.logical_ld, int64_t(2));
      if (sign) {
        rejects(matrix.hip_native_i64, nullptr);
        rejects(matrix.hip_native_i64_bytes, size_t(40));
      } else {
        rejects(matrix.hip_native_u64, nullptr);
        rejects(matrix.hip_native_u64_bytes, size_t(40));
      }
      if (wide)
        rejects(matrix.exact_range_prefix, uint32_t(0));
      CHECK_FALSE(same_source_device_rns_import_current(ctx, matrix, 0));
    }
}
TEST_CASE("canonical wrap64 host conversion qualifies byte layout and preserves all bits",
          "[wrap64][cpu-contract]") {
  using namespace rns8::detail;
  const uint8_t little[]{1, 2, 3, 4, 5, 6, 7, 8}, big[]{8, 7, 6, 5, 4, 3, 2, 1};
  CHECK(native_u64_matches_canonical_bytes(little));
  CHECK_FALSE(native_u64_matches_canonical_bytes(big));
  for (uint64_t word : {UINT64_C(0), UINT64_C(1), UINT64_MAX, UINT64_C(0x8000000000000000),
                        UINT64_C(0x8080808080808080), UINT64_C(0x0123456789abcdef)}) {
    uint8_t bytes[10]{};
    bytes[0] = bytes[9] = 99;
    store_canonical_u64(bytes + 1, word);
    CHECK(bytes[0] == 99);
    CHECK(bytes[9] == 99);
    CHECK(load_canonical_u64(bytes + 1) == word);
    for (unsigned i = 0; i < 8; ++i)
      CHECK(bytes[i + 1] == ((word >> (8 * i)) & 255));
  }
  CanonicalU64HostTransfer plan;
  REQUIRE(canonical_u64_host_transfer(1, 3, INT64_MAX, plan));
  CHECK(plan.logical.source_span_bytes == 24);
  CHECK(plan.logical.device_bytes == 24);
  CHECK(plan.native_byte_identity == host_u64_has_canonical_bytes());
  CHECK_FALSE(canonical_u64_host_transfer(2, 3, INT64_MAX, plan));
}

TEST_CASE("public host imports retain their owned matrix after simulated completion failure",
          "[api][import][lifetime]") {
  for (auto semantics :
       {RNS8_BOUNDED_I64, RNS8_BOUNDED_U64, RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED,
        RNS8_FINITE_RING_U8, RNS8_FINITE_FIELD_U8, RNS8_WRAP_U64_MOD_2_64}) {
    const bool wrap = semantics == RNS8_WRAP_U64_MOD_2_64;
    const bool finite = semantics == RNS8_FINITE_RING_U8 || semantics == RNS8_FINITE_FIELD_U8;
    const bool wide = rns8::detail::exact_wide_semantics(semantics);
    const bool sign = semantics == RNS8_BOUNDED_I64 || semantics == RNS8_EXACT_WIDE_SIGNED;
    auto* ctx = wrap ? create_wrap_context() : create_cpu_context();
    auto md = wrap     ? wrap_matrix_desc(1, 1)
              : finite ? finite_matrix_desc(1, 1, semantics)
              : wide   ? exact_matrix_desc(1, 1, semantics)
                       : bounded_matrix_desc(1, 1, semantics);
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
    const uint64_t word = 1;
    const uint8_t byte = 1;
    auto pack = [&](uint64_t version) {
      return finite ? rns8_pack_finite_u8(ctx, matrix, 251, &byte, 1, version)
             : sign
                 ? rns8_pack_i64(ctx, matrix, reinterpret_cast<const int64_t*>(&word), 1, version)
                 : rns8_pack_u64(ctx, matrix, &word, 1, version);
    };
    REQUIRE(pack(1) == RNS8_SUCCESS);
    const auto generation = matrix->content_generation;
    SynchronousDispatchFixture state;
    state.writers = {matrix};
    state.fail = true;
    state.drain_fails = true;
    const rns8_synchronous_test_hooks hooks{&state, &SynchronousDispatchFixture::dispatch,
                                            &SynchronousDispatchFixture::drain};
    ctx->synchronous_test_hooks = &hooks;
    CHECK(pack(0) == RNS8_BACKEND_FAILURE);
    CHECK(state.calls == 1);
    CHECK(state.reservations_observed);
    CHECK(ctx->hip_failure_poisoned);
    CHECK(matrix->in_flight_writer);
    CHECK(matrix->content_generation == generation);
    CHECK_FALSE(matrix->host_residues_current);
    CHECK_FALSE(matrix->host_byte_limbs_current);
    CHECK(rns8_destroy_matrix(matrix) == RNS8_BUSY);
    state.drain_fails = false;
    REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
    CHECK_FALSE(matrix->in_flight_writer);
    CHECK(ctx->failed_work_leases.empty());
    CHECK(pack(0) == RNS8_BACKEND_FAILURE);
    CHECK(state.calls == 1);
    ctx->synchronous_test_hooks = nullptr;
    REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
}
