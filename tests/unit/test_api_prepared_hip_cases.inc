#include "core/hip_resources.hpp"
TEST_CASE("HIP submissions reserve shared readers and publish only after "
          "observed completion",
          "[api][async]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  for (int k : {17, 65537}) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_context *ctx = nullptr;
    REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
    auto desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 2, 3, k);
    desc.max_prefix = 9;
    desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_plan *plan = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_workspace *w1 = nullptr, *w2 = nullptr;
    REQUIRE(rns8_create_workspace(ctx, plan, &w1) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &w2) == RNS8_SUCCESS);
    auto ad = exact_matrix_desc(2, k, RNS8_EXACT_WIDE_SIGNED),
         bd = exact_matrix_desc(k, 3, RNS8_EXACT_WIDE_SIGNED),
         cd = exact_matrix_desc(2, 3, RNS8_EXACT_WIDE_SIGNED);
    ad.max_prefix = bd.max_prefix = cd.max_prefix = 9;
    rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr,
                *collision = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &ad, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &bd, &b) == RNS8_SUCCESS);
    for (auto **matrix : {&c, &d, &collision})
      REQUIRE(rns8_create_matrix(ctx, &cd, matrix) == RNS8_SUCCESS);
    std::vector<int64_t> av(2 * k), bv(k * 3);
    for (std::size_t i = 0; i < av.size(); ++i)
      av[i] = 1 + i % 3;
    for (std::size_t i = 0; i < bv.size(); ++i)
      bv[i] = 1 + i % 5;
    REQUIRE(rns8_pack_i64(ctx, a, av.data(), k, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, b, bv.data(), 3, 0) == RNS8_SUCCESS);
    rns8_prepared_execution *e1 = nullptr, *e2 = nullptr, *e3 = nullptr;
    REQUIRE(rns8_prepare_gemm(ctx, plan, a, b, c, w1, &e1) == RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx, plan, a, b, d, w2, &e2) == RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx, plan, a, b, collision, w1, &e3) ==
            RNS8_SUCCESS);
    for (auto *e : {e1, e2, e3})
      REQUIRE(rns8_prepare_hip_submission(e) == RNS8_SUCCESS);
    hip_unique_stream s1, s2;
    REQUIRE(s1.create_non_blocking() == hipSuccess);
    REQUIRE(s2.create_non_blocking() == hipSuccess);
    rns8_completion_token t1{}, t2{}, unused{};
    REQUIRE(rns8_submit_prepared_hip_graph(e1, s1.get(), &unused) ==
            RNS8_UNSUPPORTED_BACKEND);
    REQUIRE(rns8_submit_prepared_hip(e1, nullptr, &unused) ==
            RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8_submit_prepared_hip(e1, s1.get(), &t1) == RNS8_SUCCESS);
    REQUIRE(a->in_flight_readers == 1);
    REQUIRE(c->in_flight_writer);
    REQUIRE_FALSE(c->device_residues_current);
    REQUIRE(rns8_pack_i64(ctx, a, av.data(), k, 0) == RNS8_BUSY);
    REQUIRE(rns8_lift_exact_wide_hip(ctx, c, 20, UINT64_MAX) == RNS8_BUSY);
    REQUIRE(rns8_destroy_prepared_execution(e1) == RNS8_BUSY);
    REQUIRE(rns8_capture_prepared_hip_graph(e1) == RNS8_BUSY);
    REQUIRE(rns8_execute_prepared(e1, &unused) == RNS8_BUSY);
    REQUIRE(rns8_submit_prepared_hip(e3, s2.get(), &unused) == RNS8_BUSY);
    REQUIRE(rns8_submit_prepared_hip(e2, s2.get(), &t2) == RNS8_SUCCESS);
    REQUIRE(a->in_flight_readers == 2);
    rns8_completion_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_prepared_completion(e1, t1, &info) == RNS8_BUSY);
    REQUIRE(rns8_wait_prepared_hip(e2, t2, &info) == RNS8_SUCCESS);
    REQUIRE(info.completed);
    REQUIRE(info.output_is_current);
    REQUIRE(info.operation_status == RNS8_SUCCESS);
    REQUIRE(a->in_flight_readers == 1);
    REQUIRE(rns8_pack_i64(ctx, a, av.data(), k, 0) == RNS8_BUSY);
    REQUIRE(rns8_wait_prepared_hip(e1, t1, &info) == RNS8_SUCCESS);
    REQUIRE(info.output_is_current);
    REQUIRE(a->in_flight_readers == 0);
    REQUIRE_FALSE(w1->in_flight);
    const auto before_capture = c->content_generation;
    REQUIRE(rns8_capture_prepared_hip_graph(e1) == RNS8_SUCCESS);
    REQUIRE(c->content_generation == before_capture);
    REQUIRE(c->device_residues_current);
    hip_direct_allocation_counters_reset();
    for (int replay = 0; replay < 3; ++replay) {
      REQUIRE(rns8_submit_prepared_hip_graph(e1, s1.get(), &unused) ==
              RNS8_SUCCESS);
      REQUIRE_FALSE(c->device_residues_current);
      REQUIRE(rns8_wait_prepared_hip(e1, unused, &info) == RNS8_SUCCESS);
      REQUIRE(info.operation_status == RNS8_SUCCESS);
      REQUIRE(info.output_is_current);
    }
    REQUIRE(hip_direct_allocation_counters_snapshot().allocate_calls == 0);
    uint64_t limbs[6]{};
    for (auto *matrix : {c, d}) {
      REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, matrix, limbs, 3,
                                                  1) == RNS8_SUCCESS);
      for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 3; ++col) {
          boost::multiprecision::cpp_int expected = 0;
          for (int q = 0; q < k; ++q)
            expected += boost::multiprecision::cpp_int(av[row * k + q]) *
                        bv[q * 3 + col];
          REQUIRE(boost::multiprecision::cpp_int(limbs[row * 3 + col]) ==
                  expected);
        }
    }
    av[0] = 9;
    REQUIRE(rns8_pack_i64(ctx, a, av.data(), k, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip(e1, s1.get(), &unused) ==
            RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8_refresh_prepared_inputs(e1) == RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_graph(e1, s1.get(), &unused) ==
            RNS8_SUCCESS);
    REQUIRE(rns8_query_prepared_hip(e1, t1, &info) == RNS8_INVALID_ARGUMENT);
    const auto query = rns8_query_prepared_hip(e1, unused, &info);
    REQUIRE((query == RNS8_BUSY || query == RNS8_SUCCESS));
    REQUIRE(rns8_wait_prepared_hip(e1, unused, &info) == RNS8_SUCCESS);
    REQUIRE(info.output_is_current);
    const auto generation = c->content_generation;
    std::fill(av.begin(), av.end(), INT64_MAX);
    std::fill(bv.begin(), bv.end(), INT64_MAX);
    REQUIRE(rns8_pack_i64(ctx, a, av.data(), k, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, b, bv.data(), 3, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_refresh_prepared_inputs(e1) == RNS8_SUCCESS);
    const auto previous = unused;
    REQUIRE(rns8_submit_prepared_hip_graph(e1, s1.get(), &unused) ==
            RNS8_RANGE_ERROR);
    REQUIRE(unused.epoch == previous.epoch);
    REQUIRE(c->content_generation == generation);
    REQUIRE(c->device_residues_current);
    REQUIRE_FALSE(w1->in_flight);
    for (auto *e : {e1, e2, e3})
      REQUIRE(rns8_destroy_prepared_execution(e) == RNS8_SUCCESS);
    for (auto *matrix : {a, b, c, d, collision})
      REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(w1) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(w2) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
#else
  SKIP("HIP submission unavailable in this build");
#endif
}

TEST_CASE(
    "write reservations reject native repack and preserve the existing matrix",
    "[api][async]") {
  auto *ctx = create_cpu_context();
  auto md = exact_matrix_desc(1, 1, RNS8_EXACT_WIDE_SIGNED);
  rns8_matrix *matrix = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &matrix) == RNS8_SUCCESS);
  const int64_t value = 7;
  REQUIRE(rns8_pack_i64(ctx, matrix, &value, 1, 0) == RNS8_SUCCESS);
  const auto generation = matrix->content_generation;
  matrix->in_flight_readers = 1;
  REQUIRE(rns8_pack_i64(ctx, matrix, &value, 1, 0) == RNS8_BUSY);
  REQUIRE(matrix->content_generation == generation);
  REQUIRE(matrix->host_residues_current);
  matrix->in_flight_readers = 0;
  matrix->in_flight_writer = true;
  REQUIRE(rns8_pack_i64(ctx, matrix, &value, 1, 0) == RNS8_BUSY);
  matrix->in_flight_writer = false;
  REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}
