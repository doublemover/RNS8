#include "core/hip_resources.hpp"
TEST_CASE(
    "HIP graphs retain one shared operand and preserve full-width semantics",
    "[api][async][graph]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (bool signed_values : {false, true}) {
    rns8_context_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_context *ctx = nullptr;
    REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
    const auto semantics =
        signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
    auto desc = exact_desc(semantics, 2, 2, 2);
    desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    auto md = exact_matrix_desc(2, 2, semantics);
    rns8_plan *plan = nullptr;
    rns8_workspace *workspace = nullptr;
    rns8_matrix *a = nullptr, *c = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &md, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
    const int64_t sv[]{INT64_MIN, INT64_MAX, INT64_MAX, -1};
    const uint64_t uv[]{UINT64_MAX, UINT64_C(1) << 63, 1, UINT64_MAX};
    REQUIRE((signed_values ? rns8_pack_i64(ctx, a, sv, 2, 0)
                           : rns8_pack_u64(ctx, a, uv, 2, 0)) == RNS8_SUCCESS);
    rns8_prepared_execution *execution = nullptr;
    REQUIRE(rns8_prepare_gemm(ctx, plan, a, a, c, workspace, &execution) ==
            RNS8_SUCCESS);
    REQUIRE(a->prepared_references == 1);
    REQUIRE(rns8_capture_prepared_hip_graph(execution) == RNS8_SUCCESS);
    // Successful replacement capture must preserve the same lifetime owner.
    REQUIRE(rns8_capture_prepared_hip_graph(execution) == RNS8_SUCCESS);
    hip_unique_stream stream;
    REQUIRE(stream.create_non_blocking() == hipSuccess);
    rns8_completion_token token{77, 88};
    REQUIRE(hipStreamBeginCapture(
                stream.get(), hipStreamCaptureModeThreadLocal) == hipSuccess);
    REQUIRE(rns8_submit_prepared_hip_graph(execution, stream.get(), &token) ==
            RNS8_UNSUPPORTED_BACKEND);
    REQUIRE(token.prepared_id == 77);
    REQUIRE(token.epoch == 88);
    hipGraph_t external_graph = nullptr;
    REQUIRE(hipStreamEndCapture(stream.get(), &external_graph) == hipSuccess);
    REQUIRE(hipGraphDestroy(external_graph) == hipSuccess);
    for (bool graph : {false, true}) {
      REQUIRE((graph ? rns8_submit_prepared_hip_graph(execution, stream.get(),
                                                      &token)
                     : rns8_submit_prepared_hip(execution, stream.get(),
                                                &token)) == RNS8_SUCCESS);
      REQUIRE(a->in_flight_readers == 1);
      REQUIRE(rns8_destroy_matrix(a) == RNS8_BUSY);
      rns8_completion_info info{};
      info.struct_size = sizeof(info);
      info.abi_version = RNS8_ABI_VERSION;
      REQUIRE(rns8_wait_prepared_hip(execution, token, &info) == RNS8_SUCCESS);
      REQUIRE(info.operation_status == RNS8_SUCCESS);
      REQUIRE(info.output_is_current);
      REQUIRE(a->in_flight_readers == 0);
      uint64_t limbs[12]{};
      REQUIRE((signed_values ? rns8_export_exact_wide_signed_limbs(ctx, plan, c,
                                                                   limbs, 2, 3)
                             : rns8_export_exact_wide_unsigned_limbs(
                                   ctx, plan, c, limbs, 2, 3)) == RNS8_SUCCESS);
      for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col) {
          cpp_int expected = 0;
          for (int q = 0; q < 2; ++q)
            expected += (signed_values ? cpp_int(sv[row * 2 + q])
                                       : cpp_int(uv[row * 2 + q])) *
                        (signed_values ? cpp_int(sv[q * 2 + col])
                                       : cpp_int(uv[q * 2 + col]));
          const int cell = row * 2 + col;
          cpp_int actual = cpp_int(limbs[3 * cell]) +
                           (cpp_int(limbs[3 * cell + 1]) << 64) +
                           (cpp_int(limbs[3 * cell + 2]) << 128);
          if (signed_values && (limbs[3 * cell + 2] >> 63))
            actual -= cpp_int(1) << 192;
          REQUIRE(actual == expected);
        }
    }
    for (auto special : {hipStreamLegacy, hipStreamPerThread}) {
      const auto previous = token;
      REQUIRE(rns8_submit_prepared_hip(execution, special, &token) == RNS8_INVALID_ARGUMENT);
      CHECK(token.epoch == previous.epoch);
    }
    REQUIRE(rns8_submit_prepared_hip(execution, stream.get(), &token) == RNS8_SUCCESS);
    auto wrong_cancel = token;
    ++wrong_cancel.prepared_id;
    CHECK(rns8_cancel_prepared_hip(execution, wrong_cancel) == RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8_cancel_prepared_hip(execution, token) == RNS8_SUCCESS);
    REQUIRE(rns8_cancel_prepared_hip(execution, token) == RNS8_SUCCESS);
    CHECK(rns8_destroy_prepared_execution(execution) == RNS8_BUSY);
    CHECK(a->in_flight_readers == 1);
    rns8_completion_info canceled{};
    canceled.struct_size = sizeof(canceled); canceled.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_wait_prepared_hip(execution, token, &canceled) == RNS8_SUCCESS);
    REQUIRE(rns8_wait_prepared_hip_for(execution, token, 0, &canceled) == RNS8_SUCCESS);
    CHECK(canceled.operation_status == RNS8_CANCELED);
    CHECK(canceled.output_is_current == 0);
    CHECK_FALSE(c->device_residues_current);
    CHECK(a->in_flight_readers == 0);
    CHECK(rns8_cancel_prepared_hip(execution, token) == RNS8_SUCCESS);
    rns8_async_failure_info failure{};
    failure.struct_size = sizeof(failure); failure.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_prepared_async_failure(execution, token, &failure) == RNS8_SUCCESS);
    CHECK(failure.stage == RNS8_ASYNC_FAILURE_NONE);
    CHECK(failure.failed_node == UINT32_MAX);
    REQUIRE(rns8_submit_prepared_hip_graph(execution, stream.get(), &token) == RNS8_SUCCESS);
    REQUIRE(rns8_wait_prepared_hip(execution, token, &canceled) == RNS8_SUCCESS);
    CHECK(canceled.operation_status == RNS8_SUCCESS);
    CHECK(canceled.output_is_current == 1);
    REQUIRE(rns8_destroy_prepared_execution(execution) == RNS8_SUCCESS);
    REQUIRE(a->prepared_references == 0);
    REQUIRE(rns8_destroy_matrix(a) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
#else
  SKIP("HIP graph submission unavailable in this build");
#endif
}

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
    desc.max_prefix = 20;
    desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_plan *plan = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    rns8_workspace *w1 = nullptr, *w2 = nullptr;
    REQUIRE(rns8_create_workspace(ctx, plan, &w1) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &w2) == RNS8_SUCCESS);
    auto ad = exact_matrix_desc(2, k, RNS8_EXACT_WIDE_SIGNED),
         bd = exact_matrix_desc(k, 3, RNS8_EXACT_WIDE_SIGNED),
         cd = exact_matrix_desc(2, 3, RNS8_EXACT_WIDE_SIGNED);
    ad.max_prefix = bd.max_prefix = cd.max_prefix = 20;
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
    // Native full-width inputs fit this plan. Produce a genuinely wider A
    // through an exact outer product to exercise range rejection on replay.
    auto wide_desc = exact_desc(RNS8_EXACT_WIDE_SIGNED, 2, k, 1);
    wide_desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    wide_desc.max_prefix = 20;
    wide_desc.flags |= RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* wide_plan = nullptr;
    rns8_workspace* wide_work = nullptr;
    REQUIRE(rns8_create_plan(ctx, &wide_desc, &wide_plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, wide_plan, &wide_work) == RNS8_SUCCESS);
    auto xd = exact_matrix_desc(2, 1, RNS8_EXACT_WIDE_SIGNED);
    auto yd = exact_matrix_desc(1, k, RNS8_EXACT_WIDE_SIGNED);
    rns8_matrix *x = nullptr, *y = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &xd, &x) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &yd, &y) == RNS8_SUCCESS);
    const int64_t xv[]{INT64_MAX, INT64_MAX};
    const std::vector<int64_t> yv(k, INT64_MAX);
    REQUIRE(rns8_pack_i64(ctx, x, xv, 1, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, y, yv.data(), k, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx, wide_plan, x, y, a, wide_work) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(y) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(wide_work) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(wide_plan) == RNS8_SUCCESS);
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


TEST_CASE("prepared HIP epilogues bind addends and preserve capture expression identity", "[api][async][epilogue]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (bool signed_values : {false, true}) for (bool shared : {false, true}) {
    rns8_context_options options{};
    options.struct_size = sizeof(options); options.abi_version = RNS8_ABI_VERSION;
    options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    rns8_context* ctx = nullptr;
    REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
    const auto semantics = signed_values ? RNS8_EXACT_WIDE_SIGNED : RNS8_EXACT_WIDE_UNSIGNED;
    auto desc = exact_desc(semantics, 2, 2, 2);
    desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
    auto md = exact_matrix_desc(2, 2, semantics);
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    rns8_matrix *a = nullptr, *d = nullptr, *c = nullptr;
    for (auto** matrix : {&a, &d, &c}) REQUIRE(rns8_create_matrix(ctx, &md, matrix) == RNS8_SUCCESS);
    const auto pack = [&](rns8_matrix* matrix, const std::vector<int64_t>& values) {
      if (signed_values) return rns8_pack_i64(ctx, matrix, values.data(), 2, 0);
      const std::vector<uint64_t> converted(values.begin(), values.end());
      return rns8_pack_u64(ctx, matrix, converted.data(), 2, 0);
    };
    std::vector<int64_t> av = signed_values ? std::vector<int64_t>{-1, 2, 3, -4} : std::vector<int64_t>{1, 2, 3, 4};
    std::vector<int64_t> dv{5, 6, 7, 8};
    REQUIRE(pack(a, av) == RNS8_SUCCESS);
    REQUIRE(pack(d, dv) == RNS8_SUCCESS);
    auto* addend = shared ? a : d;
    const int64_t alpha = signed_values ? INT64_MIN : INT64_MAX, beta = INT64_MAX;
    rns8_prepared_execution* e = nullptr;
    REQUIRE(rns8_prepare_exact_axpby(ctx, plan, a, a, addend, c, alpha, beta, workspace, &e) == RNS8_SUCCESS);
    REQUIRE(a->prepared_references == 1);
    REQUIRE(addend->prepared_references == 1);
    REQUIRE(rns8_capture_prepared_hip_graph(e) == RNS8_SUCCESS);
    hip_unique_stream stream;
    REQUIRE(stream.create_non_blocking() == hipSuccess);
    rns8_completion_info info{};
    info.struct_size = sizeof(info); info.abi_version = RNS8_ABI_VERSION;
    rns8_completion_token token{};
    for (bool graph : {false, true}) {
      const auto generation = c->content_generation;
      REQUIRE((graph ? rns8_submit_prepared_hip_graph(e, stream.get(), &token)
                     : rns8_submit_prepared_hip(e, stream.get(), &token)) == RNS8_SUCCESS);
      REQUIRE(a->in_flight_readers == 1);
      REQUIRE(addend->in_flight_readers == 1);
      REQUIRE(pack(addend, dv) == RNS8_BUSY);
      REQUIRE(c->content_generation == generation);
      REQUIRE_FALSE(c->device_residues_current);
      REQUIRE(rns8_wait_prepared_hip(e, token, &info) == RNS8_SUCCESS);
      REQUIRE(info.operation_status == RNS8_SUCCESS);
      REQUIRE(info.output_is_current);
      REQUIRE(info.has_exact_epilogue == 1);
      REQUIRE(info.alpha == alpha); REQUIRE(info.beta == beta);
      REQUIRE(info.d_instance == addend->matrix_instance_id);
      REQUIRE(info.d_generation == addend->content_generation);
      REQUIRE(addend->in_flight_readers == 0);
      uint64_t limbs[12]{};
      REQUIRE((signed_values ? rns8_export_exact_wide_signed_limbs(ctx, plan, c, limbs, 2, 3)
                             : rns8_export_exact_wide_unsigned_limbs(ctx, plan, c, limbs, 2, 3)) == RNS8_SUCCESS);
      for (int row = 0; row < 2; ++row) for (int col = 0; col < 2; ++col) {
        const auto cell = row * 2 + col;
        cpp_int expected = cpp_int(beta) * (shared ? av[cell] : dv[cell]);
        for (int q = 0; q < 2; ++q) expected += cpp_int(alpha) * av[row * 2 + q] * av[q * 2 + col];
        cpp_int actual = cpp_int(limbs[3 * cell]) + (cpp_int(limbs[3 * cell + 1]) << 64) + (cpp_int(limbs[3 * cell + 2]) << 128);
        if (signed_values && limbs[3 * cell + 2] >> 63) actual -= cpp_int(1) << 192;
        REQUIRE(actual == expected);
      }
    }
    REQUIRE(pack(addend, dv) == RNS8_SUCCESS);
    const auto previous = token;
    REQUIRE(rns8_submit_prepared_hip_graph(e, stream.get(), &token) == RNS8_INVALID_ARGUMENT);
    REQUIRE(token.epoch == previous.epoch);
    REQUIRE(rns8_refresh_prepared_inputs(e) == RNS8_SUCCESS);
    REQUIRE(rns8_submit_prepared_hip_graph(e, stream.get(), &token) == RNS8_SUCCESS);
    REQUIRE(rns8_wait_prepared_hip(e, token, &info) == RNS8_SUCCESS);
    REQUIRE(info.d_generation == addend->content_generation);
    REQUIRE(info.operation_status == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_prepared_execution(e) == RNS8_SUCCESS);
    REQUIRE(a->prepared_references == 0); REQUIRE(d->prepared_references == 0);
    for (auto* matrix : {a, d, c}) REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
#else
  SKIP("HIP epilogue submission unavailable in this build");
#endif
}
#include "test_api_prepared_batch_cases.inc"
