#include "test_hip_direct_native_byte_cases.inc"
#include "../support/native_device_fixture.hpp"
template <class T> void check_native_domain_scan() {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  std::mt19937_64 random(5353);
  for (const auto shape : {std::pair<int, int>{1, 1}, {1, 3}, {17, 19}, {1025, 1025}})
    for (bool padded : {false, true})
      for (bool full_width : {false, true}) {
        const auto rows = shape.first, cols = shape.second, ld = cols + (padded ? 3 : 0);
        std::vector<T> values(size_t(rows) * ld, std::numeric_limits<T>::max());
        T minimum = std::numeric_limits<T>::max(), maximum = std::numeric_limits<T>::lowest();
        cpp_int magnitude = 0;
        for (int row = 0; row < rows; ++row)
          for (int col = 0; col < cols; ++col) {
            uint64_t word = random();
            if (word == 0 || word == UINT64_MAX || word == (UINT64_C(1) << 63) ||
                word == uint64_t(INT64_MAX))
              word = 1;
            T value;
            if (full_width)
              std::memcpy(&value, &word, sizeof(value));
            else if constexpr (std::is_unsigned_v<T>)
              value = T(word % 256);
            else
              value = T(int(word % 15) - 7);
            if (full_width && row == rows - 1 && col == cols - 1)
              value = std::numeric_limits<T>::lowest();
            if (full_width && row == rows - 1 && col == cols - 2)
              value = std::numeric_limits<T>::max();
            values[size_t(row) * ld + col] = value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            magnitude = std::max(magnitude, cpp_int(boost::multiprecision::abs(cpp_int(value))));
          }
        hip_direct_device_buffer input, scratch;
        REQUIRE(input.allocate(0, values.size() * sizeof(T)) == RNS8_SUCCESS);
        REQUIRE(scratch.allocate(0, 2 * sizeof(uint64_t)) == RNS8_SUCCESS);
        auto host = std::make_shared<std::vector<T>>(std::move(values));
        REQUIRE(input.retain_host_payload(host) == RNS8_SUCCESS);
        input.mark_pending();
        REQUIRE(hip_direct_copy_host_to_device(0, input.get(), host->data(),
                                               host->size() * sizeof(T)) == RNS8_SUCCESS);
        input.mark_completed();
        host.reset();
        NativeDomainRange result;
        const auto before = hip_direct_allocation_counters_snapshot();
        input.mark_pending();
        REQUIRE(hip_direct_scan_native_domain(0, input.get(), rows, cols, ld, std::is_unsigned_v<T>,
                                              scratch, result) == RNS8_SUCCESS);
        input.mark_completed();
        const auto after = hip_direct_allocation_counters_snapshot();
        CHECK(after.allocate_calls == before.allocate_calls);
        CHECK(after.free_calls == before.free_calls);
        CHECK(after.h2d_calls == before.h2d_calls);
        CHECK(after.d2h_calls == before.d2h_calls + 1);
        CHECK(after.d2h_bytes == before.d2h_bytes + 2 * sizeof(uint64_t));
        REQUIRE(result.well_formed());
        CHECK(result.unsigned_inputs == std::is_unsigned_v<T>);
        CHECK(cpp_int(result.maximum_magnitude()) == magnitude);
        if constexpr (std::is_unsigned_v<T>) {
          CHECK(result.unsigned_minimum == minimum);
          CHECK(result.unsigned_maximum == maximum);
        } else {
          CHECK(result.signed_minimum == minimum);
          CHECK(result.signed_maximum == maximum);
        }
        CHECK_FALSE(scratch.pending());
        NativeDomainRange unchanged = NativeDomainRange::signed_bounds(-5, 7);
        scratch.mark_pending();
        CHECK(hip_direct_scan_native_domain(0, input.get(), rows, cols, ld, std::is_unsigned_v<T>,
                                            scratch, unchanged) == RNS8_BUSY);
        scratch.mark_completed();
        CHECK(unchanged.signed_minimum == -5);
        CHECK(hip_direct_scan_native_domain(1, input.get(), rows, cols, ld, std::is_unsigned_v<T>,
                                            scratch, unchanged) == RNS8_INVALID_ARGUMENT);
        CHECK(unchanged.signed_minimum == -5);
        CHECK(unchanged.signed_maximum == 7);
        CHECK(hip_direct_scan_native_domain(0, nullptr, rows, cols, ld, std::is_unsigned_v<T>,
                                            scratch, unchanged) == RNS8_INVALID_ARGUMENT);
        CHECK(unchanged.signed_minimum == -5);
      }
}
TEST_CASE("direct HIP native domain scans preserve exact signed unsigned extrema and padding") {
  if (!hip_available())
    SKIP("no HIP device for native domain scans");
  check_native_domain_scan<int64_t>();
  check_native_domain_scan<uint64_t>();
}

TEST_CASE("direct HIP native narrow admission proves partial sums from actual device inputs") {
  if (!hip_available())
    SKIP("no HIP device for native narrow differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  auto run = [](const std::vector<int64_t>& av, const std::vector<int64_t>& bv, int64_t m,
                int64_t n, int64_t k, rns8_status expected_status) {
    const int64_t lda = k + 1, ldb = n + 1, ldc = n + 2;
    std::vector<int64_t> a(m * lda, INT64_MIN), b(k * ldb, INT64_MIN), c(m * ldc, 99);
    for (int64_t row = 0; row < m; ++row)
      std::copy_n(av.data() + row * k, k, a.data() + row * lda);
    for (int64_t row = 0; row < k; ++row)
      std::copy_n(bv.data() + row * n, n, b.data() + row * ldb);
    void *da = nullptr, *db = nullptr, *dc = nullptr;
    REQUIRE(hip_direct_allocate(0, a.size() * sizeof(int64_t), &da) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, b.size() * sizeof(int64_t), &db) == RNS8_SUCCESS);
    REQUIRE(hip_direct_allocate(0, c.size() * sizeof(int64_t), &dc) == RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, da, a.data(), a.size() * sizeof(int64_t)) ==
            RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, db, b.data(), b.size() * sizeof(int64_t)) ==
            RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0, dc, c.data(), c.size() * sizeof(int64_t)) ==
            RNS8_SUCCESS);
    uint64_t bound = UINT64_MAX;
    REQUIRE(hip_direct_gemm_i64_narrow_checked_device(
                0, static_cast<const int64_t*>(da), static_cast<const int64_t*>(db),
                static_cast<int64_t*>(dc), m, n, k, lda, ldb, ldc, &bound) == expected_status);
    REQUIRE(hip_direct_copy_device_to_host(0, c.data(), dc, c.size() * sizeof(int64_t)) ==
            RNS8_SUCCESS);
    if (expected_status == RNS8_SUCCESS) {
      REQUIRE(bound <= static_cast<uint64_t>(INT64_MAX));
      for (int64_t row = 0; row < m; ++row)
        for (int64_t col = 0; col < n; ++col) {
          cpp_int expected = 0;
          for (int64_t q = 0; q < k; ++q)
            expected += cpp_int(av[row * k + q]) * bv[q * n + col];
          REQUIRE(cpp_int(c[row * ldc + col]) == expected);
          REQUIRE(boost::multiprecision::abs(expected) <= bound);
        }
      for (int64_t row = 0; row < m; ++row)
        for (int64_t col = n; col < ldc; ++col)
          REQUIRE(c[row * ldc + col] == 99);
    } else {
      REQUIRE(bound == UINT64_MAX);
      REQUIRE(std::all_of(c.begin(), c.end(), [](int64_t value) { return value == 99; }));
    }
    CHECK(hip_direct_free(0, dc) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, db) == RNS8_SUCCESS);
    CHECK(hip_direct_free(0, da) == RNS8_SUCCESS);
  };
  run({INT64_MAX}, {1}, 1, 1, 1, RNS8_SUCCESS);
  run({INT64_MIN}, {0}, 1, 1, 1, RNS8_SUCCESS);
  run({INT64_MIN}, {-1}, 1, 1, 1, RNS8_RANGE_ERROR);
  run({INT64_MAX, INT64_MAX}, {1, -1}, 1, 1, 2, RNS8_RANGE_ERROR); // final zero is insufficient
  for (const int64_t k : {1, 7, 65537}) {
    constexpr int64_t m = 3, n = 5;
    std::vector<int64_t> a(m * k), b(k * n);
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = static_cast<int64_t>(i % 23) - 11;
    for (std::size_t i = 0; i < b.size(); ++i)
      b[i] = static_cast<int64_t>(i % 19) - 9;
    run(a, b, m, n, k, RNS8_SUCCESS);
  }
}

TEST_CASE("public narrow split-K matches full Boost output across slab and 65536 boundaries",
          "[hip][narrow][split-k]") {
  if (!hip_available())
    SKIP("no HIP device for narrow split-K differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr int m = 3, n = 5;
  for (int k : {4095, 4096, 4097, 65535, 65536, 65537}) {
    auto* ctx = create_context(RNS8_BACKEND_HIP_DIRECT);
    auto desc = exact_signed_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
    desc.max_prefix = 20;
    desc.flags |= RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* plan = nullptr;
    rns8_workspace* workspace = nullptr;
    REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
    auto am = matrix_desc(m, k, desc.semantics, RNS8_BOUND_NONE);
    auto bm = matrix_desc(k, n, desc.semantics, RNS8_BOUND_NONE);
    auto cm = matrix_desc(m, n, desc.semantics, RNS8_BOUND_NONE);
    am.max_prefix = bm.max_prefix = cm.max_prefix = 20;
    am.flags = bm.flags = RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &am, &a) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &bm, &b) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &cm, &c) == RNS8_SUCCESS);
    std::vector<int64_t> av(size_t(m) * (k + 1), INT64_MIN), bv(size_t(k) * (n + 2), INT64_MIN);
    for (int r = 0; r < m; ++r)
      for (int q = 0; q < k; ++q)
        av[size_t(r) * (k + 1) + q] = ((q + r) % 2 ? -1 : 1) * (1 + (q * 7 + r) % 113);
    for (int q = 0; q < k; ++q)
      for (int col = 0; col < n; ++col)
        bv[size_t(q) * (n + 2) + col] = (q * 3 + col * 11) % 97 - 48;
    REQUIRE(rns8_pack_i64(ctx, a, av.data(), k + 1, 0) == RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx, b, bv.data(), n + 2, 0) == RNS8_SUCCESS);
    const auto support = query_narrow_rns_support(m, n, k, k, n, 20, false, UINT64_MAX, true, true);
    const auto run = [&](uint64_t budget) {
      return rns8_gemm_narrow_exact(ctx, plan, a, b, c, RNS8_NARROW_SPLIT_K_I64_FUSED_RNS, budget,
                                    workspace);
    };
    CHECK(run(support.scratch_bytes - 1) == RNS8_WORKSPACE_TOO_SMALL);
    REQUIRE(run(support.scratch_bytes) == RNS8_SUCCESS);
    CHECK(workspace->narrow_product_owner.bytes() == support.payload_bytes);
    CHECK_FALSE(workspace->narrow_product_owner.pending());
    const auto before = hip_direct_allocation_counters_snapshot();
    REQUIRE(run(support.scratch_bytes) == RNS8_SUCCESS);
    const auto after = hip_direct_allocation_counters_snapshot();
    CHECK(after.allocate_calls == before.allocate_calls);
    CHECK(after.free_calls == before.free_calls);
    CHECK(after.h2d_calls == before.h2d_calls);
    CHECK(after.d2h_calls == before.d2h_calls);
    std::vector<uint64_t> output(m * n * 3);
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c, output.data(), n, 3) == RNS8_SUCCESS);
    for (int r = 0; r < m; ++r)
      for (int col = 0; col < n; ++col) {
        cpp_int expected = 0;
        for (int q = 0; q < k; ++q)
          expected += cpp_int(av[size_t(r) * (k + 1) + q]) * bv[size_t(q) * (n + 2) + col];
        const auto cell = size_t(r * n + col) * 3;
        cpp_int actual = cpp_int(output[cell]) + (cpp_int(output[cell + 1]) << 64) +
                         (cpp_int(output[cell + 2]) << 128);
        if (output[cell + 2] >> 63)
          actual -= cpp_int(1) << 192;
        CHECK(actual == expected);
      }
    const auto generation = c->content_generation;
    av[0] = INT64_MAX;
    REQUIRE(rns8_pack_i64(ctx, a, av.data(), k + 1, 0) == RNS8_SUCCESS);
    CHECK(run(support.scratch_bytes) == RNS8_RANGE_ERROR);
    CHECK(c->content_generation == generation);
    REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(b) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_matrix(a) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  }
}

template <class T>
void check_public_narrow_owner(rns8_narrow_product_method method, bool device_import = false) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr bool u = std::is_unsigned_v<T>;
  constexpr int side = 17, host_ld = 19;
  const auto payload =
      query_narrow_rns_support(side, side, side, side, side, 20, native_narrow_byte_wmma(method),
                               UINT64_MAX, native_narrow_fused(method), native_narrow_split(method))
          .payload_bytes;
  const auto semantics = u ? RNS8_EXACT_WIDE_UNSIGNED : RNS8_EXACT_WIDE_SIGNED;
  auto* ctx = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto desc = exact_signed_desc(side, side, side, RNS8_BACKEND_HIP_DIRECT);
  desc.semantics = semantics;
  desc.max_prefix = 20;
  desc.flags |= RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan = nullptr;
  rns8_workspace* ws = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &ws) == RNS8_SUCCESS);
  auto md = matrix_desc(side, side, semantics, RNS8_BOUND_NONE);
  md.max_prefix = 20;
  md.flags = RNS8_MATRIX_RETAIN_NATIVE_INPUT;
  rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr;
  REQUIRE(rns8_create_matrix(ctx, &md, &a) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &b) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
  const auto cleanup = [&]() {
    rns8_destroy_matrix(c);
    rns8_destroy_matrix(b);
    rns8_destroy_matrix(a);
    rns8_destroy_workspace(ws);
    rns8_destroy_plan(plan);
    rns8_destroy_context(ctx);
  };
  std::vector<T> av(side * host_ld, T(77)), bv(side * host_ld, T(77));
  for (int r = 0; r < side; ++r)
    for (int q = 0; q < side; ++q) {
      av[r * host_ld + q] = u ? T((r + q) % 2 ? 255 : 0) : T((r + q) % 2 ? -128 : 127);
      bv[r * host_ld + q] = u ? T((r + q) % 3 ? 251 : 1) : T((r + q) % 3 ? -127 : 126);
    }
  const auto pack = [&](rns8_matrix* matrix, const std::vector<T>& data) {
    if (device_import && matrix != c) {
      hip_direct_device_buffer input;
      struct Payload {
        std::vector<T> values;
        native_device_fixture::Owner owner;
      };
      auto host = std::make_shared<Payload>();
      host->values = data;
      auto status = input.allocate(0, host->values.size() * sizeof(T));
      if (status != RNS8_SUCCESS)
        return status;
      status = input.retain_host_payload(host);
      if (status != RNS8_SUCCESS)
        return status;
      input.mark_pending();
      status = hip_direct_copy_host_to_device(0, input.get(), host->values.data(),
                                              host->values.size() * sizeof(T));
      if (status != RNS8_SUCCESS)
        return status;
      input.mark_completed();
      auto view = native_device_fixture::descriptor(input.get(), host->values.size() * sizeof(T),
                                                    host->owner);
      view.byte_offset = 0;
      view.rows = view.cols = side;
      view.ld = host_ld;
      view.element_type = u ? RNS8_NATIVE_DEVICE_U64 : RNS8_NATIVE_DEVICE_I64;
      rns8_native_device_lease* lease = nullptr;
      status = rns8_create_native_device_lease(ctx, &view, &lease);
      if (status != RNS8_SUCCESS)
        return status;
      status = input.retain_host_payload(host);
      if (status != RNS8_SUCCESS) {
        (void)rns8_destroy_native_device_lease(lease);
        return status;
      }
      input.mark_pending();
      status = rns8_import_native_device(ctx, matrix, lease);
      if (status == RNS8_SUCCESS)
        input.mark_completed();
      const auto released = rns8_destroy_native_device_lease(lease);
      return status == RNS8_SUCCESS ? released : status;
    }
    if constexpr (std::is_unsigned_v<T>)
      return rns8_pack_u64(ctx, matrix, data.data(), host_ld, 0);
    else
      return rns8_pack_i64(ctx, matrix, data.data(), host_ld, 0);
  };
  const auto run = [&](uint64_t budget = UINT64_MAX) {
    return rns8_gemm_narrow_exact(ctx, plan, a, b, c, method, budget, ws);
  };
  REQUIRE(pack(a, av) == RNS8_SUCCESS);
  REQUIRE(pack(b, bv) == RNS8_SUCCESS);
  REQUIRE(pack(c, av) == RNS8_SUCCESS);
  for (const auto* input : {a, b}) {
    const auto native = u ? input->hip_native_u64 : input->hip_native_i64;
    const auto identity = rns8::detail::api::native_summary_identity(*ctx, *input, native);
    REQUIRE(input->native_domain.matches(identity) == !device_import);
    if (!device_import)
      CHECK(input->native_domain.range.byte_domain());
  }
  // Bind the explicit native producer through the same runtime shortlist.
  // Its output remains persistent RNS, and no ordinary GEMM cost prices the
  // actual scan/status/native-accumulation/encoding boundary.
  rns8_pipeline_query pipeline{};
  pipeline.struct_size = sizeof(pipeline);
  pipeline.abi_version = RNS8_ABI_VERSION;
  pipeline.operation.struct_size = sizeof(pipeline.operation);
  pipeline.operation.abi_version = RNS8_ABI_VERSION;
  pipeline.operation.operation = RNS8_OP_NARROW_NATIVE;
  pipeline.operation.algorithm = method;
  pipeline.operation.execution_modes = RNS8_EXEC_SYNC;
  pipeline.operation.problem = desc;
  pipeline.operation.scratch_budget = UINT64_MAX;
  pipeline.operation.output_limbs = 3;
  pipeline.operation.proof_flags = RNS8_QUERY_NATIVE_RETAINED | RNS8_QUERY_NARROW_INPUT_PROOF;
  pipeline.output_boundary = u ? 2 : 1;
  pipeline.output_magnitude_limbs[0] = UINT64_C(1000000000);
  pipeline.retained_operands = 3;
  pipeline.reuse_count = 3;
  pipeline.payload_budget = UINT64_MAX;
  rns8_pipeline_candidate candidates[7]{};
  uint32_t candidate_count = 0;
  REQUIRE(rns8_plan_pipeline(ctx, &pipeline, nullptr, 0, candidates, 7, &candidate_count) ==
          RNS8_SUCCESS);
  REQUIRE(candidate_count == 7);
  if (candidates[0].status == RNS8_UNSUPPORTED_BACKEND) {
    cleanup();
    SKIP("native producer pipeline target unsupported");
  }
  REQUIRE(candidates[0].status == RNS8_SUCCESS);
  CHECK_FALSE(candidates[0].cost_known);
  CHECK_FALSE(candidates[0].pareto);
  REQUIRE(candidates[0].stage_count > 0);
  CHECK(candidates[0].stages[0].kind == 11 + uint32_t(method));
  rns8_plan* selected = nullptr;
  rns8_workspace* selected_work = nullptr;
  REQUIRE(rns8_create_plan(ctx, &candidates[0].executable_problem, &selected) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, selected, &selected_work) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(ws) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  plan = selected;
  ws = selected_work;
  auto generation = c->content_generation;
  CHECK(run(0) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(c->content_generation == generation);
  CHECK(c->device_native_current);
  c->in_flight_readers = 1;
  CHECK(run() == RNS8_BUSY);
  c->in_flight_readers = 0;
  CHECK(c->content_generation == generation);
  CHECK(rns8_gemm_narrow_exact(ctx, plan, a, b, a, method, UINT64_MAX, ws) ==
        RNS8_INVALID_ARGUMENT);
  struct LeaseState {
    rns8_matrix *a, *b, *c;
    rns8_workspace* workspace;
    bool observed = false;
  } lease_state{a, b, c, ws};
  rns8_synchronous_test_hooks hooks{};
  hooks.state = &lease_state;
  hooks.dispatch = [](void* opaque, const char*, bool* started) -> rns8_status {
    auto& state = *static_cast<LeaseState*>(opaque);
    state.observed = state.a->in_flight_readers == 1 && state.b->in_flight_readers == 1 &&
                     state.c->in_flight_writer && state.workspace->in_flight && started &&
                     !*started;
    return RNS8_BACKEND_FAILURE;
  };
  hooks.drain = [](void*) noexcept { return RNS8_SUCCESS; };
  ctx->synchronous_test_hooks = &hooks;
  const auto injected = run();
  ctx->synchronous_test_hooks = nullptr;
  REQUIRE(injected == RNS8_BACKEND_FAILURE);
  CHECK(lease_state.observed);
  CHECK(c->content_generation == generation);
  CHECK(c->device_native_current);
  CHECK_FALSE(c->in_flight_writer);
  CHECK(a->in_flight_readers == 0);
  CHECK(b->in_flight_readers == 0);
  CHECK_FALSE(ws->in_flight);
  for (int mutation = 0; mutation < 2; ++mutation) {
    if (mutation) {
      av[0] = 3;
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
    }
    const auto status = run();
    if (status == RNS8_UNSUPPORTED_BACKEND) {
      cleanup();
      SKIP("compiled device lacks requested narrow method");
    }
    REQUIRE(status == RNS8_SUCCESS);
    const auto query = [&]() {
      rns8_narrow_summary_info info{};
      info.struct_size = sizeof(info);
      info.abi_version = RNS8_ABI_VERSION;
      REQUIRE(rns8_get_narrow_summary_info(ws, &info) == RNS8_SUCCESS);
      CHECK(info.a_instance == a->matrix_instance_id);
      CHECK(info.a_generation == a->content_generation);
      CHECK(info.b_instance == b->matrix_instance_id);
      CHECK(info.b_generation == b->content_generation);
      CHECK(info.retained_scan_bytes == (device_import ? 16 : 0));
      CHECK(info.retained_product_bytes == payload);
      CHECK(info.wmma_capability_observed == (uint32_t(method) & 1));
      CHECK(info.wmma_capability_supported == (uint32_t(method) & 1));
      CHECK(info.a_device_scans == (device_import ? uint64_t(mutation + 1) : 0));
      CHECK(info.b_device_scans == uint64_t(device_import));
      CHECK(info.a_host_proofs == (device_import ? 0 : uint64_t(mutation + 1)));
      CHECK(info.b_host_proofs == uint64_t(!device_import));
      return info;
    };
    auto proof = query();
    CHECK(proof.a_reuses == uint64_t(mutation));
    CHECK(proof.b_reuses == uint64_t(2 * mutation));
    // Observe another complete public product before export. A current native
    // certificate must avoid scans/readback/probes and reuse staged storage.
    const auto before = hip_direct_allocation_counters_snapshot();
    struct Timing {
      Timing() {
        hip_direct_timing_reset();
        hip_direct_timing_set_enabled(true);
      }
      ~Timing() {
        hip_direct_timing_set_enabled(false);
        hip_direct_timing_reset();
      }
    };
    {
      Timing timing;
      REQUIRE(run() == RNS8_SUCCESS);
      hip_direct_timing_flush_pending_events();
      CHECK(hip_direct_timing_failures_complete());
      CHECK(hip_direct_timing_failure_snapshot().empty());
      for (const auto& event : hip_direct_timing_snapshot()) {
        CHECK(event.label != "narrow_native_bound_scan");
        CHECK(event.label != "narrow_native_domain_scan");
        CHECK(event.label != "native_byte_wmma_admission");
        CHECK(event.label != "native_wmma_prepare_capability");
      }
    }
    const auto after = hip_direct_allocation_counters_snapshot();
    proof = query();
    CHECK(proof.a_reuses == uint64_t(mutation + 1));
    CHECK(proof.b_reuses == uint64_t(2 * mutation + 1));
    CHECK(after.allocate_calls == before.allocate_calls);
    CHECK(after.free_calls == before.free_calls);
    CHECK(after.h2d_calls == before.h2d_calls);
    CHECK(after.d2h_calls == before.d2h_calls);
    CHECK(ws->narrow_product_owner.bytes() == payload);
    CHECK_FALSE(c->device_native_current);
    std::vector<uint64_t> limbs(side * side * 3);
    if constexpr (std::is_unsigned_v<T>)
      REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c, limbs.data(), side, 3) ==
              RNS8_SUCCESS);
    else
      REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c, limbs.data(), side, 3) ==
              RNS8_SUCCESS);
    for (int r = 0; r < side; ++r)
      for (int col = 0; col < side; ++col) {
        cpp_int expected = 0;
        for (int q = 0; q < side; ++q)
          expected += cpp_int(av[r * host_ld + q]) * bv[q * host_ld + col];
        const int i = (r * side + col) * 3;
        cpp_int actual =
            cpp_int(limbs[i]) + (cpp_int(limbs[i + 1]) << 64) + (cpp_int(limbs[i + 2]) << 128);
        if constexpr (!std::is_unsigned_v<T>)
          if (limbs[i + 2] >> 63)
            actual -= cpp_int(1) << 192;
        CHECK(actual == expected);
      }
  }
  // Admission must use the current packed generation and reject invalid values.
  av[0] = (uint32_t(method) & 1) ? T(256) : std::numeric_limits<T>::max();
  REQUIRE(pack(a, av) == RNS8_SUCCESS);
  generation = c->content_generation;
  CHECK(run() == RNS8_RANGE_ERROR);
  CHECK(c->content_generation == generation);
  CHECK(c->device_residues_current);
  av[0] = 3;
  REQUIRE(pack(a, av) == RNS8_SUCCESS);
  if constexpr (!std::is_unsigned_v<T>)
    if (uint32_t(method) & 1) {
      av[0] = 128;
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
      CHECK(run() == RNS8_RANGE_ERROR);
      CHECK(c->content_generation == generation);
      av[0] = 3;
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
    }
  const auto saved_b = bv[0];
  bv[0] = (uint32_t(method) & 1) ? T(256) : std::numeric_limits<T>::max();
  REQUIRE(pack(b, bv) == RNS8_SUCCESS);
  CHECK(run() == RNS8_RANGE_ERROR);
  CHECK(c->content_generation == generation);
  bv[0] = saved_b;
  REQUIRE(pack(b, bv) == RNS8_SUCCESS);
  REQUIRE(run() == RNS8_SUCCESS);
  const auto summary = [&]() {
    rns8_narrow_summary_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_narrow_summary_info(ws, &info) == RNS8_SUCCESS);
    return info;
  };
  const auto before_b = summary();
  REQUIRE(pack(b, bv) == RNS8_SUCCESS); // Same caller version, new engine generation.
  REQUIRE(run() == RNS8_SUCCESS);
  const auto after_b = summary();
  CHECK(after_b.a_device_scans == before_b.a_device_scans);
  CHECK(after_b.a_host_proofs == before_b.a_host_proofs);
  CHECK(after_b.a_reuses == before_b.a_reuses + 1);
  CHECK(after_b.b_device_scans == before_b.b_device_scans + uint64_t(device_import));
  CHECK(after_b.b_host_proofs == before_b.b_host_proofs + uint64_t(!device_import));
  if (device_import) {
    for (const char* phase : {"narrow_domain_a", "narrow_domain_b"}) {
      REQUIRE(pack(a, av) == RNS8_SUCCESS);
      REQUIRE(pack(b, bv) == RNS8_SUCCESS);
      rns8_synchronous_test_hooks fault{};
      fault.state = const_cast<char*>(phase);
      fault.dispatch = [](void* state, const char* operation, bool*) {
        return std::strcmp(static_cast<const char*>(state), operation) == 0 ? RNS8_BACKEND_FAILURE
                                                                            : RNS8_SUCCESS;
      };
      fault.drain = [](void*) noexcept { return RNS8_SUCCESS; };
      generation = c->content_generation;
      ctx->synchronous_test_hooks = &fault;
      const auto failed = run();
      ctx->synchronous_test_hooks = nullptr;
      CHECK(failed == RNS8_BACKEND_FAILURE);
      CHECK(c->content_generation == generation);
      CHECK(c->device_residues_current);
      CHECK_FALSE(ws->in_flight);
      CHECK_FALSE(ws->narrow_scan_owner.pending());
      CHECK(ws->narrow_domains[1].identity.instance == 0);
      if (std::strcmp(phase, "narrow_domain_a") == 0)
        CHECK(ws->narrow_domains[0].identity.instance == 0);
      else
        CHECK(ws->narrow_domains[0].identity.generation == a->content_generation);
      REQUIRE(run() == RNS8_SUCCESS);
    }
  }
  generation = c->content_generation;
  a->generation_exhausted = true;
  CHECK(run() == RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation == generation);
  a->generation_exhausted = false;
  // A==B shares the immutable native proof; no second domain scan is needed.
  const auto before_self = summary();
  REQUIRE(rns8_gemm_narrow_exact(ctx, plan, a, a, c, method, UINT64_MAX, ws) == RNS8_SUCCESS);
  const auto after_self = summary();
  CHECK(after_self.a_device_scans == before_self.a_device_scans);
  CHECK(after_self.b_device_scans == before_self.b_device_scans);
  CHECK(after_self.a_instance == after_self.b_instance);
  CHECK(after_self.b_reuses == before_self.b_reuses + uint64_t(device_import));
  std::vector<uint64_t> self_limbs(side * side * 3);
  if constexpr (u)
    REQUIRE(rns8_export_exact_wide_unsigned_limbs(ctx, plan, c, self_limbs.data(), side, 3) ==
            RNS8_SUCCESS);
  else
    REQUIRE(rns8_export_exact_wide_signed_limbs(ctx, plan, c, self_limbs.data(), side, 3) ==
            RNS8_SUCCESS);
  for (int row = 0; row < side; ++row)
    for (int col = 0; col < side; ++col) {
      cpp_int expected = 0;
      for (int k = 0; k < side; ++k)
        expected += cpp_int(av[row * host_ld + k]) * av[k * host_ld + col];
      const int i = (row * side + col) * 3;
      cpp_int actual = cpp_int(self_limbs[i]) + (cpp_int(self_limbs[i + 1]) << 64) +
                       (cpp_int(self_limbs[i + 2]) << 128);
      if constexpr (!u)
        if (self_limbs[i + 2] >> 63)
          actual -= cpp_int(1) << 192;
      CHECK(actual == expected);
    }
  generation = c->content_generation;
  REQUIRE(rns8_gemm_rns(ctx, plan, b, b, a, ws) == RNS8_SUCCESS);
  CHECK_FALSE(a->device_native_current);
  CHECK(run() == RNS8_INVALID_ARGUMENT);
  CHECK(c->content_generation == generation);
  cleanup();
}
template <class T> void check_public_narrow_group(rns8_narrow_product_method method) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr bool u = std::is_unsigned_v<T>;
  constexpr int count = 3, m = 2, n = 3;
  const int k = native_narrow_split(method) ? 4097 : 17;
  const auto semantics = u ? RNS8_EXACT_WIDE_UNSIGNED : RNS8_EXACT_WIDE_SIGNED;
  auto* ctx = create_context(RNS8_BACKEND_HIP_DIRECT);
  auto desc = exact_signed_desc(m, n, k, RNS8_BACKEND_HIP_DIRECT);
  desc.semantics = semantics;
  desc.max_prefix = 20;
  desc.flags |= RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* plan = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  rns8_matrix *a[count]{}, *b[count]{}, *c[count]{};
  rns8_workspace* workspace[count]{};
  rns8_grouped_gemm_task tasks[count]{};
  std::vector<T> av[count], bv[count];
  for (int i = 0; i < count; ++i) {
    auto am = matrix_desc(m, k, semantics, RNS8_BOUND_NONE);
    auto bm = matrix_desc(k, n, semantics, RNS8_BOUND_NONE);
    auto cm = matrix_desc(m, n, semantics, RNS8_BOUND_NONE);
    am.max_prefix = bm.max_prefix = cm.max_prefix = 20;
    am.flags = bm.flags = RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    REQUIRE(rns8_create_matrix(ctx, &am, &a[i]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &bm, &b[i]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_matrix(ctx, &cm, &c[i]) == RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx, plan, &workspace[i]) == RNS8_SUCCESS);
    tasks[i] = {sizeof(tasks[i]), RNS8_ABI_VERSION, a[i], b[i], c[i], workspace[i]};
    av[i].assign(size_t(m) * (k + 1), std::numeric_limits<T>::max());
    bv[i].assign(size_t(k) * (n + 2), std::numeric_limits<T>::max());
    for (int r = 0; r < m; ++r)
      for (int q = 0; q < k; ++q)
        av[i][size_t(r) * (k + 1) + q] =
            u ? T((q + r + i) % 2 ? 255 : 0) : T((q + r + i) % 2 ? -128 : 127);
    for (int q = 0; q < k; ++q)
      for (int col = 0; col < n; ++col)
        bv[i][size_t(q) * (n + 2) + col] =
            u ? T(1 + (q + col * 11 + i) % 17) : T((q + col * 11 + i) % 17 - 8);
    if constexpr (u) {
      REQUIRE(rns8_pack_u64(ctx, a[i], av[i].data(), k + 1, 0) == RNS8_SUCCESS);
      REQUIRE(rns8_pack_u64(ctx, b[i], bv[i].data(), n + 2, 0) == RNS8_SUCCESS);
    } else {
      REQUIRE(rns8_pack_i64(ctx, a[i], av[i].data(), k + 1, 0) == RNS8_SUCCESS);
      REQUIRE(rns8_pack_i64(ctx, b[i], bv[i].data(), n + 2, 0) == RNS8_SUCCESS);
    }
  }
  // Related GEMVs may share an operand; destination identity is never merged.
  tasks[1].b = b[0];
  const auto support = query_narrow_group_support(m, n, k, 20, method, count, UINT64_MAX);
  const auto run = [&](uint64_t budget) {
    return rns8_gemm_narrow_exact_grouped(ctx, plan, tasks, count, method, budget);
  };
  const auto close = [&] {
    for (int i = 0; i < count; ++i) {
      CHECK(rns8_destroy_matrix(c[i]) == RNS8_SUCCESS);
      CHECK(rns8_destroy_matrix(b[i]) == RNS8_SUCCESS);
      CHECK(rns8_destroy_matrix(a[i]) == RNS8_SUCCESS);
      CHECK(rns8_destroy_workspace(workspace[i]) == RNS8_SUCCESS);
    }
    CHECK(rns8_destroy_plan(plan) == RNS8_SUCCESS);
    CHECK(rns8_destroy_context(ctx) == RNS8_SUCCESS);
  };
  CHECK(run(support.scratch_bytes - 1) == RNS8_WORKSPACE_TOO_SMALL);
  const auto first_status = run(support.scratch_bytes);
  if (first_status == RNS8_UNSUPPORTED_BACKEND) {
    close();
    SKIP("compiled device lacks grouped byte WMMA");
  }
  REQUIRE(first_status == RNS8_SUCCESS);
  const auto before = hip_direct_allocation_counters_snapshot();
  REQUIRE(run(support.scratch_bytes) == RNS8_SUCCESS);
  const auto after = hip_direct_allocation_counters_snapshot();
  CHECK(after.allocate_calls == before.allocate_calls);
  CHECK(after.free_calls == before.free_calls);
  CHECK(after.h2d_calls == before.h2d_calls);
  CHECK(after.d2h_calls == before.d2h_calls);
  uint64_t generations[count]{};
  for (int i = 0; i < count; ++i) {
    generations[i] = c[i]->content_generation;
    rns8_narrow_summary_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = RNS8_ABI_VERSION;
    REQUIRE(rns8_get_narrow_summary_info(workspace[i], &info) == RNS8_SUCCESS);
    CHECK(info.retained_group_directory_bytes == (i ? 0 : count * sizeof(NativeNarrowTask)));
    CHECK(info.retained_group_task_count == (i ? 0 : count));
    CHECK(info.retained_product_bytes == support.payload_bytes / count);
    std::vector<uint64_t> output(m * n * 3);
    const auto exported =
        u ? rns8_export_exact_wide_unsigned_limbs(ctx, plan, c[i], output.data(), n, 3)
          : rns8_export_exact_wide_signed_limbs(ctx, plan, c[i], output.data(), n, 3);
    REQUIRE(exported == RNS8_SUCCESS);
    for (int r = 0; r < m; ++r)
      for (int col = 0; col < n; ++col) {
        cpp_int expected = 0;
        const auto& b_values = bv[i == 1 ? 0 : i];
        for (int q = 0; q < k; ++q)
          expected += cpp_int(av[i][size_t(r) * (k + 1) + q]) * b_values[size_t(q) * (n + 2) + col];
        const auto cell = size_t(r * n + col) * 3;
        cpp_int actual = cpp_int(output[cell]) + (cpp_int(output[cell + 1]) << 64) +
                         (cpp_int(output[cell + 2]) << 128);
        if (!u && (output[cell + 2] >> 63))
          actual -= cpp_int(1) << 192;
        CHECK(actual == expected);
      }
  }
  c[2]->in_flight_readers = 1;
  CHECK(run(support.scratch_bytes) == RNS8_BUSY);
  c[2]->in_flight_readers = 0;
  const auto original = tasks[2];
  tasks[2].c = c[0];
  CHECK(run(support.scratch_bytes) == RNS8_INVALID_ARGUMENT);
  tasks[2] = original;
  tasks[2].workspace = workspace[0];
  CHECK(run(support.scratch_bytes) == RNS8_INVALID_ARGUMENT);
  tasks[2] = original;
  av[2][0] = u ? T(256) : native_narrow_byte_wmma(method) ? T(128) : T(INT64_MAX);
  if constexpr (u)
    REQUIRE(rns8_pack_u64(ctx, a[2], av[2].data(), k + 1, 0) == RNS8_SUCCESS);
  else
    REQUIRE(rns8_pack_i64(ctx, a[2], av[2].data(), k + 1, 0) == RNS8_SUCCESS);
  CHECK(run(support.scratch_bytes) == RNS8_RANGE_ERROR);
  for (int i = 0; i < count; ++i)
    CHECK(c[i]->content_generation == generations[i]);
  // Restore the changed task before injecting a late group write failure.
  av[2][0] = 1;
  if constexpr (u)
    REQUIRE(rns8_pack_u64(ctx, a[2], av[2].data(), k + 1, 0) == RNS8_SUCCESS);
  else
    REQUIRE(rns8_pack_i64(ctx, a[2], av[2].data(), k + 1, 0) == RNS8_SUCCESS);
  rns8_synchronous_test_hooks hooks{};
  hooks.dispatch = [](void*, const char* phase, bool* started) {
    if (std::strcmp(phase, "narrow_group_compute"))
      return RNS8_SUCCESS;
    *started = true;
    return RNS8_BACKEND_FAILURE;
  };
  hooks.drain = [](void*) noexcept { return RNS8_BACKEND_FAILURE; };
  ctx->synchronous_test_hooks = &hooks;
  CHECK(run(support.scratch_bytes) == RNS8_BACKEND_FAILURE);
  for (int i = 0; i < count; ++i) {
    CHECK(workspace[i]->in_flight);
    CHECK(c[i]->in_flight_writer);
    CHECK_FALSE(c[i]->device_residues_current);
  }
  CHECK(workspace[0]->narrow_group_owner.pending());
  ctx->synchronous_test_hooks = nullptr;
  REQUIRE(rns8_drain_failed_work(ctx) == RNS8_SUCCESS);
  for (int i = 0; i < count; ++i) {
    CHECK_FALSE(workspace[i]->in_flight);
    CHECK_FALSE(c[i]->in_flight_writer);
  }
  close();
}

TEST_CASE("grouped narrow kernels preserve independent proofs outputs and completion custody",
          "[hip][narrow][group]") {
  if (!hip_available())
    SKIP("no HIP device for grouped narrow differential");
  SECTION("signed scalar") {
    check_public_narrow_group<int64_t>(RNS8_NARROW_SCALAR_FUSED_RNS);
  }
  SECTION("signed split K") {
    check_public_narrow_group<int64_t>(RNS8_NARROW_SPLIT_K_I64_FUSED_RNS);
  }
  SECTION("signed byte WMMA") {
    check_public_narrow_group<int64_t>(RNS8_NARROW_BYTE_WMMA_FUSED_RNS);
  }
  SECTION("unsigned byte WMMA") {
    check_public_narrow_group<uint64_t>(RNS8_NARROW_BYTE_WMMA_FUSED_RNS);
  }
}

TEST_CASE("public narrow products preserve exact ownership and reject changed domains") {
  if (!hip_available())
    SKIP("no HIP device for public narrow ownership");
  SECTION("signed scalar") {
    check_public_narrow_owner<int64_t>(RNS8_NARROW_SCALAR_I64);
  }
  SECTION("signed scalar fused") {
    check_public_narrow_owner<int64_t>(RNS8_NARROW_SCALAR_FUSED_RNS);
  }
  SECTION("signed split-K fused") {
    check_public_narrow_owner<int64_t>(RNS8_NARROW_SPLIT_K_I64_FUSED_RNS);
  }
  SECTION("signed byte WMMA fused") {
    check_public_narrow_owner<int64_t>(RNS8_NARROW_BYTE_WMMA_FUSED_RNS);
  }
  SECTION("unsigned byte WMMA fused") {
    check_public_narrow_owner<uint64_t>(RNS8_NARROW_BYTE_WMMA_FUSED_RNS);
  }
  SECTION("signed byte WMMA") {
    check_public_narrow_owner<int64_t>(RNS8_NARROW_BYTE_WMMA);
  }
  SECTION("unsigned byte WMMA") {
    check_public_narrow_owner<uint64_t>(RNS8_NARROW_BYTE_WMMA);
  }
  SECTION("device imported signed scalar") {
    check_public_narrow_owner<int64_t>(RNS8_NARROW_SCALAR_I64, true);
  }
  SECTION("device imported signed scalar fused") {
    check_public_narrow_owner<int64_t>(RNS8_NARROW_SCALAR_FUSED_RNS, true);
  }
  SECTION("device imported signed split-K fused") {
    check_public_narrow_owner<int64_t>(RNS8_NARROW_SPLIT_K_I64_FUSED_RNS, true);
  }
  SECTION("device imported signed WMMA") {
    check_public_narrow_owner<int64_t>(RNS8_NARROW_BYTE_WMMA, true);
  }
  SECTION("device imported unsigned WMMA fused") {
    check_public_narrow_owner<uint64_t>(RNS8_NARROW_BYTE_WMMA_FUSED_RNS, true);
  }
}
