TEST_CASE("persistent native WMMA queue mixes shapes signedness groups and K slabs") {
  if (!hip_available()) SKIP("no HIP device for native WMMA queue differential");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  // Mixed dispatch plus all six uniform specializations use the same independent
  // exact oracle, padded tails and cross-slab input. Shapes remain heterogeneous.
  for (uint32_t variant = 0; variant < 9; ++variant) {
    DYNAMIC_SECTION("queue specialization " << variant) {
      struct Buffers {
        void *a = nullptr, *b = nullptr, *c = nullptr, *encoded = nullptr;
        ~Buffers() { hip_direct_free(0, encoded); hip_direct_free(0, c); hip_direct_free(0, b); hip_direct_free(0, a); }
      } buffers[4];
      std::vector<NativeWmmaTask> tasks;
      std::vector<std::vector<int8_t>> expected(4), actual(4);
      constexpr int moduli[] = {256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167};
      const auto value = [](uint64_t bits, bool is_unsigned) -> cpp_int {
        if (is_unsigned) return cpp_int(bits);
        int64_t signed_value;
        std::memcpy(&signed_value, &bits, sizeof(bits));
        return cpp_int(signed_value);
      };
      for (uint32_t id = 0; id < 4; ++id) {
        const int m = id == 3 ? 1 : 17 + id, n = id == 3 ? 1 : 19 - id;
        const int k = id == 3 ? 65537 : 15 + id, lda = k + 1, ldb = n + 2, ldc = n + 3;
        const uint32_t prefix = id == 0 ? 1 : id == 1 ? 3 : id == 2 ? 9 : 20;
        const bool uniform=variant>0 && variant<7;
        const uint32_t group = uniform ? (1u << ((variant - 1) % 3)) : (id == 0 ? 1 : id == 1 ? 2 : 4);
        const bool is_unsigned = uniform ? variant > 3 : id % 2 != 0;
        const uint64_t values[] = {0, 1, UINT64_MAX, uint64_t{1} << 63, INT64_MAX, 127, 255};
        std::vector<uint64_t> a(m * lda), b(k * ldb);
        for (int row = 0; row < m; ++row)
          for (int q = 0; q < k; ++q) a[row * lda + q] = values[(row + q * 3) % 7];
        for (int q = 0; q < k; ++q)
          for (int col = 0; col < n; ++col) b[q * ldb + col] = values[(q * 5 + col) % 7];
        expected[id].assign((prefix + 1) * m * ldc, 99);
        actual[id] = expected[id];
        for (int row = 0; row < m; ++row)
          for (int col = 0; col < n; ++col) {
            cpp_int sum = 0;
            for (int q = 0; q < k; ++q)
              sum += value(a[row * lda + q], is_unsigned) * value(b[q * ldb + col], is_unsigned);
            for (uint32_t p = 0; p < prefix; ++p) {
              int r = (sum % moduli[p]).convert_to<int>();
              if (r < 0) r += moduli[p];
              if (r >= (moduli[p] + 1) / 2) r -= moduli[p];
              expected[id][(p * m + row) * ldc + col] = static_cast<int8_t>(r);
            }
          }
        auto& buffer = buffers[id];
        REQUIRE(hip_direct_allocate(0, a.size() * 8, &buffer.a) == RNS8_SUCCESS);
        REQUIRE(hip_direct_allocate(0, b.size() * 8, &buffer.b) == RNS8_SUCCESS);
        REQUIRE(hip_direct_allocate(0, actual[id].size(), &buffer.c) == RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_host_to_device(0, buffer.a, a.data(), a.size() * 8) == RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_host_to_device(0, buffer.b, b.data(), b.size() * 8) == RNS8_SUCCESS);
        tasks.push_back({buffer.a, buffer.b, static_cast<int8_t*>(buffer.c), m, n, k, lda, ldb, ldc,
                         prefix, group, static_cast<uint32_t>(is_unsigned)});
        if(variant==7 || (variant==8 && id%2)) {
          const auto layout=encoded_b_layout(k,n,ldb,prefix,UINT64_MAX);
          REQUIRE(layout.status==RNS8_SUCCESS);
          REQUIRE(hip_direct_allocate(0,layout.total_bytes,&buffer.encoded)==RNS8_SUCCESS);
          const auto status=hip_prepare_encoded_b(0,buffer.b,buffer.encoded,k,n,ldb,prefix,is_unsigned,layout.total_bytes);
          if(status==RNS8_UNSUPPORTED_BACKEND) SKIP("compiled target lacks encoded B WMMA");
          REQUIRE(status==RNS8_SUCCESS);
          tasks.back().encoded_b=static_cast<const uint32_t*>(buffer.encoded);
          tasks.back().encoded_k_words=layout.k_words;
        }
      }
      for (uint32_t workers : {1u, 2u, 128u}) {
        for (uint32_t id = 0; id < tasks.size(); ++id) {
          std::fill(actual[id].begin(), actual[id].end(), 99);
          REQUIRE(hip_direct_copy_host_to_device(0, tasks[id].c, actual[id].data(), actual[id].size()) == RNS8_SUCCESS);
        }
        bool touched = true;
        CHECK(hip_direct_gemm_native_wmma_queue(0, tasks, workers, 0, touched) == RNS8_WORKSPACE_TOO_SMALL);
        CHECK_FALSE(touched);
        auto collision = tasks;
        collision[1].c = tasks[0].c;
        CHECK(hip_direct_gemm_native_wmma_queue(0, collision, workers, UINT64_MAX, touched) == RNS8_INVALID_ARGUMENT);
        CHECK_FALSE(touched);
        const auto status = hip_direct_gemm_native_wmma_queue(0, tasks, workers, UINT64_MAX, touched);
        for (uint32_t id = 0; id < tasks.size(); ++id) {
          REQUIRE(hip_direct_copy_device_to_host(0, actual[id].data(), tasks[id].c, actual[id].size()) == RNS8_SUCCESS);
          if (status == RNS8_UNSUPPORTED_BACKEND)
            REQUIRE(std::all_of(actual[id].begin(), actual[id].end(), [](int8_t x) { return x == 99; }));
          else REQUIRE(actual[id] == expected[id]);
        }
        if (status == RNS8_UNSUPPORTED_BACKEND) {
          REQUIRE_FALSE(touched);
          SKIP("compiled target lacks native RDNA3 WMMA");
        }
        REQUIRE(status == RNS8_SUCCESS);
        REQUIRE(touched);
        NativeWmmaPreparedQueue prepared;
        CHECK(prepared.execute(touched) == RNS8_INVALID_ARGUMENT);
        CHECK_FALSE(touched);
        CHECK(prepared.execute_ordinary(touched) == RNS8_INVALID_ARGUMENT);
        CHECK_FALSE(touched);
        uint64_t descriptor_bytes = tasks.size() * sizeof(NativeWmmaTask) + sizeof(uint32_t);
        for (const auto& task : tasks)
          descriptor_bytes += ((task.m + 15) / 16) * ((task.n + 15) / 16) *
              ((task.prefix + task.group - 1) / task.group) * sizeof(NativeWmmaWork);
        REQUIRE(prepared.prepare(0, tasks, workers, descriptor_bytes) == RNS8_SUCCESS);
        // Admission failures are transactional: the earlier descriptors survive.
        CHECK(prepared.prepare(0, tasks, workers, 0) == RNS8_WORKSPACE_TOO_SMALL);
        CHECK(prepared.prepare(0, tasks, workers, descriptor_bytes * 2 - 1) == RNS8_WORKSPACE_TOO_SMALL);
        CHECK(prepared.prepare(0, collision, workers, UINT64_MAX) == RNS8_INVALID_ARGUMENT);
        for (int repeat = 0; repeat < 4; ++repeat) {
          for (uint32_t id = 0; id < tasks.size(); ++id) {
            std::fill(actual[id].begin(), actual[id].end(), 99);
            REQUIRE(hip_direct_copy_host_to_device(0, tasks[id].c, actual[id].data(), actual[id].size()) == RNS8_SUCCESS);
          }
          REQUIRE((repeat % 2 ? prepared.execute_ordinary(touched) : prepared.execute(touched)) == RNS8_SUCCESS);
          REQUIRE(touched);
          for (uint32_t id = 0; id < tasks.size(); ++id) {
            REQUIRE(hip_direct_copy_device_to_host(0, actual[id].data(), tasks[id].c, actual[id].size()) == RNS8_SUCCESS);
            REQUIRE(actual[id] == expected[id]);
          }
        }
        REQUIRE(prepared.prepare(0, tasks, workers, descriptor_bytes * 2) == RNS8_SUCCESS);
        if (workers == 128) {
          // Contents are reread, not cached under the pointer. Mutate each A
          // after prior executions; unchanged layout needs no descriptor upload.
          for (uint32_t id = 0; id < tasks.size(); ++id) {
            const auto& task = tasks[id];
            REQUIRE(hip_direct_zero(0, const_cast<void*>(task.a), std::size_t(task.m * task.lda) * 8) == RNS8_SUCCESS);
            for (uint32_t p = 0; p < task.prefix; ++p)
              for (int64_t row = 0; row < task.m; ++row)
                for (int64_t col = 0; col < task.n; ++col)
                  expected[id][(p * task.m + row) * task.ldc + col] = 0;
          }
          REQUIRE(prepared.execute(touched) == RNS8_SUCCESS);
          REQUIRE(touched);
          for (uint32_t id = 0; id < tasks.size(); ++id) {
            REQUIRE(hip_direct_copy_device_to_host(0, actual[id].data(), tasks[id].c, actual[id].size()) == RNS8_SUCCESS);
            REQUIRE(actual[id] == expected[id]);
          }
          for (uint32_t id = 0; id < tasks.size(); ++id) {
            std::fill(actual[id].begin(), actual[id].end(), 99);
            REQUIRE(hip_direct_copy_host_to_device(0, tasks[id].c, actual[id].data(), actual[id].size()) == RNS8_SUCCESS);
          }
          REQUIRE(prepared.execute_ordinary(touched) == RNS8_SUCCESS);
          REQUIRE(touched);
          for (uint32_t id = 0; id < tasks.size(); ++id) {
            REQUIRE(hip_direct_copy_device_to_host(0, actual[id].data(), tasks[id].c, actual[id].size()) == RNS8_SUCCESS);
            REQUIRE(actual[id] == expected[id]);
          }
        }
      }
    }
  }
}
