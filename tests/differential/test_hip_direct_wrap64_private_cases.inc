TEST_CASE("private HIP wrap64 byte-limb GEMM matches CPU reference") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP smoke");
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 3;
  constexpr int64_t k = 5;
  const std::vector<uint64_t> A = {
      0,
      1,
      std::numeric_limits<uint64_t>::max(),
      0x8080808080808080ull,
      0x0102030405060708ull,
      255,
      256,
      std::numeric_limits<uint64_t>::max() - 1,
      0x7f7f7f7f7f7f7f7full,
      17};
  const std::vector<uint64_t> B = {
      3,
      std::numeric_limits<uint64_t>::max(),
      0x1112131415161718ull,
      29,
      0x8080808080808080ull,
      31,
      0x0101010101010101ull,
      0xfefdfcfbfaf9f8f7ull,
      37,
      41,
      43,
      47,
      53,
      59,
      61};
  std::vector<uint8_t> a_limbs(static_cast<std::size_t>(m * k * 8));
  std::vector<uint8_t> b_limbs(static_cast<std::size_t>(k * n * 8));
  std::vector<uint8_t> c_limbs(static_cast<std::size_t>(m * n * 8), 0xff);
  for (int64_t cell = 0; cell < m * k; ++cell) {
    store_u64_limbs(a_limbs, cell, A[static_cast<std::size_t>(cell)]);
  }
  for (int64_t cell = 0; cell < k * n; ++cell) {
    store_u64_limbs(b_limbs, cell, B[static_cast<std::size_t>(cell)]);
  }

  run_wrap64_resident_device_gemm(a_limbs, b_limbs, c_limbs, m, n, k);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected = rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), k, B.data(), n, row, col, k);
      CHECK(load_u64_limbs(c_limbs, row * n + col) == expected);
    }
  }
}

TEST_CASE("private HIP wrap64 byte-limb GEMM matches CPU reference for carry-heavy tiled data") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP carry smoke");
  }

  constexpr int64_t m = 17;
  constexpr int64_t n = 17;
  constexpr int64_t k = 19;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
  fill_wrap64_carry_heavy_inputs(A, m, k, k, B, n, n);

  std::vector<uint8_t> a_limbs(static_cast<std::size_t>(m * k * 8));
  std::vector<uint8_t> b_limbs(static_cast<std::size_t>(k * n * 8));
  std::vector<uint8_t> c_limbs(static_cast<std::size_t>(m * n * 8), 0xff);
  for (int64_t cell = 0; cell < m * k; ++cell) {
    store_u64_limbs(a_limbs, cell, A[static_cast<std::size_t>(cell)]);
  }
  for (int64_t cell = 0; cell < k * n; ++cell) {
    store_u64_limbs(b_limbs, cell, B[static_cast<std::size_t>(cell)]);
  }

  run_wrap64_resident_device_gemm(a_limbs, b_limbs, c_limbs, m, n, k);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), k, B.data(), n, row, col, k);
      CHECK(load_u64_limbs(c_limbs, row * n + col) == expected);
    }
  }
}

TEST_CASE("private HIP wrap64 byte-limb GEMM covers tile tails and signed-int8 correction") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP tile-tail smoke");
  }

  constexpr int64_t m = 31;
  constexpr int64_t n = 30;
  constexpr int64_t k = 33;
  std::vector<uint64_t> A(static_cast<std::size_t>(m * k), 0);
  std::vector<uint64_t> B(static_cast<std::size_t>(k * n), 0);
  std::mt19937_64 rng(0x7772617036345f74ull);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      uint64_t value = rng();
      if ((row + col) % 7 == 0) {
        value = std::numeric_limits<uint64_t>::max();
      } else if ((row + col) % 7 == 1) {
        value = 0x8080808080808080ull;
      } else if ((row + col) % 7 == 2) {
        value = 0x7f807f807f807f80ull;
      }
      A[static_cast<std::size_t>(row * k + col)] = value;
    }
  }
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      uint64_t value = rng();
      if ((row * 3 + col) % 11 == 0) {
        value = std::numeric_limits<uint64_t>::max();
      } else if ((row * 3 + col) % 11 == 1) {
        value = 0xfefdfcfbfaf9f8f7ull;
      } else if ((row * 3 + col) % 11 == 2) {
        value = 0x0102030405060708ull;
      }
      B[static_cast<std::size_t>(row * n + col)] = value;
    }
  }

  std::vector<uint8_t> a_limbs(static_cast<std::size_t>(m * k * 8));
  std::vector<uint8_t> b_limbs(static_cast<std::size_t>(k * n * 8));
  std::vector<uint8_t> c_limbs(static_cast<std::size_t>(m * n * 8), 0xa5);
  for (int64_t cell = 0; cell < m * k; ++cell) {
    store_u64_limbs(a_limbs, cell, A[static_cast<std::size_t>(cell)]);
  }
  for (int64_t cell = 0; cell < k * n; ++cell) {
    store_u64_limbs(b_limbs, cell, B[static_cast<std::size_t>(cell)]);
  }

  run_wrap64_resident_device_gemm(a_limbs, b_limbs, c_limbs, m, n, k);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const uint64_t expected =
          rns8::detail::wrap64_low_diagonal_byte_pair_gemm_cell(A.data(), k, B.data(), n, row, col, k);
      CHECK(load_u64_limbs(c_limbs, row * n + col) == expected);
      CHECK(expected == rns8::detail::wrap64_byte_limb_gemm_cell(A.data(), k, B.data(), n, row, col, k));
    }
  }
}

TEST_CASE("private HIP wrap64 helpers reject invalid contracts before launch") {
  std::vector<uint8_t> limbs(8, 0);
  uint64_t src = 0;
  uint64_t dst = 0;
  void* buffer = nullptr;
  std::size_t bytes = 0;
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, nullptr, limbs.data(), limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), nullptr, limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
            0, limbs.data(), limbs.data(), limbs.data(), -1, 1, 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), limbs.data(), limbs.data(), 0, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), limbs.data(), limbs.data(), 1, 0, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), limbs.data(), limbs.data(), 1, 1, 0) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(0, limbs.data(), limbs.data(), nullptr, 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
            0,
            limbs.data(),
            limbs.data(),
            limbs.data(),
            static_cast<int64_t>(std::numeric_limits<int>::max()) + 1,
            1,
            1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_gemm_byte_limbs_device_resident(
            0,
            limbs.data(),
            limbs.data(),
            limbs.data(),
            1,
            1,
            static_cast<int64_t>(std::numeric_limits<int>::max()) + 1) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, nullptr, &buffer, &bytes, limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, nullptr, &bytes, limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, nullptr, limbs.data(), 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, &bytes, limbs.data(), 1, 2, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, &bytes, limbs.data(), -1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, &bytes, limbs.data(), 0, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(0, &src, &buffer, &bytes, nullptr, 1, 1, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_pack_u64_device(
            0,
            &src,
            &buffer,
            &bytes,
            limbs.data(),
            2,
            1,
            std::numeric_limits<int64_t>::max()) == RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), nullptr, &bytes, 1, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, nullptr, 1, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, &bytes, 1, 1, nullptr, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, &bytes, 1, 2, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, &bytes, -1, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, limbs.data(), &buffer, &bytes, 0, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(0, nullptr, &buffer, &bytes, 1, 1, &dst, 1) ==
        RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::wrap64_hip_export_u64_device(
            0,
            limbs.data(),
            &buffer,
            &bytes,
            2,
            1,
            &dst,
            std::numeric_limits<int64_t>::max()) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("private HIP wrap64 pack and export helpers preserve compact device byte-limb layout") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP pack/export layout smoke");
  }

  constexpr int64_t rows = 3;
  constexpr int64_t cols = 2;
  constexpr int64_t ld = 4;
  constexpr uint64_t sentinel = 0xccccccccccccccccull;
  std::vector<uint64_t> src(static_cast<std::size_t>(rows * ld), 0xaaaaaaaaaaaaaaaaull);
  std::vector<uint64_t> out(static_cast<std::size_t>(rows * ld), sentinel);
  src[0] = 0;
  src[1] = std::numeric_limits<uint64_t>::max();
  src[static_cast<std::size_t>(ld)] = 0x8080808080808080ull;
  src[static_cast<std::size_t>(ld + 1)] = 0x7f807f807f807f80ull;
  src[static_cast<std::size_t>(2 * ld)] = 0xfefdfcfbfaf9f8f7ull;
  src[static_cast<std::size_t>(2 * ld + 1)] = 0x0102030405060708ull;

  std::vector<uint8_t> expected_limbs(static_cast<std::size_t>(rows * cols * 8), 0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      store_u64_limbs(expected_limbs, row * cols + col, src[static_cast<std::size_t>(row * ld + col)]);
    }
  }

  void* device_limbs = nullptr;
  void* upload = nullptr;
  std::size_t upload_bytes = 0;
  void* export_buffer = nullptr;
  std::size_t export_bytes = 0;
  const std::size_t compact_bytes = expected_limbs.size();
  REQUIRE(rns8::detail::hip_direct_allocate(0, compact_bytes, &device_limbs) == RNS8_SUCCESS);

  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              0,
              src.data(),
              &upload,
              &upload_bytes,
              device_limbs,
              rows,
              cols,
              ld) == RNS8_SUCCESS);
  REQUIRE(upload != nullptr);
  REQUIRE(upload_bytes >= static_cast<std::size_t>(rows * ld * static_cast<int64_t>(sizeof(uint64_t))));
  std::vector<uint8_t> device_copy(compact_bytes, 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, device_copy.data(), device_limbs, compact_bytes) ==
          RNS8_SUCCESS);
  CHECK(device_copy == expected_limbs);

  void* first_upload = upload;
  const std::size_t first_upload_bytes = upload_bytes;
  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              0,
              src.data(),
              &upload,
              &upload_bytes,
              device_limbs,
              rows,
              cols,
              ld) == RNS8_SUCCESS);
  CHECK(upload == first_upload);
  CHECK(upload_bytes == first_upload_bytes);

  std::fill(src.begin(), src.end(), 0x1111111111111111ull);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              0,
              device_limbs,
              &export_buffer,
              &export_bytes,
              rows,
              cols,
              out.data(),
              ld) == RNS8_SUCCESS);
  REQUIRE(export_buffer != nullptr);
  REQUIRE(export_bytes >= static_cast<std::size_t>(rows * cols * static_cast<int64_t>(sizeof(uint64_t))));
  void* first_export = export_buffer;
  const std::size_t first_export_bytes = export_bytes;

  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] ==
            load_u64_limbs(expected_limbs, row * cols + col));
    }
    for (int64_t col = cols; col < ld; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] == sentinel);
    }
  }

  std::fill(out.begin(), out.end(), sentinel);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              0,
              device_limbs,
              &export_buffer,
              &export_bytes,
              rows,
              cols,
              out.data(),
              ld) == RNS8_SUCCESS);
  CHECK(export_buffer == first_export);
  CHECK(export_bytes == first_export_bytes);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] ==
            load_u64_limbs(expected_limbs, row * cols + col));
    }
    for (int64_t col = cols; col < ld; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] == sentinel);
    }
  }

  if (export_buffer) {
    CHECK(rns8::detail::hip_direct_free(0, export_buffer) == RNS8_SUCCESS);
  }
  if (upload) {
    CHECK(rns8::detail::hip_direct_free(0, upload) == RNS8_SUCCESS);
  }
  CHECK(rns8::detail::hip_direct_free(0, device_limbs) == RNS8_SUCCESS);
}

TEST_CASE("private HIP wrap64 pack and export helpers cover tile-tail padded layout reuse") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP padded tile-tail layout smoke");
  }

  constexpr int64_t rows = 17;
  constexpr int64_t cols = 19;
  constexpr int64_t ld = 23;
  constexpr uint64_t padding = 0xaaaaaaaaaaaaaaaaull;
  constexpr uint64_t sentinel = 0x6d6d6d6d6d6d6d6dull;
  std::vector<uint64_t> src(static_cast<std::size_t>(rows * ld), padding);
  std::vector<uint64_t> out(static_cast<std::size_t>(rows * ld), sentinel);
  std::vector<uint8_t> expected_limbs(static_cast<std::size_t>(rows * cols * 8), 0);

  auto fill_src = [&](uint64_t salt) {
    std::mt19937_64 rng(0x7772617036345f70ull ^ salt);
    std::fill(src.begin(), src.end(), padding ^ salt);
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t col = 0; col < cols; ++col) {
        uint64_t value = rng() ^ (salt + static_cast<uint64_t>(row * 257 + col * 19));
        if ((row + col) % 11 == 0) {
          value = std::numeric_limits<uint64_t>::max();
        } else if ((row + col) % 11 == 1) {
          value = 0;
        } else if ((row + col) % 11 == 2) {
          value = 0x8080808080808080ull;
        } else if ((row + col) % 11 == 3) {
          value = 0xfefdfcfbfaf9f8f7ull;
        }
        src[static_cast<std::size_t>(row * ld + col)] = value;
      }
    }
  };

  auto refresh_expected_limbs = [&]() {
    std::fill(expected_limbs.begin(), expected_limbs.end(), 0);
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t col = 0; col < cols; ++col) {
        store_u64_limbs(expected_limbs, row * cols + col, src[static_cast<std::size_t>(row * ld + col)]);
      }
    }
  };

  void* device_limbs = nullptr;
  void* upload = nullptr;
  std::size_t upload_bytes = 0;
  void* export_buffer = nullptr;
  std::size_t export_bytes = 0;
  const std::size_t compact_bytes = expected_limbs.size();
  REQUIRE(rns8::detail::hip_direct_allocate(0, compact_bytes, &device_limbs) == RNS8_SUCCESS);

  fill_src(0x1111);
  refresh_expected_limbs();
  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              0,
              src.data(),
              &upload,
              &upload_bytes,
              device_limbs,
              rows,
              cols,
              ld) == RNS8_SUCCESS);
  REQUIRE(upload != nullptr);
  REQUIRE(upload_bytes >= static_cast<std::size_t>(rows * ld * static_cast<int64_t>(sizeof(uint64_t))));
  std::vector<uint8_t> device_copy(compact_bytes, 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, device_copy.data(), device_limbs, compact_bytes) ==
          RNS8_SUCCESS);
  CHECK(device_copy == expected_limbs);
  void* first_upload = upload;
  const std::size_t first_upload_bytes = upload_bytes;

  fill_src(0x2222);
  refresh_expected_limbs();
  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              0,
              src.data(),
              &upload,
              &upload_bytes,
              device_limbs,
              rows,
              cols,
              ld) == RNS8_SUCCESS);
  CHECK(upload == first_upload);
  CHECK(upload_bytes == first_upload_bytes);
  std::fill(device_copy.begin(), device_copy.end(), 0);
  REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, device_copy.data(), device_limbs, compact_bytes) ==
          RNS8_SUCCESS);
  CHECK(device_copy == expected_limbs);

  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              0,
              device_limbs,
              &export_buffer,
              &export_bytes,
              rows,
              cols,
              out.data(),
              ld) == RNS8_SUCCESS);
  REQUIRE(export_buffer != nullptr);
  REQUIRE(export_bytes >= static_cast<std::size_t>(rows * cols * static_cast<int64_t>(sizeof(uint64_t))));
  void* first_export = export_buffer;
  const std::size_t first_export_bytes = export_bytes;

  auto check_export = [&]() {
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t col = 0; col < cols; ++col) {
        CHECK(out[static_cast<std::size_t>(row * ld + col)] ==
              load_u64_limbs(expected_limbs, row * cols + col));
      }
      for (int64_t col = cols; col < ld; ++col) {
        CHECK(out[static_cast<std::size_t>(row * ld + col)] == sentinel);
      }
    }
  };
  check_export();

  std::fill(out.begin(), out.end(), sentinel);
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              0,
              device_limbs,
              &export_buffer,
              &export_bytes,
              rows,
              cols,
              out.data(),
              ld) == RNS8_SUCCESS);
  CHECK(export_buffer == first_export);
  CHECK(export_bytes == first_export_bytes);
  check_export();

  if (export_buffer) {
    CHECK(rns8::detail::hip_direct_free(0, export_buffer) == RNS8_SUCCESS);
  }
  if (upload) {
    CHECK(rns8::detail::hip_direct_free(0, upload) == RNS8_SUCCESS);
  }
  CHECK(rns8::detail::hip_direct_free(0, device_limbs) == RNS8_SUCCESS);
}

TEST_CASE("private HIP wrap64 padded export uses pinned host staging timing path") {
  if (!hip_available()) {
    SKIP("no HIP device available for private wrap64 HIP pinned export staging smoke");
  }

  ScopedEnvVar force_staging("RNS8_HIP_PINNED_EXPORT_STAGING", "1");
  constexpr int64_t rows = 128;
  constexpr int64_t cols = 128;
  constexpr int64_t ld = 135;
  constexpr uint64_t sentinel = 0xfedcba9876543210ull;
  std::vector<uint64_t> src(static_cast<std::size_t>(rows * ld), 0);
  std::vector<uint64_t> out(static_cast<std::size_t>(rows * ld), sentinel);
  std::vector<uint8_t> expected_limbs(static_cast<std::size_t>(rows * cols * 8), 0);

  std::mt19937_64 rng(0x7772617036345f73ull);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      uint64_t value = rng() ^ (static_cast<uint64_t>(row) << 32u) ^ static_cast<uint64_t>(col * 37);
      if ((row + col) % 29 == 0) {
        value = std::numeric_limits<uint64_t>::max();
      } else if ((row + col) % 29 == 1) {
        value = 0;
      } else if ((row + col) % 29 == 2) {
        value = 0x8080808080808080ull;
      }
      src[static_cast<std::size_t>(row * ld + col)] = value;
      store_u64_limbs(expected_limbs, row * cols + col, value);
    }
  }

  void* device_limbs = nullptr;
  void* upload = nullptr;
  std::size_t upload_bytes = 0;
  void* export_buffer = nullptr;
  std::size_t export_bytes = 0;
  const std::size_t compact_bytes = expected_limbs.size();
  REQUIRE(rns8::detail::hip_direct_allocate(0, compact_bytes, &device_limbs) == RNS8_SUCCESS);
  REQUIRE(rns8::detail::wrap64_hip_pack_u64_device(
              0,
              src.data(),
              &upload,
              &upload_bytes,
              device_limbs,
              rows,
              cols,
              ld) == RNS8_SUCCESS);

  rns8::detail::hip_direct_timing_set_enabled(true);
  rns8::detail::hip_direct_timing_reset();
  REQUIRE(rns8::detail::wrap64_hip_export_u64_device(
              0,
              device_limbs,
              &export_buffer,
              &export_bytes,
              rows,
              cols,
              out.data(),
              ld) == RNS8_SUCCESS);
  auto export_events = rns8::detail::hip_direct_timing_snapshot();
  rns8::detail::hip_direct_timing_set_enabled(false);

  CHECK(has_timing_label(export_events, "wrap64_export_kernel"));
  CHECK(has_timing_label(export_events, "wrap64_export_d2h"));
  CHECK(has_timing_label(export_events, "export_host_staging_copy"));
  REQUIRE(export_buffer != nullptr);
  REQUIRE(export_bytes >= static_cast<std::size_t>(rows * cols * static_cast<int64_t>(sizeof(uint64_t))));

  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] == load_u64_limbs(expected_limbs, row * cols + col));
    }
    for (int64_t col = cols; col < ld; ++col) {
      CHECK(out[static_cast<std::size_t>(row * ld + col)] == sentinel);
    }
  }

  if (export_buffer) {
    CHECK(rns8::detail::hip_direct_free(0, export_buffer) == RNS8_SUCCESS);
  }
  if (upload) {
    CHECK(rns8::detail::hip_direct_free(0, upload) == RNS8_SUCCESS);
  }
  CHECK(rns8::detail::hip_direct_free(0, device_limbs) == RNS8_SUCCESS);
}


TEST_CASE("signed radix and native unsigned WMMA wrap64 agree with independent oracles") {
  if (!hip_available()) SKIP("no HIP device for wrap64 matrix candidates");
  using namespace rns8::detail;
  for(int k : {1,15,16,17,4095,4096,4097,33025,33026,65535,65536,65537,131071,131072}) {
    const int m=k<32?17:1,n=k<32?19:1;
    std::vector<uint64_t> av(m*k),bv(k*n);
    const uint64_t edge[]={0,1,127,128,255,256,UINT64_MAX,UINT64_C(1)<<63,
      UINT64_C(0x7f7f7f7f7f7f7f7f),UINT64_C(0x8080808080808080)};
    for(size_t i=0;i<av.size();++i) av[i]=edge[i%10];
    for(size_t i=0;i<bv.size();++i) bv[i]=edge[(i*7+3)%10];
    std::vector<uint8_t> ah(av.size()*8),bh(bv.size()*8),actual(m*n*8+16,0xa5);
    for(size_t i=0;i<av.size();++i) store_u64_limbs(ah,i,av[i]);
    for(size_t i=0;i<bv.size();++i) store_u64_limbs(bh,i,bv[i]);
    hip_direct_device_buffer a,b,c;
    REQUIRE(a.allocate(0,ah.size())==RNS8_SUCCESS);
    REQUIRE(b.allocate(0,bh.size())==RNS8_SUCCESS);
    REQUIRE(c.allocate(0,actual.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,a.get(),ah.data(),ah.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,b.get(),bh.data(),bh.size())==RNS8_SUCCESS);
    for(bool signed_digits:{false,true}) {
      std::fill(actual.begin(),actual.end(),0xa5);
      REQUIRE(hip_direct_copy_host_to_device(0,c.get(),actual.data(),actual.size())==RNS8_SUCCESS);
      bool started=true;
      CHECK(hip_direct_wrap_radix_wmma_device(0,a.get(),b.get(),c.get(),m,n,k,signed_digits,0,started)
            ==RNS8_WORKSPACE_TOO_SMALL);
      CHECK_FALSE(started);
      CHECK(hip_direct_wrap_radix_wmma_device(0,a.get(),b.get(),a.get(),m,n,k,signed_digits,4,started)
            ==RNS8_INVALID_ARGUMENT);
      CHECK_FALSE(started);
      const auto status=hip_direct_wrap_radix_wmma_device(0,a.get(),b.get(),c.get(),m,n,k,
                                                         signed_digits,4,started);
      if(status==RNS8_UNSUPPORTED_BACKEND) {
        CHECK_FALSE(started);
        SKIP("target lacks RDNA3 native IU8 WMMA");
      }
      REQUIRE(status==RNS8_SUCCESS); REQUIRE(started);
      REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),c.get(),actual.size())==RNS8_SUCCESS);
      for(int row=0;row<m;++row) for(int col=0;col<n;++col) {
        uint64_t native=0,bytes=0;
        boost::multiprecision::cpp_int wide=0;
        for(int q=0;q<k;++q) {
          native+=av[row*k+q]*bv[q*n+col];
          bytes+=wrap64_byte_limb_product(av[row*k+q],bv[q*n+col]);
          wide+=boost::multiprecision::cpp_int(av[row*k+q])*bv[q*n+col];
        }
        REQUIRE(load_u64_limbs(actual,row*n+col)==native);
        REQUIRE(native==bytes);
        REQUIRE(native==static_cast<uint64_t>(wide & ((boost::multiprecision::cpp_int(1)<<64)-1)));
      }
      for(size_t i=m*n*8;i<actual.size();++i) REQUIRE(actual[i]==0xa5);
    }
  }
}
