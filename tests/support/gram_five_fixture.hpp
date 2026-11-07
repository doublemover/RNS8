#pragma once
#include "core/api_internal.hpp"
#include <array>
#include <cstring>
#include <vector>
namespace gram_five_fixture {
template <class T> T abi() {
  T value{};
  value.struct_size = sizeof(value);
  value.abi_version = RNS8_ABI_VERSION;
  return value;
}
inline void check_compact_prefixes(rns8_context* ctx, bool matrix) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (uint32_t prefix = 1; prefix <= 20; ++prefix)
    for (bool transpose : {false, true}) {
      constexpr int64_t n = 3, k = 5;
      const int64_t rows = transpose ? k : n, cols = transpose ? n : k;
      auto md = abi<rns8_matrix_desc>();
      md.rows = rows;
      md.cols = md.logical_ld = cols;
      md.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
      md.semantics = RNS8_EXACT_WIDE_SIGNED;
      md.bound_kind = RNS8_BOUND_NONE;
      md.max_prefix = prefix;
      rns8_matrix *x = nullptr, *c = nullptr;
      REQUIRE(rns8_create_matrix(ctx, &md, &x) == RNS8_SUCCESS);
      md.rows = md.cols = md.logical_ld = n;
      REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
      std::vector<int64_t> input(size_t(rows * cols));
      for (size_t index = 0; index < input.size(); ++index)
        input[index] = int64_t(index % 5) - 2;
      REQUIRE(rns8_pack_i64(ctx, x, input.data(), cols, 1) == RNS8_SUCCESS);
      const auto orientation =
          transpose ? RNS8_GRAM_TRANSPOSE_TIMES_SELF : RNS8_GRAM_SELF_TIMES_TRANSPOSE;
      rns8_prepared_gram* owner = nullptr;
      const auto status =
          rns8_prepare_gram_five_product(ctx, x, c, orientation, matrix, UINT64_MAX, &owner);
      uint32_t eligible = 0;
      for (uint32_t plane = 0; plane < prefix; ++plane)
        eligible += gram_five_prime(kDefaultModuli[plane]);
      if (!eligible) {
        CHECK(status == RNS8_UNSUPPORTED_BACKEND);
        CHECK(owner == nullptr);
      } else {
        REQUIRE(status == RNS8_SUCCESS);
        auto before_execution = abi<rns8_gram_classical_info>();
        REQUIRE(rns8_get_gram_classical_info(owner, &before_execution) == RNS8_SUCCESS);
        CHECK(before_execution.matrix_probe_observed == 0);
        CHECK(before_execution.matrix_probe_supported == 0);
        REQUIRE(rns8_execute_prepared_gram(owner) == RNS8_SUCCESS);
        std::vector<int8_t> result(size_t(prefix * n * n + 16), 89);
        if (ctx->backend == RNS8_BACKEND_HIP_DIRECT)
          REQUIRE(hip_direct_copy_device_to_host(ctx->device_id, result.data(), c->hip_residues,
                                                 size_t(prefix * n * n)) == RNS8_SUCCESS);
        else
          std::copy_n(c->residues.data(), size_t(prefix * n * n), result.data());
        for (int64_t row = 0; row < n; ++row)
          for (int64_t col = 0; col < n; ++col) {
            cpp_int expected = 0;
            for (int64_t inner = 0; inner < k; ++inner) {
              const auto at = [&](int64_t index) {
                return transpose ? inner * cols + index : index * cols + inner;
              };
              expected += cpp_int(input[size_t(at(row))]) * input[size_t(at(col))];
            }
            for (uint32_t plane = 0; plane < prefix; ++plane)
              CHECK(result[size_t((int64_t(plane) * n + row) * n + col)] ==
                    centered_residue(expected, kDefaultModuli[plane]));
          }
        for (size_t index = size_t(prefix * n * n); index < result.size(); ++index)
          CHECK(result[index] == 89);
        auto info = abi<rns8_prepared_gram_info>();
        REQUIRE(rns8_get_prepared_gram_info(owner, &info) == RNS8_SUCCESS);
        CHECK(info.algebraic_prime_planes == eligible);
        CHECK(info.classical_planes == prefix - eligible);
        auto classical = abi<rns8_gram_classical_info>();
        REQUIRE(rns8_get_gram_classical_info(owner, &classical) == RNS8_SUCCESS);
        const auto kind = matrix ? gram_classical_matrix_kind(ctx->device_info.gcn_arch) : 0;
        CHECK(classical.classical_planes == info.classical_planes);
        CHECK(classical.method == (kind == 2 ? RNS8_GRAM_MFMA_GFX942 : kind));
        CHECK(classical.matrix_probe_observed == uint32_t(kind != 0));
        CHECK(classical.matrix_probe_supported == uint32_t(kind != 0));
        for (unsigned malformed = 0; malformed < 3; ++malformed) {
          auto invalid = classical;
          if (malformed == 0)
            --invalid.struct_size;
          if (malformed == 1)
            ++invalid.struct_size;
          if (malformed == 2)
            ++invalid.abi_version;
          const auto unchanged = invalid;
          CHECK(rns8_get_gram_classical_info(owner, &invalid) == RNS8_INVALID_ARGUMENT);
          CHECK(std::memcmp(&invalid, &unchanged, sizeof(invalid)) == 0);
        }
        auto invalid_classical = classical;
        invalid_classical.reserved = 1;
        const auto unchanged_classical = invalid_classical;
        CHECK(rns8_get_gram_classical_info(owner, &invalid_classical) == RNS8_INVALID_ARGUMENT);
        CHECK(std::memcmp(&invalid_classical, &unchanged_classical, sizeof(invalid_classical)) ==
              0);
        alignas(rns8_gram_classical_info) uint64_t short_classical = sizeof(uint64_t);
        CHECK(rns8_get_gram_classical_info(owner, reinterpret_cast<rns8_gram_classical_info*>(
                                                      &short_classical)) == RNS8_INVALID_ARGUMENT);
        CHECK(info.algebraic_operand_bytes == 2 * uint64_t(eligible) * 2 * 4);
        CHECK(info.algebraic_product_bytes == 5 * uint64_t(eligible) * 2 * 2);
        CHECK(info.algebraic_staging_bytes == uint64_t(prefix) * n * n);
        CHECK(info.host_job_bytes >= sizeof(GramTile));
        REQUIRE(rns8_destroy_prepared_gram(owner) == RNS8_SUCCESS);
      }
      REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
    }
}
inline void check_finite_prime_owner(rns8_context* ctx, bool matrix) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (uint16_t modulus : {2, 3, 17, 251})
    for (bool transpose : {false, true})
      for (int64_t n : {1, 17}) {
        constexpr int64_t k = 33;
        const int64_t rows = transpose ? k : n, cols = transpose ? n : k, ld = cols + 2;
        auto md = abi<rns8_matrix_desc>();
        md.rows = rows;
        md.cols = cols;
        md.logical_ld = cols;
        md.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
        md.semantics = RNS8_FINITE_FIELD_U8;
        md.bound_kind = RNS8_BOUND_NONE;
        rns8_matrix *x = nullptr, *c = nullptr;
        REQUIRE(rns8_create_matrix(ctx, &md, &x) == RNS8_SUCCESS);
        md.rows = md.cols = md.logical_ld = n;
        REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
        std::vector<uint8_t> values(size_t(rows * ld), 255);
        for (int64_t row = 0; row < rows; ++row)
          for (int64_t col = 0; col < cols; ++col)
            values[size_t(row * ld + col)] = uint8_t((row * 17 + col * 13) % modulus);
        REQUIRE(rns8_pack_finite_u8(ctx, x, modulus, values.data(), ld, 0) == RNS8_SUCCESS);
        rns8_prepared_gram* owner = nullptr;
        const auto status = rns8_prepare_gram_five_product(
            ctx, x, c, transpose ? RNS8_GRAM_TRANSPOSE_TIMES_SELF : RNS8_GRAM_SELF_TIMES_TRANSPOSE,
            matrix, UINT64_MAX, &owner);
        if (matrix && modulus != 251) {
          CHECK(status == RNS8_UNSUPPORTED_BACKEND);
          CHECK(owner == nullptr);
          REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
          REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
          continue;
        }
        REQUIRE(status == RNS8_SUCCESS);
        REQUIRE(rns8_execute_prepared_gram(owner) == RNS8_SUCCESS);
        std::vector<int8_t> actual(size_t(n * n));
        if (ctx->backend == RNS8_BACKEND_HIP_DIRECT)
          REQUIRE(hip_direct_copy_device_to_host(ctx->device_id, actual.data(), c->hip_residues,
                                                 actual.size()) == RNS8_SUCCESS);
        else
          actual = c->residues;
        for (int64_t row = 0; row < n; ++row)
          for (int64_t col = 0; col < n; ++col) {
            cpp_int expected = 0;
            for (int64_t q = 0; q < k; ++q)
              expected += cpp_int(values[size_t(transpose ? q * ld + row : row * ld + q)]) *
                          values[size_t(transpose ? q * ld + col : col * ld + q)];
            CHECK(actual[size_t(row * n + col)] == centered_residue(expected, modulus));
          }
        auto info = abi<rns8_gram_classical_info>();
        REQUIRE(rns8_get_gram_classical_info(owner, &info) == RNS8_SUCCESS);
        CHECK(info.classical_planes == 0);
        CHECK(info.method == 0);
        CHECK(info.matrix_probe_observed == 0);
        CHECK(info.matrix_probe_supported == 0);
        REQUIRE(rns8_destroy_prepared_gram(owner) == RNS8_SUCCESS);
        REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
        REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
      }
}
inline void check(rns8_context* ctx, bool matrix) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  check_compact_prefixes(ctx, matrix);
  check_finite_prime_owner(ctx, matrix);
  for (bool transpose : {false, true})
    for (auto shape :
         std::array<std::array<int64_t, 2>, 4>{{{1, 1}, {3, 5}, {17, 33}, {1, 65537}}}) {
      const int64_t n = shape[0], k = shape[1], rows = transpose ? k : n, cols = transpose ? n : k,
                    ld = cols + 2;
      auto md = abi<rns8_matrix_desc>();
      md.rows = rows;
      md.cols = cols;
      md.logical_ld = cols;
      md.logical_layout = RNS8_LAYOUT_ROW_MAJOR;
      md.semantics = RNS8_EXACT_WIDE_SIGNED;
      md.bound_kind = RNS8_BOUND_NONE;
      md.max_prefix = 20;
      rns8_matrix *x = nullptr, *c = nullptr;
      REQUIRE(rns8_create_matrix(ctx, &md, &x) == RNS8_SUCCESS);
      md.rows = md.cols = md.logical_ld = n;
      REQUIRE(rns8_create_matrix(ctx, &md, &c) == RNS8_SUCCESS);
      std::vector<int64_t> source(size_t(rows * ld), 99);
      const int64_t values[]{INT64_MIN, INT64_MAX, -1, 0, 1, 17, -17};
      for (int64_t row = 0; row < rows; ++row)
        for (int64_t col = 0; col < cols; ++col)
          source[size_t(row * ld + col)] = values[(row + col) % 7];
      REQUIRE(rns8_pack_i64(ctx, x, source.data(), ld, 0) == RNS8_SUCCESS);
      const auto orientation =
          transpose ? RNS8_GRAM_TRANSPOSE_TIMES_SELF : RNS8_GRAM_SELF_TIMES_TRANSPOSE;
      rns8_prepared_gram* owner = nullptr;
      CHECK(rns8_prepare_gram_five_product(ctx, x, c, orientation, matrix, 0, &owner) ==
            RNS8_WORKSPACE_TOO_SMALL);
      REQUIRE(owner == nullptr);
      REQUIRE(rns8_prepare_gram_five_product(ctx, x, c, orientation, matrix, UINT64_MAX, &owner) ==
              RNS8_SUCCESS);
      CHECK(rns8_destroy_matrix(x) == RNS8_BUSY);
      for (int iteration = 0; iteration < 2; ++iteration) {
        if (iteration) {
          REQUIRE(rns8_pack_i64(ctx, x, source.data(), ld, 0) == RNS8_SUCCESS);
          CHECK(rns8_execute_prepared_gram(owner) == RNS8_INVALID_ARGUMENT);
          REQUIRE(rns8_refresh_prepared_gram(owner) == RNS8_SUCCESS);
        }
        REQUIRE(rns8_execute_prepared_gram(owner) == RNS8_SUCCESS);
        std::vector<int8_t> result(size_t(20 * n * n));
        if (ctx->backend == RNS8_BACKEND_HIP_DIRECT)
          REQUIRE(hip_direct_copy_device_to_host(ctx->device_id, result.data(), c->hip_residues,
                                                 result.size()) == RNS8_SUCCESS);
        else
          result = c->residues;
        for (int64_t row = 0; row < n; ++row)
          for (int64_t col = 0; col < n; ++col) {
            cpp_int expected = 0;
            for (int64_t q = 0; q < k; ++q) {
              const auto at = [&](int64_t i) { return transpose ? q * ld + i : i * ld + q; };
              expected += cpp_int(source[size_t(at(row))]) * source[size_t(at(col))];
            }
            for (uint32_t p = 0; p < 20; ++p)
              CHECK(result[size_t((int64_t(p) * n + row) * n + col)] ==
                    centered_residue(expected, kDefaultModuli[p]));
          }
        CHECK(c->exact_max_magnitude == cpp_int(k) * (cpp_int(1) << 126));
      }
      auto info = abi<rns8_prepared_gram_info>();
      REQUIRE(rns8_get_prepared_gram_info(owner, &info) == RNS8_SUCCESS);
      CHECK(info.method == 4 + uint32_t(matrix));
      CHECK(info.owned_payload_bytes > 0);
      CHECK(info.algebraic_product_count == 5);
      CHECK(info.algebraic_prime_planes > 0);
      CHECK(info.classical_planes > 0);
      CHECK(info.algebraic_prime_planes + info.classical_planes == 20);
      CHECK(info.owned_payload_bytes >= info.algebraic_operand_bytes +
                                            info.algebraic_product_bytes +
                                            info.algebraic_staging_bytes);
      REQUIRE(rns8_destroy_prepared_gram(owner) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(x) == RNS8_SUCCESS);
    }
}
} // namespace gram_five_fixture
