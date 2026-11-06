#include "../../benchmarks/native_reference_cache.hpp"
#include <catch2/catch_test_macros.hpp>

namespace {
struct ReferenceDirectory {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("rns8-reference-test-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  ReferenceDirectory() {
    if (!std::filesystem::create_directory(path))
      throw std::runtime_error("test directory collision");
  }
  ~ReferenceDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};
} // namespace

TEST_CASE("bounded chain references cover both products and retain logical output padding",
          "[native-reference-cache][vector-chain]") {
  using namespace rns8::bench;
  ReferenceDirectory directory;
  const ReferenceContract contract{ReferenceSemantics::bounded_i64};
  const std::vector<int64_t> a{1, -2, 3, 4}, b{5, 6, -7, 8}, d{2, 3, -4, 5};
  const std::vector<int64_t> c{78, 7, -991, -226, 211, -992};
  const auto first = verify_bounded_chain_reference(2, 2, 2, 3, a, b, d, c, contract,
                                                    directory.path.string(), true);
  CHECK(first.method == "boost_cpp_int_full_integer_reference");
  CHECK(first.checked_cells == 4);
  const auto hit = verify_bounded_chain_reference(2, 2, 2, 3, a, b, d, c, contract,
                                                  directory.path.string(), false);
  CHECK(hit.method == "cached_full_integer_reference");
  CHECK(hit.key == first.key);
  CHECK(hit.payload_sha256 == first.payload_sha256);
  CHECK(c[2] == -991);
  CHECK(c[5] == -992);
  auto bad = c;
  bad[4] ^= 1;
  REQUIRE_THROWS(verify_bounded_chain_reference(2, 2, 2, 3, a, b, d, bad, contract,
                                                directory.path.string(), false));
  // A cached final result must be invalidated by any of the three actual inputs.
  auto changed_a = a;
  changed_a[0] += 1;
  const auto a_result = verify_bounded_chain_reference(
      2, 2, 2, 3, changed_a, b, d, std::vector<int64_t>{64, 52, 0, -226, 211, 0}, contract,
      directory.path.string(), false);
  auto changed_b = b;
  changed_b[0] += 1;
  const auto b_result = verify_bounded_chain_reference(
      2, 2, 2, 3, a, changed_b, d, std::vector<int64_t>{80, 10, 0, -220, 220, 0}, contract,
      directory.path.string(), false);
  auto changed_d = d;
  changed_d[0] += 1;
  const auto d_result = verify_bounded_chain_reference(2, 2, 2, 3, a, b, changed_d,
                                                       std::vector<int64_t>{97, 7, 0, -239, 211, 0},
                                                       contract, directory.path.string(), false);
  CHECK(a_result.key != first.key);
  CHECK(b_result.key != first.key);
  CHECK(d_result.key != first.key);
  const auto streamed = verify_bounded_chain_reference(2, 2, 2, 3, a, b, d, c, contract, "", false,
                                                       reference_cache_header_bytes + 8);
  CHECK(streamed.method == "boost_cpp_int_full_integer_reference");
  CHECK(streamed.checked_cells == 4);
  CHECK(streamed.payload_sha256.empty());
}

TEST_CASE("bounded chain oracles reject intermediate overflow even when the final result cancels",
          "[native-reference-cache][vector-chain]") {
  using namespace rns8::bench;
  const ReferenceContract signed_contract{ReferenceSemantics::bounded_i64};
  const ReferenceContract unsigned_contract{ReferenceSemantics::bounded_u64};
  const std::vector<int64_t> maximum{INT64_MAX, INT64_MAX}, ones{1, 1}, zero{0};
  REQUIRE_THROWS(verify_bounded_chain_reference(1, 1, 2, 1, maximum, ones, zero, zero,
                                                signed_contract, "", false));
  const std::vector<int64_t> minimum{INT64_MIN}, one{1};
  CHECK(verify_bounded_chain_reference(1, 1, 1, 1, minimum, one, one, minimum, signed_contract, "",
                                       false)
            .checked_cells == 1);
  const std::vector<uint64_t> umaximum{UINT64_MAX}, uone{1}, utwo{2}, uzero{0};
  CHECK(verify_bounded_chain_reference(1, 1, 1, 1, umaximum, uone, uone, umaximum,
                                       unsigned_contract, "", false)
            .checked_cells == 1);
  REQUIRE_THROWS(verify_bounded_chain_reference(1, 1, 1, 1, umaximum, uone, utwo, uzero,
                                                unsigned_contract, "", false));
  REQUIRE_THROWS(verify_bounded_chain_reference(1, 1, 1, 1, minimum, one, one, minimum,
                                                unsigned_contract, "", false));
  REQUIRE_THROWS(verify_bounded_chain_reference(1, 1, 1, 1, minimum, one, one, minimum,
                                                {ReferenceSemantics::wrap64}, "", false));
}

TEST_CASE("unsigned bounded chain references support output padding and reject malformed shapes",
          "[native-reference-cache][vector-chain]") {
  using namespace rns8::bench;
  const ReferenceContract contract{ReferenceSemantics::bounded_u64};
  const std::vector<uint64_t> a{2}, b{3, 4}, d{5, 7, 11, 13}, c{118, 146, 991};
  CHECK(verify_bounded_chain_reference(1, 2, 1, 3, a, b, d, c, contract, "", false).checked_cells ==
        2);
  REQUIRE_THROWS(verify_bounded_chain_reference(1, 2, 1, 1, a, b, d, c, contract, "", false));
  REQUIRE_THROWS(verify_bounded_chain_reference(1, 2, 1, 3, a, b, a, c, contract, "", false));
  REQUIRE_THROWS(
      verify_bounded_chain_reference(UINT64_MAX, 2, 2, 3, a, b, d, c, contract, "", false));
  REQUIRE_THROWS(verify_bounded_chain_reference(1, 2, 1, 3, a, b, d, c, contract, "", true));
}

TEST_CASE("integer reference observations bind ordered task coverage before final policy status",
          "[native-reference-cache]") {
  using namespace rns8::bench;
  const NativeReferenceVerification first{"boost_cpp_int_full_integer_reference",
                                          std::string(64, 'a'), "", 6, 1.0};
  auto second = first;
  second.key = std::string(64, 'b');
  second.elapsed_us = 2;
  NativeReferenceVerification forward, reverse;
  append_integer_reference(forward, first);
  CHECK(complete_integer_reference(forward, 2, 3));
  CHECK_FALSE(complete_integer_reference(forward, 2, 3, 2));
  append_integer_reference(forward, second);
  append_integer_reference(reverse, second);
  append_integer_reference(reverse, first);
  CHECK(complete_integer_reference(forward, 2, 3, 2));
  CHECK(forward.checked_cells == 12);
  CHECK(forward.elapsed_us == 3);
  CHECK(forward.key != reverse.key);
  CHECK(valid_reference_sha256(forward.key));
  CHECK(forward.payload_sha256.empty());
  CHECK_FALSE(complete_integer_reference(forward, UINT64_MAX, 2));
  CHECK_FALSE(complete_integer_reference(forward, 2, 3, UINT64_MAX));
  auto missing = first;
  missing.key.clear();
  CHECK_FALSE(valid_integer_reference(missing));
  REQUIRE_THROWS(append_integer_reference(forward, missing));
  auto unbound = first;
  unbound.key[0] = 'g';
  CHECK_FALSE(valid_integer_reference(unbound));
  auto cached = first;
  cached.method = "cached_full_integer_reference";
  CHECK_FALSE(valid_integer_reference(cached));
  cached.payload_sha256 = std::string(64, 'c');
  CHECK(valid_integer_reference(cached));
  REQUIRE_THROWS(append_integer_reference(forward, cached));
  auto overflow = first;
  overflow.checked_cells = UINT64_MAX;
  NativeReferenceVerification full;
  append_integer_reference(full, overflow);
  REQUIRE_THROWS(append_integer_reference(full, first));
  CHECK(full.checked_cells == UINT64_MAX);
  auto invalid_time = first;
  invalid_time.elapsed_us = std::numeric_limits<double>::infinity();
  CHECK_FALSE(valid_integer_reference(invalid_time));
}

TEST_CASE("native reference fixtures bind actual input words and check all padded output cells",
          "[native-reference-cache]") {
  using namespace rns8::bench;
  ReferenceDirectory directory;
  const std::vector<int64_t> a{2, -3, 4, 5}, b{7, 8, -9, 10};
  std::vector<int64_t> c{41, -14, 123, -17, 82, 123};
  const auto generated =
      verify_native_reference(2, 2, 2, 3, a, b, c, directory.path.string(), true);
  REQUIRE(generated.method == "boost_cpp_int_full_integer_reference");
  const auto reused = verify_native_reference(2, 2, 2, 3, a, b, c, directory.path.string(), false);
  REQUIRE(reused.method == "cached_full_integer_reference");
  REQUIRE(reused.checked_cells == 4);
  REQUIRE(reused.payload_sha256 == generated.payload_sha256);
  c[4] ^= 1; // Interior logical element, rather than the final checksum/canary.
  REQUIRE_THROWS(verify_native_reference(2, 2, 2, 3, a, b, c, directory.path.string(), false));
  auto changed = a;
  changed[0]++;
  REQUIRE(native_reference_key(2, 2, 2, changed, b) != generated.key);
  const std::vector<int64_t> changed_c{48, -6, 123, -17, 82, 123};
  REQUIRE(verify_native_reference(2, 2, 2, 3, changed, b, changed_c, directory.path.string(), false)
              .method == "boost_cpp_int_full_integer_reference");
  const std::vector<uint64_t> ua{2, uint64_t(-3), 4, 5}, ub{7, 8, uint64_t(-9), 10};
  REQUIRE(native_reference_key(2, 2, 2, ua, ub) != generated.key);
  REQUIRE(native_reference_key(1, 4, 2, a, b) != generated.key);
}

TEST_CASE("native reference cache corruption and overflowing exact results fail closed",
          "[native-reference-cache]") {
  using namespace rns8::bench;
  ReferenceDirectory directory;
  const std::vector<uint64_t> a{UINT64_MAX, 0}, b{1, 1}, c{UINT64_MAX};
  auto generated = verify_native_reference(1, 1, 2, 1, a, b, c, directory.path.string(), true);
  const auto path = directory.path / (generated.key + ".bin");
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(144);
    file.put('\0');
  }
  REQUIRE_THROWS(verify_native_reference(1, 1, 2, 1, a, b, c, directory.path.string(), false));
  std::filesystem::remove(path);
  const std::vector<uint64_t> overflow_b{2, 1};
  REQUIRE_THROWS_AS(
      verify_native_reference(1, 1, 2, 1, a, overflow_b, c, directory.path.string(), true),
      std::overflow_error);
  REQUIRE(std::filesystem::is_empty(directory.path));
  REQUIRE_THROWS(verify_native_reference(1, 1, 2, 0, a, b, c, directory.path.string(), false));
}

TEST_CASE("integer reference cache distinguishes wrap finite and signed wide semantics",
          "[native-reference-cache]") {
  using namespace rns8::bench;
  ReferenceDirectory directory;
  const std::vector<uint64_t> a{UINT64_MAX}, b{2}, wrap{UINT64_MAX - 1};
  const auto wrapped = verify_integer_reference(
      1, 1, 1, 1, a, b, wrap, {ReferenceSemantics::wrap64}, directory.path.string(), true);
  CHECK(verify_integer_reference(1, 1, 1, 1, a, b, wrap, {ReferenceSemantics::wrap64},
                                 directory.path.string(), false)
            .method == "cached_full_integer_reference");
  REQUIRE_THROWS(verify_native_reference(1, 1, 1, 1, a, b, wrap, directory.path.string(), false));
  const std::vector<uint8_t> fa{5, 4}, fb{3, 2}, fc{5};
  const auto ring =
      verify_integer_reference(1, 1, 2, 1, fa, fb, fc, {ReferenceSemantics::finite_ring, 6, 1},
                               directory.path.string(), true);
  CHECK(ring.key != wrapped.key);
  CHECK(verify_integer_reference(1, 1, 2, 1, fa, fb, fc, {ReferenceSemantics::finite_ring, 6, 1},
                                 directory.path.string(), false)
            .checked_cells == 1);
  REQUIRE_THROWS(verify_integer_reference(1, 1, 2, 1, fa, fb, fc,
                                          {ReferenceSemantics::finite_field, 6, 1},
                                          directory.path.string(), false));
  const std::vector<int64_t> sa{INT64_MIN}, sb{2};
  const std::vector<uint64_t> wide{0, UINT64_MAX};
  const auto generated =
      verify_integer_reference(1, 1, 1, 1, sa, sb, wide, {ReferenceSemantics::wide_signed, 0, 2},
                               directory.path.string(), true);
  CHECK(verify_integer_reference(1, 1, 1, 1, sa, sb, wide, {ReferenceSemantics::wide_signed, 0, 2},
                                 directory.path.string(), false)
            .method == "cached_full_integer_reference");
  CHECK(generated.key !=
        integer_reference_key(1, 1, 1, sa, sb, {ReferenceSemantics::wide_signed, 0, 3}));
  auto corrupted = wide;
  corrupted[1] ^= 1;
  REQUIRE_THROWS(verify_integer_reference(1, 1, 1, 1, sa, sb, corrupted,
                                          {ReferenceSemantics::wide_signed, 0, 2},
                                          directory.path.string(), false));
  REQUIRE_THROWS(verify_integer_reference(1, 1, 1, 1, sa, sb, wide,
                                          {ReferenceSemantics::wide_unsigned, 0, 2},
                                          directory.path.string(), false));
}

TEST_CASE("reference payload budgets fall back to full streaming comparison",
          "[native-reference-cache]") {
  using namespace rns8::bench;
  ReferenceDirectory directory;
  const std::vector<uint64_t> a{2, 3}, b{4}, c{8, 12};
  const ReferenceContract bounded{ReferenceSemantics::bounded_u64};
  const auto budget = reference_cache_header_bytes + 8; // One word fits; this result has two.
  const auto result = verify_integer_reference(2, 1, 1, 1, a, b, c, bounded,
                                               directory.path.string(), false, budget);
  CHECK(result.method == "boost_cpp_int_full_integer_reference");
  CHECK(result.checked_cells == 2);
  CHECK(result.payload_sha256.empty());
  CHECK(std::filesystem::is_empty(directory.path));
  auto bad = c;
  bad[1] ^= 1;
  REQUIRE_THROWS(verify_integer_reference(2, 1, 1, 1, a, b, bad, bounded, directory.path.string(),
                                          false, budget));
  REQUIRE_THROWS(verify_integer_reference(2, 1, 1, 1, a, b, c, bounded, directory.path.string(),
                                          true, budget));
  CHECK(std::filesystem::is_empty(directory.path));
}
