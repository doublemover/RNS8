#include "../../benchmarks/native_reference_cache.hpp"
#include <catch2/catch_test_macros.hpp>

namespace {
struct ReferenceDirectory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("rns8-reference-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  ReferenceDirectory() { if (!std::filesystem::create_directory(path)) throw std::runtime_error("test directory collision"); }
  ~ReferenceDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
}

TEST_CASE("native reference fixtures bind actual input words and check all padded output cells", "[native-reference-cache]") {
  using namespace rns8::bench;
  ReferenceDirectory directory;
  const std::vector<int64_t> a{2, -3, 4, 5}, b{7, 8, -9, 10};
  std::vector<int64_t> c{41, -14, 123, -17, 82, 123};
  const auto generated = verify_native_reference(2, 2, 2, 3, a, b, c, directory.path.string(), true);
  REQUIRE(generated.method == "boost_cpp_int_full_integer_reference");
  const auto reused = verify_native_reference(2, 2, 2, 3, a, b, c, directory.path.string(), false);
  REQUIRE(reused.method == "cached_full_integer_reference");
  REQUIRE(reused.checked_cells == 4);
  REQUIRE(reused.payload_sha256 == generated.payload_sha256);
  c[4] ^= 1;  // Interior logical element, rather than the final checksum/canary.
  REQUIRE_THROWS(verify_native_reference(2, 2, 2, 3, a, b, c, directory.path.string(), false));
  auto changed = a; changed[0]++;
  REQUIRE(native_reference_key(2, 2, 2, changed, b) != generated.key);
  const std::vector<int64_t> changed_c{48, -6, 123, -17, 82, 123};
  REQUIRE(verify_native_reference(2, 2, 2, 3, changed, b, changed_c, directory.path.string(), false).method ==
      "boost_cpp_int_full_integer_reference");
  const std::vector<uint64_t> ua{2, uint64_t(-3), 4, 5}, ub{7, 8, uint64_t(-9), 10};
  REQUIRE(native_reference_key(2, 2, 2, ua, ub) != generated.key);
  REQUIRE(native_reference_key(1, 4, 2, a, b) != generated.key);
}

TEST_CASE("native reference cache corruption and overflowing exact results fail closed", "[native-reference-cache]") {
  using namespace rns8::bench;
  ReferenceDirectory directory;
  const std::vector<uint64_t> a{UINT64_MAX, 0}, b{1, 1}, c{UINT64_MAX};
  auto generated = verify_native_reference(1, 1, 2, 1, a, b, c, directory.path.string(), true);
  const auto path = directory.path / (generated.key + ".bin");
  { std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out); file.seekp(144); file.put('\0'); }
  REQUIRE_THROWS(verify_native_reference(1, 1, 2, 1, a, b, c, directory.path.string(), false));
  std::filesystem::remove(path);
  const std::vector<uint64_t> overflow_b{2, 1};
  REQUIRE_THROWS_AS(verify_native_reference(1, 1, 2, 1, a, overflow_b, c, directory.path.string(), true), std::overflow_error);
  REQUIRE(std::filesystem::is_empty(directory.path));
  REQUIRE_THROWS(verify_native_reference(1, 1, 2, 0, a, b, c, directory.path.string(), false));
}

TEST_CASE("integer reference cache distinguishes wrap finite and signed wide semantics", "[native-reference-cache]") {
  using namespace rns8::bench;ReferenceDirectory directory;
  const std::vector<uint64_t> a{UINT64_MAX},b{2},wrap{UINT64_MAX-1};
  const auto wrapped=verify_integer_reference(1,1,1,1,a,b,wrap,{ReferenceSemantics::wrap64},directory.path.string(),true);
  CHECK(verify_integer_reference(1,1,1,1,a,b,wrap,{ReferenceSemantics::wrap64},directory.path.string(),false).method=="cached_full_integer_reference");
  REQUIRE_THROWS(verify_native_reference(1,1,1,1,a,b,wrap,directory.path.string(),false));
  const std::vector<uint8_t> fa{5,4},fb{3,2},fc{5};
  const auto ring=verify_integer_reference(1,1,2,1,fa,fb,fc,{ReferenceSemantics::finite_ring,6,1},directory.path.string(),true);
  CHECK(ring.key!=wrapped.key);
  CHECK(verify_integer_reference(1,1,2,1,fa,fb,fc,{ReferenceSemantics::finite_ring,6,1},directory.path.string(),false).checked_cells==1);
  REQUIRE_THROWS(verify_integer_reference(1,1,2,1,fa,fb,fc,{ReferenceSemantics::finite_field,6,1},directory.path.string(),false));
  const std::vector<int64_t> sa{INT64_MIN},sb{2};const std::vector<uint64_t> wide{0,UINT64_MAX};
  const auto generated=verify_integer_reference(1,1,1,1,sa,sb,wide,{ReferenceSemantics::wide_signed,0,2},directory.path.string(),true);
  CHECK(verify_integer_reference(1,1,1,1,sa,sb,wide,{ReferenceSemantics::wide_signed,0,2},directory.path.string(),false).method=="cached_full_integer_reference");
  CHECK(generated.key!=integer_reference_key(1,1,1,sa,sb,{ReferenceSemantics::wide_signed,0,3}));
  auto corrupted=wide;corrupted[1]^=1;
  REQUIRE_THROWS(verify_integer_reference(1,1,1,1,sa,sb,corrupted,{ReferenceSemantics::wide_signed,0,2},directory.path.string(),false));
  REQUIRE_THROWS(verify_integer_reference(1,1,1,1,sa,sb,wide,{ReferenceSemantics::wide_unsigned,0,2},directory.path.string(),false));
}

TEST_CASE("reference payload budgets fall back to full streaming comparison", "[native-reference-cache]") {
  using namespace rns8::bench;ReferenceDirectory directory;
  const std::vector<uint64_t> a{2,3},b{4},c{8,12};const ReferenceContract bounded{ReferenceSemantics::bounded_u64};
  const auto budget=reference_cache_header_bytes+8; // One word fits; this result has two.
  const auto result=verify_integer_reference(2,1,1,1,a,b,c,bounded,directory.path.string(),false,budget);
  CHECK(result.method=="boost_cpp_int_full_integer_reference");CHECK(result.checked_cells==2);CHECK(result.payload_sha256.empty());
  CHECK(std::filesystem::is_empty(directory.path));
  auto bad=c;bad[1]^=1;
  REQUIRE_THROWS(verify_integer_reference(2,1,1,1,a,b,bad,bounded,directory.path.string(),false,budget));
  REQUIRE_THROWS(verify_integer_reference(2,1,1,1,a,b,c,bounded,directory.path.string(),true,budget));
  CHECK(std::filesystem::is_empty(directory.path));
}
