#include "../../benchmarks/rns8_bench_support.hpp"
#include <cmath>
#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/internal.hpp"
#include "backend_common/garner_constants.hpp"
#include "backend_common/residue_dot.hpp"
#include <random>

namespace {

std::vector<int8_t> residues_for(boost::multiprecision::cpp_int value, uint32_t prefix) {
  std::vector<int8_t> residues(prefix);
  for (uint32_t p = 0; p < prefix; ++p) {
    residues[p] = rns8::detail::centered_residue(value, rns8::detail::kDefaultModuli[p]);
  }
  return residues;
}

}  // namespace

TEST_CASE("CRT reconstruction recovers signed and unsigned bounded values") {
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(-123456789), 9);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, 9, 123456789, value) == RNS8_SUCCESS);
    CHECK(value == -123456789);
  }

  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::min()), 9);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, 9, 1ull << 63u, value) == RNS8_SUCCESS);
    CHECK(value == std::numeric_limits<int64_t>::min());
  }

  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()), 9);
    uint64_t value = 0;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_SUCCESS);
    CHECK(value == std::numeric_limits<uint64_t>::max());
  }
}

TEST_CASE("CRT canonical reconstruction matches exact representatives across prefixes") {
  for (uint32_t prefix : {1u, 2u, 4u, 8u, 9u, 12u, RNS8_MAX_SUPPORTED_PREFIX}) {
    const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(prefix);
    const std::vector<boost::multiprecision::cpp_int> values = {
        boost::multiprecision::cpp_int(0),
        boost::multiprecision::cpp_int(1),
        product / 2,
        product - 1,
    };
    for (const boost::multiprecision::cpp_int& value : values) {
      const auto residues = residues_for(value, prefix);
      CHECK(rns8::detail::reconstruct_canonical(residues, prefix) == value);
    }
  }
}

TEST_CASE("CRT signed reconstruction locks centered midpoint thresholds") {
  constexpr uint32_t prefix = 2;
  const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(prefix);
  const boost::multiprecision::cpp_int threshold = (product + 1) / 2;
  REQUIRE(product == 65280);
  REQUIRE(threshold == 32640);

  {
    const auto residues = residues_for(threshold - 1, prefix);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, prefix, 32639, value) == RNS8_SUCCESS);
    CHECK(value == 32639);
  }
  {
    const auto residues = residues_for(threshold + 1, prefix);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, prefix, 32639, value) == RNS8_SUCCESS);
    CHECK(value == -32639);
  }
  {
    const auto residues = residues_for(threshold, prefix);
    int64_t value = 123;
    CHECK(rns8::detail::reconstruct_signed(residues, prefix, 32640, value) == RNS8_RANGE_ERROR);
    CHECK(value == 123);
  }
  {
    const auto residues = residues_for(product - 1, prefix);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, prefix, 1, value) == RNS8_SUCCESS);
    CHECK(value == -1);
  }
}

TEST_CASE("CRT signed prefix-9 distinguishes int64 min from positive 2^63") {
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::min()), 9);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, 9, 1ull << 63u, value) == RNS8_SUCCESS);
    CHECK(value == std::numeric_limits<int64_t>::min());
  }
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(1) << 63u, 9);
    int64_t value = 77;
    CHECK(rns8::detail::reconstruct_signed(residues, 9, 1ull << 63u, value) == RNS8_RANGE_ERROR);
    CHECK(value == 77);
  }
}

TEST_CASE("CRT unsigned reconstruction is canonical and does not wrap past uint64") {
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()), 9);
    uint64_t value = 0;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_SUCCESS);
    CHECK(value == std::numeric_limits<uint64_t>::max());
  }
  {
    const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<uint64_t>::max()) + 1, 9);
    uint64_t value = 123;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_RANGE_ERROR);
    CHECK(value == 123);
  }
  {
    const boost::multiprecision::cpp_int product = rns8::detail::modulus_product(9);
    const auto residues = residues_for(product - 1, 9);
    uint64_t value = 456;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_RANGE_ERROR);
    CHECK(value == 456);
  }
}

TEST_CASE("CRT reconstruction reports insufficient range") {
  const auto residues = residues_for(boost::multiprecision::cpp_int(std::numeric_limits<int64_t>::max()), 8);
  int64_t value = 0;
  CHECK(rns8::detail::reconstruct_signed(
            residues, 8, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), value) == RNS8_RANGE_ERROR);
}

TEST_CASE("CRT reconstruction rejects malformed prefix inputs before range evaluation") {
  {
    const std::vector<int8_t> residues;
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(
              residues, 8, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), value) ==
          RNS8_INVALID_ARGUMENT);
  }
  {
    const std::vector<int8_t> residues(8, 0);
    uint64_t value = 0;
    CHECK(rns8::detail::reconstruct_unsigned(residues, 9, std::numeric_limits<uint64_t>::max(), value) ==
          RNS8_INVALID_ARGUMENT);
  }
  {
    const std::vector<int8_t> residues(1, 0);
    int64_t value = 0;
    CHECK(rns8::detail::reconstruct_signed(residues, 0, 1, value) == RNS8_INVALID_ARGUMENT);
    CHECK(rns8::detail::reconstruct_signed(residues, RNS8_MAX_SUPPORTED_PREFIX + 1, 1, value) ==
          RNS8_INVALID_ARGUMENT);
  }
}

TEST_CASE("immutable Garner constants and every prefix match independent congruences", "[crt][value-lifecycle]") {
  using namespace rns8::detail;
  std::mt19937_64 rng(20261023);
  cpp_int product=1;
  for (uint32_t prefix=1;prefix<=20;++prefix) {
    const auto modulus=kDefaultModuli[prefix-1];
    CHECK(garner::kConstants.moduli[prefix-1]==modulus);
    CHECK((product % modulus)*garner::kConstants.inverses[prefix-1] % modulus==1);
    product*=modulus;
    cpp_int encoded=0;
    for (int limb=2;limb>=0;--limb) { encoded<<=64; encoded+=garner::kConstants.products[prefix][limb]; }
    CHECK(encoded==product); CHECK(product < (cpp_int(1)<<160));
    for (const cpp_int value : std::vector<cpp_int>{0,1,product/2,product-1})
      CHECK(reconstruct_canonical(residues_for(value,prefix),prefix)==value);
    for (int trial=0;trial<100;++trial) {
      std::vector<int8_t> residues(prefix);
      for (uint32_t p=0;p<prefix;++p) residues[p]=centered_residue(cpp_int(rng()%kDefaultModuli[p]),kDefaultModuli[p]);
      const auto value=reconstruct_canonical(residues,prefix);
      REQUIRE(value>=0); REQUIRE(value<product);
      for (uint32_t p=0;p<prefix;++p)
        CHECK(value % kDefaultModuli[p]==canonical_from_centered(residues[p],kDefaultModuli[p]));
    }
  }
}

TEST_CASE("Garner pair constants prove both inverse levels", "[crt][value-lifecycle]") {
  using namespace rns8::detail;
  cpp_int product=1;
  for (uint32_t pair=0;pair<10;++pair) {
    const uint32_t lhs=kDefaultModuli[pair*2], rhs=kDefaultModuli[pair*2+1];
    CHECK(garner::kConstants.pair_moduli[pair]==lhs*rhs);
    CHECK((lhs*garner::kConstants.pair_inner_inverses[pair]) % rhs==1);
    CHECK((product % (lhs*rhs))*garner::kConstants.pair_prefix_inverses[pair] % (lhs*rhs)==1);
    product*=lhs*rhs;
  }
}

namespace {
struct RestoreCpuParallel {
  uint32_t requested = rns8::detail::cpu_parallel_requested_threads();
  uint32_t maximum = rns8::detail::cpu_parallel_max_threads();
  uint64_t threshold = rns8::detail::cpu_parallel_threshold_ops();
  bool progress = rns8::detail::cpu_parallel_progress_enabled();
  ~RestoreCpuParallel() {
    rns8::detail::configure_cpu_parallel(maximum, threshold, progress);
    rns8::detail::configure_cpu_parallel(requested, threshold, progress);
  }
};
}

TEST_CASE("Native CPU packing matches multiprecision at every ladder prefix") {
  using namespace rns8::detail;
  constexpr int64_t rows = 3, cols = 257, ld = cols + 7;
  std::vector<int64_t> signed_values(rows * ld, 42);
  std::vector<uint64_t> unsigned_values(rows * ld, 42);
  std::mt19937_64 random(260026);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      const uint64_t bits = random();
      unsigned_values[row * ld + col] = bits;
      signed_values[row * ld + col] = col % 2 ? -static_cast<int64_t>(bits >> 1) : static_cast<int64_t>(bits >> 1);
    }
  }
  signed_values[0] = std::numeric_limits<int64_t>::min();
  signed_values[1] = std::numeric_limits<int64_t>::max();
  signed_values[2] = -1;
  signed_values[3] = 0;
  unsigned_values[0] = std::numeric_limits<uint64_t>::max();
  unsigned_values[1] = uint64_t{1} << 63;
  unsigned_values[2] = 0;
  RestoreCpuParallel restore;
  for (uint32_t threads : {1u, 2u, 5u}) {
    configure_cpu_parallel(threads, 1, false);
    for (uint32_t prefix = 1; prefix <= RNS8_DEFAULT_MODULUS_COUNT; ++prefix) {
      rns8_matrix matrix{};
      matrix.desc.rows = rows;
      matrix.desc.cols = cols;
      matrix.prefix = prefix;
      matrix.residues.resize(prefix * rows * cols + 1, 77);
      pack_i64_matrix(matrix, signed_values.data(), ld);
      for (uint32_t p = 0; p < prefix; ++p)
        for (int64_t row = 0; row < rows; ++row)
          for (int64_t col = 0; col < cols; ++col)
            REQUIRE(matrix.residues[residue_index(matrix, p, row, col)] ==
                    centered_residue(cpp_int(signed_values[row * ld + col]), kDefaultModuli[p]));
      CHECK(matrix.residues.back() == 77);
      pack_u64_matrix(matrix, unsigned_values.data(), ld);
      for (uint32_t p = 0; p < prefix; ++p)
        for (int64_t row = 0; row < rows; ++row)
          for (int64_t col = 0; col < cols; ++col)
            REQUIRE(matrix.residues[residue_index(matrix, p, row, col)] ==
                    centered_residue(cpp_int(unsigned_values[row * ld + col]), kDefaultModuli[p]));
      CHECK(matrix.residues.back() == 77);
    }
  }
}


TEST_CASE("Finite native CPU packing exhausts every byte and modulus") {
  using namespace rns8::detail;
  std::vector<uint8_t> values(263, 199);
  for (unsigned value = 0; value < 256; ++value) values[value] = static_cast<uint8_t>(value);
  rns8_matrix matrix{};
  matrix.desc.rows = 1;
  matrix.desc.cols = 256;
  matrix.residues.resize(257, 77);
  for (uint16_t modulus = 2; modulus <= 256; ++modulus) {
    pack_finite_u8_matrix(matrix, values.data(), 263, modulus);
    for (unsigned value = 0; value < 256; ++value)
      REQUIRE(matrix.residues[value] == centered_residue(cpp_int(value), modulus));
    CHECK(matrix.residues.back() == 77);
  }
}

TEST_CASE("Address-domain qualification proves complete spans without allocations") {
  using namespace rns8::detail;
  AddressDomain domain;
  uint8_t storage[24]{};
  CHECK(nonoverlapping_byte_ranges(storage, 8, storage + 8, 16));
  CHECK_FALSE(nonoverlapping_byte_ranges(storage, 9, storage + 8, 16));
  CHECK_FALSE(nonoverlapping_byte_ranges(storage, UINT64_MAX, storage + 8, 16));
  CHECK_FALSE(nonoverlapping_byte_ranges(nullptr, 8, storage, 8));
  REQUIRE(checked_address_domain(32768, 65536, 65536, 1, 1, 0, domain));
  CHECK(domain.last_element == 2147483647ull);
  REQUIRE(checked_address_domain(32769, 65536, 65536, 1, 1, 0, domain));
  CHECK(domain.last_element == 2147549183ull);
  CHECK_FALSE(qualified_i32_matrix_address(32769, 65536, 65536));
  CHECK_FALSE(qualified_i32_matrix_address(32768, 65536, 65536)); // signed plane product overflows
  CHECK(qualified_i32_matrix_address(32767, 65536, 65536));
  CHECK_FALSE(qualified_i32_matrix_address(32767, 65536, 65536, 2));
  CHECK_FALSE(checked_address_domain(UINT64_MAX, 8, 8, 8, 1, 0, domain));
  CHECK_FALSE(checked_address_domain(1, 8, 8, 8, UINT64_MAX, 8, domain));
  CHECK_FALSE(checked_address_domain(1, UINT64_MAX, UINT64_MAX, 8, 1, 0, domain));
  CHECK_FALSE(checked_address_domain(2, 8, 8, 1, 2, 8, domain));
  for (uint64_t rows : {1ull, 3ull, 65536ull}) {
    for (uint64_t cols : {1ull, 17ull, 65536ull}) {
      for (uint64_t padding : {0ull, 7ull}) {
        for (uint64_t planes : {1ull, 20ull}) {
          const auto ld = cols + padding;
          REQUIRE(checked_address_domain(rows, cols, ld, 8, planes, 0, domain));
          const cpp_int last = cpp_int(planes - 1) * rows * ld + cpp_int(rows - 1) * ld + cols - 1;
          CHECK(cpp_int(domain.last_element) == last);
          CHECK(cpp_int(domain.byte_extent) == (last + 1) * 8);
        }
      }
    }
  }
}

TEST_CASE("Host duration conversion preserves sub-microsecond samples") {
  using rns8::bench::duration_us;
  for (int64_t ns : {0, 1, 500, 999, 1000, 1500, 1999}) {
    CHECK(std::abs(duration_us(std::chrono::nanoseconds(ns)) - static_cast<double>(ns) / 1000.0) < 1e-12);
  }
  double sum = 0;
  for (unsigned i = 0; i < 100; ++i) sum += duration_us(std::chrono::nanoseconds(500));
  CHECK(sum == 50.0);
}

TEST_CASE("one-correction reciprocal reduction covers quotient boundaries for every byte modulus") {
  using namespace rns8::detail::finite_u8;
  std::mt19937 random(6901);
  for (uint32_t modulus = 2; modulus <= 256; ++modulus) {
    const uint32_t reciprocal = modulus_reciprocal_u32(modulus);
    for (uint64_t quotient : {uint64_t{0}, uint64_t{1}, uint64_t{65535},
                              uint64_t{UINT32_MAX / modulus}}) {
      for (int offset : {-1, 0, 1, 127, 255}) {
        const int64_t wide = static_cast<int64_t>(quotient * modulus) + offset;
        if (wide >= 0 && wide <= UINT32_MAX)
          CHECK(reduce_u32_small_modulus(static_cast<uint32_t>(wide), modulus, reciprocal) ==
                static_cast<uint32_t>(wide) % modulus);
      }
    }
    CHECK(reduce_u32_small_modulus(UINT32_MAX, modulus, reciprocal) == UINT32_MAX % modulus);
    for (unsigned i = 0; i < 1024; ++i) {
      const uint32_t value = random();
      CHECK(reduce_u32_small_modulus(value, modulus, reciprocal) == value % modulus);
    }
  }
}

TEST_CASE("packed signed byte primitive and scalar control preserve every sign and tail") {
  using namespace rns8::detail;
  for (int a = -128; a <= 127; ++a) {
    for (int b = -128; b <= 127; ++b) {
      const uint32_t ap = uint32_t{static_cast<uint8_t>(a)} * 0x01010101u;
      const uint32_t bp = uint32_t{static_cast<uint8_t>(b)} * 0x01010101u;
      REQUIRE(signed_byte_dot4(ap, bp, -127) == 4 * a * b - 127);
    }
  }
  for (int64_t count : {1, 2, 3, 4, 5, 31, 32, 33, 65535, 65536, 65537}) {
    for (int64_t stride : {1, 3}) {
      std::vector<int8_t> a(count), b(count * stride, 111);
      int64_t expected = 0;
      for (int64_t i = 0; i < count; ++i) {
        a[i] = static_cast<int8_t>((i * 79) % 256 - 128);
        b[i * stride] = static_cast<int8_t>((i * 157 + 255) % 256 - 128);
        expected += int64_t{a[i]} * b[i * stride];
      }
      for (uint32_t modulus = 2; modulus <= 256; ++modulus) {
        const auto oracle = centered_residue(boost::multiprecision::cpp_int(expected), modulus);
        CHECK(centered_residue_dot(a.data(), b.data(), count, stride, modulus, false) == oracle);
        CHECK(centered_residue_dot(a.data(), b.data(), count, stride, modulus, true) == oracle);
      }
    }
  }
}

#include "backend_common/signed_pack_policy.hpp"
TEST_CASE("signed pack calibration shares unchanged public crossover selector") {
  using namespace rns8::detail;
  for (int64_t cells : {1, 255, 256, 257, 4095, 4096, 4097, 8192}) {
    CHECK(signed_pack_kernel(1, cells, cells, SignedPackVariant::fixed_prefix) == SignedPackKernel::fixed_prefix);
    CHECK(signed_pack_kernel(1, cells, cells + 7, SignedPackVariant::generic) == SignedPackKernel::generic);
    CHECK(signed_pack_kernel(1, cells, cells + 7, SignedPackVariant::public_selected) ==
          (cells <= 4096 ? SignedPackKernel::small : SignedPackKernel::fixed_prefix));
    CHECK(signed_pack_kernel(1, cells, cells, SignedPackVariant::public_selected) ==
          (cells < 256 ? SignedPackKernel::small : cells <= 4096 ? SignedPackKernel::four_wide : SignedPackKernel::eight_wide));
  }
}
