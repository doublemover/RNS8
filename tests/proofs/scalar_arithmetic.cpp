// This harness includes production headers without translating their operations.
// CBMC's exact 128-bit domains and ordinary Boost replay are independent oracles.
#include <cstdint>
#include <climits>
#if RNS8_BV_FAMILY == 1
#include "backend_common/checked_arithmetic.hpp"
#else
#include "backend_common/finite_u8_reducer.hpp"
#endif

static_assert(CHAR_BIT == 8 && sizeof(uint64_t) == 8 && sizeof(uint32_t) == 4,
              "the proof requires the recorded 8/32/64-bit integer model");
static_assert(INT64_MIN == -INT64_MAX - 1 && INT8_MIN == -128,
              "the proof models two's-complement signed representations");
#if RNS8_BV_LLP64
static_assert(sizeof(long) == 4 && sizeof(void*) == 8, "LLP64 proof model");
#else
static_assert(sizeof(long) == 8 && sizeof(void*) == 8, "LP64 proof model");
#endif

#if RNS8_BV_CBMC
using Wide = __CPROVER::unsignedbv<128>;
using SignedWide = __CPROVER::signedbv<128>;
extern uint64_t nondet_u64();
extern uint8_t nondet_u8();
#define BV_CHECK(condition, label) __CPROVER_assert(bool(condition), label)
#else
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Wide = boost::multiprecision::cpp_int;
using SignedWide = boost::multiprecision::cpp_int;
static std::vector<const char*> failed_properties;
#define BV_CHECK(condition, label)                                                          \
  do {                                                                                     \
    if (!bool(condition)) {                                                                \
      good = false;                                                                        \
      failed_properties.push_back(label);                                                  \
    }                                                                                      \
  } while (false)
#endif

bool check_case(uint64_t a, uint64_t b, uint64_t initial, uint64_t q_seed, uint64_t selector,
                uint64_t alias) {
  bool good = true;
#if RNS8_BV_CBMC
  __CPROVER_output("rns8.bv.inputs", a, b, initial, q_seed, selector, alias);
#endif
#if RNS8_BV_FAMILY == 1
  // Snapshots precede aliasing; caller storage cannot alter the specification.
  const Wide exact = selector ? Wide(a) * Wide(b) : Wide(a) + Wide(b);
  const bool fits = exact <= Wide(UINT64_MAX);
  uint64_t left = a, right = b, separate = initial;
  uint64_t& destination = alias == 1 ? left : alias == 2 ? right : separate;
  const uint64_t old = destination;
  const bool success = selector ? rns8::detail::checked::multiply(left, right, destination)
                                : rns8::detail::checked::add(left, right, destination);
  BV_CHECK(success == fits, "rns8.bv.checked.success_iff_fits");
  BV_CHECK(!success || Wide(destination) == exact, "rns8.bv.checked.exact_value");
  BV_CHECK(success || destination == old, "rns8.bv.checked.failure_preserves_output");
#else
  using namespace rns8::detail::finite_u8;
  // A total map covers every q=2..256 without a potentially vacuous assume.
  uint32_t q = q_seed == 255 ? 256 : uint32_t(q_seed) + 2;
#if RNS8_BV_FAMILY == 2
  const uint32_t value = uint32_t(a);
  if (selector >= 2)
    q = selector == 2 ? 251 : selector == 3 ? 255 : 256;
  const uint32_t reciprocal = modulus_reciprocal_u32(q);
  const uint64_t exact_reciprocal = (uint64_t(1) << 32) / q;
  BV_CHECK(reciprocal == exact_reciprocal, "rns8.bv.reducer.reciprocal_construction");
  const uint64_t quotient = (uint64_t(value) * reciprocal) >> 32;
  BV_CHECK(quotient * q <= value && uint64_t(value) - quotient * q < 2 * q,
           "rns8.bv.reducer.one_correction_precondition");
  uint32_t result = 0;
  switch (selector) {
  case 0:
    result = reduce_u32_small_modulus(value, q, reciprocal);
    break;
  case 1:
    result = reduce_u32(value, q, reciprocal);
    break;
  case 2:
    result = reduce_u32_fixed<251>(value);
    break;
  case 3:
    result = reduce_u32_fixed<255>(value);
    break;
  default:
    result = reduce_u32_fixed<256>(value);
  }
  BV_CHECK(result == uint64_t(value) % q, "rns8.bv.reducer.exact_modulo");
  BV_CHECK(result < q, "rns8.bv.reducer.canonical_interval");
#elif RNS8_BV_FAMILY == 3
  const auto reducer = native64_reducer(q);
  const uint32_t canonical = canonical_native_u64(a, reducer);
  BV_CHECK(canonical == a % q, "rns8.bv.native.unsigned_modulo");
  BV_CHECK(canonical < q, "rns8.bv.native.unsigned_interval");
  BV_CHECK(reducer.radix32 == (Wide(1) << 32) % q, "rns8.bv.native.radix");
  BV_CHECK(reducer.signed_bias == (Wide(1) << 64) % q, "rns8.bv.native.signed_bias");
  const int32_t centered_unsigned = center_native_u64(a, reducer);
  const int32_t expected_unsigned = int32_t(a % q) - int32_t(q) * (a % q >= (q + 1) / 2);
  BV_CHECK(centered_unsigned == expected_unsigned, "rns8.bv.native.unsigned_center");
  SignedWide signed_value = SignedWide(a);
  if (a >> 63)
    signed_value -= SignedWide(1) << 64;
  SignedWide signed_residue = signed_value % q;
  if (signed_residue < 0)
    signed_residue += q;
  const int32_t expected_signed =
      int32_t(signed_residue) - int32_t(q) * (signed_residue >= (q + 1) / 2);
  // This conversion is explicitly bound to the recorded two's-complement ABI.
  const int32_t centered_signed = center_native_i64(static_cast<int64_t>(a), reducer);
  BV_CHECK(centered_signed == expected_signed, "rns8.bv.native.signed_center");
  const int32_t low = -int32_t(q / 2), high = int32_t((q + 1) / 2) - 1;
  BV_CHECK(centered_unsigned >= low && centered_unsigned <= high && centered_signed >= low &&
               centered_signed <= high,
           "rns8.bv.native.centered_interval");
#else
#error Select a reviewed scalar proof family.
#endif
  (void)b;
  (void)initial;
  (void)alias;
  (void)selector;
#endif
  return good;
}

#if RNS8_BV_CBMC
int main() {
  // No assumptions, external function bodies or bounded input sampling: these
  // nondeterministic returns are the entire stated finite bitvector domains.
  return check_case(nondet_u64(), nondet_u64(), nondet_u64(), nondet_u8(),
                    nondet_u8() % (RNS8_BV_FAMILY == 1 ? 2 : 5), nondet_u8() % 3)
             ? 0
             : 1;
}
#else
int main(int argc, char** argv) {
  try {
    if (argc != 7)
      throw std::invalid_argument("six unsigned counterexample fields required");
    uint64_t fields[6]{};
    for (unsigned i = 0; i < 6; ++i) {
      const std::string text = argv[i + 1];
      if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("unsigned decimal counterexample field required");
      size_t consumed = 0;
      fields[i] = std::stoull(text, &consumed);
      if (consumed != text.size())
        throw std::invalid_argument("invalid counterexample field");
    }
    if (fields[3] > 255 || fields[4] >= (RNS8_BV_FAMILY == 1 ? 2 : 5) || fields[5] > 2)
      throw std::invalid_argument("counterexample is outside the proof domain");
    const bool passed =
        check_case(fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]);
    std::cout << "{\"exact_reference_passed\":" << (passed ? "true" : "false")
              << ",\"failed_properties\":[";
    for (size_t i = 0; i < failed_properties.size(); ++i)
      std::cout << (i ? "," : "") << '"' << failed_properties[i] << '"';
    std::cout << "]}\n";
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
#endif
