#include "core/internal.hpp"
#include "backend_common/garner_constants.hpp"
#include "reconstruct/reconstruction.hpp"

#include <limits>

namespace rns8::detail {

namespace {

struct ReconstructionPlan {
  std::array<cpp_int, RNS8_DEFAULT_MODULUS_COUNT + 1> products, thresholds;
  ReconstructionPlan() {
    products[0] = 1;
    thresholds[0] = 1;
    for (uint32_t i = 0; i < RNS8_DEFAULT_MODULUS_COUNT; ++i) {
      if (i < RNS8_MAX_SUPPORTED_PREFIX &&
          (garner::kConstants.moduli[i] != kDefaultModuli[i] ||
           (products[i] % kDefaultModuli[i]) * garner::kConstants.inverses[i] % kDefaultModuli[i] !=
               1))
        throw std::logic_error("invalid Garner specification");
      products[i + 1] = products[i] * kDefaultModuli[i];
      thresholds[i + 1] = (products[i + 1] + 1) / 2;
    }
  }
};
const ReconstructionPlan& reconstruction_plan() {
  // C++ guarantees thread-safe initialization; no mutable per-cell cache.
  static const ReconstructionPlan plan;
  return plan;
}

} // namespace

const cpp_int& reconstruction_product(uint32_t prefix) {
  if (prefix > RNS8_DEFAULT_MODULUS_COUNT)
    throw std::out_of_range("invalid modulus prefix");
  return reconstruction_plan().products[prefix];
}

rns8_status prepare_reconstruction_contract(IntegerExportKind kind, uint32_t prefix, uint64_t bound,
                                            uint32_t limbs, ReconstructionContract& out) {
  if (!prefix || prefix > RNS8_MAX_SUPPORTED_PREFIX || !limbs || limbs > 32)
    return RNS8_INVALID_ARGUMENT;
  const bool bounded =
      kind == IntegerExportKind::bounded_signed || kind == IntegerExportKind::bounded_unsigned;
  const bool sign =
      kind == IntegerExportKind::bounded_signed || kind == IntegerExportKind::wide_signed;
  if (!bounded && kind != IntegerExportKind::wide_signed &&
      kind != IntegerExportKind::wide_unsigned)
    return RNS8_INVALID_ARGUMENT;
  if (bounded) {
    if (limbs != 1)
      return RNS8_INVALID_ARGUMENT;
    const auto status = validate_bound_contract(
        sign ? RNS8_BOUNDED_I64 : RNS8_BOUNDED_U64,
        sign ? RNS8_BOUND_GLOBAL_MAX_ABS : RNS8_BOUND_GLOBAL_MAX_UNSIGNED, bound, prefix);
    if (status != RNS8_SUCCESS)
      return status;
  }
  const auto& plan = reconstruction_plan();
  ReconstructionContract result;
  result.prefix = prefix;
  result.limbs = limbs;
  result.signed_value = sign;
  result.products = plan.products.data();
  result.centering_threshold = &plan.thresholds[prefix];
  result.ring_limit = cpp_int(1) << (64u * limbs);
  if (sign) {
    result.minimum = -(result.ring_limit >> 1);
    result.maximum = (result.ring_limit >> 1) - 1;
  } else {
    result.minimum = 0;
    result.maximum = result.ring_limit - 1;
  }
  if (bounded) {
    const cpp_int magnitude = bound;
    if (sign && result.minimum < -magnitude)
      result.minimum = -magnitude;
    if (result.maximum > magnitude)
      result.maximum = magnitude;
  }
  out = std::move(result);
  return RNS8_SUCCESS;
}

namespace {
void reconstruct_into(const int8_t* first, std::size_t stride, uint32_t prefix,
                      const cpp_int* products, cpp_int& x) {
  x = 0;
  for (uint32_t i = 0; i < prefix; ++i) {
    const uint32_t modulus = kDefaultModuli[i];
    const uint32_t target =
        canonical_from_centered(first[std::size_t(i) * stride], uint16_t(modulus));
    const uint32_t current =
        static_cast<uint32_t>(x % modulus); // Garner x is canonical/nonnegative
    const uint32_t delta = (target + modulus - current) % modulus;
    const uint32_t coefficient =
        static_cast<uint32_t>((uint64_t(delta) * garner::kConstants.inverses[i]) % modulus);
    if (coefficient)
      x += products[i] * coefficient;
  }
}
} // namespace

rns8_status reconstruct_strided(const int8_t* first, std::size_t stride,
                                const ReconstructionContract& contract, cpp_int& value) {
  reconstruct_into(first, stride, contract.prefix, contract.products, value);
  if (contract.signed_value && value >= *contract.centering_threshold)
    value -= contract.products[contract.prefix];
  return value < contract.minimum || value > contract.maximum ? RNS8_RANGE_ERROR : RNS8_SUCCESS;
}
void store_reconstructed_limbs(cpp_int& value, const ReconstructionContract& contract,
                               uint64_t* out) {
  if (value < 0)
    value += contract.ring_limit;
  for (uint32_t limb = 0; limb < contract.limbs; ++limb) {
    out[limb] = static_cast<uint64_t>(value & UINT64_MAX);
    value >>= 64;
  }
}

cpp_int reconstruct_canonical(const std::vector<int8_t>& residues, uint32_t prefix) {
  if (prefix > RNS8_MAX_SUPPORTED_PREFIX || residues.size() < prefix)
    throw std::invalid_argument("invalid residue prefix");
  cpp_int value;
  reconstruct_into(residues.data(), 1, prefix, reconstruction_plan().products.data(), value);
  return value;
}

rns8_status reconstruct_unsigned(const std::vector<int8_t>& residues, uint32_t prefix,
                                 uint64_t bound, uint64_t& out) {
  if (residues.size() < prefix)
    return RNS8_INVALID_ARGUMENT;
  ReconstructionContract contract;
  auto status = prepare_reconstruction_contract(IntegerExportKind::bounded_unsigned, prefix, bound,
                                                1, contract);
  if (status != RNS8_SUCCESS)
    return status;
  cpp_int value;
  status = reconstruct_strided(residues.data(), 1, contract, value);
  if (status == RNS8_SUCCESS)
    out = static_cast<uint64_t>(value);
  return status;
}
rns8_status reconstruct_signed(const std::vector<int8_t>& residues, uint32_t prefix, uint64_t bound,
                               int64_t& out) {
  if (residues.size() < prefix)
    return RNS8_INVALID_ARGUMENT;
  ReconstructionContract contract;
  auto status = prepare_reconstruction_contract(IntegerExportKind::bounded_signed, prefix, bound, 1,
                                                contract);
  if (status != RNS8_SUCCESS)
    return status;
  cpp_int value;
  status = reconstruct_strided(residues.data(), 1, contract, value);
  if (status == RNS8_SUCCESS)
    out = static_cast<int64_t>(value);
  return status;
}
namespace {
rns8_status export_wide(const std::vector<int8_t>& residues, uint32_t prefix, uint64_t* out,
                        uint32_t limbs, IntegerExportKind kind) {
  if (!out || residues.size() < prefix)
    return RNS8_INVALID_ARGUMENT;
  ReconstructionContract contract;
  auto status = prepare_reconstruction_contract(kind, prefix, 0, limbs, contract);
  if (status != RNS8_SUCCESS)
    return status;
  cpp_int value;
  status = reconstruct_strided(residues.data(), 1, contract, value);
  if (status == RNS8_SUCCESS)
    store_reconstructed_limbs(value, contract, out);
  return status;
}
} // namespace
rns8_status export_exact_wide_unsigned_limbs(const std::vector<int8_t>& residues, uint32_t prefix,
                                             uint64_t* out, uint32_t limbs) {
  return export_wide(residues, prefix, out, limbs, IntegerExportKind::wide_unsigned);
}
rns8_status export_exact_wide_signed_limbs(const std::vector<int8_t>& residues, uint32_t prefix,
                                           uint64_t* out, uint32_t limbs) {
  return export_wide(residues, prefix, out, limbs, IntegerExportKind::wide_signed);
}
} // namespace rns8::detail
