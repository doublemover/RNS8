#pragma once
#include "rns8/rns8.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <cstddef>

namespace rns8::detail {
using boost::multiprecision::cpp_int;

enum class IntegerExportKind { bounded_signed, bounded_unsigned, wide_signed, wide_unsigned };
struct ReconstructionContract {
  uint32_t prefix = 0, limbs = 0;
  bool signed_value = false;
  const cpp_int* products = nullptr;
  const cpp_int* centering_threshold = nullptr;
  cpp_int minimum, maximum, ring_limit;
};

const cpp_int& reconstruction_product(uint32_t prefix);
rns8_status prepare_reconstruction_contract(IntegerExportKind, uint32_t prefix, uint64_t bound,
                                            uint32_t limbs, ReconstructionContract&);
// Source capacity/stride and the contract are admitted before parallel traversal.
// Each worker owns value; no tiny per-cell residue container is constructed.
rns8_status reconstruct_strided(const int8_t* first, std::size_t plane_stride,
                                const ReconstructionContract&, cpp_int& value);
void store_reconstructed_limbs(cpp_int& value, const ReconstructionContract&, uint64_t* out);
} // namespace rns8::detail
