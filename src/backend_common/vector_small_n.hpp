#pragma once
#include <cstdint>
namespace rns8::detail {
inline constexpr uint32_t vector_small_n_width(uint32_t n,bool exact_width){return exact_width?n:8;}
// Declared shared arrays only; not an ISA/resource/occupancy receipt.
inline constexpr uint64_t vector_small_n_declared_lds(uint32_t n,bool sign,bool exact_width){return uint64_t(256)*vector_small_n_width(n,exact_width)*(24+(sign?1:0));}
}
