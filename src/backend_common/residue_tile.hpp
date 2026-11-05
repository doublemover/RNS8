#pragma once

#include "finite_u8_reducer.hpp"
#include "residue_dot.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define RNS8_RESIDUE_TILE_INLINE __host__ __device__ __forceinline__
#else
#define RNS8_RESIDUE_TILE_INLINE inline
#endif

namespace rns8::detail {
enum class ResidueTileDot { configured, packed, selected_default_moduli };

template <ResidueTileDot Policy> RNS8_RESIDUE_TILE_INLINE bool residue_tile_packed(int modulus) {
  if constexpr (Policy == ResidueTileDot::packed)
    return true;
  if constexpr (Policy == ResidueTileDot::selected_default_moduli)
    return kResidueDot4Enabled || modulus == 256 || modulus == 255 || modulus == 251;
  return kResidueDot4Enabled;
}

struct CenteredResidueTileInput {
  const int8_t* values;
  int stride;
  RNS8_RESIDUE_TILE_INLINE int8_t load(int row, int col) const {
    return values[row * stride + col];
  }
};

template <uint32_t FixedModulus = 0> struct NativeByteResidueTileInput {
  const uint8_t* values;
  int stride;
  uint32_t modulus;
  RNS8_RESIDUE_TILE_INLINE int8_t load(int row, int col) const {
    if constexpr (FixedModulus)
      return finite_u8::center_u8_fixed<FixedModulus>(values[row * stride + col]);
    else
      return finite_u8::center_u8(values[row * stride + col], modulus);
  }
};

struct ResidueTileLoadWindow {
  int rows, cols, tile_row, tile_col, k_offset;
};
struct ResidueTileMaskWindow {
  int rows, cols, tile_row, tile_col, row_offset, col_offset;
};

// This predicate is uniform across the CTA. A proved-zero row axis or column
// axis permits the existing whole-tile zero store before any shared barrier.
template <int TileM, int TileN>
RNS8_RESIDUE_TILE_INLINE bool residue_tile_zero_product(const uint8_t* zero_rows,
                                                        const uint8_t* zero_cols,
                                                        ResidueTileMaskWindow window) {
  bool rows_zero = true, cols_zero = true;
  for (int row = 0; row < TileM; ++row) {
    const int local = window.tile_row + row;
    if (local < window.rows && !zero_rows[window.row_offset + local]) {
      rows_zero = false;
      break;
    }
  }
  for (int col = 0; col < TileN; ++col) {
    const int local = window.tile_col + col;
    if (local < window.cols && !zero_cols[window.col_offset + local]) {
      cols_zero = false;
      break;
    }
  }
  return rows_zero || cols_zero;
}

// The 128-thread column-pair entrypoint deliberately keeps its original scalar
// two-output arithmetic, independently of the configured packed-dot policy.
RNS8_RESIDUE_TILE_INLINE void accumulate_residue_tile_pair(const int8_t* a, const int8_t* b,
                                                           int stride, int extent, int32_t& first,
                                                           int32_t& second) {
  for (int x = 0; x < extent; ++x) {
    const int32_t value = a[x];
    first += value * int32_t(b[x * stride]);
    second += value * int32_t(b[x * stride + 1]);
  }
}

// Load/tail semantics are shared by the real GPU body and CPU differential
// controls. The caller owns synchronization and the explicit shared arrays.
// Matrix/stride arithmetic and the <=65536 signed-byte slab are admitted by
// the existing dispatcher; this is not an unsigned-byte/wrap64 accumulator.
template <int TileM, int TileN, int TileK, int PaddedN, class InputA, class InputB>
RNS8_RESIDUE_TILE_INLINE void
load_residue_tile(int8_t (&a_tile)[TileM][TileK], int8_t (&b_tile)[TileK][PaddedN],
                  ResidueTileLoadWindow window, int tile_k, int tile_extent, int lane, int threads,
                  InputA a, InputB b) {
  static_assert(PaddedN >= TileN);
  for (int index = lane; index < TileM * TileK; index += threads) {
    const int row = index / TileK, x = index - row * TileK;
    const int global_row = window.tile_row + row;
    a_tile[row][x] = global_row < window.rows && x < tile_extent
                         ? a.load(global_row, window.k_offset + tile_k + x)
                         : 0;
  }
  for (int index = lane; index < TileK * TileN; index += threads) {
    const int x = index / TileN, col = index - x * TileN;
    const int global_col = window.tile_col + col;
    b_tile[x][col] = x < tile_extent && global_col < window.cols
                         ? b.load(window.k_offset + tile_k + x, global_col)
                         : 0;
  }
}
} // namespace rns8::detail

#undef RNS8_RESIDUE_TILE_INLINE
