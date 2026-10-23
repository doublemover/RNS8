#ifndef RNS8_NATIVE_WMMA_QUEUE_HPP
#define RNS8_NATIVE_WMMA_QUEUE_HPP
#include <cstdint>
namespace rns8::detail {
struct NativeWmmaTask {
  const void* a = nullptr;
  const void* b = nullptr;
  int8_t* c = nullptr;
  int64_t m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldc = 0;
  uint32_t prefix = 0, group = 1, unsigned_inputs = 0;
};
struct NativeWmmaWork {
  uint32_t task, tile_row, tile_col, first_plane;
};
}
#endif
