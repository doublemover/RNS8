#include <rns8/integer_split.h>
#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(rns8_integer_split_format) == 4, "32-bit enum ABI required");
_Static_assert(sizeof(rns8_integer_split_i64_desc) == 56, "descriptor ABI");
_Static_assert(sizeof(rns8_integer_split_i64_info) == 72, "info ABI");
#define REQUIRE(condition) do { if (!(condition)) { \
  fprintf(stderr, "integer split C17 check failed at line %d\n", __LINE__); return 1; \
} } while (0)

int main(void) {
  rns8_integer_split_i64_desc d = {0};
  rns8_integer_split_i64_info info = {0};
  int64_t source[12] = {INT64_MIN, INT64_MAX, 9007199254740993LL, -9007199254740993LL,
      77, 77, -129, 128, 0, -1, 77, 77};
  int64_t output[12];
  int8_t components[160];
  d.struct_size = sizeof(d); d.abi_version = RNS8_ABI_VERSION;
  d.format = RNS8_INTEGER_SPLIT_SIGNED_RADIX128;
  d.rows = 2; d.cols = 4; d.native_ld = 6;
  d.component_ld = 5; d.component_plane_stride = 15;
  info.struct_size = sizeof(info); info.abi_version = RNS8_ABI_VERSION;
  REQUIRE(rns8_query_integer_split_i64(&d, &info) == RNS8_SUCCESS);
  REQUIRE(info.component_count == 10 && info.radix_bits == 7 && info.required_component_bytes == 144);
  for (int epoch = 0; epoch < 2; ++epoch) {
    memset(components, -128, sizeof(components));
    for (int i = 0; i < 12; ++i) output[i] = 88;
    REQUIRE(rns8_split_i64(&d, source, components, sizeof(components)) == RNS8_SUCCESS);
    REQUIRE(components[9 * 15] == -1 && components[9 * 15 + 1] == 0);
    for (int s = 0; s < 9; ++s) {
      REQUIRE(components[s * 15] == 0 && components[s * 15 + 1] == 127);
    }
    for (int i = 0; i < 160; ++i) {
      int logical = 0;
      for (int s = 0; s < 10; ++s) for (int row = 0; row < 2; ++row)
        logical |= i >= s * 15 + row * 5 && i < s * 15 + row * 5 + 4;
      if (!logical) REQUIRE(components[i] == -128);
    }
    REQUIRE(rns8_merge_i64(&d, components, sizeof(components), output) == RNS8_SUCCESS);
    for (int i = 0; i < 12; ++i) REQUIRE(output[i] == (i % 6 < 4 ? source[i] : 88));
    REQUIRE(rns8_split_i64(&d, source, components, 143) == RNS8_WORKSPACE_TOO_SMALL);
    source[8] = -127;
  }
  puts("integer split: C17 ABI, exact boundaries, strides and capacity passed");
  return 0;
}
