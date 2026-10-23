#pragma once
#include <cstdint>
namespace rns8::detail::strassen {
struct Product {
  int8_t a[4], b[4], c[4];
};
struct Program {
  Product products[7];
};
constexpr Program program() {
  return {{{{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}},
           {{0, 0, 1, 1}, {1, 0, 0, 0}, {0, 0, 1, -1}},
           {{1, 0, 0, 0}, {0, 1, 0, -1}, {0, 1, 0, 1}},
           {{0, 0, 0, 1}, {-1, 0, 1, 0}, {1, 0, 1, 0}},
           {{1, 1, 0, 0}, {0, 0, 0, 1}, {-1, 1, 0, 0}},
           {{-1, 0, 1, 0}, {1, 1, 0, 0}, {0, 0, 0, 1}},
           {{0, 1, 0, -1}, {0, 0, 1, 1}, {1, 0, 0, 0}}}};
}
constexpr bool verified(Program data) {
  for (int c = 0; c < 4; ++c)
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) {
        int actual = 0;
        for (const auto &product : data.products)
          actual += product.c[c] * product.a[a] * product.b[b];
        const int expected =
            (a / 2 == c / 2 && b % 2 == c % 2 && a % 2 == b / 2) ? 1 : 0;
        if (actual != expected)
          return false;
      }
  return true;
}
static_assert(verified(program()), "Strassen polynomial identity");
} // namespace rns8::detail::strassen
