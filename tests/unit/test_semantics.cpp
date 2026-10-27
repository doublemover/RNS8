#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/internal.hpp"
#include "backend_ck/input_encoding.hpp"
#include "../support/currentness_test_helpers.hpp"
#include "rns8/rns8.h"

namespace {
#include "test_semantics_support.inc"

}  // namespace

#include "test_semantics_validation_cases.inc"
#include "test_semantics_contract_selection_cases.inc"
#include "test_semantics_workspace_cases.inc"
#include "test_semantics_currentness_cases.inc"

TEST_CASE("CK explicit input interpretation preserves every byte-ring product", "[ck][cpu][signedness]") {
  for (uint32_t modulus=2;modulus<=256;++modulus) {
    const int lo=-int(modulus/2), hi=int((modulus-1)/2);
    for (int a : {lo,-1,0,1,hi}) for (int b : {lo,-1,0,1,hi}) {
      if(a<lo || a>hi || b<lo || b>hi)continue;
      for(bool sign:{false,true}) {
        const auto x=rns8::detail::ck_encode_matrix_input(int8_t(a),modulus,sign);
        const auto y=rns8::detail::ck_encode_matrix_input(int8_t(b),modulus,sign);
        const int physical_x=sign?int(x):int(uint8_t(x));
        const int physical_y=sign?int(y):int(uint8_t(y));
        const auto canonical=[modulus](int value){int r=value%int(modulus);return r<0?r+int(modulus):r;};
        CHECK(canonical(physical_x*physical_y)==canonical(a*b));
      }
    }
  }
}
