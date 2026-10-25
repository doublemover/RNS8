#include <catch2/catch_test_macros.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <random>
#include <vector>

#include "core/internal.hpp"
#include "backend_common/wrap64_signed_radix.hpp"
#include "../support/currentness_test_helpers.hpp"
#include "rns8/rns8.h"

namespace {
#include "test_wrap64_support.inc"

}  // namespace

#include "test_wrap64_oracle_cases.inc"
#include "test_wrap64_public_cases.inc"
#include "test_wrap64_residency_reject_cases.inc"

#include "test_wrap64_prepared_hip_cases.inc"

#include "test_wrap64_resident_graph_cases.inc"

#include "backend_common/mfma_i8_layout.hpp"
TEST_CASE("CDNA signed radix fragment layouts cover each scalar coordinate once", "[wrap64][mfma][layout]") {
  using namespace rns8::detail;
  auto check=[](auto width){constexpr unsigned K=decltype(width)::value;std::vector<unsigned> a(16*K),b(K*16),c(256);
    for(unsigned lane=0;lane<64;++lane){for(unsigned byte=0;byte<K/4;++byte){const auto x=MfmaI8Layout<K>::a(lane,byte),y=MfmaI8Layout<K>::b(lane,byte);++a[x.row*K+x.col];++b[y.row*16+y.col];CHECK(x.col==(lane/16)*(K/4)+byte);CHECK(y.row==x.col);}
      for(unsigned reg=0;reg<4;++reg){const auto z=MfmaI8Layout<K>::c(lane,reg);++c[z.row*16+z.col];CHECK(z.row==(lane/16)*4+reg);}}
    for(auto count:a)CHECK(count==1);for(auto count:b)CHECK(count==1);for(auto count:c)CHECK(count==1);};
  check(std::integral_constant<unsigned,16>{});check(std::integral_constant<unsigned,32>{});check(std::integral_constant<unsigned,64>{});
}
