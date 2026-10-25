#include "../../benchmarks/exact_wide_magnitude.hpp"
#include <catch2/catch_test_macros.hpp>

namespace {
using boost::multiprecision::cpp_int;
using namespace rns8::bench;
const std::vector<uint16_t> basis{256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167};
template<class T> void magnitude_fixture_case(unsigned exponent, bool mixed) {
  constexpr uint64_t m=2,n=3,k=512,ldc=5;
  std::mt19937_64 rng(91), replay(91);
  std::vector<T> a(m*k), b(k*n), identical(m*k);
  fill_exact_wide_magnitude(a,rng,exponent,mixed);
  fill_exact_wide_magnitude(b,rng,exponent,mixed);
  fill_exact_wide_magnitude(identical,replay,exponent,mixed);
  REQUIRE(a == identical);
  ExactWideMagnitudeStats stats;
  admit_exact_wide_fixture(a,b,k,basis,3,stats);
  std::vector<uint64_t> c(m*ldc*3,UINT64_C(0x5a5a5a5a5a5a5a5a));
  for(uint64_t i=0;i<m;++i) for(uint64_t j=0;j<n;++j) {
    cpp_int v=0;
    for(uint64_t q=0;q<k;++q) v+=cpp_int(a[i*k+q])*b[q*n+j];
    if(v<0) v+=cpp_int(1)<<192;
    for(unsigned limb=0;limb<3;++limb) c[(i*ldc+j)*3+limb]=static_cast<uint64_t>((v>>(64*limb))&UINT64_MAX);
  }
  verify_exact_wide_fixture(a,b,c,m,n,k,ldc,3,stats);
  REQUIRE(stats.oracle_complete);
  REQUIRE(stats.checked_limbs == m*n*3);
  if(exponent && !mixed) REQUIRE(stats.output_min_magnitude_bits == 2*exponent+10);
  if(exponent==62) REQUIRE_THROWS(admit_exact_wide_fixture(a,b,k,basis,2,stats));
  REQUIRE_THROWS(admit_exact_wide_fixture(a,b,k,std::vector<uint16_t>{251},3,stats));
  c[2]^=1; // error only in a high limb is never hidden by checksum or low-word comparison
  REQUIRE_THROWS(verify_exact_wide_fixture(a,b,c,m,n,k,ldc,3,stats));
  c[2]^=1;
  c[n*3]=0;
  REQUIRE_THROWS(verify_exact_wide_fixture(a,b,c,m,n,k,ldc,3,stats));
}
}
TEST_CASE("public wide magnitude fixtures admit and check complete multi-limb products") {
  for(unsigned exponent:{0u,40u,62u}) for(bool mixed:{false,true}) {
    magnitude_fixture_case<int64_t>(exponent,mixed);
    magnitude_fixture_case<uint64_t>(exponent,mixed);
  }
}
TEST_CASE("wide magnitude admission handles native signed and unsigned extrema exactly") {
  ExactWideMagnitudeStats stats;
  std::vector<int64_t> a{INT64_MIN},b{INT64_MIN};
  admit_exact_wide_fixture(a,b,1,basis,2,stats);
  REQUIRE(stats.admitted_abs_bound == (cpp_int(1)<<126).str());
  std::vector<uint64_t> u{UINT64_MAX};
  admit_exact_wide_fixture(u,u,1,basis,2,stats);
  REQUIRE(stats.admitted_abs_bound == (cpp_int(UINT64_MAX)*UINT64_MAX).str());
}
