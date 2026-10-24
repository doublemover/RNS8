#include "backend_common/word_mod_plan.hpp"
TEST_CASE("HIP word modular baseline matches full Boost products and preserves padding") {
  if(!hip_available()) SKIP("no HIP device for word modular baseline");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for(unsigned bits:{8u,16u,23u,26u,27u,31u,35u,39u,42u,48u,50u,52u,64u}) {
    const uint64_t top=bits==64?UINT64_MAX:(UINT64_C(1)<<bits);
    uint64_t prime=top-1;while(!word_mod::prime(prime)) --prime;
    for(uint64_t modulus:{top,prime}) for(int k:{1,17,257}) {
      word_mod::Problem p;p.modulus=modulus;p.domain=modulus==prime?word_mod::Domain::prime_field:word_mod::Domain::ring;
      p.m=3;p.n=5;p.k=k;p.lda=k+2;p.ldb=7;p.ldc=8;
      std::vector<uint64_t> a(p.m*p.lda,UINT64_MAX),b(p.k*p.ldb,UINT64_MAX),c(p.m*p.ldc,77);
      for(int i=0;i<p.m;++i) for(int h=0;h<k;++h) a[i*p.lda+h]=(i+h)%3?modulus-1:uint64_t(h)%modulus;
      for(int h=0;h<k;++h) for(int j=0;j<p.n;++j) b[h*p.ldb+j]=(h+j)%2?modulus-1:uint64_t(j)%modulus;
      hip_direct_device_buffer da,db,dc;
      REQUIRE(da.allocate(0,a.size()*8)==RNS8_SUCCESS);REQUIRE(db.allocate(0,b.size()*8)==RNS8_SUCCESS);REQUIRE(dc.allocate(0,c.size()*8)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size()*8)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),c.data(),c.size()*8)==RNS8_SUCCESS);
      bool started=true;
      REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,3,started)==RNS8_WORKSPACE_TOO_SMALL);REQUIRE_FALSE(started);
      REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),da.get(),p,4,started)==RNS8_INVALID_ARGUMENT);REQUIRE_FALSE(started);
      REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started)==RNS8_SUCCESS);REQUIRE(started);
      REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc.get(),c.size()*8)==RNS8_SUCCESS);
      for(int i=0;i<p.m;++i) for(int j=0;j<p.ldc;++j) {
        if(j>=p.n) {REQUIRE(c[i*p.ldc+j]==77);continue;}
        cpp_int sum=0;for(int h=0;h<k;++h) sum+=cpp_int(a[i*p.lda+h])*b[h*p.ldb+j];
        REQUIRE(c[i*p.ldc+j]==(sum%modulus).convert_to<uint64_t>());
      }
      const auto before=c;a[0]=modulus;
      REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
      REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started)==RNS8_INVALID_ARGUMENT);REQUIRE_FALSE(started);
      REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc.get(),c.size()*8)==RNS8_SUCCESS);REQUIRE(c==before);
    }
  }
}
