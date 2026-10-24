#include "backend_common/word_mod_plan.hpp"
#include <array>
#include <utility>
namespace word_mod_wmma_fixture {
inline void compare(int m,int n,int k,uint64_t modulus,rns8::detail::word_mod::Domain domain) {
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  word_mod::Problem p;p.modulus=modulus;p.domain=domain;p.m=m;p.n=n;p.k=k;p.lda=k+2;p.ldb=n+3;p.ldc=n+5;
  std::vector<uint64_t> a(m*p.lda,UINT64_MAX),b(k*p.ldb,UINT64_MAX),c(m*p.ldc+8,77);
  for(int row=0;row<m;++row)for(int q=0;q<k;++q)
    a[row*p.lda+q]=(q%257==0)?uint64_t(row)%modulus:q==k/2?UINT64_C(0x8080808080808080)%modulus:modulus-1;
  for(int q=0;q<k;++q)for(int col=0;col<n;++col)
    b[q*p.ldb+col]=(q%257==1)?uint64_t(col)%modulus:q==k/2?UINT64_C(0xff00ff00ff00ff00)%modulus:modulus-1;
  hip_direct_device_buffer da,db,dc;
  REQUIRE(da.allocate(0,a.size()*8)==RNS8_SUCCESS);REQUIRE(db.allocate(0,b.size()*8)==RNS8_SUCCESS);REQUIRE(dc.allocate(0,c.size()*8)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size()*8)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),c.data(),c.size()*8)==RNS8_SUCCESS);
  bool started=true;
  REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,3,started,false,nullptr,true)==RNS8_WORKSPACE_TOO_SMALL);REQUIRE_FALSE(started);
  REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),da.get(),p,4,started,false,nullptr,true)==RNS8_INVALID_ARGUMENT);REQUIRE_FALSE(started);
  REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,true,nullptr,true)==RNS8_INVALID_ARGUMENT);REQUIRE_FALSE(started);
  const word_mod::FloatingRequest floating{2,3,17};
  REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,false,&floating,true)==RNS8_INVALID_ARGUMENT);REQUIRE_FALSE(started);
  const auto status=hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,false,nullptr,true);
  if(status==RNS8_UNSUPPORTED_BACKEND){REQUIRE_FALSE(started);SKIP("target lacks admitted unsigned byte-WMMA word lowering");}
  REQUIRE(status==RNS8_SUCCESS);REQUIRE(started);
  REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc.get(),c.size()*8)==RNS8_SUCCESS);
  for(int row=0;row<m;++row)for(int col=0;col<p.ldc;++col){
    if(col>=n){REQUIRE(c[row*p.ldc+col]==77);continue;}
    cpp_int exact=0;for(int q=0;q<k;++q)exact+=cpp_int(a[row*p.lda+q])*b[q*p.ldb+col];
    REQUIRE(c[row*p.ldc+col]==(exact%modulus).convert_to<uint64_t>());
  }
  for(size_t i=m*p.ldc;i<c.size();++i)REQUIRE(c[i]==77);
  std::vector<uint64_t> aa(a.size()),bb(b.size());
  REQUIRE(hip_direct_copy_device_to_host(0,aa.data(),da.get(),aa.size()*8)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_device_to_host(0,bb.data(),db.get(),bb.size()*8)==RNS8_SUCCESS);
  REQUIRE(aa==a);REQUIRE(bb==b);
  const auto before=c;a[0]=modulus;
  REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
  REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,false,nullptr,true)==RNS8_INVALID_ARGUMENT);REQUIRE_FALSE(started);
  REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc.get(),c.size()*8)==RNS8_SUCCESS);REQUIRE(c==before);
}
}

TEST_CASE("HIP word modular unsigned byte WMMA matches Boost at every operand width and tile tail") {
  if(!hip_available())SKIP("no HIP device for word modular byte WMMA");
  using namespace rns8::detail;
  for(unsigned bits:{8u,16u,23u,26u,27u,31u,35u,39u,42u,48u,50u,52u,64u}){
    const uint64_t ring=bits==64?UINT64_MAX:(UINT64_C(1)<<bits);
    uint64_t prime=ring-1;while(!word_mod::prime(prime))--prime;
    for(const auto modulus:{ring,prime})word_mod_wmma_fixture::compare(17,19,17,modulus,modulus==prime?word_mod::Domain::prime_field:word_mod::Domain::ring);
  }
  const std::array<std::pair<int,int>,6> shapes{{{1,1},{15,15},{16,16},{31,33},{32,32},{33,17}}};
  for(const auto shape:shapes)
    word_mod_wmma_fixture::compare(shape.first,shape.second,17,257,word_mod::Domain::prime_field);
  word_mod_wmma_fixture::compare(3,5,1,2,word_mod::Domain::ring);
}

TEST_CASE("HIP word modular byte WMMA drains complete diagonals at K4096 boundaries") {
  if(!hip_available())SKIP("no HIP device for word modular byte WMMA slabs");
  for(int k:{4095,4096,4097,8193})for(const auto modulus:{UINT64_C(251),UINT64_MAX})
    word_mod_wmma_fixture::compare(17,19,k,modulus,modulus==251?rns8::detail::word_mod::Domain::prime_field:rns8::detail::word_mod::Domain::ring);
}

TEST_CASE("HIP word modular baseline matches full Boost products and preserves padding") {
  if(!hip_available()) SKIP("no HIP device for word modular baseline");
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for(unsigned bits:{8u,16u,23u,26u,27u,31u,35u,39u,42u,48u,50u,52u,64u}) {
    const uint64_t top=bits==64?UINT64_MAX:(UINT64_C(1)<<bits);
    uint64_t prime=top-1;while(!word_mod::prime(prime)) --prime;
    for(uint64_t modulus:{top,prime}) for(int k:{1,17,257}) for(bool dot4:{false,true}) {
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
      REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,3,started,dot4)==RNS8_WORKSPACE_TOO_SMALL);REQUIRE_FALSE(started);
      REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),da.get(),p,4,started,dot4)==RNS8_INVALID_ARGUMENT);REQUIRE_FALSE(started);
      const auto status=hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,dot4);
      if(dot4 && status==RNS8_UNSUPPORTED_BACKEND) {REQUIRE_FALSE(started);continue;}
      REQUIRE(status==RNS8_SUCCESS);REQUIRE(started);
      REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc.get(),c.size()*8)==RNS8_SUCCESS);
      for(int i=0;i<p.m;++i) for(int j=0;j<p.ldc;++j) {
        if(j>=p.n) {REQUIRE(c[i*p.ldc+j]==77);continue;}
        cpp_int sum=0;for(int h=0;h<k;++h) sum+=cpp_int(a[i*p.lda+h])*b[h*p.ldb+j];
        REQUIRE(c[i*p.ldc+j]==(sum%modulus).convert_to<uint64_t>());
      }
      const auto before=c;a[0]=modulus;
      REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
      REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,dot4)==RNS8_INVALID_ARGUMENT);REQUIRE_FALSE(started);
      REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc.get(),c.size()*8)==RNS8_SUCCESS);REQUIRE(c==before);
    }
  }
}

TEST_CASE("word modular packed-dot diagonals reduce across K slab boundaries") {
  if(!hip_available()) SKIP("no HIP device for word modular dot4");
  using namespace rns8::detail;
  for(int k:{4095,4096,4097,8193}) {
    word_mod::Problem p;p.modulus=UINT64_MAX;p.m=p.n=1;p.k=k;p.lda=k;p.ldb=p.ldc=1;
    std::vector<uint64_t> a(k,UINT64_MAX-1),b(k,UINT64_MAX-1);
    a[k/2]=UINT64_C(0x8080808080808080);b[k/2]=UINT64_C(0xff00ff00ff00ff00);
    hip_direct_device_buffer da,db,dc;
    REQUIRE(da.allocate(0,k*8)==RNS8_SUCCESS);REQUIRE(db.allocate(0,k*8)==RNS8_SUCCESS);REQUIRE(dc.allocate(0,16)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),k*8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),k*8)==RNS8_SUCCESS);
    std::array<uint64_t,2> out{77,77};REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),out.data(),16)==RNS8_SUCCESS);
    bool started=false;const auto status=hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,true);
    if(status==RNS8_UNSUPPORTED_BACKEND) SKIP("target lacks admitted packed-dot word lowering");
    REQUIRE(status==RNS8_SUCCESS);REQUIRE(started);
    REQUIRE(hip_direct_copy_device_to_host(0,out.data(),dc.get(),16)==RNS8_SUCCESS);
    boost::multiprecision::cpp_int sum=0;for(int q=0;q<k;++q) sum+=boost::multiprecision::cpp_int(a[q])*b[q];
    REQUIRE(out[0]==(sum%p.modulus).convert_to<uint64_t>());REQUIRE(out[1]==77);
  }
}

TEST_CASE("HIP asymmetric binary64 word-mod schedules match exact slab and tail references") {
  if(!hip_available()) SKIP("no HIP device for asymmetric word modular schedule");
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  for(uint64_t modulus:{UINT64_C(251),UINT64_C(65536),UINT64_C(2147483647),UINT64_C(4503599627370496)})
    for(const auto request:{word_mod::FloatingRequest{1,1,17},word_mod::FloatingRequest{1,2,17},
        word_mod::FloatingRequest{2,1,17},word_mod::FloatingRequest{2,3,17},word_mod::FloatingRequest{3,2,17}})
      for(int k:{1,16,17,18,257}) {
        word_mod::Problem p;p.modulus=modulus;p.domain=modulus==251?word_mod::Domain::prime_field:word_mod::Domain::ring;
        p.m=2;p.n=3;p.k=k;p.lda=k+2;p.ldb=5;p.ldc=5;
        std::vector<uint64_t> a(p.m*p.lda,UINT64_MAX),b(k*p.ldb,UINT64_MAX),c(p.m*p.ldc+8,77);
        for(int i=0;i<p.m;++i) for(int q=0;q<k;++q) a[i*p.lda+q]=q%3?modulus-1:uint64_t(i+q)%modulus;
        for(int q=0;q<k;++q) for(int j=0;j<p.n;++j) b[q*p.ldb+j]=(q+j)%2?modulus-1:uint64_t(q)%modulus;
        hip_direct_device_buffer da,db,dc;
        REQUIRE(da.allocate(0,a.size()*8)==RNS8_SUCCESS);REQUIRE(db.allocate(0,b.size()*8)==RNS8_SUCCESS);
        REQUIRE(dc.allocate(0,c.size()*8)==RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size()*8)==RNS8_SUCCESS);
        REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),c.data(),c.size()*8)==RNS8_SUCCESS);
        bool started=true;word_mod::FloatingSchedule schedule;
        const auto expected_admission=word_mod::floating_schedule(modulus,request,schedule);
        const auto status=hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,false,&request);
        REQUIRE(status==expected_admission);
        REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc.get(),c.size()*8)==RNS8_SUCCESS);
        if(status!=RNS8_SUCCESS) {
          REQUIRE_FALSE(started);for(auto value:c) REQUIRE(value==77);continue;
        }
        REQUIRE(started);
        for(int i=0;i<p.m;++i) for(int j=0;j<p.ldc;++j) {
          if(j>=p.n) {REQUIRE(c[i*p.ldc+j]==77);continue;}
          cpp_int exact=0;for(int q=0;q<k;++q) exact+=cpp_int(a[i*p.lda+q])*b[q*p.ldb+j];
          REQUIRE(c[i*p.ldc+j]==(exact%modulus).convert_to<uint64_t>());
        }
        for(size_t i=p.m*p.ldc;i<c.size();++i) REQUIRE(c[i]==77);
        const auto before=c;a[0]=modulus;
        REQUIRE(hip_direct_copy_host_to_device(0,da.get(),a.data(),a.size()*8)==RNS8_SUCCESS);
        REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,false,&request)==RNS8_INVALID_ARGUMENT);
        REQUIRE_FALSE(started);
        REQUIRE(hip_direct_copy_device_to_host(0,c.data(),dc.get(),c.size()*8)==RNS8_SUCCESS);REQUIRE(c==before);
        REQUIRE(hip_direct_word_mod_device(0,da.get(),db.get(),dc.get(),p,4,started,true,&request)==RNS8_INVALID_ARGUMENT);
        REQUIRE_FALSE(started);
      }
}
