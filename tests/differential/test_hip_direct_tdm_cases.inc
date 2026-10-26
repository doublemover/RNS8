#include "core/hip_resources.hpp"
TEST_CASE("CDNA5 TDM and multicast transfer litmus preserves full words and zero tails") {
  using namespace rns8::detail;if(!hip_available())SKIP("HIP unavailable");
  NativeTdmProblem p;p.m=17;p.n=19;p.k=33;p.lda=35;p.ldb=21;p.ldc=23;p.prefix=9;p.group=4;p.mode=1;
  hip_direct_device_buffer probe;REQUIRE(probe.allocate(0,4)==RNS8_SUCCESS);probe.mark_pending();uint32_t flags=0;
  const auto supported=hip_direct_prepare_native_tdm(0,p,probe.get(),flags);REQUIRE(hip_direct_synchronize(0)==RNS8_SUCCESS);probe.mark_completed();
  if(supported==RNS8_UNSUPPORTED_BACKEND)SKIP("compiled gfx1250 TDM code object required");REQUIRE(supported==RNS8_SUCCESS);
  for(unsigned cluster:{1u,2u,4u}) {
    if(cluster>1 && !(flags&2))continue;
    for(unsigned rows:{1u,15u,16u})for(unsigned cols:{1u,31u,32u}) {
      const unsigned ld=cols+2;std::vector<uint64_t> input(rows*ld,0xa5),output(cluster*512,99);
      for(unsigned row=0;row<rows;++row)for(unsigned col=0;col<cols;++col)input[row*ld+col]=UINT64_MAX-(row*131+col*7);
      hip_direct_device_buffer a,c;REQUIRE(a.allocate(0,input.size()*8)==RNS8_SUCCESS);REQUIRE(c.allocate(0,output.size()*8)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,a.get(),input.data(),input.size()*8)==RNS8_SUCCESS);
      a.mark_pending();c.mark_pending();probe.mark_pending();
      const auto result=hip_direct_native_tdm_litmus(0,a.get(),c.get(),probe.get(),rows,cols,ld,cluster,flags);
      REQUIRE(hip_direct_synchronize(0)==RNS8_SUCCESS);a.mark_completed();c.mark_completed();probe.mark_completed();REQUIRE(result==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0,output.data(),c.get(),output.size()*8)==RNS8_SUCCESS);
      for(unsigned rank=0;rank<cluster;++rank)for(unsigned row=0;row<16;++row)for(unsigned col=0;col<32;++col)
        CHECK(output[rank*512+row*32+col]==(row<rows && col<cols?input[row*ld+col]:0));
    }
  }
}
TEST_CASE("CDNA5 two-stage RNS consumers match Boost across cluster and slab tails") {
  using namespace rns8::detail;if(!hip_available())SKIP("HIP unavailable");
  NativeTdmProblem p;p.m=2;p.n=3;p.ldb=5;p.ldc=5;p.group=4;p.mode=1;p.prefix=9;p.k=33;p.lda=35;
  hip_direct_device_buffer probe;REQUIRE(probe.allocate(0,4)==RNS8_SUCCESS);probe.mark_pending();uint32_t flags=0;
  const auto supported=hip_direct_prepare_native_tdm(0,p,probe.get(),flags);REQUIRE(hip_direct_synchronize(0)==RNS8_SUCCESS);probe.mark_completed();
  if(supported==RNS8_UNSUPPORTED_BACKEND)SKIP("compiled gfx1250 TDM required");REQUIRE(supported==RNS8_SUCCESS);
  for(unsigned group:{1u,2u,4u})for(bool sign:{false,true})for(unsigned compute:{0u,1u})for(unsigned axis:{0u,1u})for(unsigned cluster:{1u,2u,4u})for(uint64_t k:{31u,32u,33u,63u,64u,65u,65537u}) {
    if(group!=4 && k!=33)continue;p.group=group;
    if((compute && !(flags&4)) || (axis && cluster==1))continue;p.compute=compute;p.cluster_axis=axis;p.m=k>65536?2:17;p.n=k>65536?3:19;
    if(cluster>1 && !(flags&2))continue;p.k=k;p.lda=k+2;p.cluster=cluster;p.mode=cluster>1?2:1;p.signed_input=sign;
    std::vector<uint64_t> av(p.m*p.lda,99),bv(k*p.ldb,99);std::vector<int8_t> result(p.prefix*p.m*p.ldc,99);
    for(uint64_t row=0;row<p.m;++row)for(uint64_t inner=0;inner<k;++inner)av[row*p.lda+inner]=((row+inner)%2?UINT64_MAX:uint64_t{1}<<63)^(row*UINT64_C(0x9e3779b97f4a7c15)+inner*131);
    for(uint64_t inner=0;inner<k;++inner)for(uint64_t col=0;col<p.n;++col)bv[inner*p.ldb+col]=((inner+col)%3?uint64_t(INT64_MAX):17)^(inner*17+col*UINT64_C(0x517cc1b727220a95));
    hip_direct_device_buffer a,b,c;REQUIRE(a.allocate(0,av.size()*8)==RNS8_SUCCESS);REQUIRE(b.allocate(0,bv.size()*8)==RNS8_SUCCESS);REQUIRE(c.allocate(0,result.size())==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,a.get(),av.data(),av.size()*8)==RNS8_SUCCESS);REQUIRE(hip_direct_copy_host_to_device(0,b.get(),bv.data(),bv.size()*8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,c.get(),result.data(),result.size())==RNS8_SUCCESS);
    for(auto* buffer:{&a,&b,&c,&probe})buffer->mark_pending();
    const auto outcome=hip_direct_native_tdm_admitted(0,a.get(),b.get(),c.get(),probe.get(),p,flags);
    REQUIRE(hip_direct_synchronize(0)==RNS8_SUCCESS);for(auto* buffer:{&a,&b,&c,&probe})buffer->mark_completed();REQUIRE(outcome==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0,result.data(),c.get(),result.size())==RNS8_SUCCESS);
    for(uint64_t row=0;row<p.m;++row)for(uint64_t col=0;col<p.ldc;++col) {
      boost::multiprecision::cpp_int exact=0;
      if(col<p.n)for(uint64_t inner=0;inner<k;++inner) {
        boost::multiprecision::cpp_int x=av[row*p.lda+inner],y=bv[inner*p.ldb+col];
        if(sign && (av[row*p.lda+inner]>>63))x-=boost::multiprecision::cpp_int(1)<<64;
        if(sign && (bv[inner*p.ldb+col]>>63))y-=boost::multiprecision::cpp_int(1)<<64;exact+=x*y;
      }
      for(unsigned plane=0;plane<p.prefix;++plane) {
        const auto modulus=rns8_default_modulus(plane);auto residue=(exact%modulus).convert_to<int>();if(residue<0)residue+=modulus;if(residue>=(modulus+1)/2)residue-=modulus;
        CHECK(result[(plane*p.m+row)*p.ldc+col]==(col<p.n?residue:99));
      }
    }
  }
}
