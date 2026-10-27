#include "backend_common/instruction_roof_plan.hpp"
TEST_CASE("WMMA instruction chains preserve exact accumulators and every output cell") {
  if(!hip_available()) SKIP("no HIP device for instruction chains");
  using namespace rns8::detail;
  const std::array<int8_t,8> av{-128,127,-128,1,0,-1,63,-64};
  const std::array<int8_t,8> bv{-128,127,127,-128,127,-1,-64,63};
  hip_direct_device_buffer a,b,capability;
  REQUIRE(a.allocate(0,av.size())==RNS8_SUCCESS);REQUIRE(b.allocate(0,bv.size())==RNS8_SUCCESS);
  REQUIRE(capability.allocate(0,4)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,a.get(),av.data(),av.size())==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,b.get(),bv.data(),bv.size())==RNS8_SUCCESS);
  for(uint32_t chains:{1u,2u,4u,8u}) for(uint32_t iterations:{1u,4095u,4096u})
    for(uint32_t waves:{1u,5u}) for(bool reduce:{false,true}) {
      InstructionRoofProblem p{1,chains,iterations,waves,reduce};
      const auto support=query_instruction_roof(p);REQUIRE(support.status==RNS8_SUCCESS);
      std::vector<int32_t> actual(support.output_bytes/4+17,0x12345678);
      hip_direct_device_buffer output;REQUIRE(output.allocate(0,actual.size()*4)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,output.get(),actual.data(),actual.size()*4)==RNS8_SUCCESS);
      const auto status=hip_direct_instruction_roof(0,a.get(),b.get(),output.get(),capability.get(),p);
      if(status==RNS8_UNSUPPORTED_BACKEND) SKIP("target has no qualified native INT8 WMMA builtin");
      REQUIRE(status==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),output.get(),actual.size()*4)==RNS8_SUCCESS);
      for(uint32_t chain=0;chain<chains;++chain) {
        int64_t expected=0;
        for(uint32_t i=0;i<iterations;++i) {
          expected+=int64_t(16)*av[chain]*bv[chain];
          if(reduce) {expected%=251;if(expected<0) expected+=251;if(expected>=126) expected-=251;}
        }
        for(uint32_t wave=0;wave<waves;++wave) for(int cell=0;cell<256;++cell)
          REQUIRE(actual[(wave*chains+chain)*256+cell]==expected);
      }
      for(size_t i=support.output_bytes/4;i<actual.size();++i) REQUIRE(actual[i]==0x12345678);
      CHECK(hip_direct_instruction_roof(0,a.get(),b.get(),output.get(),output.get(),p)==RNS8_INVALID_ARGUMENT);
      p.iterations=4097;
      CHECK(hip_direct_instruction_roof(0,a.get(),b.get(),output.get(),capability.get(),p)==RNS8_INVALID_ARGUMENT);
      std::vector<int32_t> unchanged(actual.size());
      REQUIRE(hip_direct_copy_device_to_host(0,unchanged.data(),output.get(),unchanged.size()*4)==RNS8_SUCCESS);
      REQUIRE(unchanged==actual);
    }
}
