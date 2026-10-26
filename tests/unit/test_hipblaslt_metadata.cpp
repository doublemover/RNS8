#include <algorithm>
#include "backend_hipblaslt/matmul_metadata_plan.hpp"
#include "backend_hipblaslt/hipblaslt_backend.hpp"
#include <catch2/catch_test_macros.hpp>
using namespace rns8::detail;
TEST_CASE("hipBLASLt algorithm observations retain physical identities and explicit overflow") {
  HipblasLtAlgorithmInventory inventory;
  HipblasLtMetadataKey key{0,16,32,64,32,4096};
  inventory.record(key,7);inventory.record(key,7);inventory.record(key,8);
  REQUIRE(inventory.size==2);CHECK(inventory.entries[0].calls==2);CHECK(inventory.entries[1].algorithm_index==8);
  for(int index=9;index<23;++index)inventory.record(key,index);
  REQUIRE(inventory.size==16);CHECK(inventory.complete);
  inventory.record(key,23);CHECK_FALSE(inventory.complete);CHECK(inventory.size==16);
  inventory={};inventory.record(key,-1);CHECK_FALSE(inventory.complete);CHECK(inventory.entries[0].algorithm_index==-1);
  inventory={};inventory.record(key,7);inventory.entries[0].calls=UINT64_MAX;
  inventory.record(key,7);CHECK_FALSE(inventory.complete);CHECK(inventory.entries[0].calls==UINT64_MAX);
  hipblaslt_algorithm_observations_set_enabled(false);
  CHECK_FALSE(hipblaslt_algorithm_observations_snapshot().complete);
}
TEST_CASE("hipBLASLt operand retention policies are independent and reject unknown bits") {
  const auto original=hipblaslt_prepack_policy();
  struct Restore {HipblasLtPrepackPolicy policy;~Restore(){(void)hipblaslt_prepack_set_policy(policy);}} restore{original};
  for(const auto policy:{HipblasLtPrepackPolicy::none,HipblasLtPrepackPolicy::a,HipblasLtPrepackPolicy::b,HipblasLtPrepackPolicy::both}) {
    REQUIRE(hipblaslt_prepack_set_policy(policy)==RNS8_SUCCESS);CHECK(hipblaslt_prepack_policy()==policy);
    CHECK(hipblaslt_prepack_a_enabled(policy)==(policy==HipblasLtPrepackPolicy::a || policy==HipblasLtPrepackPolicy::both));
    CHECK(hipblaslt_prepack_b_enabled(policy)==(policy==HipblasLtPrepackPolicy::b || policy==HipblasLtPrepackPolicy::both));
  }
  CHECK(hipblaslt_prepack_set_policy(static_cast<HipblasLtPrepackPolicy>(4))==RNS8_INVALID_ARGUMENT);
  CHECK(hipblaslt_prepack_policy()==HipblasLtPrepackPolicy::both);
}
TEST_CASE("hipBLASLt immutable call directory shares padded geometry across every residue plane") {
  for(unsigned prefix:{1u,9u,18u,20u})for(int k:{1,15,16,17,65535,65536,65537,131072,131073}) {
    HipblasLtCallMetadataDirectory directory;unsigned builds=0,hits=0,blocks=0;
    for(unsigned p=0;p<prefix;++p)for(int start=0;start<k;start+=65536){
      const int slab=std::min(65536,k-start),padded=(slab+15)/16*16;
      HipblasLtMetadataKey key{0,16,32,padded,32,1u<<20};
      if(directory.find(key)<0){const int slot=directory.free_slot();REQUIRE(slot>=0);REQUIRE(directory.publish(slot,key));++builds;}
      else ++hits;
      ++blocks;
    }
    const unsigned geometries=(k>65536 && k%65536!=0)?2:1;
    CHECK(builds==geometries);CHECK(hits+builds==blocks);CHECK(builds<=2);
  }
}
TEST_CASE("hipBLASLt metadata identity includes all layout operation device and workspace attributes") {
  HipblasLtMetadataKey key{0,16,32,65536,32,4096};HipblasLtCallMetadataDirectory d;
  REQUIRE(d.publish(0,key));CHECK(d.find(key)==0);CHECK_FALSE(d.publish(0,key));
  for(unsigned field=0;field<7;++field){auto changed=key;
    switch(field){case 0:++changed.device_id;break;case 1:++changed.m;break;case 2:++changed.n;break;case 3:++changed.k;break;case 4:++changed.scratch_ld;break;case 5:++changed.workspace_bytes;break;case 6:++changed.attribute_abi;break;}
    CHECK_FALSE(changed==key);CHECK(d.find(changed)==-1);
  }
  CHECK_FALSE(d.publish(2,key));auto bad=key;bad.attribute_abi=2;CHECK_FALSE(d.publish(1,bad));CHECK(d.free_slot()==1);
  bad=key;bad.device_id=-1;CHECK_FALSE(d.publish(1,bad));bad=key;bad.scratch_ld=31;CHECK_FALSE(d.publish(1,bad));
  auto tail=key;tail.k=16;REQUIRE(d.publish(1,tail));CHECK(d.free_slot()==-1);CHECK(d.find(key)==0);CHECK(d.find(tail)==1);
}
#if defined(RNS8_ENABLE_HIPBLASLT) && RNS8_ENABLE_HIPBLASLT
#include "backend_hipblaslt/hipblaslt_headers.hpp"
#include <cmath>
#if __has_include(<hipblaslt/hipblaslt_e5m3.h>)
TEST_CASE("hipBLASLt host E5M3 decodes every positive subnormal exactly", "[hipblaslt][headers]") {
  for (uint8_t bits=1;bits<8;++bits) {
    hipblaslt_e5m3 value; value.data=bits;
    CHECK(float(value)==std::ldexp(float(bits),-17));
  }
}
#endif
#if defined(_MSC_VER) && !defined(__clang__)
TEST_CASE("hipBLASLt MSVC leading-zero adaptation covers unsigned boundaries", "[hipblaslt][headers]") {
  CHECK(hipblaslt_host_clz(0)==32);
  for(unsigned bit=0;bit<32;++bit) CHECK(hipblaslt_host_clz(1u<<bit)==31-int(bit));
  CHECK(hipblaslt_host_clz(UINT32_MAX)==0);
}
#endif
#endif
