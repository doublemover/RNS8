#include "../support/native_device_fixture.hpp"
TEST_CASE("native device query checks view spans without touching device or owner") {
  using namespace native_device_fixture;
  alignas(8) uint64_t words[32]{};Owner owner;
  const auto valid=descriptor(words,sizeof(words),owner);
  rns8_native_device_resources r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_query_native_device(&valid,&r)==RNS8_SUCCESS);
  REQUIRE(r.view_span_bytes==19*8);REQUIRE(r.compact_native_bytes==15*8);REQUIRE(r.export_scratch_bytes==15*8+4);REQUIRE(r.verified_import_scratch_bytes==8);
  for(unsigned bad=0;bad<12;++bad) {
    auto d=valid;
    switch(bad) {
    case 0:d.byte_offset=1;break;case 1:d.allocation_bytes=19*8;break;
    case 2:d.rows=INT64_MAX;break;case 3:d.ld=-1;break;
    case 4:d.producer_complete=0;break;case 5:d.owner_identity=0;break;
    case 6:d.generation=0;break;case 7:d.acquire=nullptr;break;
    case 8:d.release=nullptr;break;case 9:d.access=static_cast<rns8_native_device_access>(3);break;
    case 10:d.element_type=static_cast<rns8_native_device_type>(0);break;case 11:d.allocation_bytes=UINT64_MAX;break;
    }
    const auto previous=r;REQUIRE(rns8_query_native_device(&d,&r)!=RNS8_SUCCESS);
    REQUIRE(r.view_span_bytes==previous.view_span_bytes);REQUIRE(r.export_scratch_bytes==previous.export_scratch_bytes);
  }
  REQUIRE(owner.acquired==0);REQUIRE(owner.released==0);
  auto* ctx=create_cpu_context();rns8_native_device_lease* lease=nullptr;
  REQUIRE(rns8_create_native_device_lease(ctx,&valid,&lease)==RNS8_UNSUPPORTED_BACKEND);REQUIRE(lease==nullptr);
  REQUIRE(owner.acquired==0);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
