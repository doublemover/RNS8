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

TEST_CASE("copy-owned native leases require a read snapshot budget before owner acquisition") {
  using namespace native_device_fixture;
  alignas(8) uint64_t words[32]{};Owner owner;auto d=descriptor(words,sizeof(words),owner);
  rns8_native_device_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_query_native_device(&d,&resources)==RNS8_SUCCESS);REQUIRE(resources.snapshot_bytes==0);
  d.policy=RNS8_NATIVE_DEVICE_COPY_OWNED;
  REQUIRE(rns8_query_native_device(&d,&resources)==RNS8_WORKSPACE_TOO_SMALL);
  d.snapshot_budget_bytes=119;REQUIRE(rns8_query_native_device(&d,&resources)==RNS8_WORKSPACE_TOO_SMALL);
  d.snapshot_budget_bytes=120;REQUIRE(rns8_query_native_device(&d,&resources)==RNS8_SUCCESS);
  REQUIRE(resources.snapshot_bytes==120);REQUIRE(resources.view_span_bytes==152);
  d.access=RNS8_NATIVE_DEVICE_WRITE;REQUIRE(rns8_query_native_device(&d,&resources)==RNS8_INVALID_ARGUMENT);
  d.access=RNS8_NATIVE_DEVICE_READ;d.policy=RNS8_NATIVE_DEVICE_BORROW;
  REQUIRE(rns8_query_native_device(&d,&resources)==RNS8_INVALID_ARGUMENT);
  REQUIRE(owner.acquired==0);REQUIRE(owner.released==0);
}

TEST_CASE("native limb descriptors bind integer widths and reject unsupported ownership") {
  using namespace native_device_fixture;
  alignas(8) uint64_t words[2048]{}; Owner owner;
  for (auto type : {RNS8_NATIVE_DEVICE_SIGNED_LIMBS, RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS})
    for (uint32_t width : {1u, 2u, 3u, 32u}) {
      auto d=descriptor(words,sizeof(words),owner);
      d.element_type=type; d.limb_count=width; d.access=RNS8_NATIVE_DEVICE_WRITE;
      rns8_native_device_resources r{}; r.struct_size=sizeof(r); r.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_query_native_device(&d,&r)==RNS8_SUCCESS);
      CHECK(r.view_span_bytes==19*8*width);
      CHECK(r.export_scratch_bytes==15*8*width+4);
      CHECK(r.verified_import_scratch_bytes==0);
      for (unsigned bad=0;bad<7;++bad) {
        auto invalid=d;
        switch(bad) {
          case 0: invalid.limb_count=0; break;
          case 1: invalid.limb_count=33; break;
          case 2: invalid.access=RNS8_NATIVE_DEVICE_READ; break;
          case 3: invalid.policy=RNS8_NATIVE_DEVICE_COPY_OWNED; break;
          case 4: invalid.element_type=RNS8_NATIVE_DEVICE_I64; break;
          case 5: invalid.allocation_bytes=d.byte_offset+r.view_span_bytes-1; break;
          case 6: invalid.ld=INT64_MAX; break;
        }
        CHECK(rns8_query_native_device(&invalid,&r)!=RNS8_SUCCESS);
      }
    }
  CHECK(owner.acquired==0); CHECK(owner.released==0);
}
