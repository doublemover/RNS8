#include "../support/finite_verify_fixture.hpp"
TEST_CASE("HIP finite Freivalds projections agree with independent ring products") {
  if(!hip_available()) SKIP("no HIP device for finite Freivalds");
  rns8_context_options o{}; o.struct_size=sizeof(o); o.abi_version=RNS8_ABI_VERSION;
  o.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr; REQUIRE(rns8_create_context(0,&o,&ctx)==RNS8_SUCCESS);
  for(uint16_t q : {2,4,6,251,256}) finite_verify_fixture::public_roundtrip(ctx,q,65);
  finite_verify_fixture::public_roundtrip(ctx,256,65537);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
