TEST_CASE("private HIP buffer moves and rejected replacement preserve allocation ownership") {
  using namespace rns8::detail;
  if(!hip_available()) SKIP("no HIP device for buffer ownership");
  hip_direct_device_buffer first,second;
  REQUIRE(first.allocate(0,128)==RNS8_SUCCESS);
  void* original=first.get();
  REQUIRE(first.allocate(0,0)==RNS8_INVALID_ARGUMENT);
  REQUIRE(first.get()==original); REQUIRE(first.bytes()==128);
  REQUIRE(first.allocate(-1,256)==RNS8_INVALID_ARGUMENT);
  REQUIRE(first.get()==original);
  hip_direct_device_buffer moved(std::move(first));
  REQUIRE_FALSE(first); REQUIRE(first.bytes()==0);
  REQUIRE(moved.get()==original); REQUIRE(moved.bytes()==128);
  REQUIRE(second.allocate(0,64)==RNS8_SUCCESS);
  const auto before=hip_direct_allocation_counters_snapshot();
  second=std::move(moved);
  REQUIRE_FALSE(moved); REQUIRE(moved.bytes()==0);
  REQUIRE(second.get()==original); REQUIRE(second.bytes()==128);
  const auto after=hip_direct_allocation_counters_snapshot();
  REQUIRE(after.free_calls-before.free_calls==1);
  REQUIRE(after.allocate_calls==before.allocate_calls);
  REQUIRE(second.reset()==RNS8_SUCCESS);
  REQUIRE_FALSE(second); REQUIRE(second.bytes()==0);
  REQUIRE(second.reset()==RNS8_SUCCESS);
}

#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
TEST_CASE("pinned host buffer reuse and moves preserve staging ownership") {
  using namespace rns8::detail;
  if(!hip_available()) SKIP("no HIP device for pinned staging ownership");
  hip_pinned_host_buffer first,second;
  REQUIRE(first.ensure(0,128)==hipSuccess);
  auto* original=first.get();
  REQUIRE(first.ensure(0,64)==hipSuccess);
  REQUIRE(first.get()==original); REQUIRE(first.capacity()==128);
  REQUIRE(first.ensure(0,0)==hipErrorInvalidValue);
  REQUIRE(first.get()==original); REQUIRE(first.capacity()==128);
  REQUIRE(first.ensure(-1,256)==hipErrorInvalidValue);
  REQUIRE(first.get()==original);
  hip_pinned_host_buffer moved(std::move(first));
  REQUIRE(first.get()==nullptr); REQUIRE(first.capacity()==0); REQUIRE(first.device_id()==-1);
  REQUIRE(second.ensure(0,64)==hipSuccess);
  second=std::move(moved);
  REQUIRE(moved.get()==nullptr); REQUIRE(moved.capacity()==0);
  REQUIRE(second.get()==original); REQUIRE(second.capacity()==128); REQUIRE(second.device_id()==0);
  REQUIRE(second.reset()==hipSuccess); REQUIRE(second.reset()==hipSuccess);
  REQUIRE(second.get()==nullptr); REQUIRE(second.capacity()==0);
}
#endif
