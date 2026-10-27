#include <catch2/catch_test_macros.hpp>
#include "backend_common/allocation_observations.hpp"
#include "backend_hip_direct/hip_backend.hpp"
#include "core/hip_resources.hpp"
#include "rns8/rns8.h"
#include <atomic>
#include <thread>
using namespace rns8::detail;
TEST_CASE("allocation observations retain live ownership across peak resets and pointer reuse") {
  AllocationObservationLedger ledger;char first=0,second=0;
  {AllocationObservationLedger::Transaction tx(&ledger);tx.allocated(&first,128,AllocationObservationDomain::device);tx.allocated(&second,64,AllocationObservationDomain::pinned_host);}
  auto before=ledger.snapshot();REQUIRE(before.complete);CHECK(before.domains[0].live_bytes==128);CHECK(before.domains[1].live_bytes==64);
  uint64_t epoch=99;REQUIRE(ledger.reset_peaks(epoch));CHECK(epoch==1);
  // A failed or deferred physical free never calls released: reset must retain
  // the entire live allocation, rather than pretending that it was freed.
  auto live=ledger.snapshot();CHECK(live.domains[0].peak_bytes==128);CHECK(live.domains[0].allocated_bytes==128);
  {AllocationObservationLedger::Transaction tx(&ledger);tx.released(&first,AllocationObservationDomain::device);tx.allocated(&first,32,AllocationObservationDomain::device);}
  live=ledger.snapshot();CHECK(live.domains[0].live_bytes==32);CHECK(live.domains[0].peak_bytes==128);CHECK(live.domains[0].allocation_calls==2);CHECK(live.domains[0].allocated_bytes==160);
  REQUIRE(ledger.reset_peaks(epoch));CHECK(epoch==2);CHECK(ledger.snapshot().domains[0].peak_bytes==32);
  {AllocationObservationLedger::Transaction tx(&ledger);tx.released(&first,AllocationObservationDomain::device);tx.released(&second,AllocationObservationDomain::pinned_host);}
  const auto done=ledger.snapshot();CHECK(done.complete);CHECK(done.domains[0].live_bytes==0);CHECK(done.domains[1].live_bytes==0);CHECK(done.domains[0].free_calls==2);
}
TEST_CASE("allocation observation overflow and unmatched releases remain explicitly incomplete") {
  char first=0,second=0;
  for(bool overflow:{false,true}) {
    AllocationObservationLedger ledger;
    {AllocationObservationLedger::Transaction tx(&ledger);tx.allocated(&first,overflow?UINT64_MAX:1,AllocationObservationDomain::device);
      if(overflow)tx.allocated(&second,1,AllocationObservationDomain::device);else tx.released(&second,AllocationObservationDomain::device);}
    CHECK_FALSE(ledger.snapshot().complete);uint64_t epoch=99;CHECK_FALSE(ledger.reset_peaks(epoch));CHECK(epoch==99);
  }
}
TEST_CASE("allocation observation transactions reconcile concurrent owners without conflating views") {
  AllocationObservationLedger ledger;std::atomic<unsigned> ready{0};char addresses[2]{};
  const auto owner=[&](unsigned i){
    {AllocationObservationLedger::Transaction tx(&ledger);tx.allocated(&addresses[i],16*(i+1),AllocationObservationDomain::device);}
    ready.fetch_add(1);while(ready.load()!=2)std::this_thread::yield();
    {AllocationObservationLedger::Transaction tx(&ledger);tx.released(&addresses[i],AllocationObservationDomain::device);}
  };
  std::thread first(owner,0),second(owner,1);first.join();second.join();
  const auto state=ledger.snapshot();REQUIRE(state.complete);CHECK(state.domains[0].peak_bytes==48);CHECK(state.domains[0].peak_allocations==2);
  CHECK(state.domains[0].live_bytes==0);CHECK(state.domains[0].allocation_calls==2);CHECK(state.domains[0].free_calls==2);
}
TEST_CASE("allocation observation query rejects unsupported contexts without fabricated zero observations") {
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* context=nullptr;REQUIRE(rns8_create_context(-1,&options,&context)==RNS8_SUCCESS);
  rns8_allocation_observation observation{};observation.struct_size=sizeof(observation);observation.abi_version=RNS8_ABI_VERSION;observation.epoch=99;
  CHECK(rns8_get_allocation_observation(context,&observation)==RNS8_UNSUPPORTED_BACKEND);CHECK(observation.epoch==99);
  uint64_t epoch=99;CHECK(rns8_reset_allocation_peaks(context,&epoch)==RNS8_UNSUPPORTED_BACKEND);CHECK(epoch==99);
  observation.struct_size=0;CHECK(rns8_get_allocation_observation(context,&observation)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_context(context)==RNS8_SUCCESS);
}
TEST_CASE("HIP owned device and pinned allocations reconcile public live and peak observations") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* context=nullptr;const auto status=rns8_create_context(0,&options,&context);
  if(status==RNS8_UNSUPPORTED_BACKEND)SKIP("no applicable HIP device");REQUIRE(status==RNS8_SUCCESS);
  auto snapshot=[&]{rns8_allocation_observation value{};value.struct_size=sizeof(value);value.abi_version=RNS8_ABI_VERSION;REQUIRE(rns8_get_allocation_observation(context,&value)==RNS8_SUCCESS);return value;};
  const auto before=snapshot();hip_direct_device_buffer device;hip_pinned_host_buffer pinned;
  REQUIRE(device.allocate(0,128)==RNS8_SUCCESS);REQUIRE(pinned.ensure(0,64)==hipSuccess);
  const auto allocated=snapshot();CHECK(allocated.device.live_bytes==before.device.live_bytes+128);CHECK(allocated.pinned_host.live_bytes==before.pinned_host.live_bytes+64);
  hip_direct_device_buffer moved(std::move(device));CHECK(snapshot().device.live_allocations==before.device.live_allocations+1);
  moved.mark_pending();CHECK(moved.reset()==RNS8_BUSY);CHECK(snapshot().device.live_bytes==allocated.device.live_bytes);moved.mark_completed();
  pinned.mark_pending();CHECK(pinned.reset()==hipErrorNotReady);CHECK(snapshot().pinned_host.live_bytes==allocated.pinned_host.live_bytes);pinned.mark_completed();
  uint64_t epoch=0;REQUIRE(rns8_reset_allocation_peaks(context,&epoch)==RNS8_SUCCESS);const auto reset=snapshot();CHECK(reset.epoch==epoch);CHECK(reset.device.peak_bytes==allocated.device.live_bytes);
  REQUIRE(moved.reset()==RNS8_SUCCESS);REQUIRE(pinned.reset()==hipSuccess);
  const auto after=snapshot();CHECK(after.device.live_bytes==before.device.live_bytes);CHECK(after.pinned_host.live_bytes==before.pinned_host.live_bytes);
  CHECK(after.device.free_calls==before.device.free_calls+1);CHECK(after.pinned_host.free_calls==before.pinned_host.free_calls+1);
  REQUIRE(rns8_destroy_context(context)==RNS8_SUCCESS);
#else
  SKIP("HIP allocation observation requires a compiled HIP backend");
#endif
}
TEST_CASE("timing failures remain censored diagnostics and resets isolate capture epochs") {
  hip_direct_timing_set_enabled(false);hip_direct_timing_record_failure("disabled","create",2);CHECK(hip_direct_timing_failure_snapshot().empty());
  hip_direct_timing_set_enabled(true);hip_direct_timing_reset();
  hip_direct_timing_record_sample("valid",1);hip_direct_timing_record_sample("invalid",std::numeric_limits<double>::quiet_NaN());
  hip_direct_timing_record_failure("operation","create_or_record_start",2);
  const auto samples=hip_direct_timing_snapshot();REQUIRE(samples.size()==1);CHECK(samples[0].label=="valid");
  const auto failures=hip_direct_timing_failure_snapshot();REQUIRE(failures.size()==2);CHECK(failures[0].label=="invalid");CHECK(failures[1].backend_status==2);
  for(unsigned i=0;i<100;++i)hip_direct_timing_record_failure("overflow","record_stop",3);
  CHECK_FALSE(hip_direct_timing_failures_complete());CHECK(hip_direct_timing_failure_snapshot().size()==64);
  hip_direct_timing_reset();CHECK(hip_direct_timing_failure_snapshot().empty());CHECK(hip_direct_timing_failures_complete());
  hip_direct_timing_set_enabled(false);
}
