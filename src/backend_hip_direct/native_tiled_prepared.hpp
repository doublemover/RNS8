#pragma once
#include "backend_common/native_tiled_resources.hpp"
#include "backend_common/native_wmma_queue.hpp"
#include "core/hip_resources.hpp"
#include <mutex>
namespace rns8::detail {
struct NativeTiledConfig {
  uint32_t depth=16, buffers=0, waves=4;
  NativeConversion conversion=NativeConversion::replicated;
  uint64_t lds_budget=65536;
};
// Device/code-module-lifetime-bound synchronous research owner. Native input
// values may change between completed calls; pointers/layouts must stay alive.
// No encoded values or input-dependent facts are cached. Device reset/module
// unload while this owner or its borrowed allocations is live is prohibited.
// No public matrix generation/lifetime ownership is implied by this private API.
class NativeTiledPrepared {
 public:
  NativeTiledPrepared()=default;
  NativeTiledPrepared(const NativeTiledPrepared&)=delete;
  NativeTiledPrepared& operator=(const NativeTiledPrepared&)=delete;
  // Rebinding is transactional; failure preserves prior preparation/output.
  rns8_status prepare(int device_id,const NativeWmmaTask& task,const NativeTiledConfig& config);
  // No allocation, capability launch/readback, symbolic work, or descriptor
  // upload. One admitted grid and one completion wait. Failure poisons owner.
  rns8_status execute(bool& outputs_may_be_written);
 private:
  std::mutex mutex_;
  hip_direct_device_buffer capability_lifetime_;
  NativeWmmaTask task_{};
  NativeTiledConfig config_{};
  int device_id_=-1;
  bool ready_=false,poisoned_=false;
};
rns8_status hip_direct_gemm_native_conversion_device(int device_id,const NativeWmmaTask& task,
    const NativeTiledConfig& config,bool& outputs_may_be_written);
}
