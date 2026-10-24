#ifndef RNS8_NATIVE_WMMA_PREPARED_HPP
#define RNS8_NATIVE_WMMA_PREPARED_HPP

#include "core/hip_resources.hpp"
#include <mutex>

namespace rns8::detail {
// Private synchronous descriptor owner, not a value/materialization cache.
// All referenced allocations must stay alive at their original addresses and
// layouts. Raw operand contents may change between completed calls and are reread.
// Optional encoded-B payloads are caller-owned immutable materializations; their
// logical generation binding must be maintained by the caller. Caller owns range proofs and exclusive output/input access during use.
// Public matrix generation/reader/writer ownership is not supplied by this API.
class NativeWmmaPreparedQueue {
 public:
  NativeWmmaPreparedQueue() = default;
  NativeWmmaPreparedQueue(const NativeWmmaPreparedQueue&) = delete;
  NativeWmmaPreparedQueue& operator=(const NativeWmmaPreparedQueue&) = delete;
  // Failed preparation preserves the previous prepared queue and all outputs.
  // Budget includes old plus replacement descriptors during transactional upload.
  rns8_status prepare(int device_id, const std::vector<NativeWmmaTask>& tasks,
                      uint32_t workers, uint64_t max_scratch_bytes);
  // Concurrent prepare/execute returns BUSY. A failed launch poisons the queue;
  // successful preparation is required before another execution. Destruction
  // requires external exclusion, like destruction of the referenced buffers.
  rns8_status execute(bool& outputs_may_be_written);
  // Matched scheduling control: reuse the same admitted preparation but launch
  // one ordinary grid per task, with one final synchronization. No queue cursor.
  rns8_status execute_ordinary(bool& outputs_may_be_written);

 private:
  std::mutex mutex_;
  hip_direct_device_buffer storage_;
  std::vector<NativeWmmaTask> host_tasks_;
  int device_id_ = -1;
  NativeWmmaTask* tasks_ = nullptr;
  NativeWmmaWork* work_ = nullptr;
  uint32_t* cursor_ = nullptr;
  uint32_t count_ = 0, workers_ = 0, uniform_group_ = 0;
  bool unsigned_inputs_ = false, poisoned_ = false;
};
}
#endif
