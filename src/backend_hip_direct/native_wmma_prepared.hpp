#ifndef RNS8_NATIVE_WMMA_PREPARED_HPP
#define RNS8_NATIVE_WMMA_PREPARED_HPP

#include "core/hip_resources.hpp"
#include "backend_common/native_wmma_schedule.hpp"
#include <mutex>

namespace rns8::detail {
// Private descriptor owner, not a value/materialization cache. Synchronous and
// borrowed-stream operations share the same immutable admitted descriptors.
// All referenced allocations must stay alive at their original addresses and
// layouts. Raw operand contents may change between completed calls and are reread.
// Optional encoded-B payloads are caller-owned immutable materializations; their
// logical generation binding must be maintained by the caller. Caller owns range proofs and exclusive output/input access during use.
// Public matrix generation/reader/writer ownership is not supplied by this API.
struct NativeWmmaCompletionTelemetry {
  std::vector<uint32_t> completed_work;
  std::vector<uint64_t> completion_ticks;
  int wall_clock_rate_khz=0;
  uint64_t diagnostic_device_bytes=0;
};
class NativeWmmaPreparedQueue {
 public:
  NativeWmmaPreparedQueue() = default;
  NativeWmmaPreparedQueue(const NativeWmmaPreparedQueue&) = delete;
  NativeWmmaPreparedQueue& operator=(const NativeWmmaPreparedQueue&) = delete;
  // Failed preparation preserves the previous prepared queue and all outputs.
  // Budget includes old plus replacement descriptors during transactional upload.
  rns8_status prepare(int device_id, const std::vector<NativeWmmaTask>& tasks,
                      uint32_t workers, uint64_t max_scratch_bytes,
                      NativeWmmaScheduleConfig config = {},const rns8_locality_desc* locality=nullptr);
  // Concurrent prepare/execute returns BUSY. A failed launch poisons the queue;
  // successful preparation is required before another execution. Destruction
  // requires external exclusion, like destruction of the referenced buffers.
  rns8_status execute(bool& outputs_may_be_written);
  // Borrowed stream launch: no allocation, synchronization or ownership
  // publication. Caller serializes cursor/storage reuse until real completion.
  // Owned capture may use this body; raw caller capture is not admitted publicly.
  rns8_status enqueue_borrowed_stream(void* stream,bool ordinary,bool& outputs_may_be_written);
  // Failure preserves this owner for retry; drain poisoned work before freeing.
  rns8_status close();
  // Matched scheduling control: reuse the same admitted preparation but launch
  // one ordinary grid per task, with one final synchronization. No queue cursor.
  rns8_status execute_ordinary(bool& outputs_may_be_written);
  // Separate diagnostic submission: allocation/readback and per-task atomics
  // are intentionally absent from execute(). No production latency claim.
  rns8_status execute_completion(NativeWmmaCompletionTelemetry& result, bool& outputs_may_be_written);
  // Immutable accounting after successful preparation; caller externally
  // excludes replacement while inspecting, as with borrowed descriptor views.
  uint64_t device_metadata_bytes() const {return storage_.bytes();}
  uint64_t retained_host_metadata_bytes() const {return host_tasks_.capacity()*sizeof(NativeWmmaTask)+sizeof(header_)+(locality_.struct_size?sizeof(locality_):0);}
  uint64_t preparation_materialized_host_payload_bytes() const {return materialized_host_bytes_;}
  uint64_t preparation_peak_host_payload_bytes() const {return peak_host_payload_bytes_;}
  uint64_t preparation_peak_device_metadata_bytes() const {return peak_device_bytes_;}
  NativeWmmaScheduleHeader schedule_header() const {return header_;}
  rns8_locality_info locality_info() const {return locality_;}

 private:
  std::mutex mutex_;
  hip_direct_device_buffer storage_;
  std::vector<NativeWmmaTask> host_tasks_;
  int device_id_ = -1;
  NativeWmmaTask* tasks_ = nullptr;
  NativeWmmaWork* work_ = nullptr;
  NativeWmmaTaskRange* ranges_ = nullptr;
  NativeWmmaRoundPhase* phases_ = nullptr;
  NativeWmmaScheduleHeader header_{};
  rns8_locality_info locality_{};
  uint64_t materialized_host_bytes_=0,peak_device_bytes_=0,peak_host_payload_bytes_=0;
  uint32_t* cursor_ = nullptr;
  uint32_t count_ = 0, workers_ = 0, uniform_group_ = 0;
  bool unsigned_inputs_ = false, poisoned_ = false;
};
}
#endif
