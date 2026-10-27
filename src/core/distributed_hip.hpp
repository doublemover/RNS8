#pragma once
#include "backend_common/distributed_inputs.hpp"
#include <memory>

namespace rns8::detail::distributed {
struct HipProgress {
  State state = State::idle;
  uint64_t epoch = 0, local_completed = 0, transfers_queued = 0, expected = 0;
  uint64_t published_epoch = 0;
  Admission operation;
  int runtime_error = 0;
  bool output_current = false;
  bool integrity_requested = false, integrity_completed = false;
  uint32_t integrity_failure_mask = 0;
};
// One process, explicit modulus/M/N/2-D/K shards and serialized host operations.
// No RCCL, cross-device event timing, partial-result export or implicit fallback.
class HipExecution {
 public:
  static bool compiled() noexcept;
  // Discovery reports runtime ordinals and BDF identities, never physical-card
  // counts. Unknown package/partition fields remain unknown.
  static Admission discover(const std::vector<uint32_t>& ordinals,
                            std::vector<Rank>& ranks, std::vector<Edge>& edges);
  static Admission create(std::shared_ptr<const Plan>, std::shared_ptr<const Inputs>,
                          std::unique_ptr<HipExecution>& output);
  ~HipExecution();
  HipExecution(const HipExecution&) = delete;
  HipExecution& operator=(const HipExecution&) = delete;
  // timeout_ns is a positive duration on the host monotonic clock. A failed
  // partial enqueue still accepts its epoch; inspect progress/operation status.
  Admission submit(uint64_t epoch, uint64_t timeout_ns);
  Admission query(uint64_t epoch, HipProgress&);
  Admission wait(uint64_t epoch, uint64_t observation_timeout_ns, HipProgress&);
  Admission cancel(uint64_t epoch);
  Admission retain(uint64_t epoch);
  Admission release(uint64_t epoch);
  // Always three little-endian limbs/cell, two's complement for signed output.
  // Caller publication occurs only after a complete checked host copy.
  Admission export_limbs(uint64_t epoch, uint64_t* destination, uint64_t ld);
  // Borrow requires retain(epoch) through the last actual consumer. No local
  // shard is exposed as a short prefix. The result is on the explicit owner.
  Admission resident_limbs(uint64_t epoch, int& device, const void*& pointer) const;
  Admission materialize(uint64_t epoch, uint32_t prefix, uint64_t max_bytes);
  Admission resident_residues(uint64_t epoch, int& device, const void*& pointer, uint32_t& prefix) const;
  const std::vector<std::string>& arithmetic_certificates() const;
  Admission close(); // BUSY while work/readers survive; failed frees retain ownership
  static void reap_quarantined() noexcept;
 private:
  struct Impl;
  explicit HipExecution(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
} // namespace rns8::detail::distributed
