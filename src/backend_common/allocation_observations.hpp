#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
namespace rns8::detail {
enum class AllocationObservationDomain:uint32_t { device=0,pinned_host=1 };
struct AllocationDomainObservation {
  uint64_t live_bytes=0,peak_bytes=0,live_allocations=0,peak_allocations=0;
  uint64_t allocation_calls=0,free_calls=0,allocated_bytes=0;
};
struct AllocationObservation {
  bool complete=false;
  uint64_t epoch=0;
  std::array<AllocationDomainObservation,2> domains{};
};
// Per-device, process-owned requested allocations. Transactions include the
// allocator call: a concurrent malloc cannot reuse a just-freed address before
// its retirement is recorded. No lock or lookup is added to compute/copy paths.
class AllocationObservationLedger {
 public:
  class Transaction {
   public:
    explicit Transaction(AllocationObservationLedger* ledger) noexcept:ledger_(ledger) {
      if(ledger_)try{lock_=std::unique_lock<std::mutex>(ledger_->mutex_);}catch(...){ledger_->complete_=false;ledger_=nullptr;}
    }
    void allocated(void* pointer,uint64_t bytes,AllocationObservationDomain domain) noexcept {
      if(!ledger_)return;
      const auto index=static_cast<uint32_t>(domain);
      if(!pointer || !bytes || index>=2){ledger_->complete_=false;return;}
      auto& s=ledger_->domains_[index];
      if(bytes>UINT64_MAX-s.live_bytes || bytes>UINT64_MAX-s.allocated_bytes || s.live_allocations==UINT64_MAX || s.allocation_calls==UINT64_MAX) {
        ledger_->complete_=false;return;
      }
      try {
        if(!ledger_->live_.emplace(std::make_pair(index,reinterpret_cast<uintptr_t>(pointer)),bytes).second){ledger_->complete_=false;return;}
      }catch(...){ledger_->complete_=false;return;}
      s.live_bytes+=bytes;s.allocated_bytes+=bytes;++s.live_allocations;++s.allocation_calls;
      s.peak_bytes=std::max(s.peak_bytes,s.live_bytes);s.peak_allocations=std::max(s.peak_allocations,s.live_allocations);
    }
    // Call only after successful physical release. Failed/deferred frees retain
    // their live record and cannot make the observation look artificially small.
    void released(void* pointer,AllocationObservationDomain domain) noexcept {
      if(!ledger_ || !pointer)return;
      const auto index=static_cast<uint32_t>(domain);
      if(index>=2){ledger_->complete_=false;return;}
      const auto found=ledger_->live_.find({index,reinterpret_cast<uintptr_t>(pointer)});
      if(found==ledger_->live_.end()){ledger_->complete_=false;return;}
      auto& s=ledger_->domains_[index];
      if(s.live_bytes<found->second || !s.live_allocations || s.free_calls==UINT64_MAX){ledger_->complete_=false;return;}
      s.live_bytes-=found->second;--s.live_allocations;++s.free_calls;ledger_->live_.erase(found);
    }
   private:
    AllocationObservationLedger* ledger_;
    std::unique_lock<std::mutex> lock_;
  };
  AllocationObservation snapshot() const noexcept {
    try{std::lock_guard<std::mutex> lock(mutex_);return {complete_.load(),epoch_,domains_};}catch(...){return {};}
  }
  bool reset_peaks(uint64_t& epoch) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if(!complete_.load() || epoch_==UINT64_MAX)return false;
      for(auto& d:domains_){d.peak_bytes=d.live_bytes;d.peak_allocations=d.live_allocations;}
      epoch=++epoch_;return true;
    }catch(...){complete_=false;return false;}
  }
 private:
  mutable std::mutex mutex_;
  std::atomic<bool> complete_{true};
  uint64_t epoch_=0;
  std::array<AllocationDomainObservation,2> domains_{};
  std::map<std::pair<uint32_t,uintptr_t>,uint64_t> live_;
};
} // namespace rns8::detail
