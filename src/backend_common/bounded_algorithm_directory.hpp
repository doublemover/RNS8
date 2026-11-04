#pragma once
#include <array>
#include <cstddef>
#include <mutex>
#include <type_traits>

namespace rns8::detail {
// Four bounded probes, no allocation or callbacks under the directory lock.
// Eviction is round-robin within a set; copied selections outlive eviction and
// reset. Concurrent identical misses may independently select outside the lock.
template<class Key,class Value,class Hash,std::size_t Sets=64,std::size_t Ways=4>
class BoundedAlgorithmDirectory {
  static_assert(Sets>0 && Ways>0);
  static_assert(std::is_trivially_copyable_v<Key> && std::is_trivially_copyable_v<Value> &&
    std::is_nothrow_copy_assignable_v<Key> && std::is_nothrow_copy_assignable_v<Value>,
    "algorithm directory entries must be owned nonallocating values");
  struct Entry {Key key{};Value value{};bool ready=false;};
  struct Set {std::array<Entry,Ways> entries{};std::size_t next=0;};
 public:
  static constexpr std::size_t capacity=Sets*Ways;
  bool find(const Key& key,Value& value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for(const auto& entry:sets_[Hash{}(key)%Sets].entries)
      if(entry.ready && entry.key==key){value=entry.value;return true;}
    return false;
  }
  void remember(const Key& key,const Value& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& set=sets_[Hash{}(key)%Sets];
    for(auto& entry:set.entries)if(entry.ready && entry.key==key){entry.value=value;return;}
    for(auto& entry:set.entries)if(!entry.ready){entry.key=key;entry.value=value;entry.ready=true;return;}
    auto& entry=set.entries[set.next];entry.key=key;entry.value=value;entry.ready=true;
    set.next=(set.next+1)%Ways;
  }
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for(auto& set:sets_){for(auto& entry:set.entries)entry.ready=false;set.next=0;}
  }
  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);std::size_t size=0;
    for(const auto& set:sets_)for(const auto& entry:set.entries)size+=entry.ready;
    return size;
  }
 private:
  mutable std::mutex mutex_;
  std::array<Set,Sets> sets_{};
};
}
