#pragma once
#include <condition_variable>
#include <mutex>

namespace rns8::test {
// Establish overlap without sleeps or synchronization around the tested work.
class ThreadStartGate {
public:
  explicit ThreadStartGate(unsigned expected) : expected_(expected) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++ready_;
    changed_.notify_all();
    changed_.wait(lock, [&] { return open_; });
  }
  void release() {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [&] { return ready_ == expected_; });
    open_ = true;
    changed_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable changed_;
  const unsigned expected_;
  unsigned ready_ = 0;
  bool open_ = false;
};
} // namespace rns8::test
