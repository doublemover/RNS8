#pragma once
#include "rns8/status.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace rns8::detail {
// Stack-scoped to one submission, never retained with an unowned external stream.
class BorrowedStreamAdmission {
public:
  template <class Check> rns8_status admit(const void* context, void* stream, Check&& check) {
    if (!context || !stream)
      return RNS8_INVALID_ARGUMENT;
    if (context_)
      return context_ == context && stream_ == stream ? RNS8_SUCCESS : RNS8_INVALID_ARGUMENT;
    const auto status = check();
    if (status == RNS8_SUCCESS) {
      context_ = context;
      stream_ = stream;
    }
    return status;
  }

private:
  const void* context_ = nullptr;
  void* stream_ = nullptr;
};
template <class Reader>
inline std::vector<std::pair<Reader*, uint64_t>>
compile_reader_multiplicities(std::vector<Reader*> readers) {
  std::sort(readers.begin(), readers.end(), std::less<Reader*>{});
  size_t unique = 0;
  for (size_t i = 0; i < readers.size(); ++i)
    unique += i == 0 || readers[i] != readers[i - 1];
  std::vector<std::pair<Reader*, uint64_t>> result;
  result.reserve(unique);
  for (auto* reader : readers) {
    if (result.empty() || result.back().first != reader)
      result.emplace_back(reader, 1);
    else
      ++result.back().second;
  }
  return result;
}
template <class Reader>
inline bool
reader_multiplicities_fit(const std::vector<std::pair<Reader*, uint64_t>>& readers) noexcept {
  for (const auto& entry : readers)
    if (!entry.first || entry.first->in_flight_readers > UINT64_MAX - entry.second)
      return false;
  return true;
}
} // namespace rns8::detail
