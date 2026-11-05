#pragma once

#include <array>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <nlohmann/json.hpp>
#if !defined(_WIN32)
extern "C" {
extern char** environ;
}
#endif

namespace rns8::tools {
namespace environment_observation_detail {
inline bool relevant(std::string_view key) noexcept {
  constexpr std::string_view prefixes[]{"RNS8_",   "OMP_", "HIP_", "ROCR_", "ROCP_",
                                        "ROCPROF", "HSA_", "KMP_", "PERF_"};
  const auto matches = [&](std::string_view name, bool exact) {
    if (key.size() < name.size() || (exact && key.size() != name.size()))
      return false;
    for (std::size_t i = 0; i < name.size(); ++i) {
      char c = key[i];
#if defined(_WIN32)
      if (c >= 'a' && c <= 'z')
        c = static_cast<char>(c - 'a' + 'A');
#endif
      if (c != name[i])
        return false;
    }
    return true;
  };
  for (const auto prefix : prefixes)
    if (matches(prefix, false))
      return true;
  return matches("CUDA_VISIBLE_DEVICES", true) || matches("LD_LIBRARY_PATH", true) ||
         matches("LD_PRELOAD", true);
}
std::string utf8(const char* text, std::size_t size);
inline nlohmann::json snapshot() {
  char** entries = nullptr;
#if defined(_WIN32)
  entries = _environ;
#else
  entries = ::environ;
#endif
  if (!entries)
    throw std::runtime_error("native C runtime environment is unavailable");
  nlohmann::json result = nlohmann::json::object();
  std::size_t retained_bytes = 0;
  for (char** entry = entries; *entry; ++entry) {
    const char* separator = std::strchr(*entry, '=');
    if (!separator || separator == *entry)
      continue;
    const auto key_size = static_cast<std::size_t>(separator - *entry);
    if (!relevant(std::string_view(*entry, key_size)))
      continue;
    if (key_size > 256)
      throw std::runtime_error("environment observation key exceeds its budget");
    auto key = utf8(*entry, key_size);
#if defined(_WIN32)
    // Python's Windows environment mapping canonicalizes variable names.
    for (char& c : key)
      if (c >= 'a' && c <= 'z')
        c = static_cast<char>(c - 'a' + 'A');
#endif
    const auto size = std::strlen(separator + 1);
    if (key.size() > 256 || size > 4096 || result.size() == 256)
      throw std::runtime_error("relevant environment exceeds the observation budget");
    auto value = utf8(separator + 1, size);
    if (value.size() > 4096)
      throw std::runtime_error("environment observation value exceeds its budget");
    retained_bytes += key.size() + value.size();
    if (retained_bytes > 65536 || result.contains(key))
      throw std::runtime_error("relevant environment is over budget or ambiguous");
    result.emplace(std::move(key), std::move(value));
  }
  return result;
}
} // namespace environment_observation_detail

class ExecutionEnvironmentWitness {
public:
  ExecutionEnvironmentWitness(bool requested, int argc, char** argv) : requested_(requested) {
    if (!requested_)
      return;
    if (!argv || argc < 1 || argc > 513)
      throw std::runtime_error("argument observation exceeds its budget");
    arguments_ = nlohmann::json::array();
    std::size_t bytes = 0;
    for (int i = 1; i < argc; ++i) {
      if (!argv[i])
        throw std::runtime_error("native argument observation contains a null argument");
      const auto size = std::strlen(argv[i]);
      if (size > 4096)
        throw std::runtime_error("argument observation exceeds its byte budget");
      auto value = environment_observation_detail::utf8(argv[i], size);
      bytes += value.size();
      if (value.size() > 4096 || bytes > 65536)
        throw std::runtime_error("argument observation exceeds its byte budget");
      arguments_.push_back(std::move(value));
    }
    before_ = environment_observation_detail::snapshot();
  }
  void finish() {
    if (!requested_)
      return;
    after_ = environment_observation_detail::snapshot();
    finished_ = true;
  }
  void print(std::ostream& out) const {
    const char* state = !requested_         ? "not_requested"
                        : !finished_        ? "incomplete"
                        : before_ == after_ ? "stable"
                                            : "changed";
    out << nlohmann::json{{"schema", "rns8_execution_environment_witness_v1"},
                          {"state", state},
                          {"arguments", arguments_},
                          {"before", before_},
                          {"after", after_}}
               .dump();
  }

private:
  bool requested_ = false, finished_ = false;
  nlohmann::json arguments_, before_, after_;
};
} // namespace rns8::tools
