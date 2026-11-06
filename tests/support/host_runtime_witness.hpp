#pragma once

// Opt-in observations in the real test/consumer process. No sidecar probe can
// stand in for these bookends, and they make no claim about transient images.
#include "host_runtime_inventory.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#ifndef __has_feature
#define __has_feature(value) 0
#endif
#if defined(__linux__) && __has_feature(thread_sanitizer)
extern "C" void __tsan_init();
#elif defined(__linux__) && __has_feature(address_sanitizer)
extern "C" void __asan_init();
#endif
#if defined(__linux__) && defined(_OPENMP)
#include <omp.h>
#endif

namespace rns8::test {
inline uint64_t host_parent_process_id() {
#ifdef _WIN32
  const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE)
    throw std::runtime_error("native parent snapshot failed");
  struct Owner {
    HANDLE value;
    ~Owner() { CloseHandle(value); }
  } owner{snapshot};
  PROCESSENTRY32W entry{};
  entry.dwSize = sizeof(entry);
  if (!Process32FirstW(snapshot, &entry))
    throw std::runtime_error("native parent enumeration failed");
  unsigned count = 0;
  do {
    if (++count > 131072)
      throw std::runtime_error("native parent inventory limit exceeded");
    if (entry.th32ProcessID == GetCurrentProcessId()) {
      if (!entry.th32ParentProcessID)
        throw std::runtime_error("native parent identity absent");
      return entry.th32ParentProcessID;
    }
  } while (Process32NextW(snapshot, &entry));
  throw std::runtime_error("native parent identity unavailable");
#elif defined(__linux__)
  const auto pid = getppid();
  if (pid <= 0)
    throw std::runtime_error("native parent identity absent");
  return static_cast<uint64_t>(pid);
#else
  throw std::runtime_error("native parent identity unsupported");
#endif
}

inline std::string host_executable_path() {
#ifdef _WIN32
  std::vector<wchar_t> path(32768);
  const auto size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (!size || size >= path.size())
    throw std::runtime_error("native executable path unavailable or truncated");
  return std::filesystem::canonical(std::filesystem::path(std::wstring(path.data(), size)))
      .u8string();
#elif defined(__linux__)
  return std::filesystem::canonical("/proc/self/exe").u8string();
#else
  throw std::runtime_error("native executable path unsupported");
#endif
}

class HostRuntimeWitness {
  using Clock = std::chrono::steady_clock;
  std::string token_, executable_;
  nlohmann::json arguments_;
  Clock::time_point start_{};
  uint64_t parent_ = 0;
  int exceptions_ = 0;

  void emit(const char* phase) const {
    auto modules = host_runtime_inventory();
    modules["scope"] = "application_process_loader_bookend_not_transient_image_history";
    modules["openmp_workers"] = 0; // Do not create unrelated work to initialize a runtime.
    modules["symbol_owners"] = nlohmann::json::object();
#if defined(__linux__) && __has_feature(thread_sanitizer)
    modules["symbol_owners"]["sanitizer"] =
        host_symbol_owner(reinterpret_cast<const void*>(&__tsan_init));
#elif defined(__linux__) && __has_feature(address_sanitizer)
    modules["symbol_owners"]["sanitizer"] =
        host_symbol_owner(reinterpret_cast<const void*>(&__asan_init));
#endif
#if defined(__linux__) && defined(_OPENMP)
    modules["symbol_owners"]["openmp"] =
        host_symbol_owner(reinterpret_cast<const void*>(&omp_get_num_threads));
#endif
    if (parent_ != host_parent_process_id() || executable_ != host_executable_path())
      throw std::runtime_error("native witness process identity changed");
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count();
    if (elapsed < 0)
      throw std::runtime_error("native witness clock regressed");
    nlohmann::json value = {{"schema", "rns8_host_application_witness_v1"},
                            {"token", token_},
                            {"phase", phase},
                            {"process_id", rns8::tools::process_id()},
                            {"parent_process_id", parent_},
                            {"executable", executable_},
                            {"arguments", arguments_},
                            {"elapsed_ns", elapsed},
                            {"unwinding", std::uncaught_exceptions() != exceptions_},
                            {"address_feature", bool(__has_feature(address_sanitizer))},
                            {"thread_feature", bool(__has_feature(thread_sanitizer))},
                            {"loaded_modules", std::move(modules)}};
    const auto encoded = value.dump();
    if (encoded.size() > 256 * 1024)
      throw std::runtime_error("native witness record budget exceeded");
    std::cout << '\n' << "RNS8_HOST_RUNTIME_WITNESS " << encoded << '\n' << std::flush;
    if (!std::cout)
      throw std::runtime_error("native witness output failed");
  }

public:
  HostRuntimeWitness(int argc, char** argv) {
#ifdef _WIN32
    std::array<char, 65> buffer{};
    const auto size = GetEnvironmentVariableA("RNS8_HOST_RUNTIME_WITNESS_TOKEN", buffer.data(),
                                              DWORD(buffer.size()));
    if (!size && GetLastError() == ERROR_ENVVAR_NOT_FOUND)
      return;
    if (!size || size >= buffer.size())
      throw std::invalid_argument("native witness token missing or exceeds budget");
    const char* token = buffer.data();
#else
    const char* token = std::getenv("RNS8_HOST_RUNTIME_WITNESS_TOKEN");
    if (!token)
      return; // Catch discovery and ordinary executions have no observations.
#endif
    for (size_t i = 0; i != 64; ++i)
      if (!token[i] ||
          !((token[i] >= '0' && token[i] <= '9') || (token[i] >= 'a' && token[i] <= 'f')))
        throw std::invalid_argument("native witness requires a lowercase 256-bit token");
    if (token[64] || argc < 1 || argc > 512 || !argv)
      throw std::invalid_argument("native witness invocation invalid");
    token_ = token;
    arguments_ = nlohmann::json::array();
    size_t bytes = 0;
    for (int i = 1; i < argc; ++i) {
      if (!argv[i])
        throw std::invalid_argument("native witness argument absent");
      std::string argument(argv[i]);
      bytes += argument.size();
      if (argument.size() > 4096 || bytes > 32768)
        throw std::invalid_argument("native witness argument budget exceeded");
      arguments_.push_back(std::move(argument));
    }
    executable_ = host_executable_path();
    parent_ = host_parent_process_id();
    exceptions_ = std::uncaught_exceptions();
    start_ = Clock::now();
    emit("enter");
  }
  HostRuntimeWitness(const HostRuntimeWitness&) = delete;
  HostRuntimeWitness& operator=(const HostRuntimeWitness&) = delete;
  ~HostRuntimeWitness() noexcept {
    if (token_.empty())
      return;
    try {
      emit("leave");
    } catch (const std::exception& error) {
      std::cerr << "host runtime witness: " << error.what() << '\n';
      std::_Exit(2); // A missing bookend must never be an ordinary PASS.
    } catch (...) {
      std::_Exit(2);
    }
  }
};
} // namespace rns8::test
