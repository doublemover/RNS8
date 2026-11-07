#pragma once

// One native process snapshot. Callers retain their control/application scope.
#include "tool_common/process_identity.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <tlhelp32.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <exception>
#include <link.h>
#endif

namespace rns8::test {
inline nlohmann::json host_runtime_inventory() {
  std::set<std::string> paths;
  nlohmann::json virtual_images = nlohmann::json::array();
#ifdef _WIN32
  const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
  if (snapshot == INVALID_HANDLE_VALUE)
    throw std::runtime_error("native module snapshot failed");
  // Close the native owner even when path conversion/allocation throws.
  struct Owner {
    HANDLE value;
    ~Owner() { CloseHandle(value); }
  } owner{snapshot};
  MODULEENTRY32W entry{};
  entry.dwSize = sizeof(entry);
  if (!Module32FirstW(snapshot, &entry))
    throw std::runtime_error("native module enumeration failed");
  do {
    if (paths.size() >= 256)
      throw std::runtime_error("native module inventory limit exceeded");
    paths.insert(std::filesystem::canonical(entry.szExePath).u8string());
  } while (Module32NextW(snapshot, &entry));
  if (GetLastError() != ERROR_NO_MORE_FILES)
    throw std::runtime_error("native module enumeration incomplete");
#elif defined(__linux__)
  struct State {
    std::set<std::string>& paths;
    nlohmann::json& virtual_images;
    std::exception_ptr error;
    size_t count = 0;
  } state{paths, virtual_images, {}};
  const auto visit = [](dl_phdr_info* image, size_t, void* data) noexcept {
    auto& state = *static_cast<State*>(data);
    try {
      if (++state.count > 256)
        throw std::runtime_error("native module inventory limit exceeded");
      const std::string name = image->dlpi_name ? image->dlpi_name : "";
      if (name == "linux-vdso.so.1" || name == "linux-gate.so.1")
        state.virtual_images.push_back(name);
      else
        state.paths.insert(
            std::filesystem::canonical(name.empty() ? "/proc/self/exe" : name).u8string());
      return 0;
    } catch (...) {
      state.error = std::current_exception();
      return 1;
    }
  };
  if (dl_iterate_phdr(visit, &state) != 0 || state.error) {
    if (state.error)
      std::rethrow_exception(state.error);
    throw std::runtime_error("native module enumeration incomplete");
  }
#else
  throw std::runtime_error("native module inventory unsupported");
#endif
  if (paths.empty())
    throw std::runtime_error("native module inventory empty");
  return {{"schema", "rns8_host_loaded_modules_v1"},
          {"process_id", rns8::tools::process_id()},
          {"modules", paths},
          {"virtual_images", virtual_images},
          {"scope", "control_process_loader_snapshot_not_all_application_processes"}};
}

#ifdef __linux__
inline std::string host_symbol_owner(const void* address) {
  Dl_info info{};
  if (!address || !dladdr(address, &info) || !info.dli_fname)
    throw std::runtime_error("native runtime symbol owner unavailable");
  return std::filesystem::canonical(info.dli_fname).u8string();
}
#endif
} // namespace rns8::test
