#pragma once

#include "backend_common/sha256.hpp"
#include <array>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <nlohmann/json.hpp>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rns8::tools {

struct ExecutionHostSnapshot {
  const char* status = "unavailable";
  std::string platform, source, os_instance_sha256;
  uint32_t native_status = 0;

  static ExecutionHostSnapshot from_material(std::string platform, std::string source,
                                             std::string_view material) {
    ExecutionHostSnapshot value;
    value.platform = std::move(platform);
    value.source = std::move(source);
    while (!material.empty() && (material.back() == '\n' || material.back() == '\r'))
      material.remove_suffix(1);
    if (material.size() == 38 && material.front() == '{' && material.back() == '}') {
      material.remove_prefix(1);
      material.remove_suffix(1);
    }
    if (material.size() != 32 && material.size() != 36)
      return value;
    std::string canonical;
    canonical.reserve(32);
    bool nonzero = false;
    for (std::size_t i = 0; i < material.size(); ++i) {
      if (material.size() == 36 && (i == 8 || i == 13 || i == 18 || i == 23)) {
        if (material[i] != '-')
          return value;
        continue;
      }
      char c = material[i];
      if (c >= 'A' && c <= 'F')
        c = static_cast<char>(c - 'A' + 'a');
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
        return value;
      nonzero = nonzero || c != '0';
      canonical.push_back(c);
    }
    if (!nonzero || canonical.size() != 32)
      return value;
    value.os_instance_sha256 = detail::hmac_sha256("RNS8 benchmark observed OS instance v1",
                                                   value.platform + "\n" + canonical);
    value.status = "observed";
    return value;
  }

  static ExecutionHostSnapshot observe() {
#if defined(_WIN32)
    ExecutionHostSnapshot value;
    value.platform = "windows";
    value.source = "native_registry_machine_guid_64_view";
    HMODULE module = LoadLibraryExW(L"advapi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module) {
      value.native_status = GetLastError();
      return value;
    }
    const auto query =
        reinterpret_cast<decltype(&RegGetValueW)>(GetProcAddress(module, "RegGetValueW"));
    std::array<wchar_t, 128> bytes{};
    DWORD size = static_cast<DWORD>(sizeof(bytes)), type = 0;
    const LSTATUS status =
        query ? query(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Cryptography", L"MachineGuid",
                      RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY, &type, bytes.data(), &size)
              : ERROR_PROC_NOT_FOUND;
    FreeLibrary(module);
    value.native_status = static_cast<uint32_t>(status);
    if (status != ERROR_SUCCESS || type != REG_SZ || size < sizeof(wchar_t) ||
        size > sizeof(bytes) || size % sizeof(wchar_t))
      return value;
    const auto count = size / sizeof(wchar_t);
    if (bytes[count - 1] != L'\0')
      return value;
    std::string material;
    for (std::size_t i = 0; i + 1 < count; ++i) {
      if (bytes[i] > 127 || !bytes[i])
        return value;
      material.push_back(static_cast<char>(bytes[i]));
    }
    return from_material(value.platform, value.source, material);
#elif defined(__linux__)
    ExecutionHostSnapshot value;
    value.platform = "linux";
    for (const auto path : {"/etc/machine-id", "/var/lib/dbus/machine-id"}) {
      errno = 0;
      std::ifstream input(path, std::ios::binary);
      if (!input) {
        value.native_status = static_cast<uint32_t>(errno);
        continue;
      }
      value.native_status = 0;
      std::array<char, 128> bytes{};
      input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      value.source = path;
      if (!input.eof() || input.bad())
        return value;
      return from_material(
          value.platform, value.source,
          std::string_view(bytes.data(), static_cast<std::size_t>(input.gcount())));
    }
    return value;
#else
    ExecutionHostSnapshot value;
    value.status = "unsupported_platform";
    value.platform = "unsupported";
    return value;
#endif
  }

  void print(std::ostream& out) const {
    out << nlohmann::json{
        {"schema", "rns8_execution_host_identity_v1"},
        {"scope", "observed_os_instance"},
        {"status", status},
        {"platform", platform},
        {"source", source},
        {"native_status", native_status},
        {"os_instance_sha256",
         os_instance_sha256.empty() ? nlohmann::json(nullptr) : nlohmann::json(os_instance_sha256)},
    };
  }
};

} // namespace rns8::tools
