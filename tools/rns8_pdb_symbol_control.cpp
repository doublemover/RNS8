// Test-only optimized symbol/line control, separate from every benchmark.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

#ifndef RNS8_PDB_CONTROL_SOURCE_SHA256
#error "CMake must bind the native PDB control source"
#endif

constexpr uint32_t control_first_line = __LINE__;
extern "C" __declspec(dllexport) __declspec(noinline) uint64_t
rns8_pdb_symbol_control(uint64_t value) {
  return value * UINT64_C(0x9e3779b97f4a7c15) + UINT64_C(0x123456789abcdef);
}
constexpr uint32_t control_last_line = __LINE__;

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) != "--describe")
    return 2;
  const auto module = GetModuleHandleW(nullptr);
  if (!module)
    return 2;
  const auto* bytes = reinterpret_cast<const unsigned char*>(module);
  const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(bytes);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0)
    return 2;
  const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(bytes + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE)
    return 2;
  const uintptr_t base = reinterpret_cast<uintptr_t>(module);
  const uintptr_t address = reinterpret_cast<uintptr_t>(&rns8_pdb_symbol_control);
  if (address < base || address - base >= nt->OptionalHeader.SizeOfImage)
    return 2;
  std::cout << nlohmann::json{{"schema", "rns8_native_pdb_source_control_v1"},
                              {"process_id", GetCurrentProcessId()},
                              {"pointer_bytes", sizeof(void*)},
                              {"source_sha256", RNS8_PDB_CONTROL_SOURCE_SHA256},
                              {"rva", address - base},
                              {"image_size", nt->OptionalHeader.SizeOfImage},
                              {"source_file", __FILE__},
                              {"first_line", control_first_line},
                              {"last_line", control_last_line},
                              {"symbol_name", "rns8_pdb_symbol_control"},
                              {"test_only", true},
                              {"benchmark_executed", false}}
                   .dump()
            << '\n';
  return 0;
}
