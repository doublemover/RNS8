#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <catch2/catch_test_macros.hpp>
#include "backend_common/sha256.hpp"

namespace {
std::filesystem::path module_path(HMODULE module) {
  std::array<wchar_t,32768> path{};
  const auto length=GetModuleFileNameW(module,path.data(),DWORD(path.size()));
  REQUIRE(length>0);REQUIRE(length<path.size());
  return std::filesystem::path(std::wstring(path.data(),length));
}
void require_selected_runtime(const char* name,const char* expected_hash) {
  CAPTURE(name);
  const auto module=GetModuleHandleA(name);REQUIRE(module!=nullptr);
  const auto path=module_path(module),executable=module_path(nullptr);
  INFO(path.u8string());
  REQUIRE(std::filesystem::equivalent(path.parent_path(),executable.parent_path()));
  std::ifstream input(path,std::ios::binary);REQUIRE(input.good());
  const std::string bytes{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
  REQUIRE_FALSE(input.bad());CHECK(rns8::detail::sha256(bytes)==expected_hash);
}
}

TEST_CASE("loaded HIP runtime matches the selected app-local SDK artifacts", "[hip][runtime][discovery]") {
  require_selected_runtime(RNS8_TEST_HIP_DLL_NAME,RNS8_TEST_HIP_DLL_SHA256);
#ifdef RNS8_TEST_KPACK_SHA256
  require_selected_runtime("rocm_kpack.dll",RNS8_TEST_KPACK_SHA256);
#endif
}
