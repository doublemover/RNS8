#include "tool_common/execution_environment_json.hpp"
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rns8::tools::environment_observation_detail {
std::string utf8(const char* text, std::size_t size) {
#if defined(_WIN32)
  if (!size)
    return {};
  if (size > INT_MAX)
    throw std::runtime_error("environment observation string is too large");
  const auto count = static_cast<int>(size);
  const int wide_size = MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS, text, count, nullptr, 0);
  if (!wide_size)
    throw std::runtime_error("native environment encoding is unavailable");
  std::wstring wide(static_cast<std::size_t>(wide_size), L'\0');
  if (MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS, text, count, wide.data(), wide_size) !=
      wide_size)
    throw std::runtime_error("native environment conversion failed");
  const int encoded_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(),
                                               wide_size, nullptr, 0, nullptr, nullptr);
  if (!encoded_size)
    throw std::runtime_error("native environment UTF-8 conversion failed");
  std::string result(static_cast<std::size_t>(encoded_size), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), wide_size, result.data(),
                          encoded_size, nullptr, nullptr) != encoded_size)
    throw std::runtime_error("native environment UTF-8 conversion failed");
  return result;
#else
  std::string result(text, size);
  (void)nlohmann::json(result).dump(); // Validate UTF-8 before any workload begins.
  return result;
#endif
}
} // namespace rns8::tools::environment_observation_detail
