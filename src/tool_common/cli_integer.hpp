#ifndef RNS8_CLI_INTEGER_HPP
#define RNS8_CLI_INTEGER_HPP
#include <charconv>
#include <cstdint>
#include <limits>
#include <string_view>
#include <system_error>

namespace rns8::cli {
// A complete decimal token, optionally prefixed by minus. No whitespace,
// base prefixes, saturation or partial conversion; failure preserves output.
inline bool parse_i64(const char* text,int64_t& out) noexcept {
  if(!text || !*text)return false;
  const std::string_view token(text);
  int64_t value=0;
  const auto parsed=std::from_chars(token.data(),token.data()+token.size(),value,10);
  if(parsed.ec!=std::errc{} || parsed.ptr!=token.data()+token.size())return false;
  out=value;return true;
}
inline bool parse_device(const char* text,int& out) noexcept {
  int64_t value=0;
  if(!parse_i64(text,value) || value < -1 || value > std::numeric_limits<int>::max())return false;
  out=static_cast<int>(value);return true;
}
}
#endif
