#include "tool_common/cli_integer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <string>

TEST_CASE("CLI integers reject incomplete tokens and overflow without changing the output") {
  for(const auto* text:{"","junk","1junk"," 1","1 ","+1","+-1","0x10","9223372036854775808","-9223372036854775809"}) {
    int64_t value=73;CHECK_FALSE(rns8::cli::parse_i64(text,value));CHECK(value==73);
  }
  int64_t value=73;CHECK_FALSE(rns8::cli::parse_i64(nullptr,value));CHECK(value==73);
  REQUIRE(rns8::cli::parse_i64("9223372036854775807",value));CHECK(value==INT64_MAX);
  REQUIRE(rns8::cli::parse_i64("-9223372036854775808",value));CHECK(value==INT64_MIN);
  REQUIRE(rns8::cli::parse_i64("-0",value));CHECK(value==0);
}
TEST_CASE("CLI device ordinals admit only the explicit sentinel or representable nonnegative ints") {
  for(const auto* text:{"","not-a-number","1junk","-2","-2147483648","2147483648","4294967296","9223372036854775808","-9223372036854775809"}) {
    int value=73;CHECK_FALSE(rns8::cli::parse_device(text,value));CHECK(value==73);
  }
  for(int expected:{-1,0,1,std::numeric_limits<int>::max()}) {
    int value=73;const auto text=std::to_string(expected);
    REQUIRE(rns8::cli::parse_device(text.c_str(),value));CHECK(value==expected);
  }
}
