#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "rns8/rns8.h"

namespace rns8::bench {

uint64_t elapsed_us(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end);

[[noreturn]] void usage_error(const std::string& message);

void fail_status(const char* label, rns8_status status);

void fail_hip_runtime(const char* label, int status);

void mix_checksum(uint64_t& checksum, uint64_t value);

std::string json_escape(const std::string& input);

std::string command_line(int argc, char** argv);

std::string trim_ascii_whitespace(std::string value);

std::string runtime_git_commit();

std::string compiler_id();

std::string compiler_version();

void print_nullable_string(const char* value);

void print_json_string_or_null(const char* value);

std::size_t checked_elements(int64_t rows, int64_t cols, const char* label);

std::size_t checked_limb_elements(int64_t rows, int64_t cols, uint32_t limb_count, const char* label);

std::size_t checked_bytes(std::size_t elements, std::size_t element_size, const char* label);

std::size_t checked_add_bytes(std::size_t lhs, std::size_t rhs, const char* label);

}  // namespace rns8::bench
