#pragma once

#include <array>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>

namespace rns8::tools {
namespace json_output_detail {
class FileBuffer final : public std::streambuf {
public:
  explicit FileBuffer(std::FILE* file) noexcept : file_(file) {
    setp(buffer_.data(), buffer_.data() + buffer_.size());
  }

protected:
  int_type overflow(int_type value) override {
    if (!drain())
      return traits_type::eof();
    if (!traits_type::eq_int_type(value, traits_type::eof())) {
      *pptr() = traits_type::to_char_type(value);
      pbump(1);
    }
    return traits_type::not_eof(value);
  }
  int sync() override { return drain() && std::fflush(file_) == 0 ? 0 : -1; }

private:
  bool drain() {
    const auto count = static_cast<std::size_t>(pptr() - pbase());
    if (count && std::fwrite(pbase(), 1, count, file_) != count)
      return false;
    setp(buffer_.data(), buffer_.data() + buffer_.size());
    return true;
  }
  std::FILE* file_;
  std::array<char, 8192> buffer_{};
};
struct Close {
  void operator()(std::FILE* file) const noexcept {
    if (file)
      std::fclose(file);
  }
};
} // namespace json_output_detail

// Reserve a fresh destination before work. Serialization owns its stream;
// concurrent profiler logging never shares the native JSON destination.
class JsonOutput {
public:
  explicit JsonOutput(const std::filesystem::path& path) : to_file_(!path.empty()) {
    if (!to_file_)
      return;
    std::FILE* opened = nullptr;
#if defined(_WIN32)
    if (_wfopen_s(&opened, path.c_str(), L"wbx") != 0)
      opened = nullptr;
#else
    opened = std::fopen(path.c_str(), "wbx");
#endif
    file_.reset(opened);
    if (!file_)
      throw std::runtime_error("JSON output needs a writable, previously absent file");
  }
  JsonOutput(const JsonOutput&) = delete;
  JsonOutput& operator=(const JsonOutput&) = delete;
  template <class Writer> void emit(Writer&& writer) {
    if (emitted_)
      throw std::runtime_error("native JSON output already consumed");
    emitted_ = true;
    if (!to_file_) {
      std::forward<Writer>(writer)(std::cout);
      return;
    }
    json_output_detail::FileBuffer buffer(file_.get());
    {
      std::ostream output(&buffer);
      std::forward<Writer>(writer)(output);
      output.flush();
      if (!output)
        throw std::runtime_error("native JSON output write failed");
    }
    if (std::fclose(file_.release()) != 0)
      throw std::runtime_error("native JSON output close failed");
  }

private:
  std::unique_ptr<std::FILE, json_output_detail::Close> file_;
  bool to_file_ = false, emitted_ = false;
};
template <class Writer>
void emit_json_file_or_stdout(const std::filesystem::path& path, Writer&& writer) {
  JsonOutput output(path);
  output.emit(std::forward<Writer>(writer));
}
} // namespace rns8::tools
