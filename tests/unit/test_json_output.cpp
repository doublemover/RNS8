#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "tool_common/json_output.hpp"

namespace {
struct OutputFiles {
  std::filesystem::path directory, native, partial;
  OutputFiles() {
    const auto root = std::filesystem::current_path() / "temp";
    std::filesystem::create_directories(root);
    directory =
        root / ("json-output-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("unique test output directory unavailable");
    native = directory / "native.json";
    partial = directory / "partial.json";
  }
  ~OutputFiles() {
    std::error_code ignored;
    std::filesystem::remove(native, ignored);
    std::filesystem::remove(partial, ignored);
    std::filesystem::remove(directory, ignored);
  }
};
std::string read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
} // namespace

TEST_CASE("native JSON file output preserves evidence without redirecting stdout", "[cpu][tools]") {
  OutputFiles files;
  auto* original = std::cout.rdbuf();
  const std::string payload = "{\"authored\":\"" + std::string(20000, 'a') + "\"}\n";
  rns8::tools::emit_json_file_or_stdout(files.native, [&](std::ostream& output) {
    CHECK(std::cout.rdbuf() == original);
    output << payload;
  });
  CHECK(std::cout.rdbuf() == original);
  CHECK(read(files.native) == payload);
  CHECK_THROWS_AS(rns8::tools::emit_json_file_or_stdout(
                      files.native, [](std::ostream& output) { output << "must not overwrite"; }),
                  std::runtime_error);
  CHECK(read(files.native) == payload);
  CHECK_THROWS_AS(rns8::tools::emit_json_file_or_stdout(files.partial,
                                                        [](std::ostream& output) {
                                                          output << "{\"partial\":";
                                                          throw std::runtime_error(
                                                              "authored writer failure");
                                                        }),
                  std::runtime_error);
  CHECK(std::cout.rdbuf() == original);
}
