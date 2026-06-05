#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace {

constexpr std::size_t kMaxJsonBytes = 4096;

void touch_metadata_shape(const nlohmann::json& value) {
  if (!value.is_object()) {
    return;
  }
  const auto backend = value.value("backend", std::string());
  const auto selected_backend = value.value("selected_backend", std::string());
  const auto schema_version = value.value("schema_version", 0);
  const auto semantic_contract = value.value("semantic_contract", nlohmann::json::object());
  const auto gpu_events = value.value("gpu_events", nlohmann::json::array());
  const auto grouped_dispatch = value.value("grouped_dispatch", nlohmann::json::object());
  const auto output_policy = value.value("output_policy", nlohmann::json::object());
  const auto metadata_registry = value.value("metadata_registry", nlohmann::json::object());

  volatile std::size_t sink = backend.size();
  sink ^= selected_backend.size();
  sink ^= static_cast<std::size_t>(schema_version);
  sink ^= semantic_contract.size();
  sink ^= gpu_events.size();
  sink ^= grouped_dispatch.size();
  sink ^= output_policy.size();
  sink ^= metadata_registry.size();
  (void)sink;
}

void fuzz_metadata_json(const uint8_t* data, std::size_t size) {
  const std::size_t clamped_size = std::min(size, kMaxJsonBytes);
  std::string text(reinterpret_cast<const char*>(data), clamped_size);
  nlohmann::json parsed = nlohmann::json::parse(text, nullptr, false);
  if (parsed.is_discarded()) {
    return;
  }
  touch_metadata_shape(parsed);
  if (parsed.is_array()) {
    const std::size_t limit = std::min<std::size_t>(parsed.size(), 32);
    for (std::size_t i = 0; i < limit; ++i) {
      touch_metadata_shape(parsed[i]);
    }
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
  fuzz_metadata_json(data, size);
  return 0;
}
