#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace {

constexpr std::size_t kMaxJsonBytes = 4096;

// Dependency parser/typed-access smoke only. Production Python capture
// admission requires the separate property driver, not this adapter.
void touch_metadata_shape(const nlohmann::json& value) {
  if (!value.is_object()) {
    return;
  }
  volatile std::size_t sink = 0;
  for (const char* key : {"backend", "selected_backend", "schema_version", "semantic_contract",
                          "gpu_events", "grouped_dispatch", "output_policy", "metadata_registry"}) {
    const auto item = value.find(key);
    if (item != value.end()) {
      // size() is defined for every JSON type. No harness-only unchecked
      // value<T>() conversion can turn a valid wrong-type input into a crash.
      sink ^= item->size();
    }
  }
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
