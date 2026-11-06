#include "core/matrix_cleanup.hpp"
#include "core/export_metadata_transaction.hpp"
#include "core/host_readback_slot.hpp"
#include "core/host_transfer_payload.hpp"
#include "core/host_upload_packet.hpp"
#include "backend_wrap64/canonical_host_transfer.hpp"
#include "backend_common/completed_observation.hpp"
#include <array>
#include <memory>

TEST_CASE("failed completion cannot publish synchronized timing evidence",
          "[api][lifetime][evidence]") {
  using rns8::detail::publish_after_completion;
  unsigned drains = 0, samples = 0;
  int status = 0;
  const auto observe = [&] {
    return publish_after_completion(
        [&] {
          ++drains;
          return status;
        },
        [&] { ++samples; });
  };
  for (const int failure : {-1, 1, 3, INT32_MAX}) {
    status = failure;
    CHECK(observe() == failure);
    CHECK(samples == 0);
  }
  CHECK(drains == 4);
  status = 0;
  CHECK(observe() == 0);
  CHECK(drains == 5);
  CHECK(samples == 1);
  status = 3;
  CHECK(observe() == 3);
  CHECK(drains == 6);
  CHECK(samples == 1);
}

namespace {
struct MockDeviceBufferOperations {
  static inline void* fail_pointer = nullptr;
  static inline bool fail_all = false;
  static inline std::array<rns8::detail::hip_deferred_buffer_cleanup*, 8> deferred{};
  static inline unsigned deferred_count = 0;
  static rns8_status allocate(int, std::size_t bytes, void** out) {
    *out = new unsigned char[bytes];
    return RNS8_SUCCESS;
  }
  static rns8_status release(int, void* pointer) {
    if (fail_all || pointer == fail_pointer)
      return RNS8_BACKEND_FAILURE;
    delete[] static_cast<unsigned char*>(pointer);
    return RNS8_SUCCESS;
  }
  static void defer(rns8::detail::hip_deferred_buffer_cleanup* node) noexcept {
    deferred[deferred_count++] = node;
  }
};
} // namespace

TEST_CASE("device allocation owner retains failed frees and exception cleanup records",
          "[api][lifetime]") {
  using Buffer = rns8::detail::basic_hip_device_buffer<MockDeviceBufferOperations>;
  using Ops = MockDeviceBufferOperations;
  Ops::fail_pointer = nullptr;
  Ops::fail_all = false;
  Ops::deferred_count = 0;
  {
    Buffer a, b;
    void* pa = nullptr;
    void* pb = nullptr;
    std::size_t ba = 16, bb = 32;
    REQUIRE(a.allocate_alias(0, ba, pa) == RNS8_SUCCESS);
    REQUIRE(b.allocate_alias(0, bb, pb) == RNS8_SUCCESS);
    Ops::fail_pointer = pa;
    CHECK(a.reset_alias(pa, ba) == RNS8_BACKEND_FAILURE);
    CHECK(pa == a.get());
    CHECK(ba == 16);
    CHECK(b.reset_alias(pb, bb) == RNS8_SUCCESS);
    CHECK(pb == nullptr);
    CHECK(bb == 0);
    Ops::fail_pointer = nullptr;
    CHECK(a.reset_alias(pa, ba) == RNS8_SUCCESS);
    CHECK(pa == nullptr);
    CHECK(ba == 0);
  }
  try {
    Buffer packed, native;
    REQUIRE(packed.allocate(0, 16) == RNS8_SUCCESS);
    REQUIRE(native.allocate(0, 32) == RNS8_SUCCESS);
    Ops::fail_all = true;
    throw std::bad_alloc(); // Equivalent host staging failure, no HIP involved.
  } catch (const std::bad_alloc&) {
  }
  REQUIRE(Ops::deferred_count == 2);
  Ops::fail_all = false;
  for (unsigned i = 0; i < Ops::deferred_count; ++i) {
    auto* node = Ops::deferred[i];
    REQUIRE(Ops::release(node->device_id, node->pointer) == RNS8_SUCCESS);
    delete node;
    Ops::deferred[i] = nullptr;
  }
  Ops::deferred_count = 0;
}

TEST_CASE("tiled export replacement cannot certify partially uploaded metadata",
          "[api][lifetime]") {
  rns8_matrix matrix;
  for (unsigned fail_after = 0; fail_after < 4; ++fail_after) {
    REQUIRE(rns8::detail::replace_export_metadata(matrix, 1000, 1, 4096,
                                                  [] { return RNS8_SUCCESS; }) == RNS8_SUCCESS);
    std::array<unsigned, 4> device_metadata{2, 1000, 0, 0};
    const auto failed = rns8::detail::replace_export_metadata(matrix, 10, 1, 4096, [&] {
      for (unsigned stage = 0; stage < device_metadata.size(); ++stage) {
        device_metadata[stage] = stage == 0 ? 1 : 10;
        if (stage == fail_after)
          return RNS8_BACKEND_FAILURE;
      }
      return RNS8_SUCCESS;
    });
    REQUIRE(failed == RNS8_BACKEND_FAILURE);
    CHECK(matrix.hip_export_schedule_fingerprint == 0);
    CHECK(matrix.hip_export_tile_schedule_count == 0);
    CHECK(matrix.hip_export_tile_bounds_count == 0);
    // Repacking a value changes resident identity, not this independently invalid
    // metadata cache. Equal capacities cannot resurrect the old plan key.
    matrix.source_version = 7;
    CHECK(matrix.hip_export_schedule_fingerprint != 1000);
    for (uint64_t plan_key : {1000u, 10u, 1000u}) {
      REQUIRE(rns8::detail::replace_export_metadata(matrix, plan_key, 1, 4096,
                                                    [] { return RNS8_SUCCESS; }) == RNS8_SUCCESS);
      CHECK(matrix.hip_export_schedule_fingerprint == plan_key);
      CHECK(matrix.hip_export_tile_schedule_count == 1);
    }
  }
  CHECK(256 % 256 == 0);
  CHECK(256 % 255 == 1);
}

TEST_CASE("matrix cleanup retains only failed release ownership for retry", "[api][lifetime]") {
  // Pure bookkeeping fixture: these addresses are host markers, never HIP allocations.
  rns8_matrix matrix;
  std::array<int, 3> markers{};
  matrix.hip_upload_buffer = &markers[0];
  matrix.hip_upload_bytes = 16;
  matrix.hip_export_tile_schedule = &markers[1];
  matrix.hip_export_tile_schedule_bytes = 32;
  matrix.hip_export_tile_schedule_count = 2;
  matrix.hip_residues = &markers[2];
  matrix.hip_residue_bytes = 64;
  matrix.hip_export_schedule_fingerprint = 99;
  std::vector<void*> released;
  const auto first = rns8::detail::release_matrix_allocations(matrix, [&](int, void* pointer) {
    if (pointer == &markers[1])
      return RNS8_BACKEND_FAILURE;
    released.push_back(pointer);
    return RNS8_SUCCESS;
  });
  REQUIRE(first == RNS8_BACKEND_FAILURE);
  REQUIRE(released == std::vector<void*>{&markers[0], &markers[2]});
  CHECK(matrix.hip_upload_buffer == nullptr);
  CHECK(matrix.hip_upload_bytes == 0);
  CHECK(matrix.hip_residues == nullptr);
  CHECK(matrix.hip_residue_bytes == 0);
  CHECK(matrix.hip_export_tile_schedule == &markers[1]);
  CHECK(matrix.hip_export_tile_schedule_bytes == 32);
  CHECK(matrix.hip_export_tile_schedule_count == 2);
  CHECK(matrix.hip_export_schedule_fingerprint == 0);
  released.clear();
  REQUIRE(rns8::detail::release_matrix_allocations(matrix, [&](int, void* pointer) {
            released.push_back(pointer);
            return RNS8_SUCCESS;
          }) == RNS8_SUCCESS);
  REQUIRE(released == std::vector<void*>{&markers[1]});
  CHECK(matrix.hip_export_tile_schedule == nullptr);
  CHECK(matrix.hip_export_tile_schedule_bytes == 0);
  CHECK(matrix.hip_export_tile_schedule_count == 0);
  REQUIRE(rns8::detail::release_matrix_allocations(matrix, [](int, void*) {
            FAIL("an empty owner must never release an address");
            return RNS8_INTERNAL_ERROR;
          }) == RNS8_SUCCESS);
}

TEST_CASE("matrix destruction rejects active readers and writers without mutation",
          "[api][lifetime]") {
  auto* matrix = new rns8_matrix;
  matrix->in_flight_readers = 1;
  REQUIRE(rns8_destroy_matrix(matrix) == RNS8_BUSY);
  CHECK(matrix->host_residues_current);
  matrix->in_flight_readers = 0;
  matrix->in_flight_writer = true;
  REQUIRE(rns8_destroy_matrix(matrix) == RNS8_BUSY);
  CHECK(matrix->host_residues_current);
  matrix->in_flight_writer = false;
  REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
}

TEST_CASE("context destruction drains unpublished failed matrix owners", "[api][lifetime]") {
  auto* context = new rns8_context;
  auto* first = new rns8_matrix;
  auto* second = new rns8_matrix;
  first->failed_cleanup_next = second;
  context->failed_matrix_cleanup = first;
  context->hip_failure_poisoned = true;
  // Empty owners exercise the actual list drain without invoking the HIP runtime.
  context->prepared_references = 1;
  REQUIRE(rns8_destroy_context(context) == RNS8_BUSY);
  REQUIRE(context->failed_matrix_cleanup == first);
  context->prepared_references = 0;
  REQUIRE(rns8_destroy_context(context) == RNS8_SUCCESS);
}

TEST_CASE("pending device buffers defer instead of freeing unobserved GPU work",
          "[api][lifetime]") {
  using Buffer = rns8::detail::basic_hip_device_buffer<MockDeviceBufferOperations>;
  using Ops = MockDeviceBufferOperations;
  Ops::fail_pointer = nullptr;
  Ops::fail_all = false;
  Ops::deferred_count = 0;
  {
    Buffer buffer;
    REQUIRE(buffer.allocate(0, 64) == RNS8_SUCCESS);
    buffer.mark_pending();
    CHECK(buffer.reset() == RNS8_BUSY);
    Buffer moved = std::move(buffer);
    CHECK(moved.pending());
    CHECK_FALSE(buffer.pending());
  }
  REQUIRE(Ops::deferred_count == 1);
  auto* node = Ops::deferred[0];
  REQUIRE(Ops::release(node->device_id, node->pointer) == RNS8_SUCCESS);
  delete node;
  Ops::deferred[0] = nullptr;
  Ops::deferred_count = 0;
  {
    Buffer buffer;
    REQUIRE(buffer.allocate(0, 64) == RNS8_SUCCESS);
    buffer.mark_pending();
    buffer.mark_completed();
    CHECK(buffer.reset() == RNS8_SUCCESS);
  }
  CHECK(Ops::deferred_count == 0);
}

TEST_CASE("device retry records retain pageable DMA sources across unwind and failed free",
          "[api][lifetime]") {
  using Buffer = rns8::detail::basic_hip_device_buffer<MockDeviceBufferOperations>;
  using Ops = MockDeviceBufferOperations;
  Ops::fail_pointer = nullptr;
  Ops::fail_all = false;
  Ops::deferred_count = 0;
  std::weak_ptr<const void> retained;
  for (bool failed_free : {false, true}) {
    try {
      Buffer source;
      REQUIRE(source.allocate(0, 64) == RNS8_SUCCESS);
      auto payload = std::make_shared<std::vector<int>>(16, 123);
      retained = payload;
      REQUIRE(source.retain_host_payload(payload) == RNS8_SUCCESS);
      CHECK(source.retain_host_payload(payload) == RNS8_BUSY);
      payload.reset();
      if (failed_free)
        Ops::fail_all = true;
      else
        source.mark_pending();
      Buffer moved = std::move(source);
      CHECK_FALSE(retained.expired());
      CHECK(moved.reset() == (failed_free ? RNS8_BACKEND_FAILURE : RNS8_BUSY));
      throw std::bad_alloc();
    } catch (const std::bad_alloc&) {
    }
    REQUIRE(Ops::deferred_count == 1);
    CHECK_FALSE(retained.expired());
    Ops::fail_all = false;
    auto* node = Ops::deferred[0];
    REQUIRE(Ops::release(node->device_id, node->pointer) == RNS8_SUCCESS);
    delete node; // Simulated observed completion followed by successful release.
    Ops::deferred[0] = nullptr;
    Ops::deferred_count = 0;
    CHECK(retained.expired());
  }
  Buffer source;
  REQUIRE(source.allocate(0, 64) == RNS8_SUCCESS);
  {
    auto payload = std::make_shared<int>(7);
    retained = payload;
    REQUIRE(source.retain_host_payload(payload) == RNS8_SUCCESS);
  }
  source.mark_pending();
  source.release_host_payload(); // The copy completed; another kernel still owns the device buffer.
  CHECK(retained.expired());
  CHECK(source.pending());
  CHECK(source.reset() == RNS8_BUSY);
  source.mark_completed();
  CHECK(source.reset() == RNS8_SUCCESS);
}

TEST_CASE("repeated verifier uploads retain the current challenge after any failed trial",
          "[api][lifetime][finite-verify]") {
  using Buffer = rns8::detail::basic_hip_device_buffer<MockDeviceBufferOperations>;
  using Ops = MockDeviceBufferOperations;
  Ops::fail_pointer = nullptr;
  Ops::fail_all = false;
  Ops::deferred_count = 0;
  for (unsigned failed_trial : {0u, 1u, 2u}) {
    std::weak_ptr<std::vector<uint8_t>> retained;
    {
      Buffer device_challenge;
      REQUIRE(device_challenge.allocate(0, 3) == RNS8_SUCCESS);
      auto challenge = std::make_shared<std::vector<uint8_t>>(3);
      retained = challenge;
      for (unsigned trial = 0; trial <= failed_trial; ++trial) {
        (*challenge)[0] = uint8_t(trial);
        REQUIRE(device_challenge.retain_host_payload(challenge) == RNS8_SUCCESS);
        device_challenge.mark_pending();
        CHECK(challenge.use_count() == 2);
        if (trial != failed_trial) {
          device_challenge.mark_completed();
          CHECK(challenge.use_count() == 1);
        }
      }
      // Model unwinding after uncertain H2D/projection completion, without
      // touching a GPU or releasing a source still needed by a retry record.
    }
    REQUIRE(Ops::deferred_count == 1);
    auto* record = Ops::deferred[0];
    REQUIRE(record->host_payload);
    {
      const auto challenge = retained.lock();
      REQUIRE(challenge);
      CHECK((*challenge)[0] == failed_trial);
    }
    REQUIRE(Ops::release(record->device_id, record->pointer) == RNS8_SUCCESS);
    delete record; // A later observed drain can now retire the allocation and source.
    Ops::deferred[0] = nullptr;
    Ops::deferred_count = 0;
    CHECK(retained.expired());
  }
}

TEST_CASE("large owned readbacks retain their actual destination through failure and unwind",
          "[api][lifetime][export]") {
  using Payload = rns8::detail::HostTransferPayload;
  using Ops = MockDeviceBufferOperations;
  Ops::deferred_count = 0;
  for (bool unwinding : {false, true}) {
    std::weak_ptr<std::vector<uint64_t>> retained;
    try {
      Payload transfer(Ops::defer);
      auto destination = std::make_shared<std::vector<uint64_t>>(512, 77);
      retained = destination;
      REQUIRE(transfer.begin(7, std::shared_ptr<void>(destination, destination->data()),
                             destination->size() * sizeof(uint64_t)) == RNS8_SUCCESS);
      CHECK(transfer.data() == destination->data());
      CHECK(transfer.begin(7, std::shared_ptr<void>(destination, destination->data()), 8) ==
            RNS8_BUSY);
      destination.reset();
      if (unwinding)
        throw std::bad_alloc();
    } catch (const std::bad_alloc&) {
    }
    REQUIRE(Ops::deferred_count == 1);
    auto* record = Ops::deferred[0];
    CHECK(record->device_id == 7);
    CHECK(record->kind == rns8::detail::hip_deferred_allocation_kind::host_payload);
    {
      const auto destination = retained.lock();
      REQUIRE(destination);
      REQUIRE(record->pointer == destination->data());
      // A simulated late DMA writes only the retained allocation, after its
      // original stack owner and transfer scope have gone away.
      std::fill_n(static_cast<uint64_t*>(record->pointer), destination->size(), uint64_t(123));
      CHECK(destination->front() == 123);
      CHECK(destination->back() == 123);
    }
    delete record; // Model a later proven completion; no HIP resource release.
    Ops::deferred[0] = nullptr;
    Ops::deferred_count = 0;
    CHECK(retained.expired());
  }
  {
    Payload transfer(Ops::defer);
    auto destination = std::make_shared<uint64_t>(9);
    CHECK(transfer.begin(-1, destination, 8) == RNS8_INVALID_ARGUMENT);
    CHECK(transfer.begin(7, destination, 0) == RNS8_INVALID_ARGUMENT);
    CHECK(transfer.begin(7, {}, 8) == RNS8_INVALID_ARGUMENT);
    REQUIRE(transfer.begin(7, destination, 8) == RNS8_SUCCESS);
    transfer.complete();
    CHECK(transfer.data() == nullptr);
    CHECK(destination.use_count() == 1);
  }
  CHECK(Ops::deferred_count == 0);
}

TEST_CASE("private converted upload sources survive the original call scope",
          "[api][lifetime][import]") {
  using Ops = MockDeviceBufferOperations;
  Ops::deferred_count = 0;
  std::weak_ptr<std::vector<int8_t>> retained;
  {
    rns8::detail::HostTransferPayload transfer(Ops::defer);
    auto converted = std::make_shared<std::vector<int8_t>>(4096, int8_t(-17));
    retained = converted;
    REQUIRE(transfer.begin(3, std::shared_ptr<void>(converted, converted->data()),
                           converted->size()) == RNS8_SUCCESS);
    converted.reset();
    // An uncertain upload returns without successful observation.
  }
  REQUIRE(Ops::deferred_count == 1);
  auto* record = Ops::deferred[0];
  {
    const auto converted = retained.lock();
    REQUIRE(converted);
    REQUIRE(record->pointer == converted->data());
    int64_t sum = 0;
    const auto* bytes = static_cast<const int8_t*>(record->pointer);
    for (size_t i = 0; i < converted->size(); ++i)
      sum += bytes[i]; // Model the DMA's remaining source reads after stack exit.
    CHECK(sum == -69632);
    CHECK(record->device_id == 3);
  }
  delete record; // A later completion observation can release the source.
  Ops::deferred[0] = nullptr;
  Ops::deferred_count = 0;
  CHECK(retained.expired());
}

TEST_CASE("pageable readback cache reuses complete storage and isolates failed destinations",
          "[api][lifetime][export]") {
  using Ops = MockDeviceBufferOperations;
  Ops::deferred_count = 0;
  rns8::detail::HostTransferStorage cache;
  CHECK(cache.reserve(0) == RNS8_INVALID_ARGUMENT);
  REQUIRE(cache.reserve(4096) == RNS8_SUCCESS);
  auto* first = cache.data();
  REQUIRE(cache.reserve(128) == RNS8_SUCCESS);
  CHECK(cache.data() == first);
  CHECK(cache.capacity() == 4096);
  {
    rns8::detail::HostTransferPayload pending(Ops::defer);
    REQUIRE(pending.begin(7, cache.owner(), 4096) == RNS8_SUCCESS);
    CHECK(cache.reserve(128) == RNS8_BUSY);
    CHECK(cache.reserve(8192) == RNS8_BUSY);
    cache.discard(); // An uncertain DMA cannot share storage with the next export.
    CHECK(cache.data() == nullptr);
    CHECK(cache.capacity() == 0);
    REQUIRE(cache.reserve(4096) == RNS8_SUCCESS);
    REQUIRE(cache.data() != first);
    std::fill_n(static_cast<unsigned char*>(pending.data()), 4096, 123);
    std::fill_n(static_cast<unsigned char*>(cache.data()), 4096, 77);
  }
  REQUIRE(Ops::deferred_count == 1);
  auto* record = Ops::deferred[0];
  REQUIRE(record->pointer == first);
  CHECK(static_cast<unsigned char*>(record->pointer)[4095] == 123);
  CHECK(static_cast<unsigned char*>(cache.data())[4095] == 77);
  delete record; // A later observed drain releases only the old destination.
  Ops::deferred[0] = nullptr;
  Ops::deferred_count = 0;
  cache.trim(4096);
  CHECK(cache.capacity() == 4096);
  cache.trim(1024);
  CHECK(cache.capacity() == 0);
  CHECK(cache.data() == nullptr);
}

TEST_CASE("borrowed upload snapshots compact only logical rows and retain uncertain source bytes",
          "[api][lifetime][upload]") {
  using namespace rns8::detail;
  using Ops = MockDeviceBufferOperations;
  Ops::deferred_count = 0;
  HostTransferStorage cache;
  HostUploadPlan plan;
  REQUIRE(make_host_upload_plan(3, 2, 5, sizeof(uint64_t), plan));
  // No trailing padding is allocated for the final row.
  std::array<uint64_t, 12> source{11, 12, 99, 99, 99, 21, 22, 99, 99, 99, 31, 32};
  const std::array<uint64_t, 6> expected{11, 12, 21, 22, 31, 32};
  REQUIRE(cache.snapshot(source.data(), plan) == RNS8_SUCCESS);
  auto* first = cache.data();
  CHECK(std::memcmp(first, expected.data(), sizeof(expected)) == 0);
  source.fill(123); // Caller storage may change after the synchronous API returns.
  {
    HostTransferPayload pending(Ops::defer);
    REQUIRE(pending.begin(2, cache.owner(), plan.device_bytes) == RNS8_SUCCESS);
    CHECK(cache.snapshot(source.data(), plan) == RNS8_BUSY);
    CHECK(std::memcmp(first, expected.data(), sizeof(expected)) == 0);
    cache.discard(); // Model failed completion without inducing a device fault.
    REQUIRE(cache.snapshot(source.data(), plan) == RNS8_SUCCESS);
    CHECK(cache.data() != first);
    CHECK(std::memcmp(pending.data(), expected.data(), sizeof(expected)) == 0);
  }
  REQUIRE(Ops::deferred_count == 1);
  REQUIRE(Ops::deferred[0]->device_id == 2);
  CHECK(std::memcmp(Ops::deferred[0]->pointer, expected.data(), sizeof(expected)) == 0);
  delete Ops::deferred[0];
  Ops::deferred[0] = nullptr;
  Ops::deferred_count = 0;

  // Completed transfers can reuse capacity; invalid requests preserve it.
  HostUploadPlan compact;
  REQUIRE(make_host_upload_plan(1, 6, INT64_MAX, sizeof(uint64_t), compact));
  REQUIRE(cache.snapshot(expected.data(), compact) == RNS8_SUCCESS);
  auto* reused = cache.data();
  CHECK(cache.snapshot(cache.data(), compact) == RNS8_INVALID_ARGUMENT);
  CHECK(cache.snapshot(nullptr, compact) == RNS8_INVALID_ARGUMENT);
  for (unsigned field = 0; field < 5; ++field) {
    auto invalid = plan;
    if (field == 0)
      invalid.rows = SIZE_MAX;
    if (field == 1)
      invalid.row_bytes = SIZE_MAX;
    if (field == 2)
      invalid.source_pitch_bytes = 1;
    if (field == 3)
      --invalid.source_span_bytes;
    if (field == 4)
      --invalid.device_bytes;
    CHECK(cache.snapshot(source.data(), invalid) == RNS8_INVALID_ARGUMENT);
    CHECK(cache.data() == reused);
    CHECK(std::memcmp(cache.data(), expected.data(), sizeof(expected)) == 0);
  }
  {
    HostTransferPayload completed(Ops::defer);
    REQUIRE(completed.begin(2, cache.owner(), compact.device_bytes) == RNS8_SUCCESS);
    completed.complete();
  }
  CHECK(Ops::deferred_count == 0);
  REQUIRE(cache.snapshot(expected.data(), compact) == RNS8_SUCCESS);
  CHECK(cache.data() == reused);
  cache.trim(sizeof(expected) - 1);
  CHECK(cache.data() == nullptr);
}

TEST_CASE("workspace upload packets preserve offsets padding and independent source ownership",
          "[api][workspace][lifetime]") {
  using rns8::detail::make_host_upload_packet;
  std::array<unsigned char, 3> first{1, 2, 3};
  const std::array<unsigned char, 2> last{4, 5};
  std::shared_ptr<void> packet;
  REQUIRE(make_host_upload_packet(16, {{8, last.data(), 2}, {0, first.data(), 3}, {0, nullptr, 0}},
                                  packet) == RNS8_SUCCESS);
  first.fill(9); // Device preparation must not borrow the caller's vectors.
  const auto* bytes = static_cast<const unsigned char*>(packet.get());
  CHECK(bytes[0] == 1);
  CHECK(bytes[2] == 3);
  CHECK(bytes[8] == 4);
  CHECK(bytes[9] == 5);
  for (size_t i : {3u, 4u, 5u, 6u, 7u, 10u, 11u, 12u, 13u, 14u, 15u})
    CHECK(bytes[i] == 0);
  const auto original = packet.get();
  const auto rejected = [&](std::initializer_list<rns8::detail::HostUploadRegion> regions) {
    CHECK(make_host_upload_packet(16, regions, packet) == RNS8_INVALID_ARGUMENT);
    CHECK(packet.get() == original);
    CHECK(bytes[0] == 1);
  };
  rejected({{0, nullptr, 1}});
  rejected({{17, first.data(), 0}});
  rejected({{15, first.data(), 3}});
  rejected({{0, first.data(), 3}, {2, last.data(), 2}});
  rejected({{0, nullptr, 0}});
  CHECK(make_host_upload_packet(0, {{0, first.data(), 3}}, packet) == RNS8_INVALID_ARGUMENT);
  CHECK(make_host_upload_packet(SIZE_MAX, {{0, first.data(), 3}}, packet) == RNS8_RANGE_ERROR);
  CHECK(packet.get() == original);
}

TEST_CASE("canonical wrap transfer payloads preserve late encoded reads and writes",
          "[api][wrap64][lifetime]") {
  using Ops = MockDeviceBufferOperations;
  Ops::deferred_count = 0;
  const std::array<uint64_t, 8> words{0,
                                      1,
                                      UINT64_MAX,
                                      uint64_t(1) << 63,
                                      UINT64_C(0x8080808080808080),
                                      UINT64_C(0x0123456789abcdef),
                                      255,
                                      256};
  for (bool readback : {false, true}) {
    {
      auto canonical = std::shared_ptr<uint8_t[]>(new uint8_t[sizeof(words)]);
      if (!readback)
        for (size_t i = 0; i < words.size(); ++i)
          rns8::detail::store_canonical_u64(canonical.get() + 8 * i, words[i]);
      rns8::detail::HostTransferPayload transfer(Ops::defer);
      REQUIRE(transfer.begin(0, std::shared_ptr<void>(canonical, canonical.get()), sizeof(words)) ==
              RNS8_SUCCESS);
    }
    REQUIRE(Ops::deferred_count == 1);
    auto* record = Ops::deferred[0];
    auto* bytes = static_cast<uint8_t*>(record->pointer);
    if (readback)
      for (size_t i = 0; i < words.size(); ++i)
        rns8::detail::store_canonical_u64(bytes + 8 * i, words[i]);
    for (size_t i = 0; i < words.size(); ++i)
      CHECK(rns8::detail::load_canonical_u64(bytes + 8 * i) == words[i]);
    delete record; // Simulated later observation; no device fault or DMA.
    Ops::deferred[0] = nullptr;
    Ops::deferred_count = 0;
  }
  auto source = std::make_shared<uint64_t>(7);
  uint64_t untouched = 99;
  for (const char* label : {static_cast<const char*>(nullptr), ""})
    CHECK(rns8::detail::hip_direct_copy_owned_host_to_device(
              0, &untouched, source, sizeof(untouched), label) == RNS8_INVALID_ARGUMENT);
  CHECK(untouched == 99);
}

TEST_CASE("small status readbacks publish only after completion and retain failed destinations",
          "[api][lifetime]") {
  using Slot = rns8::detail::HostReadbackSlot;
  using Ops = MockDeviceBufferOperations;
  Ops::deferred_count = 0;
  Slot slot(Ops::defer);
  CHECK(slot.begin(-1, 4) == RNS8_INVALID_ARGUMENT);
  CHECK(slot.begin(3, 0) == RNS8_INVALID_ARGUMENT);
  CHECK(slot.begin(3, Slot::capacity + 1) == RNS8_INVALID_ARGUMENT);
  uint32_t output = 0xdeadbeef;
  REQUIRE(slot.begin(3, sizeof(output)) == RNS8_SUCCESS);
  const uint32_t computed = 123;
  std::memcpy(slot.data(), &computed, sizeof(computed));
  CHECK(slot.begin(3, sizeof(output)) == RNS8_BUSY);
  CHECK(slot.complete(nullptr) == RNS8_INVALID_ARGUMENT);
  slot.fail();
  CHECK(output == 0xdeadbeef);
  REQUIRE(Ops::deferred_count == 1);
  auto* failed = Ops::deferred[0];
  CHECK(failed->kind == rns8::detail::hip_deferred_allocation_kind::host_payload);
  CHECK(failed->device_id == 3);
  std::weak_ptr<const void> retained = failed->host_payload;
  REQUIRE(slot.begin(3, sizeof(output)) == RNS8_SUCCESS);
  CHECK(slot.data() != failed->pointer);
  std::memcpy(failed->pointer, &computed, sizeof(computed)); // Simulated late copy.
  CHECK(output == 0xdeadbeef);
  auto* reused = slot.data();
  std::memcpy(slot.data(), &computed, sizeof(computed));
  REQUIRE(slot.complete(&output) == RNS8_SUCCESS);
  CHECK(output == computed);
  REQUIRE(slot.begin(3, sizeof(uint64_t)) == RNS8_SUCCESS);
  CHECK(slot.data() == reused);
  const uint64_t boundary = UINT64_MAX;
  std::memcpy(slot.data(), &boundary, sizeof(boundary));
  uint64_t wide = 0;
  REQUIRE(slot.complete(&wide) == RNS8_SUCCESS);
  CHECK(wide == boundary);
  delete failed; // The existing completion owner, not the slot, authorizes this.
  Ops::deferred[0] = nullptr;
  Ops::deferred_count = 0;
  CHECK(retained.expired());
  for (const auto bytes : {2 * sizeof(uint64_t), 4 * sizeof(uint64_t)}) {
    std::array<uint64_t, 4> output_words{9, 9, 9, 9};
    const auto unchanged_words = output_words;
    const std::array<uint64_t, 4> computed_words{0, UINT64_MAX, UINT64_C(1) << 63, 127};
    REQUIRE(slot.begin(3, bytes) == RNS8_SUCCESS);
    std::memcpy(slot.data(), computed_words.data(), bytes);
    slot.fail();
    REQUIRE(Ops::deferred_count == 1);
    auto* delayed = Ops::deferred[0];
    REQUIRE(slot.begin(3, bytes) == RNS8_SUCCESS);
    CHECK(slot.data() != delayed->pointer);
    std::memcpy(delayed->pointer, computed_words.data(), bytes);
    CHECK(output_words == unchanged_words);
    std::memcpy(slot.data(), computed_words.data(), bytes);
    REQUIRE(slot.complete(output_words.data()) == RNS8_SUCCESS);
    for (size_t i = 0; i < output_words.size(); ++i)
      CHECK(output_words[i] == (i < bytes / sizeof(uint64_t) ? computed_words[i] : 9));
    delete delayed;
    Ops::deferred[0] = nullptr;
    Ops::deferred_count = 0;
  }
  try {
    Slot unwinding(Ops::defer);
    REQUIRE(unwinding.begin(4, 4) == RNS8_SUCCESS);
    throw std::bad_alloc();
  } catch (const std::bad_alloc&) {
  }
  REQUIRE(Ops::deferred_count == 1);
  CHECK(Ops::deferred[0]->device_id == 4);
  delete Ops::deferred[0];
  Ops::deferred[0] = nullptr;
  Ops::deferred_count = 0;
}
