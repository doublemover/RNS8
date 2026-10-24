#include "backend_common/native_wmma_queue_plan.hpp"
#include <catch2/catch_test_macros.hpp>
#include <set>
#include <tuple>

using namespace rns8::detail;

TEST_CASE("native queue specialization requires uniform groups and signedness") {
  CHECK(native_wmma_uniform_group({}) == 0);
  for (uint32_t group : {1u, 2u, 4u}) {
    for (uint32_t unsigned_inputs : {0u, 1u}) {
      NativeWmmaTask first{}, second{};
      first.group = second.group = group;
      first.unsigned_inputs = second.unsigned_inputs = unsigned_inputs;
      first.m = 17; second.m = 3;
      first.prefix = 20; second.prefix = 1;
      CHECK(native_wmma_uniform_group({first}) == group);
      CHECK(native_wmma_uniform_group({first, second}) == group);
      second.unsigned_inputs ^= 1;
      CHECK(native_wmma_uniform_group({first, second}) == 0);
      second.unsigned_inputs = unsigned_inputs;
      second.group = group == 1 ? 2 : 1;
      CHECK(native_wmma_uniform_group({first, second}) == 0);
      first.group = 3;
      CHECK(native_wmma_uniform_group({first}) == 0);
      first.group = group; first.unsigned_inputs = 2;
      CHECK(native_wmma_uniform_group({first}) == 0);
    }
  }
}

TEST_CASE("native queue bounds offset K fragments through the last accessed element") {
  uint64_t allocation[12]{}, b = 1;
  NativeWmmaTask task{allocation + 3, &b, reinterpret_cast<int8_t*>(allocation + 8),
                      2, 1, 1, 4, 1, 1, 1, 1, 0};
  std::vector<NativeWmmaWork> work;
  uint64_t bytes = 0;
  REQUIRE(build_native_wmma_queue({task}, 1, UINT64_MAX, work, bytes) == RNS8_SUCCESS);
  task.c = reinterpret_cast<int8_t*>(allocation + 7);
  REQUIRE(build_native_wmma_queue({task}, 1, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("native WMMA queue covers heterogeneous tiles exactly once") {
  std::vector<uint64_t> a(33 * 19), b(19 * 35);
  std::vector<int8_t> c(20 * 33 * 35), d(c.size()), e(c.size());
  std::vector<NativeWmmaTask> tasks = {
      {a.data(), b.data(), c.data(), 33, 35, 19, 19, 35, 35, 20, 4, 0},
      {a.data(), b.data(), d.data(), 17, 21, 19, 19, 35, 35, 3, 2, 1},
      {a.data(), b.data(), e.data(), 1, 1, 19, 19, 35, 35, 1, 1, 0}};
  std::vector<NativeWmmaWork> work;
  uint64_t bytes = 0;
  REQUIRE(build_native_wmma_queue(tasks, 1024, UINT64_MAX, work, bytes) == RNS8_SUCCESS);
  REQUIRE(work.size() == 45 + 8 + 1);
  CHECK(work[0].task == 0);
  CHECK(work[1].task == 1);
  CHECK(work[2].task == 2);
  std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> expected, actual;
  for (uint32_t id = 0; id < tasks.size(); ++id) {
    const auto& t = tasks[id];
    for (uint32_t row = 0; row < (t.m + 15) / 16; ++row)
      for (uint32_t col = 0; col < (t.n + 15) / 16; ++col)
        for (uint32_t p = 0; p < t.prefix; p += t.group) expected.emplace(id, row, col, p);
  }
  for (const auto& w : work) REQUIRE(actual.emplace(w.task, w.tile_row, w.tile_col, w.first_plane).second);
  CHECK(actual == expected);
  CHECK(bytes == tasks.size() * sizeof(NativeWmmaTask) + work.size() * sizeof(NativeWmmaWork) + 4);
  const auto original_bytes = bytes;
  REQUIRE(build_native_wmma_queue(tasks, 1, bytes - 1, work, bytes) == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(bytes == original_bytes);
  CHECK(work.size() == expected.size());
  for (uint32_t workers : {0u, 1025u})
    CHECK(build_native_wmma_queue(tasks, workers, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
  tasks[1].c = tasks[0].c + 1;
  CHECK(build_native_wmma_queue(tasks, 1, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
  tasks[1].c = reinterpret_cast<int8_t*>(a.data());
  CHECK(build_native_wmma_queue(tasks, 1, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
  tasks[1].c = d.data();
  tasks[0].group = 3;
  CHECK(build_native_wmma_queue(tasks, 1, UINT64_MAX, work, bytes) == RNS8_INVALID_ARGUMENT);
  tasks[0].group = 4;
  tasks[0].lda = INT64_MAX;
  CHECK(build_native_wmma_queue(tasks, 1, UINT64_MAX, work, bytes) == RNS8_UNSUPPORTED_BACKEND);
  CHECK(bytes == original_bytes);
  CHECK(work.size() == expected.size());
}
