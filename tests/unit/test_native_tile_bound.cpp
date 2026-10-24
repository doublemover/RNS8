#include "backend_common/native_tile_bound.hpp"
#include "backend_common/adaptive_native_support.hpp"
#include "backend_common/garner_constants.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include "backend_common/native_narrow.hpp"
#include "backend_common/mixed_radix_lift.hpp"

TEST_CASE("mixed radix lift subtracts the complete source product for negatives") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const auto& moduli = garner::kConstants.moduli;
  cpp_int product = 1;
  for (uint32_t count = 1; count < 20; ++count) {
    product *= moduli[count - 1];
    const std::vector<cpp_int> values{0, 1, product - 1, (product - 1) / 2, product / 2 + 1};
    for (const auto& canonical : values) {
      uint32_t digits[20]{};
      cpp_int remaining = canonical;
      for (uint32_t i = 0; i < count; ++i) {
        digits[i] = static_cast<uint32_t>(remaining % moduli[i]);
        remaining /= moduli[i];
      }
      REQUIRE(remaining == 0);
      for (bool negative : {false, true}) for (uint32_t p = count; p < 20; ++p) {
        const cpp_int value = negative ? cpp_int(canonical - product) : canonical;
        cpp_int expected = value % moduli[p];
        if (expected < 0) expected += moduli[p];
        REQUIRE(mixed_radix_lift_residue(digits, moduli, count, moduli[p], negative) == expected);
      }
    }
  }
  // X=P-1 represents -1: reducing X alone is a different residue for every
  // coprime suffix modulus and must not be accepted as signed extension.
  const uint32_t digits[]{255}, base[]{256};
  REQUIRE(mixed_radix_lift_residue(digits, base, 1, 255, false) == 0);
  REQUIRE(mixed_radix_lift_residue(digits, base, 1, 255, true) == 254);
}

TEST_CASE("native partial sum admission agrees with arbitrary precision at caps") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for (uint64_t limit : {uint64_t{INT32_MAX}, uint64_t{INT64_MAX}, UINT64_MAX})
    for (uint64_t a : {uint64_t{0}, uint64_t{1}, uint64_t{128}, uint64_t{255}, uint64_t{1} << 63, UINT64_MAX})
      for (uint64_t b : {uint64_t{0}, uint64_t{1}, uint64_t{127}, uint64_t{255}, UINT64_MAX})
        for (uint64_t k : {uint64_t{1}, uint64_t{33025}, uint64_t{33026}, uint64_t{131071}, uint64_t{131072}, UINT64_MAX}) {
          const cpp_int expected = cpp_int(a) * b * k;
          uint64_t bound = 41;
          REQUIRE(native_partial_sum_bound(a, b, k, limit, bound) == (expected <= limit));
          REQUIRE(cpp_int(bound) == (expected <= limit ? expected : cpp_int(41)));
        }
  uint64_t bound = 41;
  REQUIRE_FALSE(native_partial_sum_bound(0, 0, 0, 0, bound));
  REQUIRE(bound == 41);
}

TEST_CASE("native L1 bounds retain all carry bits through full-width products") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const auto decode = [](const NativeTileBound& value) -> cpp_int {
    return cpp_int(value.magnitude[0]) + (cpp_int(value.magnitude[1]) << 64) +
           (cpp_int(value.magnitude[2]) << 128);
  };
  for (uint64_t maximum : {uint64_t{0}, uint64_t{1}, uint64_t{1} << 63, UINT64_MAX})
    for (uint32_t k : {1u, 17u, 65536u, UINT32_MAX}) {
      const cpp_int sum = cpp_int(maximum) * k;
      const NativeAxisSummary axis{maximum, static_cast<uint64_t>(sum & UINT64_MAX),
                                   static_cast<uint64_t>(sum >> 64)};
      REQUIRE(decode(native_axis_l1_bound(axis, axis, k)) == sum * maximum);
    }
  std::mt19937_64 rng(340034);
  for (unsigned iteration = 0; iteration < 512; ++iteration) {
    const uint64_t maximum = rng(), low = rng(), high = static_cast<uint32_t>(rng());
    REQUIRE(decode(native_l1_product_bound(maximum, low, high)) ==
            cpp_int(maximum) * (cpp_int(low) + (cpp_int(high) << 64)));
  }
  const NativeAxisSummary one_outlier{100, 100, 0}, dense{3, 300, 0};
  REQUIRE(decode(native_axis_l1_bound(one_outlier, dense, 100)) == 300);
  REQUIRE(decode(native_tile_max_bound(100, 3, 100)) == 30000);
}

TEST_CASE("native tile maximum bounds have exact multiword products and strict prefixes") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr auto basis = garner::make_constants();
  for (uint64_t a : {uint64_t{0}, uint64_t{1}, uint64_t{128}, uint64_t{1} << 32, uint64_t{1} << 63, UINT64_MAX})
    for (uint64_t b : {uint64_t{0}, uint64_t{1}, uint64_t{255}, uint64_t{1} << 32, uint64_t{1} << 63, UINT64_MAX})
      for (uint32_t k : {1u, 65535u, 65536u, 65537u, UINT32_MAX}) {
        auto bound = native_tile_max_bound(a, b, k);
        const cpp_int exact = cpp_int(a) * b * k;
        const cpp_int observed = cpp_int(bound.magnitude[0]) + (cpp_int(bound.magnitude[1]) << 64) +
                                 (cpp_int(bound.magnitude[2]) << 128);
        REQUIRE(observed == exact);
        for (bool signed_result : {false, true})
          for (uint32_t capacity = 1; capacity <= 20; ++capacity) {
            select_native_bound_prefix(bound, signed_result, capacity, basis.products);
            cpp_int product = 1;
            uint32_t expected = exact == 0 ? 0 : 21;
            for (uint32_t p = 0; p < capacity && expected == 21; ++p) {
              product *= basis.moduli[p];
              if (product > exact * (signed_result ? 2 : 1)) expected = p + 1;
            }
            REQUIRE(bound.required_prefix == expected);
          }
      }
  auto equal = native_tile_max_bound(128, 1, 1);
  select_native_bound_prefix(equal, true, 1, basis.products);
  REQUIRE(equal.required_prefix == 21); // P == 2*bound is insufficient.
}

TEST_CASE("K-panel bound summation preserves carries and disjoint zero proofs") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const auto decode = [](const NativeTileBound& value) -> cpp_int {
    return cpp_int(value.magnitude[0]) + (cpp_int(value.magnitude[1]) << 64) +
           (cpp_int(value.magnitude[2]) << 128);
  };
  for (uint64_t maximum : {uint64_t{0}, uint64_t{1}, uint64_t{1} << 63, UINT64_MAX}) {
    NativeTileBound accumulated{};
    uint64_t terms = 0;
    for (uint32_t panel : {1u, 65535u, 65536u, UINT32_MAX - 131072u}) {
      add_native_panel_bound(accumulated, native_tile_max_bound(maximum, maximum, panel));
      terms += panel;
      REQUIRE(decode(accumulated) == cpp_int(maximum) * maximum * terms);
    }
    REQUIRE(terms == UINT32_MAX);
  }
  const NativeAxisSummary zero{}, large{UINT64_MAX, UINT64_MAX, 0};
  NativeTileBound disjoint{};
  add_native_panel_bound(disjoint, native_axis_l1_bound(large, zero, 1));
  add_native_panel_bound(disjoint, native_axis_l1_bound(zero, large, 1));
  REQUIRE(decode(disjoint) == 0);
  REQUIRE(decode(native_axis_l1_bound(large, large, 2)) > 0);
}


TEST_CASE("compact adaptive scheduling covers each admitted channel group exactly once") {
  using namespace rns8::detail;
  for (uint32_t group : {1u, 2u, 4u}) {
    uint32_t offsets[22]{};
    std::vector<NativeAdaptiveWork> expected;
    for (uint32_t prefix = 0; prefix <= 20; ++prefix) {
      const uint32_t tiles = prefix == 0 ? 5 : prefix % 4 == 0 ? 0 : prefix % 3 + 1;
      offsets[prefix + 1] = offsets[prefix] + tiles;
      if (prefix)
        for (uint32_t index = offsets[prefix]; index < offsets[prefix + 1]; ++index)
          for (uint32_t plane = 0; plane < prefix; plane += group)
            expected.push_back({index, prefix, plane});
    }
    REQUIRE(native_adaptive_work_count(offsets, 20, group) == expected.size());
    for (uint32_t workers : {1u, 3u, 128u}) {
      std::vector<unsigned> visits(expected.size());
      for (uint64_t worker = 0; worker < workers; ++worker)
        for (uint64_t ordinal = worker; ordinal < expected.size(); ordinal += workers) {
          const auto actual = native_adaptive_work(offsets, 20, group, ordinal);
          CHECK(actual.compact_index == expected[ordinal].compact_index);
          CHECK(actual.prefix == expected[ordinal].prefix);
          CHECK(actual.first_plane == expected[ordinal].first_plane);
          ++visits[ordinal];
        }
      for (const auto visits_per_work : visits) REQUIRE(visits_per_work == 1);
    }
    CHECK(native_adaptive_work(offsets, 20, group, expected.size()).prefix == 0);
  }
  uint32_t empty[22]{};
  CHECK(native_adaptive_work_count(empty, 20, 1) == 0);
  uint32_t large[22]{};
  large[21] = UINT32_MAX;
  CHECK(native_adaptive_work_count(large, 20, 1) == uint64_t{UINT32_MAX} * 20);
  const auto last = native_adaptive_work(large, 20, 1, uint64_t{UINT32_MAX} * 20 - 1);
  CHECK(last.compact_index == UINT32_MAX - 1);
  CHECK(last.first_plane == 19);
  CHECK(last.prefix == 20);
}

TEST_CASE("adaptive static support reports checked resources without device qualification") {
  using namespace rns8::detail;
  const auto query = [](uint64_t budget, uint32_t workers=3) {
    return query_adaptive_native_support(35,33,17,19,36,37,20,2,NativeBoundMethod::axis_l1,256,workers,budget);
  };
  const auto valid=query(UINT64_MAX);
  REQUIRE(valid.status==RNS8_SUCCESS);
  CHECK(valid.constraint==AdaptiveNativeConstraint::none);
  CHECK(valid.version==1);
  CHECK(valid.resource_requirement_known);
  CHECK(valid.requires_device_capability_check);
  CHECK(valid.tiles==9);
  CHECK(valid.stage_bytes==35*33*20);
  CHECK(valid.bound_bytes==9*sizeof(NativeTileBound));
  CHECK(valid.metadata_bytes==32*sizeof(uint32_t));
  CHECK(valid.discovery_bytes==6*sizeof(NativeAxisSummary)+43*sizeof(uint32_t));
  CHECK(valid.peak_scratch_bytes==valid.reserved_bytes+valid.discovery_bytes);
  CHECK(query(valid.peak_scratch_bytes).status==RNS8_SUCCESS);
  const auto tight=query(valid.peak_scratch_bytes-1);
  CHECK(tight.status==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(tight.constraint==AdaptiveNativeConstraint::scratch_budget);
  CHECK(tight.resource_requirement_known);
  CHECK(tight.peak_scratch_bytes==valid.peak_scratch_bytes);
  CHECK(query(UINT64_MAX,0).peak_scratch_bytes==valid.peak_scratch_bytes);
  CHECK(query(UINT64_MAX,65536).constraint==AdaptiveNativeConstraint::workers);
  for (int bad=0; bad<8; ++bad) {
    const auto result=query_adaptive_native_support(bad==0?0:35,33,17,bad==1?16:19,36,
        bad==2?INT64_MAX:37,bad==3?21:20,bad==4?3:2,
        bad==5?static_cast<NativeBoundMethod>(99):NativeBoundMethod::panel_l1,bad==6?0:7,bad==7?65536:3,UINT64_MAX);
    CHECK(result.status!=RNS8_SUCCESS);
    CHECK_FALSE(result.resource_requirement_known);
    CHECK(result.requires_device_capability_check);
  }
  const auto panels=query_adaptive_native_support(1,1,65536,65536,1,1,20,1,NativeBoundMethod::panel_l1,1,1,UINT64_MAX);
  CHECK(panels.status==RNS8_UNSUPPORTED_BACKEND);
  CHECK(panels.constraint==AdaptiveNativeConstraint::panel_count);
}

#include "backend_common/native_tiled_resources.hpp"
TEST_CASE("multiwave native storage budgets match padded raw LDS allocations") {
  using namespace rns8::detail;
  for (uint32_t buffers : {0u,1u,2u}) for (uint32_t depth : {16u,32u}) {
    const auto result=query_native_tiled_resources(depth,buffers,UINT64_MAX);
    if (!buffers && depth==32) {
      CHECK(result.status==RNS8_INVALID_ARGUMENT);
      continue;
    }
    REQUIRE(result.status==RNS8_SUCCESS);
    CHECK(result.output_rows==32);
    CHECK(result.output_cols==32);
    CHECK(result.waves==4);
    CHECK(result.raw_lds_bytes==uint64_t(buffers)*(32*(depth+1)+depth*33)*8);
    CHECK(query_native_tiled_resources(depth,buffers,result.raw_lds_bytes).status==RNS8_SUCCESS);
    if (result.raw_lds_bytes)
      CHECK(query_native_tiled_resources(depth,buffers,result.raw_lds_bytes-1).status==RNS8_WORKSPACE_TOO_SMALL);
  }
  CHECK(query_native_tiled_resources(0,1,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_native_tiled_resources(16,3,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
}

#include "backend_common/operation_support.hpp"
#include "backend_hip_direct/hip_backend.hpp"
TEST_CASE("shared epilogue and lift providers reject before pointer consumption") {
  using namespace rns8::detail;
  int8_t a=1,b=2,c=3,d=4;
  uint64_t bound[3]{1,0,0};
  for (uint32_t prefix : {0u,21u}) {
    const auto support=query_axpby_support(1,1,1,1,1,1,1,prefix);
    CHECK(support.constraint==OperationConstraint::prefix);
    CHECK_FALSE(support.resources_known);
    CHECK(hip_direct_gemm_rns_axpby_device(-1,&a,&b,&d,&c,1,1,1,1,1,1,1,prefix,1,1)==support.status);
    CHECK(c==3);
  }
  const auto span=query_axpby_support(1,1,1,1,1,INT64_MAX,1,20);
  CHECK(span.constraint==OperationConstraint::address_span);
  CHECK(hip_direct_gemm_rns_axpby_device(-1,&a,&b,&d,&c,1,1,1,1,1,INT64_MAX,1,20,1,1)==span.status);
  const auto fused=query_axpby_support(17,19,31,32,20,21,22,20);
  REQUIRE(fused.status==RNS8_SUCCESS);
  CHECK(fused.resources_known);
  CHECK(fused.scratch_bytes==0);
  CHECK_FALSE(fused.device_qualified);
  for (uint32_t source : {0u,1u,20u}) {
    const auto support=query_exact_lift_support(1,source,source,UINT64_MAX);
    bool commit=true;
    CHECK(hip_direct_lift_exact_suffix(-1,&c,1,source,source,false,bound,commit)==support.status);
    CHECK_FALSE(commit);
    CHECK(c==3);
  }
  for (uint32_t source=1; source<20; ++source) for (uint32_t target=source+1; target<=20; ++target) {
    const auto support=query_exact_lift_support(257,source,target,UINT64_MAX);
    REQUIRE(support.status==RNS8_SUCCESS);
    CHECK(support.payload_bytes==257*(target-source));
    CHECK(support.scratch_bytes==support.payload_bytes+sizeof(int));
    CHECK(query_exact_lift_support(257,source,target,support.payload_bytes).status==RNS8_SUCCESS);
    const auto tight=query_exact_lift_support(257,source,target,support.payload_bytes-1);
    CHECK(tight.status==RNS8_WORKSPACE_TOO_SMALL);
    CHECK(tight.resources_known);
    CHECK(tight.scratch_bytes==support.scratch_bytes);
  }
  const auto large=query_exact_lift_support(INT64_MAX,1,20,UINT64_MAX);
  CHECK(large.constraint==OperationConstraint::address_span);
}

#include "backend_common/wide_gemm_support.hpp"
TEST_CASE("wide residue support proves full spans without narrowed shape products") {
  using namespace rns8::detail;
  const auto wide=query_wide_gemm_support(65536,65536,1,1,65536,65536,1);
  REQUIRE(wide.status==RNS8_SUCCESS);
  CHECK(wide.logical_outputs==(uint64_t{1}<<32));
  CHECK(wide.c.byte_extent==(uint64_t{1}<<32));
  CHECK_FALSE(qualified_i32_gemm_address(65536,65536,1,1,65536,65536,1));
  CHECK(query_wide_gemm_support(1,1,1,INT64_MAX,INT64_MAX,INT64_MAX,1).status==RNS8_SUCCESS);
  CHECK(query_wide_gemm_support(1,1,1,INT64_MAX,INT64_MAX,INT64_MAX,2).status==RNS8_RANGE_ERROR);
  CHECK(query_wide_gemm_support(INT64_MAX,2,1,1,2,2,1).status==RNS8_RANGE_ERROR);
  CHECK(query_wide_gemm_support(1,1,1,1,1,1,0).status==RNS8_INVALID_ARGUMENT);
}
