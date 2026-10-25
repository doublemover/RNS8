#include "backend_common/native_tile_bound.hpp"
#include "backend_common/native_summary_identity.hpp"
#include "backend_common/adaptive_native_support.hpp"
#include "backend_hip_direct/native_global_minimum.hpp"
#include "backend_common/garner_constants.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include "backend_common/native_narrow.hpp"
#include "backend_common/mixed_radix_lift.hpp"

TEST_CASE("native summary reuse requires complete matrix generation and policy identity") {
  using namespace rns8::detail;
  const uint64_t payload=0;
  NativeSummaryIdentity identity;
  identity.instance=7; identity.generation=9; identity.native=&payload;
  identity.rows=17; identity.cols=65; identity.ld=68; identity.device=0;
  identity.method=NativeBoundMethod::panel_l1; identity.panel_k=64; identity.row_tiles=true;
  CHECK(identity==identity);
  for (unsigned field=0; field<11; ++field) {
    auto changed=identity;
    switch (field) {
      case 0: ++changed.instance; break;
      case 1: ++changed.generation; break;
      case 2: changed.native=nullptr; break;
      case 3: ++changed.rows; break;
      case 4: ++changed.cols; break;
      case 5: ++changed.ld; break;
      case 6: ++changed.device; break;
      case 7: changed.method=NativeBoundMethod::axis_l1; break;
      case 8: ++changed.panel_k; break;
      case 9: changed.unsigned_inputs=true; break;
      case 10: changed.row_tiles=false; break;
    }
    CHECK_FALSE(identity==changed);
    CHECK_FALSE(changed==identity);
  }
  CHECK_FALSE(NativeSummaryIdentity{}==NativeSummaryIdentity{});
}

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

TEST_CASE("finite indexed windows prove original row bases at signed address boundaries") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  for(int64_t m:{32767,32768,32769}) {
    const auto wide=query_finite_indexed_window_support(m,1,65536,65536,1,3,253,m-1,1,FiniteGemmAddressWidth::unsigned64);
    REQUIRE(wide.status==RNS8_SUCCESS);
    const cpp_int last=cpp_int(m-1)*65536+65535;
    CHECK(cpp_int(wide.a.last_element)==last);CHECK(cpp_int(wide.a.byte_extent)==last+1);
    CHECK(wide.logical_outputs==1);CHECK(wide.c.byte_extent==uint64_t(m-1)*3+1);
    const auto narrow=query_finite_indexed_window_support(m,1,65536,65536,1,3,253,m-1,1,FiniteGemmAddressWidth::signed32);
    CHECK(narrow.status==(m==32767?RNS8_SUCCESS:RNS8_UNSUPPORTED_BACKEND));
  }
  const int64_t big=(int64_t{1}<<31)+1;
  CHECK(query_finite_indexed_window_support(2,1,1,big,1,1,253,1,1,FiniteGemmAddressWidth::unsigned64).a.last_element==uint64_t(big));
  CHECK(query_finite_indexed_window_support(1,1,2,2,big,1,253,0,1,FiniteGemmAddressWidth::unsigned64).b.last_element==uint64_t(big));
  CHECK(query_finite_indexed_window_support(2,1,1,1,1,big,253,1,1,FiniteGemmAddressWidth::unsigned64).c.last_element==uint64_t(big));
  for(auto first:{-1,2})CHECK(query_finite_indexed_window_support(2,1,1,1,1,1,253,first,1,FiniteGemmAddressWidth::unsigned64).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_finite_indexed_window_support(2,1,1,1,1,1,253,1,2,FiniteGemmAddressWidth::unsigned64).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_finite_indexed_window_support(2,1,1,1,1,1,1,0,1,FiniteGemmAddressWidth::unsigned64).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_finite_indexed_window_support(2,1,1,1,1,1,253,0,1,static_cast<FiniteGemmAddressWidth>(2)).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_finite_indexed_window_support(INT64_MAX,2,1,1,2,2,253,0,1,FiniteGemmAddressWidth::unsigned64).status==RNS8_RANGE_ERROR);
}

TEST_CASE("narrow persistent provider charges simultaneous staging and scan storage") {
  using namespace rns8::detail;
  for (bool wmma : {false,true}) for (uint32_t prefix=1;prefix<=20;++prefix) {
    const auto support=query_narrow_rns_support(17,19,31,32,20,prefix,wmma,UINT64_MAX);
    REQUIRE(support.status==RNS8_SUCCESS);
    CHECK(support.payload_bytes==17*19*8);
    CHECK(support.scratch_bytes==17*19*8+(wmma?32:16));
    CHECK_FALSE(support.device_qualified);
    CHECK(query_narrow_rns_support(17,19,31,32,20,prefix,wmma,support.scratch_bytes).status==RNS8_SUCCESS);
    const auto tight=query_narrow_rns_support(17,19,31,32,20,prefix,wmma,support.scratch_bytes-1);
    CHECK(tight.status==RNS8_WORKSPACE_TOO_SMALL);
    CHECK(tight.constraint==OperationConstraint::scratch_budget);
    int64_t a=1,b=2;
    int8_t c=3;
    bool started=true;
    CHECK(hip_direct_gemm_narrow_rns_device(-1,&a,&b,&c,17,19,31,32,20,prefix,false,wmma,
        support.scratch_bytes-1,started)==tight.status);
    CHECK_FALSE(started);
    CHECK(c==3);
  }
  CHECK(query_narrow_rns_support(1,1,1,1,1,0,false,100).constraint==OperationConstraint::prefix);
  CHECK(query_narrow_rns_support(INT64_MAX,1,1,1,1,20,false,UINT64_MAX).constraint==OperationConstraint::address_span);
}

#include "backend_common/encoded_b_layout.hpp"
TEST_CASE("encoded B layout pads K and accounts for capability storage") {
  using namespace rns8::detail;
  for(uint32_t prefix=1;prefix<=20;++prefix) for(int k:{1,15,16,17,65535,65536,65537}) {
    const auto layout=encoded_b_layout(k,19,21,prefix,UINT64_MAX);
    REQUIRE(layout.status==RNS8_SUCCESS);
    CHECK(layout.k_words==((uint64_t(k)+15)/16)*4);
    CHECK(layout.payload_bytes==layout.k_words*4*19*prefix);
    CHECK(layout.total_bytes==layout.payload_bytes+4);
    CHECK(encoded_b_layout(k,19,21,prefix,layout.total_bytes-1).status==RNS8_WORKSPACE_TOO_SMALL);
  }
  CHECK(encoded_b_layout(INT64_MAX,1,1,20,UINT64_MAX).status==RNS8_UNSUPPORTED_BACKEND);
  CHECK(encoded_b_layout(1,1,1,0,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
}

#include "backend_common/direct_operation_registry.hpp"
TEST_CASE("direct operation inventory separates static predicates from device evidence") {
  using namespace rns8::detail;
  DirectOperationProblem p;
  p.m=17;p.n=19;p.k=31;p.lda=31;p.ldb=19;p.ldc=19;p.ldd=19;
  p.prefix=20;p.source_prefix=9;p.modulus=250;
  for(const auto& record:direct_operation_records) {
    p.operation=record.operation;
    const auto query=query_direct_operation(p);
    REQUIRE(query.implementation==&record);
    CHECK(query.static_support.status==RNS8_SUCCESS);
    CHECK(query.static_support.resources_known);
    CHECK_FALSE(query.runtime_observed); CHECK_FALSE(query.exact_qualified);
    CHECK_FALSE(query.performance_promoted); CHECK(query.requires_operand_binding);
    for(auto execution:{DirectExecution::borrowed_stream,DirectExecution::owned_capture,DirectExecution::distributed}) {
      p.execution=execution;
      const auto rejected=query_direct_operation(p);
      CHECK(rejected.static_support.status==RNS8_UNSUPPORTED_BACKEND);
      CHECK(rejected.static_support.constraint==OperationConstraint::execution_mode);
      CHECK_FALSE(rejected.static_support.resources_known);
    }
    p.execution=DirectExecution::synchronous;
  }
  p.operation=static_cast<DirectOperation>(999);
  CHECK(query_direct_operation(p).static_support.constraint==OperationConstraint::operation);
  p.operation=DirectOperation::residue_gemm;p.version=2;
  CHECK(query_direct_operation(p).static_support.status==RNS8_INVALID_ARGUMENT);
}
TEST_CASE("shared dense CRT and finite verifier providers admit exact boundaries") {
  using namespace rns8::detail;
  CHECK(query_residue_gemm_support(17,19,31,31,19,19,20).status==RNS8_SUCCESS);
  CHECK(query_residue_gemm_support(1,1,1,INT64_MAX,1,1,20).constraint==OperationConstraint::address_span);
  for(uint32_t prefix:{0u,21u}) CHECK(query_canonical_crt_support(1,prefix).constraint==OperationConstraint::prefix);
  const auto crt=query_canonical_crt_support(257,20);
  CHECK(crt.status==RNS8_SUCCESS);CHECK(crt.payload_bytes==257*24);
  CHECK(crt.scratch_bytes==0);CHECK_FALSE(crt.requires_value_proof);
  CHECK(query_canonical_crt_support(INT64_MAX,20).constraint==OperationConstraint::address_span);
  for(bool gpu:{false,true}) for(uint32_t q:{2u,4u,251u,256u}) {
    const auto support=query_finite_verify_support(17,19,31,q,gpu,UINT64_MAX);
    REQUIRE(support.status==RNS8_SUCCESS);
    CHECK(support.scratch_bytes==uint64_t(gpu?19*2+31*4+8:19+31*4));
    CHECK(query_finite_verify_support(17,19,31,q,gpu,support.scratch_bytes).status==RNS8_SUCCESS);
    CHECK(query_finite_verify_support(17,19,31,q,gpu,support.scratch_bytes-1).constraint==OperationConstraint::scratch_budget);
  }
  for(uint32_t q:{0u,1u,257u}) CHECK(query_finite_verify_support(1,1,1,q,false,99).constraint==OperationConstraint::modulus);
  CHECK(query_finite_verify_support(INT64_MAX,2,1,251,true,UINT64_MAX).constraint==OperationConstraint::address_span);
}

#include "backend_hip_direct/hip_backend.hpp"
#include "backend_hip_direct/finite_verify_backend.hpp"
TEST_CASE("static direct provider rejection agrees with launch preflight without a device") {
  using namespace rns8::detail;
  int8_t a=1,b=2,c=37; uint64_t limbs[3]={11,22,33};
  for(uint32_t prefix:{0u,1u,20u,21u}) {
    const auto dense=query_residue_gemm_support(1,1,1,INT64_MAX,1,1,prefix);
    REQUIRE(dense.status!=RNS8_SUCCESS);
    CHECK(hip_direct_gemm_rns_device(-1,&a,&b,&c,1,1,1,INT64_MAX,1,1,prefix)==dense.status);
    const auto crt=query_canonical_crt_support(INT64_MAX,prefix);
    REQUIRE(crt.status!=RNS8_SUCCESS);
    CHECK(hip_direct_export_canonical_direct_crt_device(-1,&a,limbs,INT64_MAX,prefix)==crt.status);
    CHECK(hip_direct_export_canonical_balanced_crt_device(-1,&a,limbs,INT64_MAX,prefix)==crt.status);
    CHECK(c==37); CHECK(limbs[0]==11); CHECK(limbs[2]==33);
  }
  uint8_t r=0; uint32_t br=0,flag=0,result=77;
  const auto verify=query_finite_verify_support(INT64_MAX,2,1,251,true,UINT64_MAX);
  CHECK(hip_finite_verify_trial(-1,&a,&b,&c,INT64_MAX,2,1,251,&r,&br,&flag,result)==verify.status);
  CHECK(result==77); CHECK(c==37);
}

#include "backend_common/native_conversion_reference.hpp"
#include "backend_common/native_conversion_layout.hpp"
TEST_CASE("encode-once fragment ownership is exact and rejects mapping mutations", "[native][layout][cpu]") {
  namespace ref=rns8::detail::native_conversion_reference;
  constexpr uint32_t m=17,n=19,k=17,lda=23,ldb=29;
  std::vector<uint64_t> a(m*lda,UINT64_MAX),b(k*ldb,UINT64_MAX);
  for(uint32_t row=0;row<m;++row)for(uint32_t q=0;q<k;++q)a[row*lda+q]=(uint64_t(row+1)<<59)^(q*7919u+row);
  for(uint32_t q=0;q<k;++q)for(uint32_t col=0;col<n;++col)b[q*ldb+col]=(uint64_t(col+1)<<58)^(q*3571u+col*31u);
  auto matches=[&](const ref::Fragments& f,uint32_t row0,uint32_t col0,uint32_t base,uint32_t modulus,bool unsign) {
    for(uint32_t lane=0;lane<32;++lane)for(uint32_t word=0;word<4;++word)for(uint32_t byte=0;byte<4;++byte) {
      const auto row=row0+lane%16,col=col0+lane%16,q=base+word*4+byte;
      const auto av=row<m && q<k ? a[row*lda+q] : 0;
      const auto bv=col<n && q<k ? b[q*ldb+col] : 0;
      if(ref::byte(f.a[lane][word],byte)!=uint8_t(ref::center(av,unsign,modulus)) ||
         ref::byte(f.b[lane][word],byte)!=uint8_t(ref::center(bv,unsign,modulus)))return false;
    }
    return true;
  };
  for(bool unsign:{false,true})for(uint32_t modulus:{256u,255u,253u,251u,167u})for(uint32_t base:{0u,16u}) {
    const auto f=ref::cooperative(a,b,m,n,k,lda,ldb,0,0,base,modulus,unsign);
    CHECK(matches(f,0,0,base,modulus,unsign));CHECK(f.conversions==512);
    const auto tiled=ref::encoded_lds(a,b,m,n,k,lda,ldb,0,0,base,16,0,modulus,unsign);
    CHECK(tiled[0].conversions==1024);
    for(uint32_t wave=0;wave<4;++wave)CHECK(matches(tiled[wave],(wave/2)*16,(wave%2)*16,base,modulus,unsign));
  }
  for(auto mutation:{ref::Mutation{1},ref::Mutation{16,true},ref::Mutation{16,false,true}})
    CHECK_FALSE(matches(ref::cooperative(a,b,m,n,k,lda,ldb,0,0,16,251,false,mutation),0,0,16,251,false));
  for(auto mutation:{ref::Mutation{16,false,false,true},ref::Mutation{16,false,false,false,true}}) {
    const auto broken=ref::encoded_lds(a,b,m,n,k,lda,ldb,0,0,0,32,0,251,false,mutation);
    CHECK_FALSE(matches(broken[0],0,0,0,251,false));
  }
  for(uint32_t group:{1u,2u,4u})for(uint32_t depth:{16u,32u}) {
    const auto encoded=rns8::detail::query_native_tiled_resources(depth,1,65536,rns8::detail::NativeConversion::encoded_lds,group);
    REQUIRE(encoded.status==RNS8_SUCCESS);
    CHECK(encoded.raw_lds_bytes==0);
    CHECK(encoded.encoded_lds_bytes==uint64_t(group)*2*32*(depth/4+1)*4);
    CHECK(encoded.conversions_per_panel==uint64_t(group)*2*32*depth);
    CHECK(rns8::detail::query_native_tiled_resources(depth,1,encoded.total_lds_bytes-1,rns8::detail::NativeConversion::encoded_lds,group).status==RNS8_WORKSPACE_TOO_SMALL);
  }
  CHECK(rns8::detail::query_native_tiled_resources(16,2,65536,rns8::detail::NativeConversion::encoded_lds,1).status==RNS8_INVALID_ARGUMENT);
  CHECK(rns8::detail::query_native_tiled_resources(16,1,65536,rns8::detail::NativeConversion::cooperative,1).status==RNS8_INVALID_ARGUMENT);
}

#include "../../benchmarks/native_dataset.hpp"
TEST_CASE("versioned native datasets certify heterogeneous buckets before GPU work", "[native][bounds][fixture]") {
  using namespace rns8::bench;
  for(bool u:{false,true})for(const std::string profile:{"full-width-v1","uniform-small-v2","mixed-tiles-v2","zero-tiles-v2","localized-outlier-v2"})
    for(uint32_t k:{1u,17u,257u}) {
      constexpr uint32_t m=17,n=19;
      std::vector<uint64_t> a(m*k),b(k*n);std::mt19937_64 rng(3135);
      fill_native_dataset(a,b,m,n,k,u,profile,3135,rng);
      for(uint32_t group:{1u,2u,4u}) {
        const auto model=model_native_dataset(a,b,m,n,k,u,group,profile);
        uint64_t tiles=0,channels=0,groups=0;
        for(uint32_t p=0;p<=20;++p) {tiles+=model.required_histogram[p];channels+=p*model.required_histogram[p];groups+=((p+group-1)/group)*model.required_histogram[p];}
        CHECK(tiles==4);CHECK(channels==model.required_channels);CHECK(groups==model.group_work_items);
        CHECK(model.required_histogram==model.executed_histogram);
        if(profile=="zero-tiles-v2")CHECK(model.zero_tiles==3);
        if(profile!="full-width-v1")CHECK(model.group_work_items<model.fixed_capacity_group_work_items);
      }
      const auto first=model_native_dataset(a,b,m,n,k,u,1,profile);a[0]^=1;
      // Input mutation is visible even when a conservative bound bucket stays unchanged.
      CHECK(model_native_dataset(a,b,m,n,k,u,1,profile).input_hash!=first.input_hash);
    }
  std::vector<uint64_t> a(16*17),b(17*16);std::mt19937_64 rng(0);
  fill_native_dataset(a,b,16,16,17,false,"mixed-tiles-v2",0,rng);
  CHECK_THROWS_AS(model_native_dataset(a,b,16,16,17,false,1,"mixed-tiles-v2"),std::invalid_argument);
}

#include "backend_common/split_k_partial.hpp"
TEST_CASE("native first merge normalizes centered bytes without changing canonical trees", "[split-k][arithmetic]") {
  using namespace rns8::detail;
  for(uint32_t p=2;p<=256;++p) {
    for(uint32_t r=0;r<p;++r) {
      const int32_t centered=r>=(p+1)/2?int32_t(r)-int32_t(p):int32_t(r);
      const auto bits=static_cast<uint8_t>(centered);
      CHECK(split_k_partial_canonical(bits,p,true)==r);
      CHECK(split_k_partial_centered(bits,p,true)==centered);
      CHECK(split_k_partial_centered(static_cast<uint8_t>(r),p,false)==centered);
    }
    for(uint32_t fragments:{1u,2u,3u,127u,128u,129u,257u,4096u}) {
      std::vector<uint8_t> partials(fragments);uint64_t reference=0;
      for(uint32_t f=0;f<fragments;++f) {
        const uint32_t r=(f*127u+(f%2?p-1:0))%p;reference+=r;
        partials[f]=static_cast<uint8_t>(r>=(p+1)/2?int32_t(r)-int32_t(p):int32_t(r));
      }
      bool centered=true;
      while(partials.size()>1) {
        std::vector<uint8_t> next((partials.size()+127)/128);
        for(std::size_t first=0;first<partials.size();first+=128) {
          uint32_t sum=0;
          for(auto f=first;f<std::min(first+128,partials.size());++f)sum+=split_k_partial_canonical(partials[f],p,centered);
          REQUIRE(sum<=32640);next[first/128]=static_cast<uint8_t>(sum%p);
        }
        centered=false;partials=std::move(next);
      }
      const uint32_t canonical=reference%p;
      CHECK(split_k_partial_centered(partials[0],p,centered)==
        static_cast<int8_t>(canonical>=(p+1)/2?int32_t(canonical)-int32_t(p):int32_t(canonical)));
    }
  }
}

TEST_CASE("global minimum control admits complete simultaneous scratch before any execution") {
  using namespace rns8::detail;
  for(uint32_t capacity:{1u,2u,20u})for(uint32_t group:{1u,2u,4u}) {
    const auto support=query_native_global_minimum_support(17,19,257,260,22,23,capacity,group,UINT64_MAX);
    REQUIRE(support.status==RNS8_SUCCESS);
    CHECK(support.peak_scratch_bytes==std::max(support.discovery.peak_scratch_bytes+sizeof(NativeTileBound),support.discovery.stage_bytes+std::max<uint64_t>(support.suffix_bytes,sizeof(int))));
    CHECK(support.suffix_bytes==(capacity==1?0:17*19*(capacity-1)+sizeof(int)));
    CHECK(query_native_global_minimum_support(17,19,257,260,22,23,capacity,group,support.peak_scratch_bytes-1).status==RNS8_WORKSPACE_TOO_SMALL);
    CHECK(query_native_global_minimum_support(17,19,257,260,22,23,capacity,group,support.peak_scratch_bytes).status==RNS8_SUCCESS);
  }
  CHECK(query_native_global_minimum_support(INT64_MAX,1,1,1,1,1,20,1,UINT64_MAX).status==RNS8_UNSUPPORTED_BACKEND);
  CHECK(query_native_global_minimum_support(1,1,1,1,1,1,0,1,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_native_global_minimum_support(1,1,1,1,1,1,20,3,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
}
