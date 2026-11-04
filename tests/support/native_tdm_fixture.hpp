#pragma once
#include "backend_common/native_tdm.hpp"

namespace rns8::test {
struct NativeTdmFixtureCase {
  detail::NativeTdmProblem problem;
  detail::NativeTdmResources resources;
  std::size_t a_elements = 0, b_elements = 0, c_elements = 0;
};

// Construct and admit a complete descriptor before any host allocation/fill.
// Full final-row padding is owned here so the oracle may check every canary.
inline bool native_tdm_fixture_case(uint64_t k, uint32_t group, bool sign,
    uint32_t compute, uint32_t axis, uint32_t cluster, NativeTdmFixtureCase& out) {
  if (!k || k > UINT32_MAX - 2) return false;
  NativeTdmFixtureCase result;
  auto& p = result.problem;
  p.m = k > 65536 ? 2 : 17; p.n = k > 65536 ? 3 : 19; p.k = k;
  p.lda = k + 2; p.ldb = p.n + 2; p.ldc = p.n + 2;
  p.prefix = 9; p.group = group; p.signed_input = sign; p.compute = compute;
  p.cluster_axis = axis; p.cluster = cluster; p.mode = cluster > 1 ? 2 : 1;
  result.resources = detail::query_native_tdm(p);
  detail::AddressDomain a, b, c;
  if (result.resources.status != RNS8_SUCCESS ||
      !detail::checked_host_plane_address(p.m,p.lda,p.lda,1,8,a) ||
      !detail::checked_host_plane_address(p.k,p.ldb,p.ldb,1,8,b) ||
      !detail::checked_host_plane_address(p.m,p.ldc,p.ldc,p.prefix,1,c)) return false;
  result.a_elements = static_cast<std::size_t>(a.byte_extent / 8);
  result.b_elements = static_cast<std::size_t>(b.byte_extent / 8);
  result.c_elements = static_cast<std::size_t>(c.byte_extent);
  out = result;
  return true;
}
} // namespace rns8::test
