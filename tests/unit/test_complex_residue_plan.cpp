#include "backend_common/complex_residue_plan.hpp"
#include "backend_common/residue_epilogue.hpp"
#include <catch2/catch_test_macros.hpp>
using namespace rns8::detail;
using boost::multiprecision::cpp_int;
namespace {
int8_t *complex_address(uintptr_t slot) { return reinterpret_cast<int8_t *>(slot * 0x1000000); }
struct ComplexFixture {
  ComplexResidueBasis basis{ComplexResidueDomain::wide_signed, 1, 20, 0};
  ComplexResiduePair<const int8_t> a{{complex_address(1), 3, 7, 8, basis},
                                     {complex_address(2), 3, 7, 9, basis}};
  ComplexResiduePair<const int8_t> b{{complex_address(3), 7, 5, 6, basis},
                                     {complex_address(4), 7, 5, 7, basis}};
  ComplexResiduePair<int8_t> c{{complex_address(5), 3, 5, 6, basis},
                               {complex_address(6), 3, 5, 7, basis}};
  ComplexMagnitudeBounds bounds{2, 3, 5, 7};
  ComplexResiduePlan query(uint64_t budget = UINT64_MAX,
                           ComplexProductMethod method = ComplexProductMethod::three_real) const {
    return plan_complex_residue_product(a, b, c, bounds, method, budget);
  }
};
int center(const cpp_int &value, int modulus) {
  int residue = (value % modulus).convert_to<int>();
  if (residue < 0)
    residue += modulus;
  if (residue >= (modulus + 1) / 2)
    residue -= modulus;
  return residue;
}
} // namespace
TEST_CASE("complex residue plan proves both final bounds and peak scratch") {
  ComplexFixture f;
  const auto three = f.query();
  REQUIRE(three.status == RNS8_SUCCESS);
  CHECK(three.real_bound == 7 * (2 * 5 + 3 * 7));
  CHECK(three.imag_bound == 7 * (2 * 7 + 3 * 5));
  CHECK(three.scratch_bytes == (3 * 7 + 7 * 5 + 3 * 3 * 5) * 20);
  CHECK(f.query(three.scratch_bytes - 1).status == RNS8_WORKSPACE_TOO_SMALL);
  CHECK(f.query(three.scratch_bytes).status == RNS8_SUCCESS);
  const auto four = f.query(UINT64_MAX, ComplexProductMethod::four_real);
  CHECK(four.scratch_bytes == 4 * 3 * 5 * 20);
  CHECK(four.real_bound == three.real_bound);
  CHECK(four.imag_bound == three.imag_bound);
  f.bounds = {cpp_int(1) << 63, cpp_int(1) << 63, cpp_int(1) << 63, cpp_int(1) << 63};
  CHECK(f.query().status == RNS8_SUCCESS);
  f.bounds.ar = cpp_int(1) << 200;
  CHECK(f.query().status == RNS8_RANGE_ERROR);
  f.bounds.ar = -1;
  CHECK(f.query().status == RNS8_INVALID_ARGUMENT);
}
TEST_CASE("complex residue plan rejects mismatched views and all cross-output aliases") {
  for (int field = 0; field < 7; ++field) {
    ComplexFixture f;
    if (field == 0)
      f.a.imag.basis.prefix = 19;
    if (field == 1)
      f.b.real.basis.version = 2;
    if (field == 2)
      f.c.imag.basis.domain = ComplexResidueDomain::finite_ring;
    if (field == 3)
      f.a.real.ld = 6;
    if (field == 4)
      f.b.imag.rows = 8;
    if (field == 5)
      f.c.real.cols = 6;
    if (field == 6)
      f.a.real.data = nullptr;
    CHECK(f.query().status == RNS8_INVALID_ARGUMENT);
  }
  for (int output = 0; output < 2; ++output)
    for (int source = 0; source < 4; ++source) {
      ComplexFixture f;
      auto &destination = output ? f.c.imag : f.c.real;
      destination.data = complex_address(source + 1) + 1;
      CHECK(f.query().status == RNS8_INVALID_ARGUMENT);
    }
  ComplexFixture f;
  f.c.imag.data = f.c.real.data + 1;
  CHECK(f.query().status == RNS8_INVALID_ARGUMENT);
  CHECK(f.query(UINT64_MAX, static_cast<ComplexProductMethod>(99)).status == RNS8_INVALID_ARGUMENT);
}
TEST_CASE("division-free complex 3M identity holds for every byte ring and signed extrema") {
  const cpp_int values[]{0, 1, -1, INT64_MIN, INT64_MAX, cpp_int(1) << 70, -(cpp_int(1) << 70)};
  bool detects_bad_sum = false, detects_bad_combine = false;
  for (int modulus = 2; modulus <= 256; ++modulus)
    for (std::size_t i = 0; i < 7; ++i) {
      const cpp_int ar = values[i], ai = values[(i + 2) % 7], br = values[(i + 3) % 7],
                    bi = values[(i + 5) % 7];
      const int ra = center(ar, modulus), ia = center(ai, modulus), rb = center(br, modulus),
                ib = center(bi, modulus);
      const int sa = centered_residue_axpby(int8_t(ra), int8_t(ia), 1, 1, modulus);
      const int sb = centered_residue_axpby(int8_t(rb), int8_t(ib), 1, 1, modulus);
      const int p0 = center(cpp_int(ra) * rb, modulus), p1 = center(cpp_int(ia) * ib, modulus),
                p2 = center(cpp_int(sa) * sb, modulus);
      CHECK(center(p0 - p1, modulus) == center(ar * br - ai * bi, modulus));
      CHECK(center(p2 - p0 - p1, modulus) == center(ar * bi + ai * br, modulus));
      detects_bad_sum |=
          center(cpp_int(ra - ia) * sb - p0 - p1, modulus) != center(ar * bi + ai * br, modulus);
      detects_bad_combine |= center(p2 - p0 + p1, modulus) != center(ar * bi + ai * br, modulus);
    }
  CHECK(detects_bad_sum);
  CHECK(detects_bad_combine);
}
TEST_CASE("complex admission distinguishes exact signed ranges from finite rings") {
  ComplexFixture f;
  const auto set_basis = [&](ComplexResidueBasis basis) {
    f.a.real.basis = f.a.imag.basis = f.b.real.basis = f.b.imag.basis = f.c.real.basis =
        f.c.imag.basis = basis;
  };
  f.bounds = {0, cpp_int(1) << 80, cpp_int(1) << 80, 0};
  CHECK(f.query().status == RNS8_RANGE_ERROR); // imaginary component alone exceeds range
  f.bounds = {cpp_int(1) << 80, 0, cpp_int(1) << 80, 0};
  CHECK(f.query().status == RNS8_RANGE_ERROR); // real component alone exceeds range
  set_basis({ComplexResidueDomain::bounded_signed, 1, 20, 0});
  f.bounds = {cpp_int(1) << 63, 0, 1, 0};
  CHECK(f.query().status == RNS8_RANGE_ERROR);
  set_basis({ComplexResidueDomain::wide_signed, 1, 1, 0});
  f.a.real.cols = f.a.imag.cols = f.b.real.rows = f.b.imag.rows = 1;
  f.bounds = {128, 0, 1, 0};
  CHECK(f.query().status == RNS8_RANGE_ERROR);
  f.bounds.ar = 127;
  CHECK(f.query().status == RNS8_SUCCESS);
  f.bounds = {-1, -1, -1, -1}; // finite rings intentionally need no integer range certificate
  for (uint32_t modulus = 2; modulus <= 256; ++modulus) {
    set_basis({ComplexResidueDomain::finite_ring, 1, 1, modulus});
    CHECK(f.query().status == RNS8_SUCCESS);
  }
  set_basis({ComplexResidueDomain::finite_ring, 1, 1, 257});
  CHECK(f.query().status == RNS8_INVALID_ARGUMENT);
}

#include "backend_common/complex_residue_algebra.hpp"
#include <algorithm>
#include <numeric>
#include <vector>
TEST_CASE("complex internal dead-slot recombination preserves every channel under reordered cell ownership") {
  for(unsigned modulus=2;modulus<=256;++modulus)for(bool three:{false,true})for(unsigned cells:{1u,15u,323u}){
    const unsigned products=three?3:4;std::vector<int8_t> storage(products*cells+2,91),expected(2*cells);
    for(unsigned p=0;p<products;++p)for(unsigned i=0;i<cells;++i)storage[1+p*cells+i]=int8_t(center(cpp_int(i*73+p*111),modulus));
    auto* values=storage.data()+1;
    for(unsigned i=0;i<cells;++i){
      expected[i]=int8_t(center(cpp_int(values[i])-values[cells+i],modulus));
      cpp_int imaginary=values[2*cells+i];if(three)imaginary-=cpp_int(values[i])+values[cells+i];else imaginary+=values[3*cells+i];
      expected[cells+i]=int8_t(center(imaginary,modulus));
    }
    std::vector<unsigned> order(cells);std::iota(order.begin(),order.end(),0);std::reverse(order.begin(),order.end());
    for(auto i:order){
      const auto pair=complex_recombine_cell(values[i],values[cells+i],values[2*cells+i],three?0:values[3*cells+i],three,modulus,finite_u8::modulus_reciprocal_u32(modulus));
      values[i]=pair.real;values[cells+i]=pair.imag;
    }
    CHECK(std::equal(expected.begin(),expected.end(),values));CHECK(storage.front()==91);CHECK(storage.back()==91);
  }
}
TEST_CASE("complex queried storage admits exactly the shared reduced scratch envelope") {
  ComplexFixture f;
  for(auto method:{ComplexProductMethod::three_real,ComplexProductMethod::four_real}){
    const auto plan=f.query(UINT64_MAX,method);REQUIRE(plan.status==RNS8_SUCCESS);
    CHECK(plan.layout_version==2);CHECK(plan.staged_real_offset()==plan.products_offset());
    CHECK(plan.staged_imag_offset()+plan.product_bytes<=plan.scratch_bytes);
    CHECK(plan.scratch_bytes==plan.sum_a_bytes+plan.sum_b_bytes+plan.products*plan.product_bytes);
    CHECK(f.query(plan.scratch_bytes,method).status==RNS8_SUCCESS);
    CHECK(f.query(plan.scratch_bytes-1,method).status==RNS8_WORKSPACE_TOO_SMALL);
  }
}
