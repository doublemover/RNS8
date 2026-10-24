#pragma once
#include "backend_common/finite_verify.hpp"
#include "core/internal.hpp"
#include <memory>
namespace finite_verify_fixture {
inline void public_roundtrip(rns8_context* ctx, uint16_t q, int64_t k) {
  constexpr int64_t m=2,n=3;
  const auto matrix = [&](int64_t rows,int64_t cols) {
    rns8_matrix_desc d{}; d.struct_size=sizeof(d); d.abi_version=RNS8_ABI_VERSION;
    d.rows=rows; d.cols=cols; d.logical_ld=cols; d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
    d.semantics=RNS8_FINITE_RING_U8; d.bound_kind=RNS8_BOUND_NONE;
    rns8_matrix* p=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&d,&p)==RNS8_SUCCESS);
    return std::unique_ptr<rns8_matrix,decltype(&rns8_destroy_matrix)>(p,rns8_destroy_matrix);
  };
  auto a=matrix(m,k),b=matrix(k,n),c=matrix(m,n);
  std::vector<uint8_t> av(size_t(m*k)),bv(size_t(k*n)),cv(size_t(m*n));
  for(size_t i=0;i<av.size();++i) av[i]=uint8_t((i*71+q-1)%q);
  for(size_t i=0;i<bv.size();++i) bv[i]=uint8_t((i*37+q-1)%q);
  // Independent uint64 modular dot reference; dimensions here cannot overflow.
  for(int64_t row=0;row<m;++row) for(int64_t col=0;col<n;++col) {
    uint64_t sum=0;
    for(int64_t j=0;j<k;++j) sum+=uint64_t(av[size_t(row*k+j)])*bv[size_t(j*n+col)];
    cv[size_t(row*n+col)]=uint8_t(sum%q);
  }
  REQUIRE(rns8_pack_finite_u8(ctx,a.get(),q,av.data(),k,1)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(ctx,b.get(),q,bv.data(),n,1)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_finite_u8(ctx,c.get(),q,cv.data(),n,1)==RNS8_SUCCESS);
  rns8_finite_verify_options o{}; o.struct_size=sizeof(o); o.abi_version=RNS8_ABI_VERSION;
  o.repetition_count=32; o.random_seed=3135; o.sampler=RNS8_VERIFY_SPLITMIX64_BINARY_V1;
  o.max_workspace_bytes=UINT64_MAX; o.flags=RNS8_VERIFY_CHECK_GENERATIONS;
  const rns8_matrix* matrices[]={a.get(),b.get(),c.get()};
  for(int i=0;i<3;++i) o.expected_generations[i]=matrices[i]->content_generation;
  rns8_finite_verify_report r{}; r.struct_size=sizeof(r); r.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_verify_finite_product(ctx,a.get(),b.get(),c.get(),&o,&r)==RNS8_SUCCESS);
  CHECK(r.passed==1); CHECK(r.completed_trials==32); CHECK(r.modulus==q);
  CHECK(r.ideal_failure_bound_exponent==32); CHECK(r.ideal_failure_probability==1.0/4294967296.0);
  for(int i=0;i<3;++i) { CHECK(r.generations[i]==o.expected_generations[i]); CHECK(r.matrix_instance_ids[i]==matrices[i]->matrix_instance_id); }
  const auto bytes=r.workspace_bytes;
  o.max_workspace_bytes=bytes-1; r.passed=99;
  CHECK(rns8_verify_finite_product(ctx,a.get(),b.get(),c.get(),&o,&r)==RNS8_RANGE_ERROR);
  CHECK(r.passed==99); o.max_workspace_bytes=bytes;
  c->in_flight_writer=true;
  CHECK(rns8_verify_finite_product(ctx,a.get(),b.get(),c.get(),&o,&r)==RNS8_BUSY);
  CHECK(r.passed==99); c->in_flight_writer=false;
  cv[1]=uint8_t((cv[1]+1)%q);
  REQUIRE(rns8_pack_finite_u8(ctx,c.get(),q,cv.data(),n,2)==RNS8_SUCCESS);
  CHECK(rns8_verify_finite_product(ctx,a.get(),b.get(),c.get(),&o,&r)==RNS8_INVALID_ARGUMENT);
  CHECK(r.passed==99);
  o.expected_generations[2]=c->content_generation;
  REQUIRE(rns8_verify_finite_product(ctx,a.get(),b.get(),c.get(),&o,&r)==RNS8_VERIFICATION_FAILED);
  CHECK(r.passed==0); const auto trials=r.completed_trials;
  REQUIRE(rns8_verify_finite_product(ctx,a.get(),b.get(),c.get(),&o,&r)==RNS8_VERIFICATION_FAILED);
  CHECK(r.completed_trials==trials);
  CHECK(c->content_generation==o.expected_generations[2]);
  if(q<128) {
    const int8_t bad=127;
    if(ctx->backend==RNS8_BACKEND_CPU_REFERENCE) c->residues[0]=bad;
    else REQUIRE(rns8::detail::hip_direct_copy_host_to_device(ctx->device_id,c->hip_residues,&bad,1)==RNS8_SUCCESS);
    r.passed=99;
    CHECK(rns8_verify_finite_product(ctx,a.get(),b.get(),c.get(),&o,&r)==RNS8_INVALID_ARGUMENT);
    CHECK(r.passed==99);
  }
}
}
