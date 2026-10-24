#pragma once
#include "core/internal.hpp"
#include "rns8/rns8.h"
namespace ragged_owner_fixture {
inline void roundtrip(rns8_context* ctx,bool unsigned_values) {
  constexpr int64_t m=17,k=19,n=7;
  const auto semantics=unsigned_values?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
  const auto make=[&](int64_t rows,int64_t cols) {
    rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
    d.rows=rows;d.cols=cols;d.logical_ld=cols;d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
    d.semantics=semantics;d.bound_kind=RNS8_BOUND_NONE;d.max_prefix=20;
    rns8_matrix* x=nullptr;REQUIRE(rns8_create_matrix(ctx,&d,&x)==RNS8_SUCCESS);return x;
  };
  auto* source=make(m,k);auto* restored=make(m,k);auto* b=make(k,n);auto* c=make(m,n);
  std::vector<uint64_t> au(size_t(m*k)),bu(size_t(k*n),1);
  std::vector<int64_t> as(size_t(m*k)),bs(size_t(k*n),1);
  for(int64_t row=0;row<m;++row) for(int64_t col=0;col<k;++col) {
    const size_t index=size_t(row*k+col);
    const uint64_t value=col<5?0:col<10?uint64_t(row+col):((uint64_t{1}<<62)+index);
    au[index]=value;as[index]=(row+col)%2?-int64_t(value):int64_t(value);
  }
  const auto pack_source=[&](uint64_t version) {
    return unsigned_values?rns8_pack_u64(ctx,source,au.data(),k,version):rns8_pack_i64(ctx,source,as.data(),k,version);
  };
  REQUIRE(pack_source(1)==RNS8_SUCCESS);
  REQUIRE((unsigned_values?rns8_pack_u64(ctx,b,bu.data(),n,1):rns8_pack_i64(ctx,b,bs.data(),n,1))==RNS8_SUCCESS);
  rns8_ragged_materialization* owner=nullptr;
  CHECK(rns8_prepare_ragged_materialization(ctx,source,3,5,0,&owner)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(owner==nullptr);
  REQUIRE(rns8_prepare_ragged_materialization(ctx,source,3,5,UINT64_MAX,&owner)==RNS8_SUCCESS);
  rns8_ragged_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_get_ragged_resources(owner,&resources)==RNS8_SUCCESS);
  CHECK(resources.tile_count==24);CHECK(resources.minimum_prefix==1);CHECK(resources.maximum_prefix>1);
  CHECK(resources.payload_bytes<uint64_t(m*k*20));
  const uint64_t preparation_bytes=resources.payload_bytes+resources.directory_bytes*
      (ctx->backend==RNS8_BACKEND_HIP_DIRECT?2:1)+(ctx->backend==RNS8_BACKEND_HIP_DIRECT?4:0);
  rns8_ragged_materialization* exact_budget=nullptr;
  REQUIRE(rns8_prepare_ragged_materialization(ctx,source,3,5,preparation_bytes-1,&exact_budget)==RNS8_WORKSPACE_TOO_SMALL);
  REQUIRE(exact_budget==nullptr);CHECK(source->in_flight_readers==0);
  REQUIRE(rns8_prepare_ragged_materialization(ctx,source,3,5,preparation_bytes,&exact_budget)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_ragged_materialization(exact_budget)==RNS8_SUCCESS);
  CHECK(resources.source_instance==source->matrix_instance_id);CHECK(resources.source_generation==source->content_generation);
  CHECK(rns8_destroy_matrix(source)==RNS8_BUSY);CHECK(rns8_destroy_context(ctx)==RNS8_BUSY);
  CHECK(rns8_materialize_ragged_exact(ctx,owner,source,UINT64_MAX)==RNS8_INVALID_ARGUMENT);
  const auto generation=restored->content_generation;
  CHECK(rns8_materialize_ragged_exact(ctx,owner,restored,0)==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(restored->content_generation==generation);
  restored->in_flight_readers=1;
  CHECK(rns8_materialize_ragged_exact(ctx,owner,restored,UINT64_MAX)==RNS8_BUSY);
  restored->in_flight_readers=0;
  for(int repeat=0;repeat<2;++repeat) {
    const auto before=rns8::detail::hip_direct_allocation_counters_snapshot();
    REQUIRE(rns8_materialize_ragged_exact(ctx,owner,restored,UINT64_MAX)==RNS8_SUCCESS);
    const auto after=rns8::detail::hip_direct_allocation_counters_snapshot();
    if(repeat==1) {CHECK(after.allocate_calls==before.allocate_calls);CHECK(after.free_calls==before.free_calls);}
    REQUIRE(rns8_get_ragged_resources(owner,&resources)==RNS8_SUCCESS);
    CHECK(resources.materialization_capacity_bytes==uint64_t(ctx->backend==RNS8_BACKEND_HIP_DIRECT?m*k*20:0));
    CHECK(resources.device_metadata_bytes==(ctx->backend==RNS8_BACKEND_HIP_DIRECT?resources.directory_bytes+4:0));
    std::vector<int8_t> got(size_t(m*k*20)),expected(size_t(m*k*20));
    if(ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
      REQUIRE(rns8::detail::hip_direct_copy_device_to_host(ctx->device_id,got.data(),restored->hip_residues,got.size())==RNS8_SUCCESS);
      REQUIRE(rns8::detail::hip_direct_copy_device_to_host(ctx->device_id,expected.data(),source->hip_residues,expected.size())==RNS8_SUCCESS);
    } else {got=restored->residues;expected=source->residues;}
    CHECK(got==expected);
  }
  rns8_gemm_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;
  gd.m=m;gd.n=n;gd.k=k;gd.semantics=semantics;gd.bound_kind=RNS8_BOUND_NONE;
  gd.max_prefix=20;gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;gd.requested_backend=ctx->backend;
  rns8_plan* plan=nullptr;rns8_workspace* ws=nullptr;
  REQUIRE(rns8_create_plan(ctx,&gd,&plan)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,plan,&ws)==RNS8_SUCCESS);
  REQUIRE(rns8_gemm_rns(ctx,plan,restored,b,c,ws)==RNS8_SUCCESS);
  std::vector<uint64_t> limbs(size_t(m*n*3));
  REQUIRE((unsigned_values?rns8_export_exact_wide_unsigned_limbs(ctx,plan,c,limbs.data(),n,3):
      rns8_export_exact_wide_signed_limbs(ctx,plan,c,limbs.data(),n,3))==RNS8_SUCCESS);
  using boost::multiprecision::cpp_int;
  for(int64_t row=0;row<m;++row) for(int64_t col=0;col<n;++col) {
    cpp_int expected=0;
    for(int64_t j=0;j<k;++j) expected+=unsigned_values?cpp_int(au[size_t(row*k+j)]):cpp_int(as[size_t(row*k+j)]);
    const size_t at=size_t((row*n+col)*3);
    cpp_int actual=cpp_int(limbs[at])+(cpp_int(limbs[at+1])<<64)+(cpp_int(limbs[at+2])<<128);
    if(!unsigned_values && limbs[at+2]>>63) actual-=cpp_int(1)<<192;
    CHECK(actual==expected);
  }
  REQUIRE(pack_source(2)==RNS8_SUCCESS);
  const auto previous=restored->content_generation;
  CHECK(rns8_materialize_ragged_exact(ctx,owner,restored,UINT64_MAX)==RNS8_INVALID_ARGUMENT);
  CHECK(restored->content_generation==previous);
  REQUIRE(rns8_destroy_ragged_materialization(owner)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(ws)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  for(auto* matrix:{source,restored,b,c}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
}
}

namespace ragged_owner_fixture {
inline void wide_roundtrip(rns8_context* ctx,bool unsigned_values) {
  using boost::multiprecision::cpp_int;
  constexpr int rows=9,cols=11,prefix=20,cells=rows*cols;
  rns8_matrix_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;
  desc.rows=rows;desc.cols=cols;desc.logical_ld=cols;desc.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
  desc.semantics=unsigned_values?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
  desc.bound_kind=RNS8_BOUND_NONE;desc.max_prefix=prefix;
  rns8_matrix *source=nullptr,*output=nullptr;
  REQUIRE(rns8_create_matrix(ctx,&desc,&source)==RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx,&desc,&output)==RNS8_SUCCESS);
  std::vector<int8_t> encoded(cells*prefix);
  cpp_int bound=0;
  for(int cell=0;cell<cells;++cell) {
    cpp_int value=cell%7==0?cpp_int(0):(cpp_int(1)<<140)+cell;
    if(!unsigned_values && cell%2) value=-value;
    const cpp_int magnitude=boost::multiprecision::abs(value);
    if(magnitude>bound) bound=magnitude;
    for(int p=0;p<prefix;++p) {
      const int modulus=rns8::detail::kDefaultModuli[p];
      int residue=(value%modulus).convert_to<int>();if(residue<0) residue+=modulus;
      if(residue>=(modulus+1)/2) residue-=modulus;
      encoded[p*cells+cell]=int8_t(residue);
    }
  }
  if(ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(ctx->device_id,source->hip_residues,encoded.data(),encoded.size())==RNS8_SUCCESS);
    rns8::detail::api::mark_output_device_residues_current(*source);
  } else {source->residues=encoded;rns8::detail::api::mark_output_host_residues_current(*source);}
  source->exact_max_magnitude=bound;source->exact_range_prefix=prefix;
  rns8_ragged_materialization* owner=nullptr;
  // A forged smaller proof must reject on the discovery pass, before an owner exists.
  source->exact_max_magnitude=1;
  REQUIRE(rns8_prepare_ragged_materialization(ctx,source,3,4,UINT64_MAX,&owner)==RNS8_RANGE_ERROR);
  REQUIRE(owner==nullptr);CHECK(source->in_flight_readers==0);
  source->exact_max_magnitude=bound;
  REQUIRE(rns8_prepare_ragged_materialization(ctx,source,3,4,UINT64_MAX,&owner)==RNS8_SUCCESS);
  REQUIRE(rns8_materialize_ragged_exact(ctx,owner,output,UINT64_MAX)==RNS8_SUCCESS);
  std::vector<int8_t> actual(encoded.size());
  if(ctx->backend==RNS8_BACKEND_HIP_DIRECT)
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(ctx->device_id,actual.data(),output->hip_residues,actual.size())==RNS8_SUCCESS);
  else actual=output->residues;
  REQUIRE(actual==encoded);CHECK(output->exact_max_magnitude==bound);
  REQUIRE(rns8_destroy_ragged_materialization(owner)==RNS8_SUCCESS);owner=nullptr;
  // The second default modulus is below 256; -128 is outside its centered alphabet.
  REQUIRE(rns8::detail::kDefaultModuli[1]<256);
  encoded[cells]=-128;
  if(ctx->backend==RNS8_BACKEND_HIP_DIRECT)
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(ctx->device_id,source->hip_residues,encoded.data(),encoded.size())==RNS8_SUCCESS);
  else source->residues=encoded;
  REQUIRE(rns8_prepare_ragged_materialization(ctx,source,3,4,UINT64_MAX,&owner)==RNS8_INVALID_ARGUMENT);
  REQUIRE(owner==nullptr);CHECK(source->in_flight_readers==0);
  REQUIRE(rns8_destroy_matrix(output)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(source)==RNS8_SUCCESS);
}
}
