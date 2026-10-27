#pragma once
#include "rns8/experimental_basis.h"
#include "core/experimental_basis.hpp"
#include "backend_common/basis_word_arithmetic.hpp"
#include <algorithm>
#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <vector>
namespace experimental_basis_fixture {
inline rns8_basis* create(const std::array<uint16_t,20>& moduli) {
  rns8_basis_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.representation=1;
  d.moduli=moduli.data();d.count=20;rns8_basis* basis=nullptr;REQUIRE(rns8_create_basis(&d,&basis)==RNS8_SUCCESS);return basis;
}
inline rns8_basis_plan* plan(rns8_context* ctx,const rns8_basis* basis,rns8_semantics semantics,int64_t m,int64_t n,int64_t k,uint32_t prefix) {
  rns8_basis_plan_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.semantics=semantics;
  d.backend=ctx->backend;d.m=m;d.n=n;d.k=k;d.prefix=prefix;
  rns8_basis_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_query_basis_plan(basis,&d,&resources)==RNS8_WORKSPACE_TOO_SMALL);
  d.max_device_bytes=resources.required_device_bytes;d.max_host_bytes=resources.required_host_bytes;
  rns8_basis_plan* owner=nullptr;REQUIRE(rns8_create_basis_plan(ctx,basis,&d,&owner)==RNS8_SUCCESS);return owner;
}
inline rns8_matrix* matrix(rns8_context* ctx,const rns8_basis* basis,rns8_semantics semantics,int64_t rows,int64_t cols) {
  rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.rows=rows;d.cols=cols;
  d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;d.logical_ld=cols;d.semantics=semantics;d.bound_kind=RNS8_BOUND_NONE;d.max_prefix=20;
  rns8_matrix* value=nullptr;REQUIRE(rns8_create_basis_matrix(ctx,basis,&d,&value)==RNS8_SUCCESS);return value;
}
inline std::array<uint16_t,20> moduli() {
  std::array<uint16_t,20> out{};std::copy_n(rns8::detail::kDefaultModuli,20,out.begin());std::swap(out[0],out[1]);return out;
}
inline void roundtrip(rns8_context* ctx,bool sign) {
  using boost::multiprecision::cpp_int;
  auto ordered=moduli(),reversed=ordered;std::reverse(reversed.begin(),reversed.end());
  auto* first=create(ordered);auto* second=create(reversed);
  const auto semantics=sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
  auto* p=plan(ctx,first,semantics,3,2,5,20);auto* q=plan(ctx,second,semantics,3,2,5,20);
  auto* defaults=plan(ctx,nullptr,semantics,3,2,5,20);
  auto* a=matrix(ctx,first,semantics,3,5);auto* b=matrix(ctx,first,semantics,5,2);auto* c=matrix(ctx,first,semantics,3,2);
  auto* converted=matrix(ctx,second,semantics,3,2);auto* restored=matrix(ctx,nullptr,semantics,3,2);
  REQUIRE(rns8_destroy_basis(first)==RNS8_SUCCESS);REQUIRE(rns8_destroy_basis(second)==RNS8_SUCCESS);
  rns8_basis_info identity{};identity.struct_size=sizeof(identity);identity.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_get_matrix_basis(a,&identity)==RNS8_SUCCESS);CHECK(identity.count==20);
  REQUIRE(std::equal(ordered.begin(),ordered.end(),identity.moduli));
  cpp_int product=1;for(auto modulus:ordered)product*=modulus;
  CHECK(cpp_int(identity.prefix_product[0])+(cpp_int(identity.prefix_product[1])<<64)+(cpp_int(identity.prefix_product[2])<<128)==product);
  const uint64_t edges[]{0,1,UINT64_MAX,UINT64_C(1)<<63,UINT64_C(0x7fffffffffffffff)};
  std::vector<uint64_t> av(3*7,99),bv(5*5,99);
  for(int row=0;row<3;++row)for(int col=0;col<5;++col)av[row*7+col]=edges[(row+col)%5];
  for(int row=0;row<5;++row)for(int col=0;col<2;++col)bv[row*5+col]=edges[(row*2+col+1)%5];
  const auto pack=[&](rns8_matrix* value,const std::vector<uint64_t>& data,int64_t ld) {
    return sign?rns8_pack_basis_i64(p,value,reinterpret_cast<const int64_t*>(data.data()),ld,1):rns8_pack_basis_u64(p,value,data.data(),ld,1);
  };
  REQUIRE(pack(a,av,7)==RNS8_SUCCESS);REQUIRE(pack(b,bv,5)==RNS8_SUCCESS);
  const auto generation=a->content_generation;REQUIRE(pack(a,av,7)==RNS8_SUCCESS);CHECK(a->content_generation==generation);
  REQUIRE(rns8_gemm_basis(p,a,b,c)==RNS8_SUCCESS);
  std::vector<uint64_t> output(27,99),other(27,99),restored_output(27,99);
  REQUIRE(rns8_export_basis_limbs(p,c,output.data(),3,3)==RNS8_SUCCESS);
  const auto exported_generation=c->content_generation;const auto exported=output;
  REQUIRE(rns8_export_basis_limbs(p,c,output.data(),3,1)==RNS8_RANGE_ERROR);
  CHECK(output==exported);CHECK(c->content_generation==exported_generation);
  for(int row=0;row<3;++row)for(int col=0;col<2;++col) {
    cpp_int expected=0;
    for(int k=0;k<5;++k)expected+=(sign?cpp_int(static_cast<int64_t>(av[row*7+k])):cpp_int(av[row*7+k]))*
        (sign?cpp_int(static_cast<int64_t>(bv[k*5+col])):cpp_int(bv[k*5+col]));
    const auto at=(row*3+col)*3;cpp_int actual=cpp_int(output[at])+(cpp_int(output[at+1])<<64)+(cpp_int(output[at+2])<<128);
    if(sign && output[at+2]>>63)actual-=cpp_int(1)<<192;
    REQUIRE(actual==expected);
  }
  auto* ragged_copy=matrix(ctx,nullptr,semantics,3,2);
  // A differently ordered destination must never reinterpret the directory.
  rns8_ragged_materialization* ragged=nullptr;
  REQUIRE(rns8_prepare_ragged_materialization(ctx,c,2,1,UINT64_MAX,&ragged)==RNS8_SUCCESS);
  CHECK(rns8_materialize_ragged_exact(ctx,ragged,ragged_copy,UINT64_MAX)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_matrix(ragged_copy)==RNS8_SUCCESS);
  auto same_desc=c->desc;REQUIRE(rns8_create_matrix(ctx,&same_desc,&ragged_copy)==RNS8_SUCCESS);
  ragged_copy->experimental_basis=c->experimental_basis;
  REQUIRE(rns8_materialize_ragged_exact(ctx,ragged,ragged_copy,UINT64_MAX)==RNS8_SUCCESS);
  std::vector<uint64_t> ragged_result(27,99);
  REQUIRE(rns8_export_basis_limbs(p,ragged_copy,ragged_result.data(),3,3)==RNS8_SUCCESS);CHECK(ragged_result==output);
  REQUIRE(rns8_destroy_ragged_materialization(ragged)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(ragged_copy)==RNS8_SUCCESS);
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(ctx->backend==RNS8_BACKEND_HIP_DIRECT) {
    rns8_prepared_execution* prepared=nullptr;
    REQUIRE(rns8_prepare_basis_gemm_hip(p,a,b,c,&prepared)==RNS8_SUCCESS);
    REQUIRE(rns8_capture_prepared_hip_graph(prepared)==RNS8_SUCCESS);
    rns8::detail::hip_unique_stream stream;REQUIRE(stream.create_non_blocking()==hipSuccess);
    for(bool graph:{false,true}) {
      rns8_completion_token token{};
      REQUIRE((graph?rns8_submit_prepared_hip_graph(prepared,stream.get(),&token):rns8_submit_prepared_hip(prepared,stream.get(),&token))==RNS8_SUCCESS);
      CHECK(rns8_destroy_matrix(a)==RNS8_BUSY);CHECK(pack(a,av,7)==RNS8_BUSY);
      rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_wait_prepared_hip(prepared,token,&info)==RNS8_SUCCESS);REQUIRE(info.operation_status==RNS8_SUCCESS);CHECK(info.output_is_current);
      REQUIRE(rns8_export_basis_limbs(p,c,other.data(),3,3)==RNS8_SUCCESS);CHECK(other==output);
    }
    rns8_completion_token canceled{};REQUIRE(rns8_submit_prepared_hip(prepared,stream.get(),&canceled)==RNS8_SUCCESS);
    REQUIRE(rns8_cancel_prepared_hip(prepared,canceled)==RNS8_SUCCESS);
    rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_wait_prepared_hip(prepared,canceled,&info)==RNS8_SUCCESS);CHECK(info.operation_status==RNS8_CANCELED);CHECK_FALSE(info.output_is_current);
    rns8_completion_token restored_token{};REQUIRE(rns8_execute_prepared(prepared,&restored_token)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_prepared_execution(prepared)==RNS8_SUCCESS);
  }
#endif
  const auto prior=converted->content_generation;
  REQUIRE(rns8_gemm_basis(q,a,b,converted)==RNS8_INVALID_ARGUMENT);CHECK(converted->content_generation==prior);
  const auto source_generation=c->content_generation;
  REQUIRE(rns8_convert_basis_values(q,c,converted)==RNS8_SUCCESS);CHECK(c->content_generation==source_generation);
  REQUIRE(rns8_export_basis_limbs(q,converted,other.data(),3,3)==RNS8_SUCCESS);REQUIRE(other==output);
  uint16_t tiny_modulus=251;rns8_basis_desc tiny_desc{};tiny_desc.struct_size=sizeof(tiny_desc);tiny_desc.abi_version=RNS8_ABI_VERSION;
  tiny_desc.representation=1;tiny_desc.moduli=&tiny_modulus;tiny_desc.count=1;rns8_basis* tiny=nullptr;
  REQUIRE(rns8_create_basis(&tiny_desc,&tiny)==RNS8_SUCCESS);
  auto* tiny_plan=plan(ctx,tiny,semantics,3,2,5,1);
  auto tiny_md=c->desc;tiny_md.max_prefix=1;rns8_matrix* narrow=nullptr;
  REQUIRE(rns8_create_basis_matrix(ctx,tiny,&tiny_md,&narrow)==RNS8_SUCCESS);
  const auto narrow_generation=narrow->content_generation;
  REQUIRE(rns8_convert_basis_values(tiny_plan,c,narrow)==RNS8_RANGE_ERROR);CHECK(narrow->content_generation==narrow_generation);
  REQUIRE(rns8_destroy_matrix(narrow)==RNS8_SUCCESS);REQUIRE(rns8_destroy_basis_plan(tiny_plan)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_basis(tiny)==RNS8_SUCCESS);
  REQUIRE(rns8_convert_basis_values(defaults,converted,restored)==RNS8_SUCCESS);
  REQUIRE(rns8_export_basis_limbs(defaults,restored,restored_output.data(),3,3)==RNS8_SUCCESS);REQUIRE(restored_output==output);
  for(int row=0;row<3;++row)for(int limb=0;limb<3;++limb)CHECK(output[(row*3+2)*3+limb]==99);
  // The unchanged API cannot reinterpret experimental plane order.
  rns8_gemm_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;gd.semantics=semantics;
  gd.bound_kind=RNS8_BOUND_NONE;gd.requested_backend=ctx->backend;gd.m=3;gd.n=2;gd.k=5;gd.max_prefix=20;gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
  rns8_plan* ordinary=nullptr;rns8_workspace* workspace=nullptr;
  REQUIRE(rns8_create_plan(ctx,&gd,&ordinary)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,ordinary,&workspace)==RNS8_SUCCESS);
  const auto restored_generation=restored->content_generation;
  REQUIRE(rns8_gemm_rns(ctx,ordinary,a,b,restored,workspace)==RNS8_INVALID_ARGUMENT);CHECK(restored->content_generation==restored_generation);
  std::fill(restored_output.begin(),restored_output.end(),99);
  REQUIRE((sign?rns8_export_exact_wide_signed_limbs(ctx,ordinary,restored,restored_output.data(),3,3):
      rns8_export_exact_wide_unsigned_limbs(ctx,ordinary,restored,restored_output.data(),3,3))==RNS8_SUCCESS);
  REQUIRE(restored_output==output);
  REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(ordinary)==RNS8_SUCCESS);
  for(auto* value:{a,b,c,converted,restored})REQUIRE(rns8_destroy_matrix(value)==RNS8_SUCCESS);
  for(auto* owner:{p,q,defaults})REQUIRE(rns8_destroy_basis_plan(owner)==RNS8_SUCCESS);
}
inline void slab_edges(rns8_context* ctx) {
  using boost::multiprecision::cpp_int;
  auto* basis=create(moduli());
  for(int64_t k:{65535,65536,65537}) {
    auto* p=plan(ctx,basis,RNS8_EXACT_WIDE_SIGNED,1,1,k,9);
    auto* a=matrix(ctx,basis,RNS8_EXACT_WIDE_SIGNED,1,k);auto* b=matrix(ctx,basis,RNS8_EXACT_WIDE_SIGNED,k,1);
    auto* c=matrix(ctx,basis,RNS8_EXACT_WIDE_SIGNED,1,1);std::vector<int64_t> values(size_t(k),128);
    REQUIRE(rns8_pack_basis_i64(p,a,values.data(),k,1)==RNS8_SUCCESS);REQUIRE(rns8_pack_basis_i64(p,b,values.data(),1,1)==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_basis(p,a,b,c)==RNS8_SUCCESS);uint64_t result=0;
    REQUIRE(rns8_export_basis_limbs(p,c,&result,1,1)==RNS8_SUCCESS);CHECK(cpp_int(result)==cpp_int(k)*128*128);
    CHECK(c->exact_range_prefix==9);
    for(auto* value:{a,b,c})REQUIRE(rns8_destroy_matrix(value)==RNS8_SUCCESS);REQUIRE(rns8_destroy_basis_plan(p)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_basis(basis)==RNS8_SUCCESS);
}
}
