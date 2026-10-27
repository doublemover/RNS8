#pragma once
#include "core/api_internal.hpp"
#include <array>
#include <vector>
namespace gram_five_fixture {
template<class T>T abi(){T value{};value.struct_size=sizeof(value);value.abi_version=RNS8_ABI_VERSION;return value;}
inline void check(rns8_context* ctx,bool matrix) {
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  for(bool transpose:{false,true})for(auto shape:std::array<std::array<int64_t,2>,4>{{{1,1},{3,5},{17,33},{1,65537}}}) {
    const int64_t n=shape[0],k=shape[1],rows=transpose?k:n,cols=transpose?n:k,ld=cols+2;
    auto md=abi<rns8_matrix_desc>();md.rows=rows;md.cols=cols;md.logical_ld=cols;
    md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.semantics=RNS8_EXACT_WIDE_SIGNED;md.bound_kind=RNS8_BOUND_NONE;md.max_prefix=20;
    rns8_matrix *x=nullptr,*c=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&x)==RNS8_SUCCESS);
    md.rows=md.cols=md.logical_ld=n;REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
    std::vector<int64_t> source(size_t(rows*ld),99);
    const int64_t values[]{INT64_MIN,INT64_MAX,-1,0,1,17,-17};
    for(int64_t row=0;row<rows;++row)for(int64_t col=0;col<cols;++col)source[size_t(row*ld+col)]=values[(row+col)%7];
    REQUIRE(rns8_pack_i64(ctx,x,source.data(),ld,0)==RNS8_SUCCESS);
    const auto orientation=transpose?RNS8_GRAM_TRANSPOSE_TIMES_SELF:RNS8_GRAM_SELF_TIMES_TRANSPOSE;
    rns8_prepared_gram* owner=nullptr;
    CHECK(rns8_prepare_gram_five_product(ctx,x,c,orientation,matrix,0,&owner)==RNS8_WORKSPACE_TOO_SMALL);
    REQUIRE(owner==nullptr);REQUIRE(rns8_prepare_gram_five_product(ctx,x,c,orientation,matrix,UINT64_MAX,&owner)==RNS8_SUCCESS);
    CHECK(rns8_destroy_matrix(x)==RNS8_BUSY);
    for(int iteration=0;iteration<2;++iteration) {
      if(iteration){REQUIRE(rns8_pack_i64(ctx,x,source.data(),ld,0)==RNS8_SUCCESS);
        CHECK(rns8_execute_prepared_gram(owner)==RNS8_INVALID_ARGUMENT);REQUIRE(rns8_refresh_prepared_gram(owner)==RNS8_SUCCESS);}
      REQUIRE(rns8_execute_prepared_gram(owner)==RNS8_SUCCESS);
      std::vector<int8_t> result(size_t(20*n*n));
      if(ctx->backend==RNS8_BACKEND_HIP_DIRECT)REQUIRE(hip_direct_copy_device_to_host(ctx->device_id,result.data(),c->hip_residues,result.size())==RNS8_SUCCESS);
      else result=c->residues;
      for(int64_t row=0;row<n;++row)for(int64_t col=0;col<n;++col){cpp_int expected=0;
        for(int64_t q=0;q<k;++q){const auto at=[&](int64_t i){return transpose?q*ld+i:i*ld+q;};expected+=cpp_int(source[size_t(at(row))])*source[size_t(at(col))];}
        for(uint32_t p=0;p<20;++p)CHECK(result[size_t((int64_t(p)*n+row)*n+col)]==centered_residue(expected,kDefaultModuli[p]));
      }
      CHECK(c->exact_max_magnitude==cpp_int(k)*(cpp_int(1)<<126));
    }
    auto info=abi<rns8_prepared_gram_info>();REQUIRE(rns8_get_prepared_gram_info(owner,&info)==RNS8_SUCCESS);
    CHECK(info.method==4+uint32_t(matrix));CHECK(info.owned_payload_bytes>0);
    CHECK(info.algebraic_product_count==5);CHECK(info.algebraic_prime_planes>0);CHECK(info.classical_planes>0);
    CHECK(info.algebraic_prime_planes+info.classical_planes==20);
    CHECK(info.owned_payload_bytes>=info.algebraic_operand_bytes+info.algebraic_product_bytes+info.algebraic_staging_bytes);
    REQUIRE(rns8_destroy_prepared_gram(owner)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(x)==RNS8_SUCCESS);
  }
}
}
