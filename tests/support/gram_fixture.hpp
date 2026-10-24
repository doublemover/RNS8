#pragma once
#include "core/api_internal.hpp"
#include "backend_common/gram_plan.hpp"
inline void check_public_gram(rns8_backend_kind backend,bool prepared=false) {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  rns8_context_options cd{};cd.struct_size=sizeof(cd);cd.abi_version=RNS8_ABI_VERSION;
  cd.requested_backend=backend;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(backend==RNS8_BACKEND_HIP_DIRECT?0:-1,&cd,&ctx)==RNS8_SUCCESS);
  for(auto semantics:{RNS8_BOUNDED_I64,RNS8_EXACT_WIDE_SIGNED,RNS8_FINITE_RING_U8,RNS8_FINITE_FIELD_U8})
    for(auto orientation:{RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SELF_TIMES_TRANSPOSE})
      for(int n:{1,15,16,17,31,32,33}) {
        const int k=n==1?65537:17,rows=orientation==RNS8_GRAM_TRANSPOSE_TIMES_SELF?k:n;
        const int cols=orientation==RNS8_GRAM_TRANSPOSE_TIMES_SELF?n:k,ld=cols+2;
        const bool bounded=semantics==RNS8_BOUNDED_I64;
        const bool finite=semantics==RNS8_FINITE_RING_U8 || semantics==RNS8_FINITE_FIELD_U8;
        const uint16_t modulus=semantics==RNS8_FINITE_RING_U8?250:251;
        rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;
        md.rows=rows;md.cols=cols;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
        md.semantics=semantics;md.bound_kind=bounded?RNS8_BOUND_GLOBAL_MAX_ABS:RNS8_BOUND_NONE;md.max_prefix=finite?0:20;
        rns8_matrix *x=nullptr,*c=nullptr;
        REQUIRE(rns8_create_matrix(ctx,&md,&x)==RNS8_SUCCESS);
        md.rows=md.cols=n;REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
        std::vector<int64_t> values(rows*ld,INT64_MAX);
        std::vector<uint8_t> bytes(rows*ld,255);
        for(int r=0;r<rows;++r) for(int col=0;col<cols;++col) {
          values[r*ld+col]=((r*7+col*3)%11)-5;
          if(!bounded && n==17 && r==0) values[r*ld+col]=col%2?INT64_MIN:INT64_MAX;
          bytes[r*ld+col]=uint8_t((r*7+col*3)%modulus);
        }
        if(finite) REQUIRE(rns8_pack_finite_u8(ctx,x,modulus,bytes.data(),ld,0)==RNS8_SUCCESS);
        else REQUIRE(rns8_pack_i64(ctx,x,values.data(),ld,0)==RNS8_SUCCESS);
        const auto old=c->content_generation;
        CHECK(rns8_gram_residues(ctx,x,c,orientation,RNS8_GRAM_SCALAR,0)==RNS8_WORKSPACE_TOO_SMALL);
        CHECK(c->content_generation==old);
        CHECK(rns8_gram_residues(ctx,x,x,orientation,RNS8_GRAM_SCALAR,UINT64_MAX)==RNS8_INVALID_ARGUMENT);
        for(auto method:{RNS8_GRAM_SCALAR,RNS8_GRAM_WMMA}) {
          if(backend==RNS8_BACKEND_CPU_REFERENCE && method==RNS8_GRAM_WMMA) continue;
          rns8_prepared_gram* owner=nullptr;
          if(prepared) {
            REQUIRE(rns8_prepare_gram(ctx,x,c,orientation,method,UINT64_MAX,&owner)==RNS8_SUCCESS);
            CHECK(rns8_destroy_matrix(x)==RNS8_BUSY);
            CHECK(rns8_destroy_matrix(c)==RNS8_BUSY);
            CHECK(rns8_destroy_context(ctx)==RNS8_BUSY);
          }
          const auto execute=[&] {return prepared?rns8_execute_prepared_gram(owner):
            rns8_gram_residues(ctx,x,c,orientation,method,UINT64_MAX);};
          const auto status=execute();
          if(status==RNS8_UNSUPPORTED_BACKEND && method==RNS8_GRAM_WMMA) {
            REQUIRE(rns8_destroy_prepared_gram(owner)==RNS8_SUCCESS);continue;
          }
          if(prepared && status==RNS8_SUCCESS) {
            const auto before=hip_direct_allocation_counters_snapshot();
            REQUIRE(execute()==RNS8_SUCCESS);
            const auto after=hip_direct_allocation_counters_snapshot();
            CHECK(after.allocate_calls==before.allocate_calls);
            CHECK(after.free_calls==before.free_calls);
          }
          REQUIRE(status==RNS8_SUCCESS);
          std::vector<int8_t> actual;
          if(backend==RNS8_BACKEND_HIP_DIRECT) {
            actual.resize(c->hip_residue_bytes);
            REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),c->hip_residues,actual.size())==RNS8_SUCCESS);
          } else actual=c->residues;
          const auto get=[&](int index,int q)->cpp_int {
            const int at=orientation==RNS8_GRAM_TRANSPOSE_TIMES_SELF?q*ld+index:index*ld+q;
            return finite?cpp_int(bytes[at]):cpp_int(values[at]);
          };
          for(int row=0;row<n;++row) for(int col=0;col<n;++col) {
            cpp_int expected=0;
            for(int q=0;q<k;++q) expected+=get(row,q)*get(col,q);
            for(uint32_t p=0;p<(finite?1:20);++p) {
              const auto canonical=centered_residue(expected,finite?modulus:kDefaultModuli[p]);
              REQUIRE(actual[(p*n+row)*n+col]==canonical);
              REQUIRE(actual[(p*n+row)*n+col]==actual[(p*n+col)*n+row]);
            }
          }
          c->in_flight_readers=1;
          CHECK(execute()==RNS8_BUSY);
          c->in_flight_readers=0;
          if(prepared) {
            if(finite) REQUIRE(rns8_pack_finite_u8(ctx,x,modulus,bytes.data(),ld,0)==RNS8_SUCCESS);
            else REQUIRE(rns8_pack_i64(ctx,x,values.data(),ld,0)==RNS8_SUCCESS);
            const auto generation=c->content_generation;
            CHECK(execute()==RNS8_INVALID_ARGUMENT);
            CHECK(c->content_generation==generation);
            REQUIRE(rns8_refresh_prepared_gram(owner)==RNS8_SUCCESS);
            REQUIRE(execute()==RNS8_SUCCESS);
            REQUIRE(rns8_destroy_prepared_gram(owner)==RNS8_SUCCESS);
          }
        }
        REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(x)==RNS8_SUCCESS);
      }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
