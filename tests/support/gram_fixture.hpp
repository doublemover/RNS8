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
        for(auto method:{RNS8_GRAM_SCALAR,RNS8_GRAM_WMMA,RNS8_GRAM_SCALAR_SEPARATE_MIRROR,RNS8_GRAM_WMMA_SEPARATE_MIRROR}) {
          if(backend==RNS8_BACKEND_CPU_REFERENCE && (uint32_t(method)&1)) continue;
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
          if(status==RNS8_UNSUPPORTED_BACKEND && (uint32_t(method)&1)) {
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

// Independent full Boost oracle for every new ownership/preparation ablation.
inline void check_gram_prepared_ablations(rns8_backend_kind backend) {
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=backend;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(backend==RNS8_BACKEND_HIP_DIRECT?0:-1,&options,&ctx)==RNS8_SUCCESS);
  for(auto orientation:{RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SELF_TIMES_TRANSPOSE})
  for(int n:{1,15,16,17,31,32,33})for(int k:(n==1?std::vector<int>{1,65535,65536,65537}:std::vector<int>{17})) {
    const int rows=orientation==RNS8_GRAM_TRANSPOSE_TIMES_SELF?k:n,cols=orientation==RNS8_GRAM_TRANSPOSE_TIMES_SELF?n:k,ld=cols+2;
    rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.rows=rows;d.cols=cols;d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
    d.semantics=RNS8_EXACT_WIDE_SIGNED;d.bound_kind=RNS8_BOUND_NONE;d.max_prefix=20;d.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    rns8_matrix *x=nullptr,*c=nullptr;REQUIRE(rns8_create_matrix(ctx,&d,&x)==RNS8_SUCCESS);
    d.rows=d.cols=n;d.flags=0;REQUIRE(rns8_create_matrix(ctx,&d,&c)==RNS8_SUCCESS);
    std::vector<int64_t> values(rows*ld,INT64_MAX);
    for(int r=0;r<rows;++r)for(int col=0;col<cols;++col)values[r*ld+col]=(r+col)%3?int64_t((r*7+col*3)%11)-5:0;
    values[0]=INT64_MIN;if(cols>1)values[1]=INT64_MAX;
    REQUIRE(rns8_pack_i64(ctx,x,values.data(),ld,1)==RNS8_SUCCESS);
    for(auto policy:{RNS8_GRAM_RESIDENT_RESIDUES,RNS8_GRAM_RETAINED_NATIVE_I64,RNS8_GRAM_RETAINED_TRANSPOSE})
    for(auto method:{RNS8_GRAM_SCALAR,RNS8_GRAM_WMMA,RNS8_GRAM_SCALAR_SEPARATE_MIRROR,RNS8_GRAM_WMMA_SEPARATE_MIRROR}) {
      if(backend==RNS8_BACKEND_CPU_REFERENCE && (uint32_t(method)&1))continue;
      rns8_prepared_gram* owner=nullptr;
      CHECK(rns8_prepare_gram_ex(ctx,x,c,orientation,method,policy,1,UINT64_MAX,&owner)==RNS8_RANGE_ERROR);
      REQUIRE(owner==nullptr);
      REQUIRE(rns8_prepare_gram_ex(ctx,x,c,orientation,method,policy,3,UINT64_MAX,&owner)==RNS8_SUCCESS);
      rns8_prepared_gram_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_get_prepared_gram_info(owner,&info)==RNS8_SUCCESS);
      CHECK(info.independent_cells==uint64_t(n)*(n+1)/2);CHECK(info.dense_cells==uint64_t(n)*n);
      CHECK(info.orientation_adapter_bytes==(policy==RNS8_GRAM_RETAINED_TRANSPOSE?uint64_t(rows)*cols*20:0));
      rns8_prepared_gram* rejected=nullptr;
      CHECK(rns8_prepare_gram_ex(ctx,x,c,orientation,method,policy,3,info.owned_payload_bytes-1,&rejected)==RNS8_WORKSPACE_TOO_SMALL);
      CHECK(rejected==nullptr);
      constexpr uint64_t poison=UINT64_C(0x82a5e67f938c10d4);std::vector<uint64_t> out(uint64_t(n)*(n+2)*3,poison);
      CHECK(rns8_export_prepared_gram_limbs(owner,out.data(),n+2)==RNS8_INVALID_ARGUMENT);
      const auto generation=c->content_generation;const auto status=rns8_execute_prepared_gram(owner);
      if(status==RNS8_UNSUPPORTED_BACKEND && (uint32_t(method)&1)){CHECK(c->content_generation==generation);REQUIRE(rns8_destroy_prepared_gram(owner)==RNS8_SUCCESS);continue;}
      REQUIRE(status==RNS8_SUCCESS);const auto allocations=hip_direct_allocation_counters_snapshot();
      REQUIRE(rns8_execute_prepared_gram(owner)==RNS8_SUCCESS);
      REQUIRE(rns8_export_prepared_gram_limbs(owner,out.data(),n+2)==RNS8_SUCCESS);
      CHECK(hip_direct_allocation_counters_snapshot().allocate_calls==allocations.allocate_calls);
      std::vector<int8_t> residues;
      if(backend==RNS8_BACKEND_HIP_DIRECT){residues.resize(c->hip_residue_bytes);REQUIRE(hip_direct_copy_device_to_host(0,residues.data(),c->hip_residues,residues.size())==RNS8_SUCCESS);}else residues=c->residues;
      const cpp_int mask=(cpp_int(1)<<64)-1;
      for(int r=0;r<n;++r)for(int col=0;col<n+2;++col){
        if(col>=n){for(int limb=0;limb<3;++limb)CHECK(out[(r*(n+2)+col)*3+limb]==poison);continue;}
        cpp_int expected=0;for(int q=0;q<k;++q){const auto at=[&](int i){return orientation==RNS8_GRAM_TRANSPOSE_TIMES_SELF?q*ld+i:i*ld+q;};expected+=cpp_int(values[at(r)])*values[at(col)];}
        for(uint32_t plane=0;plane<20;++plane)REQUIRE(residues[(plane*n+r)*n+col]==centered_residue(expected,kDefaultModuli[plane]));
        cpp_int bits=expected;if(bits<0)bits+=cpp_int(1)<<192;
        for(int limb=0;limb<3;++limb)REQUIRE(out[(r*(n+2)+col)*3+limb]==((bits>>(64*limb))&mask).convert_to<uint64_t>());
      }
      if(backend==RNS8_BACKEND_CPU_REFERENCE)CHECK(rns8_export_prepared_gram_limbs(owner,reinterpret_cast<uint64_t*>(x->native_i64.data()),n)==RNS8_INVALID_ARGUMENT);
      // A different producer of C invalidates the symmetry certificate even if X is unchanged.
      std::vector<int64_t> unrelated(n*n,0);REQUIRE(rns8_pack_i64(ctx,c,unrelated.data(),n,9)==RNS8_SUCCESS);
      const auto preserved=out;CHECK(rns8_export_prepared_gram_limbs(owner,out.data(),n+2)==RNS8_INVALID_ARGUMENT);CHECK(out==preserved);
      REQUIRE(rns8_pack_i64(ctx,x,values.data(),ld,2)==RNS8_SUCCESS);
      CHECK(rns8_execute_prepared_gram(owner)==RNS8_INVALID_ARGUMENT);REQUIRE(rns8_refresh_prepared_gram(owner)==RNS8_SUCCESS);
      CHECK(rns8_export_prepared_gram_limbs(owner,out.data(),n+2)==RNS8_INVALID_ARGUMENT);
      REQUIRE(rns8_execute_prepared_gram(owner)==RNS8_SUCCESS);REQUIRE(rns8_export_prepared_gram_limbs(owner,out.data(),n+2)==RNS8_SUCCESS);
      CHECK(out==preserved);REQUIRE(rns8_destroy_prepared_gram(owner)==RNS8_SUCCESS);
    }
    REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(x)==RNS8_SUCCESS);
  }
  // Every default prefix is numerically exercised with a representable rank-one
  // signed Gram; the extrema/long-K fixtures above require the full basis.
  for(uint32_t prefix=1;prefix<=20;++prefix){
    rns8_matrix_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.rows=1;d.cols=17;d.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
    d.semantics=RNS8_EXACT_WIDE_SIGNED;d.bound_kind=RNS8_BOUND_NONE;d.max_prefix=prefix;d.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    rns8_matrix *x=nullptr,*c=nullptr;REQUIRE(rns8_create_matrix(ctx,&d,&x)==RNS8_SUCCESS);d.rows=17;d.flags=0;REQUIRE(rns8_create_matrix(ctx,&d,&c)==RNS8_SUCCESS);
    std::vector<int64_t> values(17);for(int i=0;i<17;++i)values[i]=i%3-1;
    REQUIRE(rns8_pack_i64(ctx,x,values.data(),17,1)==RNS8_SUCCESS);
    for(auto policy:{RNS8_GRAM_RESIDENT_RESIDUES,RNS8_GRAM_RETAINED_NATIVE_I64,RNS8_GRAM_RETAINED_TRANSPOSE}){
      rns8_prepared_gram* owner=nullptr;REQUIRE(rns8_prepare_gram_ex(ctx,x,c,RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SCALAR_SEPARATE_MIRROR,policy,1,UINT64_MAX,&owner)==RNS8_SUCCESS);
      REQUIRE(rns8_execute_prepared_gram(owner)==RNS8_SUCCESS);std::vector<uint64_t> actual(17*17);
      REQUIRE(rns8_export_prepared_gram_limbs(owner,actual.data(),17)==RNS8_SUCCESS);
      for(int r=0;r<17;++r)for(int col=0;col<17;++col)REQUIRE(actual[r*17+col]==uint64_t(values[r]*values[col]));
      REQUIRE(rns8_destroy_prepared_gram(owner)==RNS8_SUCCESS);
    }
    REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(x)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
