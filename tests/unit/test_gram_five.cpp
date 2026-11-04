#include <catch2/catch_test_macros.hpp>
#include "backend_common/gram_five_product.hpp"
#include "backend_common/finite_u8_reducer.hpp"
#include "backend_amdgpu_builtins/amdgpu_builtins_backend.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <vector>
#include "../support/gram_five_fixture.hpp"
TEST_CASE("indexed plane partitions retain logical identity and compact only eligible products") {
  using namespace rns8::detail;
  const uint16_t moduli[]{251,256,239,255,253,233};IndexedModulusPlanes planes;
  REQUIRE(partition_modulus_planes(6,moduli,[](uint32_t q,uint32_t){return gram_five_prime(q);},planes));
  CHECK(planes.logical_count==6);CHECK(planes.eligible_count==3);CHECK(planes.fallback_count==3);
  for(unsigned compact=0;compact<3;++compact){CHECK(planes.eligible[compact]==2*compact+(compact==2));
    CHECK(planes.moduli[compact]==moduli[planes.eligible[compact]]);CHECK(planes.compact_index[planes.eligible[compact]]==compact);}
  for(unsigned index=0;index<planes.fallback_count;++index)CHECK(planes.compact_index[planes.fallback[index]]==UINT8_MAX);
  const auto saved=planes;const uint16_t invalid[]{0};CHECK_FALSE(partition_modulus_planes(1,invalid,[](uint32_t,uint32_t){return true;},planes));
  CHECK(planes.logical_count==saved.logical_count);CHECK(planes.eligible_count==saved.eligible_count);
  REQUIRE(partition_modulus_planes(6,moduli,[](uint32_t q,uint32_t){return q%2!=0;},planes));
  CHECK(planes.eligible_count==5);CHECK(planes.fallback_count==1);CHECK(planes.fallback[0]==1);
}

TEST_CASE("five product compact storage omits composite planes and preserves reordered moduli") {
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  const uint16_t basis[]{251,256,239,255,253,233};
  GramProblem p;p.rows=5;p.cols=3;p.source_ld=3;p.output_ld=3;p.prefix=6;
  const auto r=query_gram_five_product(p,basis,UINT64_MAX,false,false);REQUIRE(r.status==RNS8_SUCCESS);
  CHECK(r.packet.planes.eligible_count==3);CHECK(r.panel==3*2*4);CHECK(r.product==3*2*2);
  CHECK(r.output==6*3*3);CHECK(r.total==2*r.panel+5*r.product+r.output);
  CHECK(r.retained_bytes==r.total+r.host_job_bytes);
  CHECK(query_gram_five_product(p,basis,r.retained_bytes-1,false,false).status==RNS8_WORKSPACE_TOO_SMALL);
  CHECK(query_gram_five_product(p,basis,r.retained_bytes,false,false).status==RNS8_SUCCESS);
  std::vector<int8_t> source(6*5*3),scratch(size_t(r.total+32),85);
  for(uint32_t plane=0;plane<6;++plane)for(unsigned index=0;index<15;++index)
    source[plane*15+index]=int8_t(gram_five_center(int(index*97)-128,basis[plane]));
  const auto jobs=gram_tiles(query_gram_support(p,UINT64_MAX,false,false));
  gram_five_product_cpu(source.data(),scratch.data()+16,r,jobs);
  for(unsigned index=0;index<16;++index){CHECK(scratch[index]==85);CHECK(scratch[size_t(16+r.total+index)]==85);}
  for(uint32_t plane=0;plane<6;++plane)for(unsigned row=0;row<3;++row)for(unsigned col=0;col<3;++col){
    cpp_int expected=0;for(unsigned inner=0;inner<5;++inner)expected+=cpp_int(source[plane*15+inner*3+row])*source[plane*15+inner*3+col];
    CHECK(scratch[size_t(16+r.output_offset+plane*9+row*3+col)]==centered_residue(expected,basis[plane]));
  }
}

TEST_CASE("indexed matrix batches reject malformed identities before device work", "[indexed][admission]") {
  using namespace rns8::detail;const uint16_t moduli[]{251,256,239,255,253,233};IndexedModulusPlanes planes;
  REQUIRE(partition_modulus_planes(6,moduli,[](uint32_t q,uint32_t){return gram_five_prime(q);},planes));
  REQUIRE(valid_indexed_modulus_planes(planes));int8_t a[64]{},b[64]{},c[64]{};
  auto even=planes;even.moduli[0]=256;CHECK(valid_indexed_modulus_planes(even));CHECK_FALSE(valid_indexed_modulus_planes(even,true));
  auto invalid=planes;invalid.eligible[1]=invalid.eligible[0];CHECK_FALSE(valid_indexed_modulus_planes(invalid));
  CHECK(amdgpu_builtins_gemm_indexed_batch_device(0,a,b,c,1,1,1,1,1,1,invalid,1)==RNS8_INVALID_ARGUMENT);
  invalid=planes;invalid.moduli[0]=0;CHECK_FALSE(valid_indexed_modulus_planes(invalid));
  CHECK(amdgpu_builtins_gemm_indexed_batch_device(0,a,b,c,1,1,1,1,1,1,invalid,1)==RNS8_INVALID_ARGUMENT);
  CHECK(amdgpu_builtins_gemm_indexed_batch_device(0,a,b,c,1,1,1,1,1,1,planes,65)==RNS8_INVALID_ARGUMENT);
  CHECK(amdgpu_builtins_gemm_indexed_batch_device(0,a,b,c,1,1,65537,65537,1,1,planes,1)==RNS8_RANGE_ERROR);
  CHECK(amdgpu_builtins_gemm_indexed_batch_device(0,a,b,a,1,1,1,1,1,1,planes,1)==RNS8_INVALID_ARGUMENT);
}
TEST_CASE("five-product Gram verifies skew matrices and independent prime-field products") {
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  for(uint32_t modulus=2;modulus<=251;++modulus)if(gram_five_prime(modulus))
    for(bool transpose:{false,true})for(int64_t n:{1,3,5})for(int64_t k:{1,3,5,7}) {
      GramProblem p;p.source_instance=7;p.source_generation=9;p.prefix=1;p.modulus=modulus;p.semantics=RNS8_FINITE_FIELD_U8;
      p.orientation=transpose?GramOrientation::transpose_times_self:GramOrientation::self_times_transpose;
      p.rows=transpose?k:n;p.cols=transpose?n:k;p.source_ld=p.cols+3;p.output_ld=n;
      const auto r=query_gram_five_product(p,kDefaultModuli,UINT64_MAX,false,false);
      REQUIRE(r.status==RNS8_SUCCESS);REQUIRE(r.packet.certified[0]==1);
      const uint32_t a=r.packet.a[0],b=r.packet.b[0];CHECK((a*a+b*b)%modulus==modulus-1);CHECK(r.packet.hk%2==0);
      CHECK(query_gram_five_product(p,kDefaultModuli,r.total-1,false,false).status==RNS8_WORKSPACE_TOO_SMALL);
      std::vector<int8_t> source(size_t(p.rows*p.source_ld),99),scratch(size_t(r.total),77);
      for(int64_t row=0;row<p.rows;++row)for(int64_t col=0;col<p.cols;++col)source[size_t(row*p.source_ld+col)]=int8_t(gram_five_center(int(row*97+col*53)-128,int(modulus)));
      const auto saved=source;gram_five_product_cpu(source.data(),scratch.data(),r,{});CHECK(source==saved);
      for(int64_t row=0;row<n;++row)for(int64_t col=0;col<n;++col){cpp_int expected=0;
        for(int64_t q=0;q<k;++q){const auto at=[&](int64_t i){return transpose?q*p.source_ld+i:i*p.source_ld+q;};expected+=cpp_int(source[size_t(at(row))])*source[size_t(at(col))];}
        expected%=modulus;if(expected<0)expected+=modulus;int value=expected.convert_to<int>();if(value>=int((modulus+1)/2))value-=int(modulus);
        CHECK(scratch[size_t(r.output_offset+row*n+col)]==value);
      }
    }
}
TEST_CASE("five-product Gram public CPU owner retains the original integer proof and generations") {
  auto options=gram_five_fixture::abi<rns8_context_options>();options.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(-1,&options,&ctx)==RNS8_SUCCESS);
  gram_five_fixture::check(ctx,false);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
