#include <catch2/catch_test_macros.hpp>
#include "backend_common/gram_five_product.hpp"
#include "backend_common/finite_u8_reducer.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <vector>
#include "../support/gram_five_fixture.hpp"
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
      const auto saved=source;gram_five_product_cpu(source.data(),scratch.data(),r);CHECK(source==saved);
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
