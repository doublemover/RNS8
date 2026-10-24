TEST_CASE("single-source Gram CPU products preserve full signed and finite semantics") {
  check_public_gram(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("Gram triangular jobs own every symmetric output exactly once") {
  using namespace rns8::detail;
  for(int n:{1,15,16,17,31,32,33,129}) {
    GramProblem p;p.rows=19;p.cols=n;p.source_ld=n+3;p.output_ld=n+2;
    p.source_instance=1;p.prefix=20;
    const auto support=query_gram_support(p,UINT64_MAX,false,false);
    REQUIRE(support.status==RNS8_SUCCESS);
    const auto jobs=gram_tiles(support);
    const uint64_t q=(n+15)/16;
    REQUIRE(jobs.size()==q*(q+1)/2);
    std::vector<int> owners(n*n,0);
    for(auto job:jobs) for(int r=0;r<16;++r) for(int c=0;c<16;++c) {
      const int row=job.row*16+r,col=job.col*16+c;
      if(row>=n || col>=n || row>col) continue;
      ++owners[row*n+col];if(row!=col) ++owners[col*n+row];
    }
    for(auto count:owners) REQUIRE(count==1);
    CHECK(query_gram_support(p,support.job_bytes-1,false,false).status==RNS8_WORKSPACE_TOO_SMALL);
    p.orientation=static_cast<GramOrientation>(2);
    CHECK(query_gram_support(p,UINT64_MAX,false,false).status==RNS8_INVALID_ARGUMENT);
  }
}

TEST_CASE("bounded Gram proves the accumulation instead of trusting a small result claim") {
  auto* ctx=create_cpu_context();
  auto md=bounded_matrix_desc(1,1,RNS8_BOUNDED_I64);
  rns8_matrix *x=nullptr,*c=nullptr;
  REQUIRE(rns8_create_matrix(ctx,&md,&x)==RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
  int64_t value=INT64_MAX;
  REQUIRE(rns8_pack_i64(ctx,x,&value,1,0)==RNS8_SUCCESS);
  const auto before=c->residues;
  CHECK(rns8_gram_residues(ctx,x,c,RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SCALAR,UINT64_MAX)
        ==RNS8_RANGE_ERROR);
  CHECK(c->residues==before);
  value=7;REQUIRE(rns8_pack_i64(ctx,x,&value,1,0)==RNS8_SUCCESS);
  REQUIRE(rns8_gram_residues(ctx,x,c,RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SCALAR,UINT64_MAX)
          ==RNS8_SUCCESS);
  CHECK(c->exact_max_magnitude==49);
  x->exact_range_prefix=0;
  CHECK(rns8_gram_residues(ctx,x,c,RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SCALAR,UINT64_MAX)
        ==RNS8_RANGE_ERROR);
  REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(x)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
