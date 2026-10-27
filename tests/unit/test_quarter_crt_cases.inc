#include "backend_common/quarter_crt_certificate.hpp"
TEST_CASE("quarter CRT full weights quotient and delayed borrows agree with exact integers","[crt][certificate]") {
  using namespace rns8::detail;using boost::multiprecision::cpp_int;
  for(uint32_t prefix=1;prefix<=20;++prefix) {
    const cpp_int product=modulus_product(prefix),bound=product/4;
    quarter_crt::Certificate c;c.prefix=prefix;c.limbs=3;c.bound=bound;c.source_instance=1;c.source_generation=2;
    c.target="host-model";c.build="fixture";c.plan_hash=std::string(64,'a');
    for(uint32_t mode:{0u,1u,2u}) {
      c.mode=mode;c=quarter_crt::certify(c);REQUIRE(c.admitted);
      std::vector<cpp_int> values{0,1,bound,bound-1};if(mode!=1){values.push_back(-1);values.push_back(-bound);values.push_back(1-bound);}
      for(const auto& value:values) {
        uint32_t residues[20]{};cpp_int sum=0;
        for(uint32_t p=0;p<prefix;++p){cpp_int r=value%kDefaultModuli[p];if(r<0)r+=kDefaultModuli[p];residues[p]=static_cast<uint32_t>(r);
          sum+=r*quarter_crt::from_digits(quarter_crt::kConstants.weights[prefix][p]);}
        CHECK(quarter_crt::host_quotient(quarter_crt::kConstants,residues,prefix)==static_cast<uint32_t>((sum-value)/product));
        uint64_t output[4]{0,0,0,0xcafebabe};REQUIRE(quarter_crt::reconstruct_host(c,residues,output));CHECK(output[3]==0xcafebabe);
        cpp_int observed=cpp_int(output[0])+(cpp_int(output[1])<<64)+(cpp_int(output[2])<<128);
        cpp_int expected=value;if(mode==2 && expected<0)expected+=product;else if(expected<0)expected+=cpp_int(1)<<192;
        CHECK(observed==expected);
      }
    }
    c.mode=0;c.bound=bound+1;CHECK_FALSE(quarter_crt::certify(c).admitted);
    c.bound=bound;c.source_generation=0;CHECK_FALSE(quarter_crt::certify(c).admitted);c.source_generation=2;
    auto table=quarter_crt::kConstants;table.weights[prefix][0][0]^=1;
    CHECK(quarter_crt::certify(c,table).failure=="full_weight_table_corruption");
    table=quarter_crt::kConstants;table.fractions[prefix][0]^=1;
    CHECK(quarter_crt::certify(c,table).failure=="rne_fraction_table_corruption");
    if(prefix>8){c.limbs=1;CHECK(quarter_crt::certify(c).failure=="integer_destination_width");}
  }
  for(int32_t x:{INT32_C(-335600000),-65537,-65536,-65535,-1,0,1,65535,65536,INT32_C(335600000)}) {
    const auto carry=quarter_crt::euclidean_carry(x),digit=x-carry*65536;
    CHECK(digit>=0);CHECK(digit<65536);CHECK(int64_t(carry)*65536+digit==x);
  }
}
TEST_CASE("public quarter export owns source identity and publishes complete padded results","[api][crt]") {
  auto* ctx=create_cpu_context();
  for(uint32_t prefix=1;prefix<=20;++prefix) {
    auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,3,1);desc.max_prefix=prefix;
    desc.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND;rns8_plan* plan=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    auto md=exact_matrix_desc(1,3,RNS8_EXACT_WIDE_SIGNED);md.max_prefix=prefix;rns8_matrix* matrix=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    int64_t values[3]{-63,0,63};REQUIRE(rns8_pack_i64(ctx,matrix,values,3,1)==RNS8_SUCCESS);
    rns8_quarter_crt_desc config{};config.struct_size=sizeof(config);config.abi_version=RNS8_ABI_VERSION;config.limb_count=1;config.max_scratch_bytes=1024;
    rns8_quarter_crt_export* owner=nullptr;REQUIRE(rns8_prepare_quarter_crt_export(ctx,plan,matrix,&config,&owner)==RNS8_SUCCESS);
    rns8_quarter_crt_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_quarter_crt_info(owner,&info)==RNS8_SUCCESS);CHECK(info.delayed_digits==1);CHECK(info.target_execution_verdict==RNS8_PROOF_NOT_RUN);
    uint64_t output[5]{99,99,99,99,99};REQUIRE(rns8_export_quarter_crt_limbs(owner,output,5)==RNS8_SUCCESS);
    CHECK(output[0]==static_cast<uint64_t>(-63));CHECK(output[1]==0);CHECK(output[2]==63);CHECK(output[3]==99);CHECK(output[4]==99);
    CHECK(rns8_destroy_plan(plan)==RNS8_BUSY);CHECK(rns8_destroy_matrix(matrix)==RNS8_BUSY);
    values[0]=-62;REQUIRE(rns8_pack_i64(ctx,matrix,values,3,2)==RNS8_SUCCESS);std::fill(std::begin(output),std::end(output),99);
    CHECK(rns8_export_quarter_crt_limbs(owner,output,5)==RNS8_INVALID_ARGUMENT);for(auto word:output)CHECK(word==99);
    REQUIRE(rns8_destroy_quarter_crt_export(owner)==RNS8_SUCCESS);
    if(prefix==1){values[0]=65;REQUIRE(rns8_pack_i64(ctx,matrix,values,3,3)==RNS8_SUCCESS);
      CHECK(rns8_prepare_quarter_crt_export(ctx,plan,matrix,&config,&owner)==RNS8_RANGE_ERROR);
      REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,matrix,output,5,1)==RNS8_SUCCESS);CHECK(output[0]==65);}
    rns8_destroy_matrix(matrix);rns8_destroy_plan(plan);
  }
  rns8_destroy_context(ctx);
}
TEST_CASE("quarter CRT HIP signed canonical and tail exports use the public owner","[hip][crt]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr;if(rns8_create_context(0,&options,&ctx)!=RNS8_SUCCESS)SKIP("HIP device unavailable");
  for(uint32_t prefix=1;prefix<=20;++prefix)for(uint32_t mode:{0u,2u}) {
    auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,2,17,1);desc.max_prefix=prefix;desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    desc.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    auto md=exact_matrix_desc(2,17,RNS8_EXACT_WIDE_SIGNED);md.max_prefix=prefix;rns8_matrix* matrix=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    std::vector<int64_t> input(34);for(size_t i=0;i<input.size();++i)input[i]=i%3==0?-63:i%3==1?0:63;
    REQUIRE(rns8_pack_i64(ctx,matrix,input.data(),17,1)==RNS8_SUCCESS);
    rns8_quarter_crt_desc config{};config.struct_size=sizeof(config);config.abi_version=RNS8_ABI_VERSION;config.limb_count=3;config.mode=mode;config.max_scratch_bytes=16384;
    rns8_quarter_crt_export* owner=nullptr;REQUIRE(rns8_prepare_quarter_crt_export(ctx,plan,matrix,&config,&owner)==RNS8_SUCCESS);
    std::vector<uint64_t> output(2*19*3,99);REQUIRE(rns8_export_quarter_crt_limbs(owner,output.data(),19)==RNS8_SUCCESS);
    for(int row=0;row<2;++row)for(int col=0;col<19;++col)for(int limb=0;limb<3;++limb) {
      if(col>=17){CHECK(output[(row*19+col)*3+limb]==99);continue;}
      boost::multiprecision::cpp_int value=input[row*17+col];if(value<0)value+=mode==2?rns8::detail::modulus_product(prefix):(boost::multiprecision::cpp_int(1)<<192);
      CHECK(output[(row*19+col)*3+limb]==static_cast<uint64_t>((value>>(64*limb))&UINT64_MAX));
    }
    REQUIRE(rns8_destroy_quarter_crt_export(owner)==RNS8_SUCCESS);rns8_destroy_matrix(matrix);rns8_destroy_plan(plan);
  }
  rns8_destroy_context(ctx);
#else
  SKIP("HIP was not compiled");
#endif
}


namespace {
void check_quarter_residue_producer(rns8_backend_kind backend) {
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=backend;
  const bool host=backend==RNS8_BACKEND_CPU_REFERENCE || backend==RNS8_BACKEND_CPU_AMX;
  rns8_context* ctx=nullptr;const auto created=rns8_create_context(host?-1:0,&options,&ctx);
  if(created==RNS8_UNSUPPORTED_BACKEND)SKIP("quarter producer backend unavailable on this build/host");
  REQUIRE(created==RNS8_SUCCESS);
  auto d=exact_desc(RNS8_EXACT_WIDE_SIGNED,2,3,1);d.requested_backend=backend;d.max_prefix=3;
  d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND;
  rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&d,&plan)==RNS8_SUCCESS);
  auto md=exact_matrix_desc(2,3,d.semantics);md.max_prefix=3;rns8_matrix* matrix=nullptr;
  REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);const int64_t values[]={-31,0,31,-7,1,19};
  REQUIRE(rns8_pack_i64(ctx,matrix,values,3,1)==RNS8_SUCCESS);
  rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.problem=d;
  q.operation=RNS8_OP_QUARTER_CRT;q.execution_modes=RNS8_EXEC_SYNC;q.output_limbs=1;q.ldc=5;q.scratch_budget=65536;
  q.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF|RNS8_QUERY_QUARTER_RANGE_PROOF;
  rns8_operation_support support{};uint32_t count=0;REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
  REQUIRE(count==1);REQUIRE(support.status==RNS8_SUCCESS);CHECK(support.required_host_bytes==48);CHECK(support.required_device_bytes==(host?0:48));
  rns8_quarter_crt_desc config{};config.struct_size=sizeof(config);config.abi_version=RNS8_ABI_VERSION;config.mode=0;config.limb_count=1;config.max_scratch_bytes=65536;
  rns8_quarter_crt_export* owner=nullptr;REQUIRE(rns8_prepare_quarter_crt_export(ctx,plan,matrix,&config,&owner)==RNS8_SUCCESS);
  uint64_t output[10];std::fill(std::begin(output),std::end(output),99);REQUIRE(rns8_export_quarter_crt_limbs(owner,output,5)==RNS8_SUCCESS);
  for(unsigned r=0;r<2;++r)for(unsigned c=0;c<3;++c)CHECK(output[r*5+c]==static_cast<uint64_t>(values[r*3+c]));
  CHECK(output[3]==99);CHECK(output[4]==99);CHECK(output[8]==99);CHECK(output[9]==99);
  rns8_quarter_crt_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
  REQUIRE(rns8_get_quarter_crt_info(owner,&info)==RNS8_SUCCESS);CHECK(info.retained_payload_bytes>=support.required_host_bytes+support.required_device_bytes);
  REQUIRE(rns8_destroy_quarter_crt_export(owner)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
}
TEST_CASE("quarter owner accepts CPU reference residue producers","[api][crt][provider]") {check_quarter_residue_producer(RNS8_BACKEND_CPU_REFERENCE);}
TEST_CASE("quarter owner accepts direct HIP residue producers","[api][crt][provider][hip]") {check_quarter_residue_producer(RNS8_BACKEND_HIP_DIRECT);}
TEST_CASE("quarter owner accepts hipBLASLt residue producers","[api][crt][provider][hip]") {check_quarter_residue_producer(RNS8_BACKEND_HIPBLASLT);}
TEST_CASE("quarter owner accepts CK residue producers","[api][crt][provider][hip]") {check_quarter_residue_producer(RNS8_BACKEND_CK);}
TEST_CASE("quarter owner accepts rocWMMA residue producers","[api][crt][provider][hip]") {check_quarter_residue_producer(RNS8_BACKEND_ROCWMMA);}
TEST_CASE("quarter owner accepts builtin residue producers","[api][crt][provider][hip]") {check_quarter_residue_producer(RNS8_BACKEND_AMDGPU_BUILTINS);}
TEST_CASE("quarter owner accepts CPU AMX residue producers","[api][crt][provider][amx]") {check_quarter_residue_producer(RNS8_BACKEND_CPU_AMX);}
