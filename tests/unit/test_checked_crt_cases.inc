#include "backend_common/checked_crt_export.hpp"
#include "core/prepared_export_read.hpp"

namespace {
using CheckedCrtInteger=boost::multiprecision::cpp_int;
std::vector<uint64_t> checked_crt_input(const std::vector<CheckedCrtInteger>& values,int64_t rows,int64_t cols,int64_t ld) {
  std::vector<uint64_t> result(size_t(rows*ld)*3,UINT64_C(0xfedcba9876543210));
  for(int64_t r=0;r<rows;++r)for(int64_t c=0;c<cols;++c) {
    CheckedCrtInteger bits=values[size_t(r*cols+c)];if(bits<0)bits+=CheckedCrtInteger(1)<<192;
    for(unsigned l=0;l<3;++l)result[size_t(r*ld+c)*3+l]=static_cast<uint64_t>((bits>>(64*l))&UINT64_MAX);
  }
  return result;
}
void exercise_checked_crt_policies(rns8_backend_kind backend) {
  using namespace rns8::detail;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=backend;
  rns8_context* ctx=nullptr;const auto created=rns8_create_context(backend==RNS8_BACKEND_CPU_REFERENCE?-1:0,&options,&ctx);
  if(backend==RNS8_BACKEND_HIP_DIRECT && created!=RNS8_SUCCESS)SKIP("HIP device unavailable for checked CRT owner");
  REQUIRE(created==RNS8_SUCCESS);
  constexpr int64_t rows=2,cols=5,input_ld=7,output_ld=8;
  for(uint32_t prefix=1;prefix<=20;++prefix)for(bool signed_values:{false,true}) {
    CAPTURE(backend,prefix,signed_values);
    const auto semantics=signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
    const auto product=modulus_product(prefix);
    const CheckedCrtInteger bound=signed_values?CheckedCrtInteger((product-1)/2):CheckedCrtInteger(product-1);
    std::vector<CheckedCrtInteger> values={0,1,bound,bound-1,signed_values?CheckedCrtInteger(-1):CheckedCrtInteger(2),
        bound/2,signed_values?CheckedCrtInteger(-bound):bound,signed_values?CheckedCrtInteger(1-bound):CheckedCrtInteger(bound-1),0,1};
    auto desc=exact_desc(semantics,rows,cols,1);desc.max_prefix=prefix;desc.requested_backend=backend;
    desc.flags=RNS8_PLAN_FORCE_FIXED_PREFIX|RNS8_PLAN_EXACT_VALUE_BOUND;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    auto md=exact_matrix_desc(rows,cols,semantics);md.max_prefix=prefix;
    rns8_matrix* matrix=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    const auto input=checked_crt_input(values,rows,cols,input_ld);
    REQUIRE(api::import_exact_wide_limbs(*ctx,*matrix,input.data(),input_ld,3,signed_values,1)==RNS8_SUCCESS);
    for(uint32_t algorithm=0;algorithm<=RNS8_CRT_GROUP_SPLIT;++algorithm)
      for(uint32_t mode:{signed_values?0u:1u,2u})for(uint32_t limbs:{1u,3u,4u,32u}) {
        if(limbs>3 && prefix!=20)continue;
        CAPTURE(algorithm,mode,limbs);
        rns8_crt_export_desc config{};config.struct_size=sizeof(config);config.abi_version=RNS8_ABI_VERSION;
        config.algorithm=algorithm;config.mode=mode;config.limb_count=limbs;config.max_scratch_bytes=UINT64_MAX;
        if(algorithm==RNS8_CRT_GROUP_SPLIT)config.producer_order=2*((prefix+7)/8)-1;
        rns8_crt_export* owner=nullptr;
        REQUIRE(rns8_prepare_crt_export(ctx,plan,matrix,&config,&owner)==RNS8_SUCCESS);
        std::vector<uint64_t> output(size_t(rows*output_ld)*limbs,99);
        const CheckedCrtInteger limit=CheckedCrtInteger(1)<<(64*limbs-(mode==0?1:0));
        bool fits=true;
        for(auto value:values) {
          if(mode==2 && value<0)value+=product;
          if(value>=limit || value<(mode==0?CheckedCrtInteger(-limit):CheckedCrtInteger(0)))fits=false;
        }
        const auto status=rns8_export_crt_limbs(owner,output.data(),output_ld);
        REQUIRE(status==(fits?RNS8_SUCCESS:RNS8_RANGE_ERROR));
        for(int64_t r=0;r<rows;++r)for(int64_t c=0;c<output_ld;++c)for(uint32_t l=0;l<limbs;++l) {
          if(!fits || c>=cols) {CHECK(output[size_t(r*output_ld+c)*limbs+l]==99);continue;}
          CheckedCrtInteger expected=values[size_t(r*cols+c)];
          if(expected<0)expected+=mode==2?product:(CheckedCrtInteger(1)<<(64*limbs));
          CHECK(output[size_t(r*output_ld+c)*limbs+l]==static_cast<uint64_t>((expected>>(64*l))&UINT64_MAX));
        }
        CHECK(matrix->in_flight_readers==0);
        rns8_crt_export_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
        REQUIRE(rns8_get_crt_export_info(owner,&info)==RNS8_SUCCESS);
        CHECK(info.successful_exports==uint64_t(fits));CHECK(info.source_generation==matrix->content_generation);
        CHECK(std::string(info.certificate_sha256).size()==64);
        uint32_t channels=0;
        for(uint32_t g=0;g<info.group_count;++g) {
          CHECK((channels&info.group_channel_masks[g])==0);channels|=info.group_channel_masks[g];CHECK(info.group_products[g]>1);
        }
        if(algorithm>=RNS8_CRT_GROUP_LOCAL)CHECK(channels==((1u<<prefix)-1));
        CHECK(rns8_destroy_plan(plan)==RNS8_BUSY);CHECK(rns8_destroy_matrix(matrix)==RNS8_BUSY);
        REQUIRE(rns8_destroy_crt_export(owner)==RNS8_SUCCESS);
      }
    REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
void exercise_checked_crt_rebind(rns8_backend_kind backend) {
  using namespace rns8::detail;
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=backend;
  rns8_context* ctx=nullptr;const auto created=rns8_create_context(backend==RNS8_BACKEND_CPU_REFERENCE?-1:0,&options,&ctx);
  if(backend==RNS8_BACKEND_HIP_DIRECT && created!=RNS8_SUCCESS)SKIP("HIP device unavailable for checked CRT rebind");
  REQUIRE(created==RNS8_SUCCESS);
  auto d=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,3,1);d.requested_backend=backend;
  rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&d,&plan)==RNS8_SUCCESS);
  auto md=exact_matrix_desc(1,3,RNS8_EXACT_WIDE_SIGNED);
  rns8_matrix *a=nullptr,*b=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);
  const int64_t first[]={INT64_MIN,0,INT64_MAX},second[]={-17,0,31};
  REQUIRE(rns8_pack_i64(ctx,a,first,3,1)==RNS8_SUCCESS);REQUIRE(rns8_pack_i64(ctx,b,second,3,1)==RNS8_SUCCESS);
  for(uint32_t algorithm=0;algorithm<=RNS8_CRT_GROUP_SPLIT;++algorithm) {
    rns8_crt_export_desc config{};config.struct_size=sizeof(config);config.abi_version=RNS8_ABI_VERSION;
    config.algorithm=algorithm;config.mode=0;config.limb_count=1;config.max_scratch_bytes=UINT64_MAX;
    rns8_crt_export* owner=nullptr;REQUIRE(rns8_prepare_crt_export(ctx,plan,a,&config,&owner)==RNS8_SUCCESS);
    uint64_t result[5]{99,99,99,99,99};REQUIRE(rns8_export_crt_limbs(owner,result,5)==RNS8_SUCCESS);
    CHECK(result[0]==uint64_t(INT64_MIN));CHECK(result[2]==uint64_t(INT64_MAX));CHECK(result[3]==99);
    rns8_allocation_observation before{},after{};before.struct_size=after.struct_size=sizeof(before);before.abi_version=after.abi_version=RNS8_ABI_VERSION;
    if(backend==RNS8_BACKEND_HIP_DIRECT)REQUIRE(rns8_get_allocation_observation(ctx,&before)==RNS8_SUCCESS);
    for(unsigned round=0;round<3;++round) {
      REQUIRE(rns8_rebind_crt_export(owner,b)==RNS8_SUCCESS);
      REQUIRE(rns8_export_crt_limbs(owner,result,5)==RNS8_SUCCESS);CHECK(result[0]==uint64_t(-17));CHECK(result[2]==31);
      REQUIRE(rns8_rebind_crt_export(owner,a)==RNS8_SUCCESS);
      REQUIRE(rns8_export_crt_limbs(owner,result,5)==RNS8_SUCCESS);CHECK(result[0]==uint64_t(INT64_MIN));
    }
    if(backend==RNS8_BACKEND_HIP_DIRECT) {
      REQUIRE(rns8_get_allocation_observation(ctx,&after)==RNS8_SUCCESS);
      CHECK(after.device.allocation_calls==before.device.allocation_calls);CHECK(after.device.free_calls==before.device.free_calls);
    }
    REQUIRE(rns8_pack_i64(ctx,a,first,3,2+algorithm)==RNS8_SUCCESS);
    std::fill(std::begin(result),std::end(result),99);
    CHECK(rns8_export_crt_limbs(owner,result,5)==RNS8_INVALID_ARGUMENT);for(auto value:result)CHECK(value==99);
    REQUIRE(rns8_rebind_crt_export(owner,a)==RNS8_SUCCESS);
    REQUIRE(rns8_export_crt_limbs(owner,result,5)==RNS8_SUCCESS);CHECK(result[0]==uint64_t(INT64_MIN));
    CHECK(rns8_export_crt_limbs(owner,result,INT64_MAX)==RNS8_SUCCESS); // One row has no trailing pitch access.
    REQUIRE(rns8_destroy_crt_export(owner)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
} // namespace

TEST_CASE("checked CRT policies preserve every prefix and explicit signed unsigned canonical width", "[api][crt]") {
  exercise_checked_crt_policies(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("checked CRT rebind retains buffers and handles the signed native boundary", "[api][crt]") {
  exercise_checked_crt_rebind(RNS8_BACKEND_CPU_REFERENCE);
}
TEST_CASE("checked CRT HIP policies match independent full-width references and canaries", "[hip][crt]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  exercise_checked_crt_policies(RNS8_BACKEND_HIP_DIRECT);
#else
  SKIP("HIP was not compiled");
#endif
}
TEST_CASE("checked CRT HIP rebind and replay perform no additional device allocations", "[hip][crt]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  exercise_checked_crt_rebind(RNS8_BACKEND_HIP_DIRECT);
#else
  SKIP("HIP was not compiled");
#endif
}
TEST_CASE("prepared export readers unwind CPU exceptions and retain failed-drain ownership", "[api][crt][lifetime]") {
  rns8_matrix source{};bool pending=false;
  CHECK_THROWS_AS(([&] {rns8::detail::api::PreparedExportReadLease read(source,pending);throw std::bad_alloc();}()),std::bad_alloc);
  CHECK(source.in_flight_readers==0);
  {rns8::detail::api::PreparedExportReadLease read(source,pending);pending=true;}
  CHECK(source.in_flight_readers==1);
  pending=false;--source.in_flight_readers; // Models only an owner's successful drain, not a real GPU fault.
}

TEST_CASE("checked CRT discovery shares resource budgets and preflight preserves bindings", "[api][crt][provider]") {
  using namespace rns8::detail;
  auto* ctx=create_cpu_context();auto d=exact_desc(RNS8_EXACT_WIDE_SIGNED,2,3,1);
  rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&d,&plan)==RNS8_SUCCESS);
  auto md=exact_matrix_desc(2,3,RNS8_EXACT_WIDE_SIGNED);
  rns8_matrix *source=nullptr,*larger=nullptr;
  REQUIRE(rns8_create_matrix(ctx,&md,&source)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&md,&larger)==RNS8_SUCCESS);
  const int64_t values[]={-63,0,63,-1,1,17},outside[]={-64,0,64,-1,1,17};
  REQUIRE(rns8_pack_i64(ctx,source,values,3,1)==RNS8_SUCCESS);REQUIRE(rns8_pack_i64(ctx,larger,outside,3,1)==RNS8_SUCCESS);
  for(uint32_t algorithm=0;algorithm<=RNS8_CRT_GROUP_SPLIT;++algorithm) {
    const auto resources=checked_crt::resources(6,20,3,algorithm,false,UINT64_MAX);REQUIRE(resources.status==RNS8_SUCCESS);
    rns8_operation_query q{};q.struct_size=sizeof(q);q.abi_version=RNS8_ABI_VERSION;q.problem=d;
    q.operation=RNS8_OP_CHECKED_CRT;q.execution_modes=RNS8_EXEC_SYNC;q.algorithm=algorithm;
    q.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF;q.output_limbs=3;q.ldc=5;q.scratch_budget=resources.total_bytes;
    rns8_operation_support support{};uint32_t count=0;
    REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
    CHECK(support.status==RNS8_SUCCESS);CHECK(support.resource_status==1);CHECK(support.required_host_bytes==resources.host_bytes);
    CHECK(support.required_device_bytes==0);CHECK(support.supported_execution_modes==RNS8_EXEC_SYNC);
    --q.scratch_budget;REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
    CHECK(support.status==RNS8_WORKSPACE_TOO_SMALL);CHECK(support.required_host_bytes==resources.host_bytes);
    rns8_crt_export_desc config{};config.struct_size=sizeof(config);config.abi_version=RNS8_ABI_VERSION;
    config.algorithm=algorithm;config.mode=0;config.limb_count=3;config.max_scratch_bytes=q.scratch_budget;
    rns8_crt_export* owner=nullptr;
    CHECK(rns8_prepare_crt_export(ctx,plan,source,&config,&owner)==RNS8_WORKSPACE_TOO_SMALL);CHECK(owner==nullptr);
    CHECK(source->prepared_references==0);CHECK(plan->prepared_references==0);
    config.max_scratch_bytes=resources.total_bytes;
    REQUIRE(rns8_prepare_crt_export(ctx,plan,source,&config,&owner)==RNS8_SUCCESS);
    rns8_crt_export_info before{},after{};before.struct_size=after.struct_size=sizeof(before);before.abi_version=after.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_crt_export_info(owner,&before)==RNS8_SUCCESS);
    CHECK(before.retained_host_bytes==resources.host_bytes);
    CHECK(rns8_rebind_crt_export(owner,larger)==RNS8_RANGE_ERROR);
    REQUIRE(rns8_get_crt_export_info(owner,&after)==RNS8_SUCCESS);
    CHECK(after.source_instance==before.source_instance);CHECK(after.source_generation==before.source_generation);
    CHECK(std::string(after.certificate_sha256)==before.certificate_sha256);
    uint64_t output[30];std::fill(std::begin(output),std::end(output),99);
    CHECK(rns8_export_crt_limbs(owner,output,INT64_MAX)==RNS8_INVALID_ARGUMENT);for(auto word:output)CHECK(word==99);
    REQUIRE(rns8_export_crt_limbs(owner,output,5)==RNS8_SUCCESS);
    CHECK(output[0]==uint64_t(-63));CHECK(output[15]==uint64_t(-1));CHECK(output[9]==99);
    q.proof_flags=0;q.scratch_budget=UINT64_MAX;
    REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
    CHECK(support.reason_code==RNS8_SUPPORT_MISSING_PROOF);CHECK(support.resource_status==3);
    q.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF;q.execution_modes=RNS8_EXEC_DAG;
    REQUIRE(rns8_query_operation_support(ctx,&q,&support,1,&count)==RNS8_SUCCESS);
    CHECK(support.reason_code==RNS8_SUPPORT_EXECUTION_MODE);CHECK(support.resource_status==3);
    REQUIRE(rns8_destroy_crt_export(owner)==RNS8_SUCCESS);
  }
  CHECK(checked_crt::resources(INT64_MAX,20,32,4,false,UINT64_MAX).status==RNS8_RANGE_ERROR);
  CHECK(checked_crt::resources(1,0,3,0,false,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
  CHECK(checked_crt::resources(1,20,0,0,false,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
  CHECK(checked_crt::resources(1,20,3,5,false,UINT64_MAX).status==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_matrix(source)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(larger)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
