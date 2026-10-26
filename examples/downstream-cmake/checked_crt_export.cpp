#include <rns8/rns8.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include "noninteractive_errors.hpp"

namespace {
void check(rns8_status status) {
  if(status!=RNS8_SUCCESS)throw std::runtime_error(rns8_status_string(status));
}
void require(bool condition,const char* message) {
  if(!condition)throw std::runtime_error(message);
}
struct Owners {
  rns8_context* ctx=nullptr;
  rns8_plan* plan=nullptr;
  rns8_matrix* matrix=nullptr;
  rns8_crt_export* exporter=nullptr;
  rns8_status close() noexcept {
    if(exporter) {const auto status=rns8_destroy_crt_export(exporter);if(status!=RNS8_SUCCESS)return status;exporter=nullptr;}
    if(matrix) {const auto status=rns8_destroy_matrix(matrix);if(status!=RNS8_SUCCESS)return status;matrix=nullptr;}
    if(plan) {const auto status=rns8_destroy_plan(plan);if(status!=RNS8_SUCCESS)return status;plan=nullptr;}
    if(ctx) {const auto status=rns8_destroy_context(ctx);if(status!=RNS8_SUCCESS)return status;ctx=nullptr;}
    return RNS8_SUCCESS;
  }
  ~Owners() {(void)close();}
};
}
int main(int argc,char** argv) {
  rns8_example::configure_noninteractive_errors();
  const bool gpu=argc==2 && std::strcmp(argv[1],"--hip")==0;
  if(argc>2 || (argc==2 && !gpu))return 2;
  Owners owners;
  try {
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=gpu?RNS8_BACKEND_HIP_DIRECT:RNS8_BACKEND_CPU_REFERENCE;
    const auto created=rns8_create_context(gpu?0:-1,&options,&owners.ctx);
    if(gpu && created==RNS8_UNSUPPORTED_BACKEND)return 77;
    check(created);
    rns8_gemm_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
    d.requested_backend=options.requested_backend;d.semantics=RNS8_EXACT_WIDE_SIGNED;d.bound_kind=RNS8_BOUND_NONE;
    d.m=1;d.n=3;d.k=1;d.max_prefix=20;d.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
    check(rns8_create_plan(owners.ctx,&d,&owners.plan));
    rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;
    md.semantics=d.semantics;md.bound_kind=RNS8_BOUND_NONE;md.rows=1;md.cols=3;md.logical_ld=3;
    md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=20;
    check(rns8_create_matrix(owners.ctx,&md,&owners.matrix));
    for(uint32_t algorithm=0;algorithm<=RNS8_CRT_GROUP_SPLIT;++algorithm) {
      const int64_t first[]={-31,0,31},second[]={-7,0,19};
      check(rns8_pack_i64(owners.ctx,owners.matrix,first,3,2*algorithm+1));
      rns8_operation_query query{};query.struct_size=sizeof(query);query.abi_version=RNS8_ABI_VERSION;
      query.operation=RNS8_OP_CHECKED_CRT;query.execution_modes=RNS8_EXEC_SYNC;query.problem=d;
      query.proof_flags=RNS8_QUERY_CURRENT_RANGE_PROOF;query.algorithm=algorithm;query.output_limbs=1;query.scratch_budget=65536;
      rns8_operation_support support{};uint32_t count=0;
      check(rns8_query_operation_support(owners.ctx,&query,&support,1,&count));
      require(count==1 && support.status==RNS8_SUCCESS,"installed checked CRT discovery");
      rns8_crt_export_desc config{};config.struct_size=sizeof(config);config.abi_version=RNS8_ABI_VERSION;
      config.algorithm=algorithm;config.mode=0;config.limb_count=1;config.max_scratch_bytes=65536;
      config.producer_order=algorithm==RNS8_CRT_GROUP_SPLIT?5u:0u;
      check(rns8_prepare_crt_export(owners.ctx,owners.plan,owners.matrix,&config,&owners.exporter));
      uint64_t result[5]{99,99,99,99,99};check(rns8_export_crt_limbs(owners.exporter,result,5));
      require(result[0]==uint64_t(-31) && result[1]==0 && result[2]==31 && result[3]==99 && result[4]==99,"installed checked CRT exact result/padding");
      check(rns8_pack_i64(owners.ctx,owners.matrix,second,3,2*algorithm+2));
      std::fill(std::begin(result),std::end(result),99);
      require(rns8_export_crt_limbs(owners.exporter,result,5)==RNS8_INVALID_ARGUMENT,"stale prepared export must reject");
      for(auto word:result)require(word==99,"stale export preserves caller output");
      check(rns8_rebind_crt_export(owners.exporter,owners.matrix));
      check(rns8_export_crt_limbs(owners.exporter,result,5));
      require(result[0]==uint64_t(-7) && result[1]==0 && result[2]==19 && result[3]==99 && result[4]==99,"rebound checked CRT exact result/padding");
      rns8_crt_export_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      check(rns8_get_crt_export_info(owners.exporter,&info));
      require(info.successful_exports==2 && info.prefix==20 && info.algorithm==algorithm,"installed checked CRT observation");
      check(rns8_destroy_crt_export(owners.exporter));owners.exporter=nullptr;
    }
    check(owners.close());std::cout<<"checked CRT policies, rebind and padded publication passed\n";return 0;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';const auto cleanup=owners.close();
    if(cleanup!=RNS8_SUCCESS)std::cerr<<"cleanup retained pending ownership: "<<rns8_status_string(cleanup)<<'\n';
    return 1;
  }
}
