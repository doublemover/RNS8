#include "rns8/pipeline_recipe.h"
#include "core/hip_resources.hpp"
TEST_CASE("bound whole routes execute checked exports and reject stale or forged bindings", "[api][pipeline-recipe]") {
  for(bool hip:{false,true}) {
    if(hip && !rns8::detail::api::direct_hip_compiled())continue;
    rns8_context* ctx=nullptr;
    if(hip){rns8_context_options co{};co.struct_size=sizeof(co);co.abi_version=RNS8_ABI_VERSION;co.requested_backend=RNS8_BACKEND_HIP_DIRECT;REQUIRE(rns8_create_context(0,&co,&ctx)==RNS8_SUCCESS);}
    else ctx=create_cpu_context();
    auto gd=exact_desc(RNS8_EXACT_WIDE_SIGNED,2,2,3);if(hip)gd.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&gd,&plan)==RNS8_SUCCESS);
    rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
    auto am=exact_matrix_desc(2,3,gd.semantics),bm=exact_matrix_desc(3,2,gd.semantics),cm=exact_matrix_desc(2,2,gd.semantics);
    if(hip){am.flags|=RNS8_MATRIX_RETAIN_NATIVE_INPUT;bm.flags|=RNS8_MATRIX_RETAIN_NATIVE_INPUT;}
    REQUIRE(rns8_create_matrix(ctx,&am,&a)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&bm,&b)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&cm,&c)==RNS8_SUCCESS);
    int64_t av[]={1,2,3,4,5,6},bv[]={7,8,9,10,11,12};
    REQUIRE(rns8_pack_i64(ctx,a,av,3,0)==RNS8_SUCCESS);REQUIRE(rns8_pack_i64(ctx,b,bv,2,0)==RNS8_SUCCESS);
    rns8_pipeline_recipe_options o{};o.struct_size=sizeof(o);o.abi_version=RNS8_ABI_VERSION;
    o.route_mask=31;o.execution_mode=RNS8_EXEC_SYNC;o.output_mode=1;o.output_limbs=1;o.output_ld=3;
    o.modulus_group=4;o.workers=2;o.fragments=2;o.fragment_window=1;o.locality_order_mask=1;o.reuse_count=10;o.max_extra_payload_bytes=1<<26;
    rns8_pipeline_recipe_candidate candidates[16]{};uint32_t count=0;
    REQUIRE(rns8_query_pipeline_recipes(ctx,plan,a,b,c,&o,nullptr,0,candidates,16,&count)==RNS8_SUCCESS);REQUIRE(count==8);
    uint32_t chosen=123;CHECK(rns8_select_pipeline_recipe(candidates,count,&chosen)==RNS8_UNSUPPORTED_BACKEND);CHECK(chosen==123);
    rns8_pipeline_recipe_prices price{};price.struct_size=sizeof(price);price.abi_version=RNS8_ABI_VERSION;price.known_mask=7;price.origin=1;
    std::strcpy(price.identity_sha256,candidates[0].cost_identity_sha256);std::strcpy(price.build_identity_sha256,candidates[0].build_identity_sha256);
    std::strcpy(price.target_id,candidates[0].target_id);std::strcpy(price.calibration_id,"explicit-fixture-model");
    for(int stage=0;stage<3;++stage){price.lower_ns[stage]=10;price.upper_ns[stage]=20;}
    REQUIRE(rns8_query_pipeline_recipes(ctx,plan,a,b,c,&o,&price,1,candidates,16,&count)==RNS8_SUCCESS);
    REQUIRE(rns8_select_pipeline_recipe(candidates,count,&chosen)==RNS8_SUCCESS);CHECK(chosen==0);CHECK(candidates[0].cost_origin==1);
    auto forged=candidates[0];forged.identity_sha256[0]=forged.identity_sha256[0]=='0'?'1':'0';rns8_pipeline_recipe* p=nullptr;
    CHECK(rns8_prepare_pipeline_recipe(ctx,plan,a,b,c,&o,&forged,&p)==RNS8_INVALID_ARGUMENT);CHECK(p==nullptr);
    for(uint32_t i=0;i<count;++i)if(candidates[i].status==RNS8_SUCCESS)for(unsigned algorithm=0;algorithm<5;++algorithm) {
      auto selected_options=o;selected_options.export_algorithm=algorithm;
      rns8_pipeline_recipe_candidate route[16]{};uint32_t total=0;
      REQUIRE(rns8_query_pipeline_recipes(ctx,plan,a,b,c,&selected_options,nullptr,0,route,16,&total)==RNS8_SUCCESS);
      REQUIRE(rns8_prepare_pipeline_recipe(ctx,plan,a,b,c,&selected_options,&route[i],&p)==RNS8_SUCCESS);
      CHECK(rns8_destroy_plan(plan)==RNS8_BUSY);CHECK(rns8_destroy_matrix(a)==RNS8_BUSY);
      uint64_t output[6]={0,0,0xfeed,0,0,0xfeed};
      REQUIRE(rns8_execute_pipeline_recipe(p,output,3)==RNS8_SUCCESS);
      CHECK(output[0]==58);CHECK(output[1]==64);CHECK(output[3]==139);CHECK(output[4]==154);CHECK(output[2]==0xfeed);CHECK(output[5]==0xfeed);
      REQUIRE(rns8_execute_pipeline_recipe(p,output,3)==RNS8_SUCCESS); // Rebinds the retained checked export.
      REQUIRE(rns8_destroy_pipeline_recipe(p)==RNS8_SUCCESS);p=nullptr;
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(hip)for(auto mode:{RNS8_EXEC_BORROWED_STREAM,RNS8_EXEC_OWNED_GRAPH}) {
      auto resident=o;resident.execution_mode=mode;resident.output_mode=0;resident.output_limbs=0;resident.output_ld=0;
      rns8_pipeline_recipe_candidate routes[16]{};uint32_t total=0;
      REQUIRE(rns8_query_pipeline_recipes(ctx,plan,a,b,c,&resident,nullptr,0,routes,16,&total)==RNS8_SUCCESS);
      rns8::detail::hip_unique_stream stream;REQUIRE(stream.create_non_blocking()==hipSuccess);
      for(uint32_t i=0;i<total;++i)if(routes[i].status==RNS8_SUCCESS) {
        REQUIRE(rns8_prepare_pipeline_recipe(ctx,plan,a,b,c,&resident,&routes[i],&p)==RNS8_SUCCESS);
        for(unsigned replay=0;replay<2;++replay) {
          rns8_completion_token token{};REQUIRE(rns8_submit_pipeline_recipe_hip(p,stream.get(),&token)==RNS8_SUCCESS);CHECK(token.epoch>0);
          rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
          REQUIRE(rns8_wait_pipeline_recipe_hip(p,token,&info)==RNS8_SUCCESS);
          uint64_t output[4]{};REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,c,output,2,1)==RNS8_SUCCESS);
          CHECK(output[0]==58);CHECK(output[1]==64);CHECK(output[2]==139);CHECK(output[3]==154);
        }
        REQUIRE(rns8_destroy_pipeline_recipe(p)==RNS8_SUCCESS);p=nullptr;
      }
    }
#endif
    REQUIRE(rns8_prepare_pipeline_recipe(ctx,plan,a,b,c,&o,&candidates[0],&p)==RNS8_SUCCESS);
    REQUIRE(rns8_pack_i64(ctx,a,av,3,0)==RNS8_SUCCESS);uint64_t unchanged[6]={};
    CHECK(rns8_execute_pipeline_recipe(p,unchanged,3)==RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8_destroy_pipeline_recipe(p)==RNS8_SUCCESS);rns8_reap_pipeline_recipe_cleanup();
    for(auto* matrix:{c,b,a})REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}
