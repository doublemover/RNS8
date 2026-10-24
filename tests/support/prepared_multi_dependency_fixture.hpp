#pragma once
#include <array>
inline void check_prepared_multi_dependencies(rns8_backend_kind backend,bool pipeline=false,bool cancel_producer=false) {
  rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=backend;
  rns8_context* ctx=nullptr;REQUIRE(rns8_create_context(backend==RNS8_BACKEND_HIP_DIRECT?0:-1,&options,&ctx)==RNS8_SUCCESS);
  auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,1);desc.requested_backend=backend;
  auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);
  rns8_plan* plan=nullptr;REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  std::array<rns8_workspace*,4> work{};std::array<rns8_matrix*,7> matrix{};
  for(auto& w:work) REQUIRE(rns8_create_workspace(ctx,plan,&w)==RNS8_SUCCESS);
  for(auto& m:matrix) {REQUIRE(rns8_create_matrix(ctx,&md,&m)==RNS8_SUCCESS);int64_t zero=0;REQUIRE(rns8_pack_i64(ctx,m,&zero,1,0)==RNS8_SUCCESS);}
  for(int i=0;i<3;++i) {const int64_t value=i+2;REQUIRE(rns8_pack_i64(ctx,matrix[i],&value,1,0)==RNS8_SUCCESS);}
  std::array<rns8_prepared_execution*,3> producers{};rns8_prepared_execution* consumer=nullptr;
  for(int i=0;i<3;++i) REQUIRE(rns8_prepare_gemm(ctx,plan,matrix[i],matrix[i],matrix[3+i],work[i],&producers[i])==RNS8_SUCCESS);
  REQUIRE(rns8_prepare_exact_axpby(ctx,plan,matrix[3],matrix[4],matrix[5],matrix[6],1,1,work[3],&consumer)==RNS8_SUCCESS);
  std::array<rns8_completion_token,3> tokens{};rns8_completion_token result{77,88};
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  std::array<rns8::detail::hip_unique_stream,4> streams;
  if(backend==RNS8_BACKEND_HIP_DIRECT) {
    for(auto& stream:streams) REQUIRE(stream.create_non_blocking()==hipSuccess);
    for(auto* p:producers) REQUIRE(rns8_prepare_hip_submission(p)==RNS8_SUCCESS);
    REQUIRE(rns8_prepare_hip_submission(consumer)==RNS8_SUCCESS);
    for(int i=0;i<3;++i) REQUIRE(rns8_submit_prepared_hip(producers[i],streams[i].get(),&tokens[i])==RNS8_SUCCESS);
  } else
#endif
  for(int i=0;i<3;++i) REQUIRE(rns8_execute_prepared(producers[i],&tokens[i])==RNS8_SUCCESS);
  REQUIRE(rns8_bind_prepared_dependencies(consumer,producers.data(),tokens.data(),3)==RNS8_SUCCESS);
  auto bad=tokens;++bad[2].epoch;
  REQUIRE(rns8_bind_prepared_dependencies(consumer,producers.data(),bad.data(),3)==RNS8_INVALID_ARGUMENT);
  auto duplicate=producers;duplicate[2]=producers[0];bad=tokens;bad[2]=tokens[0];
  REQUIRE(rns8_bind_prepared_dependencies(consumer,duplicate.data(),bad.data(),3)==RNS8_INVALID_ARGUMENT);
  for(auto* p:producers) REQUIRE(rns8_destroy_prepared_execution(p)==RNS8_BUSY);
  REQUIRE(rns8_refresh_prepared_inputs(consumer)==RNS8_BUSY);
  rns8_completion_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(backend==RNS8_BACKEND_HIP_DIRECT) {
    // Observe independent producers in reverse binding order, then consume all
    // publications. No assumption that completion order equals submission order.
    if(pipeline) {
      REQUIRE(rns8_bind_prepared_dependencies(consumer,producers.data(),tokens.data(),2)==RNS8_SUCCESS);
      const auto generation=matrix[6]->content_generation;
      REQUIRE(rns8_submit_prepared_hip_pipeline(consumer,streams[3].get(),&result)==RNS8_BUSY);
      REQUIRE(result.prepared_id==77);REQUIRE(result.epoch==88);
      REQUIRE(matrix[6]->content_generation==generation);
      REQUIRE(rns8_bind_prepared_dependencies(consumer,producers.data(),tokens.data(),3)==RNS8_SUCCESS);
      REQUIRE(rns8_submit_prepared_hip_pipeline(consumer,streams[3].get(),&result)==RNS8_SUCCESS);
      for(auto* p:producers) REQUIRE(rns8_destroy_prepared_execution(p)==RNS8_BUSY);
      for(int i=3;i<7;++i) REQUIRE_FALSE(matrix[i]->device_residues_current);
      if(cancel_producer) REQUIRE(rns8_cancel_prepared_hip(producers[1],tokens[1])==RNS8_SUCCESS);
    } else {
      for(int i=2;i>=0;--i) REQUIRE(rns8_wait_prepared_hip(producers[i],tokens[i],&info)==RNS8_SUCCESS);
      REQUIRE(rns8_submit_prepared_hip(consumer,streams[3].get(),&result)==RNS8_SUCCESS);
    }
    REQUIRE(rns8_wait_prepared_hip(consumer,result,&info)==RNS8_SUCCESS);
  } else
#endif
  {REQUIRE(rns8_execute_prepared(consumer,&result)==RNS8_SUCCESS);REQUIRE(rns8_get_prepared_completion(consumer,result,&info)==RNS8_SUCCESS);}
  if(cancel_producer) {
    REQUIRE(info.operation_status==RNS8_CANCELED);REQUIRE_FALSE(info.output_is_current);
    REQUIRE_FALSE(matrix[6]->device_residues_current);
    REQUIRE(rns8_destroy_prepared_execution(consumer)==RNS8_SUCCESS);
    for(auto* p:producers) REQUIRE(rns8_destroy_prepared_execution(p)==RNS8_SUCCESS);
    for(auto* m:matrix) REQUIRE(rns8_destroy_matrix(m)==RNS8_SUCCESS);
    for(auto* w:work) REQUIRE(rns8_destroy_workspace(w)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
    return;
  }
  REQUIRE(info.operation_status==RNS8_SUCCESS);REQUIRE(info.output_is_current);
  uint64_t value=0;REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,matrix[6],&value,1,1)==RNS8_SUCCESS);REQUIRE(value==52);
  // A cycle reachable only through the second join branch must reject.
  rns8_prepared_execution* cycle=nullptr;rns8_completion_token cycle_token{};
  REQUIRE(rns8_prepare_gemm(ctx,plan,matrix[6],matrix[0],matrix[1],work[3],&cycle)==RNS8_SUCCESS);
  REQUIRE(rns8_execute_prepared(cycle,&cycle_token)==RNS8_SUCCESS);
  REQUIRE(rns8_bind_prepared_dependency(cycle,consumer,result)==RNS8_SUCCESS);
  REQUIRE(rns8_bind_prepared_dependency(producers[1],cycle,cycle_token)==RNS8_SUCCESS);
  REQUIRE(rns8_bind_prepared_dependencies(consumer,producers.data(),tokens.data(),3)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_bind_prepared_dependencies(producers[1],nullptr,nullptr,0)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_prepared_execution(cycle)==RNS8_SUCCESS);
  REQUIRE(rns8_bind_prepared_dependencies(consumer,producers.data(),tokens.data(),3)==RNS8_SUCCESS);
  const auto old_generation=matrix[6]->content_generation;const int64_t changed=0;
  REQUIRE(rns8_pack_i64(ctx,matrix[4],&changed,1,0)==RNS8_SUCCESS);result={77,88};
  REQUIRE(rns8_execute_prepared(consumer,&result)==RNS8_INVALID_ARGUMENT);
  REQUIRE(result.prepared_id==77);REQUIRE(result.epoch==88);REQUIRE(matrix[6]->content_generation==old_generation);
  REQUIRE(rns8_bind_prepared_dependencies(consumer,nullptr,nullptr,0)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_prepared_execution(consumer)==RNS8_SUCCESS);
  for(auto* p:producers) REQUIRE(rns8_destroy_prepared_execution(p)==RNS8_SUCCESS);
  for(auto* m:matrix) REQUIRE(rns8_destroy_matrix(m)==RNS8_SUCCESS);
  for(auto* w:work) REQUIRE(rns8_destroy_workspace(w)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
