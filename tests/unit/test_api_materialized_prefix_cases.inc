#include <array>
#include "../support/native_device_fixture.hpp"

TEST_CASE("bounded computed prefix is independent of allocation capacity", "[api][bounded][prefix]") {
  for (bool hip : {false, true}) for (bool signed_values : {false, true}) {
#if !defined(RNS8_ENABLE_HIP) || !RNS8_ENABLE_HIP
    if (hip) continue;
#endif
    const auto semantics = signed_values ? RNS8_BOUNDED_I64 : RNS8_BOUNDED_U64;
    rns8_context_options options{}; options.struct_size=sizeof(options); options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=hip?RNS8_BACKEND_HIP_DIRECT:RNS8_BACKEND_CPU_REFERENCE;
    rns8_context* ctx=nullptr; REQUIRE(rns8_create_context(hip?0:-1,&options,&ctx)==RNS8_SUCCESS);
    auto desc=bounded_gemm_desc(semantics,2,2,2); desc.requested_backend=options.requested_backend; desc.bound=31;
    rns8_plan *narrow=nullptr,*wide=nullptr; rns8_workspace *nw=nullptr,*ww=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&narrow)==RNS8_SUCCESS);
    desc.bound=signed_values?uint64_t(INT64_MAX):UINT64_MAX;
    REQUIRE(rns8_create_plan(ctx,&desc,&wide)==RNS8_SUCCESS);
    REQUIRE(narrow->prefix<wide->prefix);
    REQUIRE(rns8_create_workspace(ctx,narrow,&nw)==RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx,wide,&ww)==RNS8_SUCCESS);
    std::array<rns8_matrix*,4> matrix{};
    const auto md=bounded_matrix_desc(2,2,semantics);
    const int64_t input[]{1,-1,2,1},identity[]{1,0,0,1},zero[4]{};
    const uint64_t input_u[]{1,3,2,1},identity_u[]{1,0,0,1},zero_u[4]{};
    for(size_t i=0;i<matrix.size();++i) {
      REQUIRE(rns8_create_matrix(ctx,&md,&matrix[i])==RNS8_SUCCESS);
      REQUIRE((signed_values?rns8_pack_i64(ctx,matrix[i],i==0?input:i==1?identity:zero,2,1):
          rns8_pack_u64(ctx,matrix[i],i==0?input_u:i==1?identity_u:zero_u,2,1))==RNS8_SUCCESS);
      REQUIRE(matrix[i]->materialized_rns_prefix==matrix[i]->prefix);
    }
    rns8_prepared_execution* retained=nullptr;
    REQUIRE(rns8_prepare_gemm(ctx,wide,matrix[2],matrix[1],matrix[3],ww,&retained)==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx,narrow,matrix[0],matrix[1],matrix[2],nw)==RNS8_SUCCESS);
    REQUIRE(matrix[2]->prefix==RNS8_DEFAULT_BOUNDED_PREFIX);
    REQUIRE(matrix[2]->materialized_rns_prefix==narrow->prefix);
    REQUIRE(matrix[2]->exact_range_prefix==0);
    rns8_matrix_storage_info info{}; info.struct_size=sizeof(info); info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_matrix_storage_info(matrix[2],&info)==RNS8_SUCCESS);
    REQUIRE(info.max_prefix>info.materialized_rns_prefix); REQUIRE(info.has_materialized_rns_schedule==0);
    int64_t out[]{99,99,99,99,99,99}; uint64_t out_u[]{99,99,99,99,99,99};
    REQUIRE((signed_values?rns8_export_i64(ctx,narrow,matrix[2],out,3):
        rns8_export_u64(ctx,narrow,matrix[2],out_u,3))==RNS8_SUCCESS);
    for(int row=0;row<2;++row) for(int col=0;col<2;++col)
      REQUIRE((signed_values?boost::multiprecision::cpp_int(out[row*3+col]):boost::multiprecision::cpp_int(out_u[row*3+col]))==
          (signed_values?boost::multiprecision::cpp_int(input[row*2+col]):boost::multiprecision::cpp_int(input_u[row*2+col])));
    REQUIRE(out[2]==99); REQUIRE(out[5]==99); REQUIRE(out_u[2]==99); REQUIRE(out_u[5]==99);
    std::fill(std::begin(out),std::end(out),99); std::fill(std::begin(out_u),std::end(out_u),99);
    REQUIRE((signed_values?rns8_export_i64(ctx,wide,matrix[2],out,3):
        rns8_export_u64(ctx,wide,matrix[2],out_u,3))==RNS8_RANGE_ERROR);
    for(auto v:out) REQUIRE(v==99); for(auto v:out_u) REQUIRE(v==99);
    const auto generation=matrix[3]->content_generation;
    const auto source_version=matrix[3]->source_version;
    REQUIRE(rns8_gemm_rns(ctx,wide,matrix[2],matrix[1],matrix[3],ww)==RNS8_RANGE_ERROR);
    REQUIRE(rns8_gemm_rns(ctx,wide,matrix[1],matrix[2],matrix[3],ww)==RNS8_RANGE_ERROR);
    REQUIRE(rns8_refresh_prepared_inputs(retained)==RNS8_RANGE_ERROR);
    rns8_completion_token token{77,88};
    REQUIRE(rns8_execute_prepared(retained,&token)==RNS8_RANGE_ERROR);
    REQUIRE(token.prepared_id==77); REQUIRE(token.epoch==88);
    rns8_prepared_execution* rejected=nullptr;
    REQUIRE(rns8_prepare_gemm(ctx,wide,matrix[2],matrix[1],matrix[3],ww,&rejected)==RNS8_RANGE_ERROR);
    REQUIRE(rejected==nullptr);
    rns8_resident_node node{}; node.struct_size=sizeof(node); node.abi_version=RNS8_ABI_VERSION;
    node.plan=wide; node.workspace=ww; node.a=matrix[2]; node.b=matrix[1]; node.c=matrix[3];
    rns8_resident_graph* graph=nullptr;
    REQUIRE(rns8_create_resident_graph(ctx,&node,1,&graph)==RNS8_RANGE_ERROR); REQUIRE(graph==nullptr);
    REQUIRE(matrix[3]->content_generation==generation); REQUIRE(matrix[3]->source_version==source_version);
    REQUIRE(matrix[3]->materialized_rns_prefix==matrix[3]->prefix);
    REQUIRE_FALSE(matrix[2]->in_flight_writer); REQUIRE(matrix[2]->in_flight_readers==0);
    REQUIRE_FALSE(ww->in_flight);
    if(hip) {
      // AUTO's native-conversion bridge must not turn a narrowed computed RNS
      // result with no native owner into a different incidental failure status.
      ctx->auto_backend_selection=true;
      REQUIRE(rns8_gemm_rns(ctx,wide,matrix[2],matrix[1],matrix[3],ww)==RNS8_RANGE_ERROR);
      REQUIRE(matrix[3]->content_generation==generation);
      ctx->auto_backend_selection=false;
    }
    // Explicit native import encodes the complete capacity again; it is not a
    // lift inferred from the old narrowed computation's storage allocation.
    REQUIRE((signed_values?rns8_pack_i64(ctx,matrix[2],input,2,2):
        rns8_pack_u64(ctx,matrix[2],input_u,2,2))==RNS8_SUCCESS);
    REQUIRE(matrix[2]->materialized_rns_prefix==matrix[2]->prefix);
    REQUIRE(rns8_refresh_prepared_inputs(retained)==RNS8_SUCCESS);
    REQUIRE(rns8_execute_prepared(retained,&token)==RNS8_SUCCESS);
    REQUIRE(matrix[3]->materialized_rns_prefix==wide->prefix);
    REQUIRE(rns8_destroy_prepared_execution(retained)==RNS8_SUCCESS);
    rns8::detail::api::invalidate_output_currentness(*matrix[2]);
    REQUIRE(matrix[2]->materialized_rns_prefix==0); REQUIRE_FALSE(matrix[2]->materialized_rns_schedule);
    for(auto* m:matrix) REQUIRE(rns8_destroy_matrix(m)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(nw)==RNS8_SUCCESS); REQUIRE(rns8_destroy_workspace(ww)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(narrow)==RNS8_SUCCESS); REQUIRE(rns8_destroy_plan(wide)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}

TEST_CASE("adaptive bounded exports retain complete owned write schedule provenance", "[api][bounded][prefix][tiled]") {
  for(bool hip:{false,true}) for(bool signed_values:{false,true}) {
#if !defined(RNS8_ENABLE_HIP) || !RNS8_ENABLE_HIP
    if(hip) continue;
#endif
    const auto semantics=signed_values?RNS8_BOUNDED_I64:RNS8_BOUNDED_U64;
    rns8_context_options options{}; options.struct_size=sizeof(options); options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=hip?RNS8_BACKEND_HIP_DIRECT:RNS8_BACKEND_CPU_REFERENCE;
    rns8_context* ctx=nullptr; REQUIRE(rns8_create_context(hip?0:-1,&options,&ctx)==RNS8_SUCCESS);
    auto desc=bounded_gemm_desc(semantics,64,128,1); desc.requested_backend=options.requested_backend;
    desc.bound_kind=signed_values?RNS8_BOUND_PER_TILE_MAX_ABS:RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
    desc.bound=0; desc.tile_m=desc.tile_n=64;
    const uint64_t bounds[]{31,signed_values?uint64_t(INT64_MAX):UINT64_MAX};
    desc.tile_bounds=bounds; desc.tile_bounds_count=2;
    rns8_plan* plan=nullptr; rns8_workspace* workspace=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    REQUIRE(plan->schedule_min_selected_prefix<plan->schedule_max_selected_prefix);
    REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
    std::array<rns8_matrix*,3> matrix{};
    for(size_t i=0;i<3;++i) {
      auto md=bounded_matrix_desc(i==1?1:64,i==0?1:128,semantics);
      md.bound_kind=desc.bound_kind; md.tile_m=md.tile_n=64;
      REQUIRE(rns8_create_matrix(ctx,&md,&matrix[i])==RNS8_SUCCESS);
      std::vector<int64_t> values(size_t(md.rows*md.cols),i==2?7:1);
      std::vector<uint64_t> values_u(values.size(),i==2?7:1);
      REQUIRE((signed_values?rns8_pack_i64(ctx,matrix[i],values.data(),md.cols,1):
          rns8_pack_u64(ctx,matrix[i],values_u.data(),md.cols,1))==RNS8_SUCCESS);
    }
    REQUIRE(rns8_gemm_rns(ctx,plan,matrix[0],matrix[1],matrix[2],workspace)==RNS8_SUCCESS);
    REQUIRE(matrix[2]->materialized_rns_prefix==plan->schedule_min_selected_prefix);
    REQUIRE(matrix[2]->materialized_rns_schedule==plan->validated_contract);
    REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS); // Provenance owns the seal.
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    REQUIRE(matrix[2]->materialized_rns_schedule!=plan->validated_contract);
    std::vector<int64_t> out(64*129,99); std::vector<uint64_t> out_u(out.size(),99);
    REQUIRE((signed_values?rns8_export_i64(ctx,plan,matrix[2],out.data(),129):
        rns8_export_u64(ctx,plan,matrix[2],out_u.data(),129))==RNS8_SUCCESS);
    for(int row=0;row<64;++row) {
      for(int col=0;col<128;++col) REQUIRE((signed_values?out[row*129+col]:int64_t(out_u[row*129+col]))==1);
      REQUIRE(out[row*129+128]==99); REQUIRE(out_u[row*129+128]==99);
    }
    desc.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* wider=nullptr; REQUIRE(rns8_create_plan(ctx,&desc,&wider)==RNS8_SUCCESS);
    // Force a diagnostic hash collision; structural equality must still reject.
    auto collision=std::make_shared<rns8_validated_plan_contract>(*wider->validated_contract);
    collision->fingerprint=matrix[2]->materialized_rns_schedule->fingerprint;
    wider->validated_contract=collision;
    std::fill(out.begin(),out.end(),99); std::fill(out_u.begin(),out_u.end(),99);
    REQUIRE((signed_values?rns8_export_i64(ctx,wider,matrix[2],out.data(),129):
        rns8_export_u64(ctx,wider,matrix[2],out_u.data(),129))==RNS8_RANGE_ERROR);
    for(auto value:out) REQUIRE(value==99); for(auto value:out_u) REQUIRE(value==99);
    REQUIRE(rns8_destroy_plan(wider)==RNS8_SUCCESS); REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
    rns8::detail::api::invalidate_output_currentness(*matrix[2]);
    REQUIRE(matrix[2]->materialized_rns_prefix==0); REQUIRE_FALSE(matrix[2]->materialized_rns_schedule);
    for(auto* m:matrix) REQUIRE(rns8_destroy_matrix(m)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}

TEST_CASE("whole-basis consumers reject narrowed exact-wide computed storage", "[api][prefix][exact]") {
  for(bool hip:{false,true}) for(bool signed_values:{false,true}) {
#if !defined(RNS8_ENABLE_HIP) || !RNS8_ENABLE_HIP
    if(hip) continue;
#endif
    const auto semantics=signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
    rns8_context_options options{}; options.struct_size=sizeof(options); options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=hip?RNS8_BACKEND_HIP_DIRECT:RNS8_BACKEND_CPU_REFERENCE;
    rns8_context* ctx=nullptr; REQUIRE(rns8_create_context(hip?0:-1,&options,&ctx)==RNS8_SUCCESS);
    auto desc=exact_desc(semantics,2,2,2); desc.max_prefix=9; desc.requested_backend=options.requested_backend;
    rns8_plan* plan=nullptr; rns8_workspace* workspace=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
    const auto md=exact_matrix_desc(2,2,semantics);
    std::array<rns8_matrix*,4> matrix{};
    const int64_t input[]{1,-1,2,1},identity[]{1,0,0,1},zero[4]{};
    const uint64_t input_u[]{1,3,2,1},identity_u[]{1,0,0,1},zero_u[4]{};
    for(size_t i=0;i<4;++i) {
      REQUIRE(rns8_create_matrix(ctx,&md,&matrix[i])==RNS8_SUCCESS);
      REQUIRE((signed_values?rns8_pack_i64(ctx,matrix[i],i==0?input:i==1?identity:zero,2,1):
          rns8_pack_u64(ctx,matrix[i],i==0?input_u:i==1?identity_u:zero_u,2,1))==RNS8_SUCCESS);
    }
    REQUIRE(rns8_gemm_rns(ctx,plan,matrix[0],matrix[1],matrix[2],workspace)==RNS8_SUCCESS);
    REQUIRE(matrix[2]->exact_range_prefix==plan->prefix); REQUIRE(matrix[2]->prefix>plan->prefix);
    uint64_t output[12]{};
    REQUIRE((signed_values?rns8_export_exact_wide_signed_limbs(ctx,plan,matrix[2],output,2,3):
        rns8_export_exact_wide_unsigned_limbs(ctx,plan,matrix[2],output,2,3))==RNS8_SUCCESS);
    for(size_t i=0;i<4;++i) {
      boost::multiprecision::cpp_int value=0;
      for(int limb=2;limb>=0;--limb) {value<<=64; value+=output[i*3+limb];}
      if(signed_values && (output[i*3+2]>>63)) value-=boost::multiprecision::cpp_int(1)<<192;
      REQUIRE(value==(signed_values?boost::multiprecision::cpp_int(input[i]):boost::multiprecision::cpp_int(input_u[i])));
    }
    const auto generation=matrix[3]->content_generation;
    if(signed_values) {
      REQUIRE(rns8_gram_residues(ctx,matrix[2],matrix[3],RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SCALAR,1<<20)==RNS8_RANGE_ERROR);
      rns8_prepared_gram* owner=nullptr;
      REQUIRE(rns8_prepare_gram(ctx,matrix[2],matrix[3],RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SCALAR,1<<20,&owner)==RNS8_RANGE_ERROR);
      REQUIRE(owner==nullptr); REQUIRE(matrix[3]->content_generation==generation);
    }
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
    if(hip) {
      rns8::detail::hip_direct_device_buffer external;
      std::array<uint64_t,6> canary{77,77,77,77,77,77},actual{};
      REQUIRE(external.allocate(0,sizeof(canary))==RNS8_SUCCESS);
      REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0,external.get(),canary.data(),sizeof(canary))==RNS8_SUCCESS);
      native_device_fixture::Owner owner;
      auto native=native_device_fixture::descriptor(external.get(),sizeof(canary),owner);
      native.byte_offset=0; native.rows=native.cols=2; native.ld=3; native.access=RNS8_NATIVE_DEVICE_WRITE;
      native.element_type=signed_values?RNS8_NATIVE_DEVICE_I64:RNS8_NATIVE_DEVICE_U64;
      rns8_native_device_lease* lease=nullptr;
      REQUIRE(rns8_create_native_device_lease(ctx,&native,&lease)==RNS8_SUCCESS);
      REQUIRE(rns8_export_native_device(ctx,matrix[2],lease,1<<20)==RNS8_RANGE_ERROR);
      REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0,actual.data(),external.get(),sizeof(actual))==RNS8_SUCCESS);
      REQUIRE(actual==canary); REQUIRE(matrix[2]->in_flight_readers==0);
      REQUIRE(rns8_destroy_native_device_lease(lease)==RNS8_SUCCESS);
      REQUIRE(owner.acquired==1); REQUIRE(owner.released==1);
    }
#endif
    // A complete native import supplies the whole basis and restores admission.
    REQUIRE((signed_values?rns8_pack_i64(ctx,matrix[2],input,2,2):
        rns8_pack_u64(ctx,matrix[2],input_u,2,2))==RNS8_SUCCESS);
    if(signed_values) REQUIRE(rns8_gram_residues(ctx,matrix[2],matrix[3],RNS8_GRAM_TRANSPOSE_TIMES_SELF,RNS8_GRAM_SCALAR,1<<20)==RNS8_SUCCESS);
    for(auto* m:matrix) REQUIRE(rns8_destroy_matrix(m)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS); REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}
