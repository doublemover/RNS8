#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
template<class T> void check_public_native_group() {
  using boost::multiprecision::cpp_int;constexpr bool u=std::is_unsigned_v<T>;
  constexpr int k=17,n=19,prefix=20;const int rows[]{2,17};
  const auto semantics=u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED;
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  rns8_plan* plans[2]{};rns8_workspace* work[2]{};rns8_prepared_execution* execution[2]{};
  rns8_matrix *a[2]{},*c[2]{},*b=nullptr;
  std::vector<T> av[2],bv(k*n);for(size_t i=0;i<bv.size();++i)bv[i]=T(i%7);
  const auto pack=[&](rns8_matrix* matrix,const std::vector<T>& values,int ld) {
    if constexpr(std::is_unsigned_v<T>)return rns8_pack_u64(ctx,matrix,values.data(),ld,0);
    else return rns8_pack_i64(ctx,matrix,values.data(),ld,0);
  };
  auto bdesc=matrix_desc(k,n,semantics,RNS8_BOUND_NONE);bdesc.max_prefix=prefix;bdesc.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
  REQUIRE(rns8_create_matrix(ctx,&bdesc,&b)==RNS8_SUCCESS);REQUIRE(pack(b,bv,n)==RNS8_SUCCESS);
  for(int i=0;i<2;++i) {
    auto desc=exact_signed_desc(rows[i],n,k,RNS8_BACKEND_HIP_DIRECT);desc.semantics=semantics;
    desc.max_prefix=prefix;desc.flags|=RNS8_PLAN_FORCE_FIXED_PREFIX;
    REQUIRE(rns8_create_plan(ctx,&desc,&plans[i])==RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx,plans[i],&work[i])==RNS8_SUCCESS);
    auto md=matrix_desc(rows[i],k,semantics,RNS8_BOUND_NONE);md.max_prefix=prefix;md.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    REQUIRE(rns8_create_matrix(ctx,&md,&a[i])==RNS8_SUCCESS);
    md.cols=n;md.logical_ld=n;md.flags=0;REQUIRE(rns8_create_matrix(ctx,&md,&c[i])==RNS8_SUCCESS);
    av[i].resize(rows[i]*k);for(size_t j=0;j<av[i].size();++j)av[i][j]=u?T(j%5):T(int(j%9)-4);
    REQUIRE(pack(a[i],av[i],k)==RNS8_SUCCESS);
    REQUIRE(rns8_prepare_gemm(ctx,plans[i],a[i],b,c[i],work[i],&execution[i])==RNS8_SUCCESS);
  }
  rns8_prepared_group* group=nullptr;REQUIRE(rns8_create_prepared_group(execution,2,&group)==RNS8_SUCCESS);
  rns8_encoded_b* cache=nullptr;rns8_native_group_binding bindings[2]{};
  for(int i=0;i<2;++i)bindings[i]={sizeof(bindings[i]),RNS8_ABI_VERSION,i?4u:1u,nullptr};
  const auto cleanup=[&]() {
    REQUIRE(rns8_destroy_prepared_group(group)==RNS8_SUCCESS);REQUIRE(rns8_destroy_encoded_b(cache)==RNS8_SUCCESS);
    for(int i=0;i<2;++i){REQUIRE(rns8_destroy_prepared_execution(execution[i])==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_matrix(c[i])==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(a[i])==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(work[i])==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plans[i])==RNS8_SUCCESS);}
    REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  };
  const auto audit=[&](int i) {
    constexpr uint64_t sentinel=0xfedcba9876543210ull;std::vector<uint64_t> limbs(rows[i]*(n+3)*3,sentinel);
    const auto status=u?rns8_export_exact_wide_unsigned_limbs(ctx,plans[i],c[i],limbs.data(),n+3,3):
        rns8_export_exact_wide_signed_limbs(ctx,plans[i],c[i],limbs.data(),n+3,3);
    REQUIRE(status==RNS8_SUCCESS);
    for(int row=0;row<rows[i];++row)for(int col=0;col<n+3;++col) {
      const auto index=(row*(n+3)+col)*3;
      if(col>=n){for(int l=0;l<3;++l)CHECK(limbs[index+l]==sentinel);continue;}
      cpp_int expected=0;for(int q=0;q<k;++q)expected+=cpp_int(av[i][row*k+q])*bv[q*n+col];
      cpp_int actual=cpp_int(limbs[index])+(cpp_int(limbs[index+1])<<64)+(cpp_int(limbs[index+2])<<128);
      if constexpr(!std::is_unsigned_v<T>)if(limbs[index+2]>>63)actual-=cpp_int(1)<<192;
      CHECK(actual==expected);
    }
  };
  rns8::detail::hip_unique_stream stream;REQUIRE(stream.create_non_blocking()==hipSuccess);
  rns8_native_group_desc desc{sizeof(desc),RNS8_ABI_VERSION,2,4,3,0,UINT64_MAX};
  auto prepared=rns8_prepare_native_group(group,&desc,bindings,2);
  if(prepared==RNS8_UNSUPPORTED_BACKEND){cleanup();SKIP("compiled target lacks native WMMA");}
  REQUIRE(prepared==RNS8_SUCCESS);
  for(bool encoded:{false,true}) {
    if(encoded){REQUIRE(rns8_prepare_encoded_b(ctx,b,UINT64_MAX,&cache)==RNS8_SUCCESS);for(auto& binding:bindings)binding.encoded_b=cache;}
    for(auto order:{RNS8_LOCALITY_ORDINARY,RNS8_LOCALITY_OUTPUT_NEIGHBOR,RNS8_LOCALITY_MODULUS,RNS8_LOCALITY_SAME_OUTPUT}){
      auto local=desc;local.schedule_mode=0;local.ordinary_grid_control=0;rns8_locality_desc layout{sizeof(layout),RNS8_ABI_VERSION,order,0,1,1,3,1024};
      REQUIRE(rns8_prepare_native_group_locality(group,&local,bindings,2,&layout)==RNS8_SUCCESS);
      rns8_locality_info info{sizeof(info),RNS8_ABI_VERSION};REQUIRE(rns8_get_native_group_locality_info(group,&info)==RNS8_SUCCESS);CHECK(info.physical_affinity_established==0);
      REQUIRE(rns8_capture_native_group_hip_graph(group)==RNS8_SUCCESS);rns8_completion_token tokens[2]{};uint32_t submitted=0;
      REQUIRE(rns8_submit_native_group_hip_graph(group,stream.get(),tokens,2,&submitted)==RNS8_SUCCESS);REQUIRE(submitted==2);
      for(int i=0;i<2;++i){rns8_completion_info completion{sizeof(completion),RNS8_ABI_VERSION};REQUIRE(rns8_wait_prepared_hip(execution[i],tokens[i],&completion)==RNS8_SUCCESS);REQUIRE(completion.operation_status==RNS8_SUCCESS);audit(i);}
      const auto fingerprint=info.permutation_fingerprint;REQUIRE(rns8_refresh_prepared_group_inputs(group)==RNS8_SUCCESS);
      REQUIRE(rns8_get_native_group_locality_info(group,&info)==RNS8_SUCCESS);CHECK(info.requested_order==order);CHECK(info.permutation_fingerprint==fingerprint);
      local.schedule_mode=2;CHECK(rns8_prepare_native_group_locality(group,&local,bindings,2,&layout)==RNS8_INVALID_ARGUMENT);
    }
    for(uint32_t mode=0;mode<3;++mode)for(bool ordinary:{false,true}) {
      desc.schedule_mode=mode;desc.ordinary_grid_control=ordinary;
      REQUIRE(rns8_prepare_native_group(group,&desc,bindings,2)==RNS8_SUCCESS);
      if(encoded){CHECK(rns8_destroy_encoded_b(cache)==RNS8_BUSY);
        rns8_encoded_b_info cached{sizeof(cached),RNS8_ABI_VERSION};REQUIRE(rns8_get_encoded_b_info(cache,&cached)==RNS8_SUCCESS);
        CHECK(cached.source_is_current==1);CHECK(cached.layout_version==1);CHECK(cached.prefix==prefix);
        CHECK(cached.source_generation==b->content_generation);CHECK(cached.retained_device_bytes>0);}
      rns8_native_group_info info{sizeof(info),RNS8_ABI_VERSION};REQUIRE(rns8_get_native_group_info(group,&info)==RNS8_SUCCESS);
      CHECK(info.task_count==2);CHECK(info.schedule_mode==mode);CHECK(info.ordinary_grid_control==ordinary);
      auto small=desc;small.max_metadata_bytes=info.retained_device_metadata_bytes;
      CHECK(rns8_prepare_native_group(group,&small,bindings,2)==RNS8_WORKSPACE_TOO_SMALL);
      REQUIRE(rns8_capture_native_group_hip_graph(group)==RNS8_SUCCESS);
      for(bool graph:{false,true})for(int repeat=0;repeat<3;++repeat) {
        const auto before=rns8::detail::hip_direct_allocation_counters_snapshot();
        rns8_completion_token tokens[2]{};uint32_t submitted=0;
        REQUIRE((graph?rns8_submit_native_group_hip_graph(group,stream.get(),tokens,2,&submitted):
            rns8_submit_native_group_hip(group,stream.get(),tokens,2,&submitted))==RNS8_SUCCESS);
        REQUIRE(submitted==2);CHECK(b->in_flight_readers==2);
        CHECK(rns8_destroy_prepared_group(group)==RNS8_BUSY);CHECK(rns8_refresh_prepared_group_inputs(group)==RNS8_BUSY);
        CHECK(pack(b,bv,n)==RNS8_BUSY);
        if(repeat==1)REQUIRE(rns8_cancel_prepared_hip(execution[0],tokens[0])==RNS8_SUCCESS);
        // Observe in reverse order; one completion must not release peer leases.
        for(int i=1;i>=0;--i) {
          rns8_completion_info completion{sizeof(completion),RNS8_ABI_VERSION};
          REQUIRE(rns8_wait_prepared_hip(execution[i],tokens[i],&completion)==RNS8_SUCCESS);
          CHECK(completion.operation_status==(repeat==1 && i==0?RNS8_CANCELED:RNS8_SUCCESS));
          CHECK(completion.output_is_current==!(repeat==1 && i==0));
          CHECK(b->in_flight_readers==uint64_t(i));
        }
        const auto after=rns8::detail::hip_direct_allocation_counters_snapshot();
        CHECK(after.allocate_calls==before.allocate_calls);CHECK(after.free_calls==before.free_calls);
        audit(1);if(repeat!=1)audit(0);
      }
      rns8_completion_token tokens[2]{};uint32_t completed=0;
      REQUIRE(rns8_execute_prepared_group(group,tokens,2,&completed)==RNS8_SUCCESS);CHECK(completed==2);audit(0);audit(1);
    }
  }
  const auto generation=c[0]->content_generation;bv[0]=T(11);REQUIRE(pack(b,bv,n)==RNS8_SUCCESS);
  rns8_completion_token tokens[2]{};uint32_t submitted=99;
  CHECK(rns8_submit_native_group_hip(group,stream.get(),tokens,2,&submitted)==RNS8_INVALID_ARGUMENT);
  CHECK(submitted==0);CHECK(c[0]->content_generation==generation);
  rns8_encoded_b_info stale{sizeof(stale),RNS8_ABI_VERSION};REQUIRE(rns8_get_encoded_b_info(cache,&stale)==RNS8_SUCCESS);
  CHECK(stale.source_is_current==0);
  CHECK(rns8_refresh_prepared_group_inputs(group)==RNS8_INVALID_ARGUMENT); // stale materialization cannot be blessed
  rns8_encoded_b* replacement=nullptr;REQUIRE(rns8_prepare_encoded_b(ctx,b,UINT64_MAX,&replacement)==RNS8_SUCCESS);
  for(auto& binding:bindings)binding.encoded_b=replacement;
  REQUIRE(rns8_prepare_native_group(group,&desc,bindings,2)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_encoded_b(cache)==RNS8_SUCCESS);cache=replacement;
  CHECK(rns8_submit_native_group_hip_graph(group,stream.get(),tokens,2,&submitted)==RNS8_UNSUPPORTED_BACKEND);
  REQUIRE(rns8_execute_prepared_group(group,tokens,2,&submitted)==RNS8_SUCCESS);audit(0);audit(1);
  cleanup();
}
#endif
TEST_CASE("public native prepared queues own generation, cancellation, graph and encoded B lifetimes", "[hip][native][prepared]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(!hip_available())SKIP("no HIP device for native public group");
  SECTION("signed"){check_public_native_group<int64_t>();}
  SECTION("unsigned"){check_public_native_group<uint64_t>();}
#else
  SKIP("native queue requires a HIP build");
#endif
}
