TEST_CASE("direct HIP rejects unsupported pack strides before touching prior values") {
  if (!hip_available()) SKIP("no HIP device for source-stride qualification");
  auto* ctx = create_context(RNS8_BACKEND_HIP_DIRECT);
  const int64_t signed_values[] = {1, -2, 3, -4};
  const uint64_t unsigned_values[] = {1, 2, 3, 4};
  for (const auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = matrix_desc(2, 2, semantics, RNS8_BOUND_NONE);
    desc.max_prefix = 4;
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &desc, &matrix) == RNS8_SUCCESS);
    const bool is_signed = semantics == RNS8_EXACT_WIDE_SIGNED;
    REQUIRE((is_signed ? rns8_pack_i64(ctx, matrix, signed_values, 2, 1)
                       : rns8_pack_u64(ctx, matrix, unsigned_values, 2, 1)) == RNS8_SUCCESS);
    const auto generation = matrix->content_generation;
    const auto bound = matrix->exact_max_magnitude;
    CHECK((is_signed ? rns8_pack_i64(ctx, matrix, signed_values, INT32_MAX, 2)
                     : rns8_pack_u64(ctx, matrix, unsigned_values, INT32_MAX, 2)) == RNS8_RANGE_ERROR);
    CHECK(matrix->content_generation == generation);
    CHECK(matrix->source_version == 1);
    CHECK(matrix->exact_max_magnitude == bound);
    CHECK(matrix->device_residues_current);
    // One plane fits signed indexing, but two planes do not. Rejection must
    // precede any access to these deliberately tiny backing allocations.
    CHECK(rns8::detail::hip_direct_gemm_rns_device(
        0, matrix->hip_residues, matrix->hip_residues, matrix->hip_residues,
        32767, 1, 65536, 65536, 1, 1, 2) == RNS8_UNSUPPORTED_BACKEND);
    REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("direct HIP exact base extension preserves every source prefix and signed representative") {
  if (!hip_available()) SKIP("no HIP device for exact base-extension qualification");
  using boost::multiprecision::cpp_int;
  auto* ctx = create_context(RNS8_BACKEND_HIP_DIRECT);
  constexpr std::size_t cells = 7;
  rns8::detail::hip_direct_device_buffer retained_lift;
  REQUIRE(retained_lift.allocate(0,cells*19+sizeof(int))==RNS8_SUCCESS);
  for (const auto semantics : {RNS8_EXACT_WIDE_SIGNED, RNS8_EXACT_WIDE_UNSIGNED}) {
    auto desc = matrix_desc(1, cells, semantics, RNS8_BOUND_NONE);
    desc.max_prefix = 20;
    rns8_matrix* matrix = nullptr;
    REQUIRE(rns8_create_matrix(ctx, &desc, &matrix) == RNS8_SUCCESS);
    for (uint32_t source = 1; source < 20; ++source) {
      const cpp_int product = rns8::detail::modulus_product(source);
      const bool signed_value = semantics == RNS8_EXACT_WIDE_SIGNED;
      const cpp_int bound = signed_value ? cpp_int((product - 1) / 2) : cpp_int(product - 1);
      std::vector<cpp_int> values{0, 1, bound, bound - 1, bound / 2, 7, 17};
      if (signed_value) { values[3] = -bound; values[4] = -1; }
      std::vector<int8_t> expected(cells * 20);
      for (uint32_t p = 0; p < 20; ++p)
        for (std::size_t cell = 0; cell < cells; ++cell)
          expected[p * cells + cell] = rns8::detail::centered_residue(values[cell], rns8::detail::kDefaultModuli[p]);
      for (uint32_t target = source + 1; target <= 20; ++target) {
        uint64_t bound_limbs[3]{};
        for (unsigned limb = 0; limb < 3; ++limb)
          bound_limbs[limb] = static_cast<uint64_t>((bound >> (64 * limb)) & UINT64_MAX);
        for (bool mixed_radix : {false, true}) for(bool retained:{false,true}) {
          auto staged = expected;
          std::fill(staged.begin() + source * cells, staged.end(), int8_t{99});
          REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, matrix->hip_residues, staged.data(), staged.size()) == RNS8_SUCCESS);
          bool committed = false;
          const auto before=rns8::detail::hip_direct_allocation_counters_snapshot();
          const auto lift=[&](const uint64_t* words){return retained?
              rns8::detail::hip_direct_lift_exact_suffix_workspace(0,matrix->hip_residues,cells,source,target,signed_value,words,retained_lift.get(),cells*19+sizeof(int),committed,mixed_radix):
              rns8::detail::hip_direct_lift_exact_suffix(0,matrix->hip_residues,cells,source,target,signed_value,words,committed,mixed_radix);};
          REQUIRE(lift(bound_limbs)==RNS8_SUCCESS);
          const auto after=rns8::detail::hip_direct_allocation_counters_snapshot();
          if(retained){CHECK(after.allocate_calls==before.allocate_calls);CHECK(after.free_calls==before.free_calls);}
          REQUIRE(committed);
          REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, staged.data(), matrix->hip_residues, staged.size()) == RNS8_SUCCESS);
          for (std::size_t i = 0; i < staged.size(); ++i)
            REQUIRE(staged[i] == (i < target * cells ? expected[i] : int8_t{99}));
          // A dishonest magnitude proof must not commit even a partial suffix.
          auto rejected = expected;
          std::fill(rejected.begin() + source * cells, rejected.end(), int8_t{99});
          REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, matrix->hip_residues, rejected.data(), rejected.size()) == RNS8_SUCCESS);
          uint64_t zero_bound[3]{};
          committed = true;
          REQUIRE(lift(zero_bound)==RNS8_RANGE_ERROR);
          REQUIRE_FALSE(committed);
          std::vector<int8_t> unchanged(rejected.size());
          REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, unchanged.data(), matrix->hip_residues, unchanged.size()) == RNS8_SUCCESS);
          REQUIRE(unchanged == rejected);
          // Retained failure status is reset before the next successful use.
          REQUIRE(lift(bound_limbs)==RNS8_SUCCESS);REQUIRE(committed);
        }
        auto initial = expected;
        std::fill(initial.begin() + source * cells, initial.end(), int8_t{99});
        REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0, matrix->hip_residues, initial.data(), initial.size()) == RNS8_SUCCESS);
        matrix->device_residues_current = true;
        matrix->host_residues_current = false;
        matrix->host_native_current = false;
        matrix->device_native_current = false;
        matrix->exact_range_prefix = source;
        matrix->exact_max_magnitude = bound;
        const auto generation = matrix->content_generation;
        const auto identity = matrix->matrix_instance_id;
        const auto version = matrix->source_version;
        const auto bytes = cells * (target - source);
        CHECK(rns8_lift_exact_wide_hip(ctx, matrix, target, bytes - 1) == RNS8_WORKSPACE_TOO_SMALL);
        CHECK(matrix->exact_range_prefix == source);
        REQUIRE(rns8_lift_exact_wide_hip(ctx, matrix, target, bytes) == RNS8_SUCCESS);
        CHECK(matrix->exact_range_prefix == target);
        CHECK(matrix->content_generation == generation);
        CHECK(matrix->matrix_instance_id == identity);
        CHECK(matrix->source_version == version);
        CHECK(matrix->device_residues_current);
        CHECK_FALSE(matrix->host_residues_current);
        std::vector<int8_t> actual(initial.size());
        REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0, actual.data(), matrix->hip_residues, actual.size()) == RNS8_SUCCESS);
        for (std::size_t i = 0; i < actual.size(); ++i)
          REQUIRE(actual[i] == (i < target * cells ? expected[i] : int8_t{99}));
        CHECK(rns8_lift_exact_wide_hip(ctx, matrix, target, 0) == RNS8_SUCCESS);
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
        for(uint32_t algorithm:{0u,1u}) {
          REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0,matrix->hip_residues,initial.data(),initial.size())==RNS8_SUCCESS);
          matrix->exact_range_prefix=source;matrix->materialized_rns_prefix=source;
          rns8_exact_lift_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
          options.target_prefix=target;options.algorithm=algorithm;options.max_total_scratch_bytes=bytes+7;
          rns8_prepared_execution* prepared=nullptr;
          REQUIRE(rns8_prepare_exact_lift_hip(ctx,matrix,&options,&prepared)==RNS8_WORKSPACE_TOO_SMALL);
          REQUIRE(!prepared);options.max_total_scratch_bytes=bytes+8;
          REQUIRE(rns8_prepare_exact_lift_hip(ctx,matrix,&options,&prepared)==RNS8_SUCCESS);
          REQUIRE(rns8_capture_prepared_hip_graph(prepared)==RNS8_SUCCESS);
          rns8::detail::hip_unique_stream stream;REQUIRE(stream.create_non_blocking()==hipSuccess);
          for(bool graph:{false,true}) {
            matrix->exact_range_prefix=source;matrix->materialized_rns_prefix=source;
            const auto before=rns8::detail::hip_direct_allocation_counters_snapshot();
            rns8_completion_token token{};
            REQUIRE((graph?rns8_submit_prepared_hip_graph(prepared,stream.get(),&token):rns8_submit_prepared_hip(prepared,stream.get(),&token))==RNS8_SUCCESS);
            REQUIRE(matrix->in_flight_writer);REQUIRE(matrix->prepared_in_flight_writers==1);
            REQUIRE(rns8_destroy_prepared_execution(prepared)==RNS8_BUSY);
            REQUIRE(matrix->exact_range_prefix==source);
            rns8_completion_info completion{};completion.struct_size=sizeof(completion);completion.abi_version=RNS8_ABI_VERSION;
            REQUIRE(rns8_wait_prepared_hip(prepared,token,&completion)==RNS8_SUCCESS);
            REQUIRE(completion.operation_status==RNS8_SUCCESS);REQUIRE(completion.output_is_current);
            REQUIRE(matrix->content_generation==generation);REQUIRE(matrix->source_version==version);
            REQUIRE(matrix->exact_range_prefix==target);REQUIRE(matrix->materialized_rns_prefix==target);
            REQUIRE(!matrix->in_flight_writer);REQUIRE(!matrix->prepared_in_flight_writers);
            const auto after=rns8::detail::hip_direct_allocation_counters_snapshot();
            CHECK(after.allocate_calls==before.allocate_calls);CHECK(after.free_calls==before.free_calls);
            REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0,actual.data(),matrix->hip_residues,actual.size())==RNS8_SUCCESS);
            for(std::size_t i=0;i<actual.size();++i)REQUIRE(actual[i]==(i<target*cells?expected[i]:int8_t{99}));
          }
          matrix->exact_range_prefix=source;matrix->materialized_rns_prefix=source;
          rns8_completion_token canceled{};REQUIRE(rns8_submit_prepared_hip(prepared,stream.get(),&canceled)==RNS8_SUCCESS);
          REQUIRE(rns8_cancel_prepared_hip(prepared,canceled)==RNS8_SUCCESS);
          rns8_completion_info completion{};completion.struct_size=sizeof(completion);completion.abi_version=RNS8_ABI_VERSION;
          REQUIRE(rns8_wait_prepared_hip(prepared,canceled,&completion)==RNS8_SUCCESS);
          REQUIRE(completion.operation_status==RNS8_CANCELED);REQUIRE(matrix->exact_range_prefix==source);
          REQUIRE(matrix->device_residues_current);REQUIRE(matrix->content_generation==generation);
          REQUIRE(rns8_destroy_prepared_execution(prepared)==RNS8_SUCCESS);
        }
#endif

      }
      matrix->exact_range_prefix = source;
      matrix->exact_max_magnitude = product; // Cannot recover an aliased integer.
      CHECK(rns8_lift_exact_wide_hip(ctx, matrix, 20, UINT64_MAX) == RNS8_RANGE_ERROR);
    }
    REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("direct HIP exact selector stages both inputs and output before one proof publication") {
  if(!hip_available())SKIP("no HIP device for exact selector transaction");
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(bool signed_values:{false,true})for(uint32_t algorithm:{0u,1u})for(bool shared:{false,true}) {
    constexpr int64_t side=3;const auto semantics=signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
    auto gd=signed_values?exact_signed_desc(side,side,side,RNS8_BACKEND_HIP_DIRECT):exact_unsigned_desc(side,side,side,RNS8_BACKEND_HIP_DIRECT);
    gd.max_prefix=20;gd.flags=0;rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
    REQUIRE(rns8_create_plan(ctx,&gd,&plan)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
    std::array<rns8_matrix*,3> matrices{};auto md=matrix_desc(side,side,semantics,RNS8_BOUND_NONE);md.max_prefix=20;
    for(auto& matrix:matrices)REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    auto* a=matrices[0];auto* b=shared?a:matrices[1];auto* c=matrices[2];
    std::vector<uint64_t> values(side*side,100),old_values(side*side,7);
    const auto pack=[&](rns8_matrix* matrix,const std::vector<uint64_t>& x){return signed_values?rns8_pack_i64(ctx,matrix,reinterpret_cast<const int64_t*>(x.data()),side,1):rns8_pack_u64(ctx,matrix,x.data(),side,1);};
    REQUIRE(pack(a,values)==RNS8_SUCCESS);if(!shared)REQUIRE(pack(b,values)==RNS8_SUCCESS);REQUIRE(pack(c,old_values)==RNS8_SUCCESS);
    std::vector<int8_t> old_a(side*side*20,99),old_b=old_a,old_c(old_a.size());std::fill_n(old_a.begin(),side*side,int8_t(100));old_b=old_a;
    REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0,a->hip_residues,old_a.data(),old_a.size())==RNS8_SUCCESS);
    if(!shared)REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0,b->hip_residues,old_b.data(),old_b.size())==RNS8_SUCCESS);
    for(auto* input:{a,b}){input->exact_range_prefix=1;input->materialized_rns_prefix=1;input->host_residues_current=false;}
    REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0,old_c.data(),c->hip_residues,old_c.size())==RNS8_SUCCESS);
    const auto a_generation=a->content_generation,b_generation=b->content_generation,c_generation=c->content_generation;
    uint32_t selected=999;CHECK(rns8_gemm_exact_wide_hip_auto(ctx,plan,a,b,c,workspace,algorithm,0,&selected)==RNS8_WORKSPACE_TOO_SMALL);
    CHECK(selected==999);CHECK(c->content_generation==c_generation);CHECK(a->exact_range_prefix==1);
    a->in_flight_readers=1;CHECK(rns8_gemm_exact_wide_hip_auto(ctx,plan,a,b,c,workspace,algorithm,UINT64_MAX,&selected)==RNS8_BUSY);a->in_flight_readers=0;
    CHECK(rns8_gemm_exact_wide_hip_auto(ctx,plan,a,b,a,workspace,algorithm,UINT64_MAX,&selected)==RNS8_INVALID_ARGUMENT);
    // A valid first stage must not publish when the second operand fails its
    // device magnitude check. No original plane or output byte is touched.
    if(!shared){old_b[0]=127;REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0,b->hip_residues,old_b.data(),old_b.size())==RNS8_SUCCESS);
      CHECK(rns8_gemm_exact_wide_hip_auto(ctx,plan,a,b,c,workspace,algorithm,UINT64_MAX,&selected)==RNS8_RANGE_ERROR);
      CHECK(a->exact_range_prefix==1);CHECK(b->exact_range_prefix==1);CHECK(selected==999);CHECK(c->device_residues_current);
      for(auto pair:{std::pair<rns8_matrix*,const std::vector<int8_t>*>{a,&old_a},{b,&old_b},{c,&old_c}}){std::vector<int8_t> actual(old_a.size());REQUIRE(rns8::detail::hip_direct_copy_device_to_host(0,actual.data(),pair.first->hip_residues,actual.size())==RNS8_SUCCESS);CHECK(actual==*pair.second);}
      old_b[0]=100;REQUIRE(rns8::detail::hip_direct_copy_host_to_device(0,b->hip_residues,old_b.data(),old_b.size())==RNS8_SUCCESS);}
    REQUIRE(rns8_gemm_exact_wide_hip_auto(ctx,plan,a,b,c,workspace,algorithm,UINT64_MAX,&selected)==RNS8_SUCCESS);
    CHECK(selected==2);CHECK(a->exact_range_prefix==2);CHECK(b->exact_range_prefix==2);CHECK(a->content_generation==a_generation);CHECK(b->content_generation==b_generation);CHECK(c->content_generation==c_generation+1);
    const auto counters=rns8::detail::hip_direct_allocation_counters_snapshot();
    REQUIRE(rns8_gemm_exact_wide_hip_auto(ctx,plan,a,b,c,workspace,algorithm,UINT64_MAX,&selected)==RNS8_SUCCESS);
    const auto after=rns8::detail::hip_direct_allocation_counters_snapshot();CHECK(after.allocate_calls==counters.allocate_calls);CHECK(after.free_calls==counters.free_calls);
    gd.max_prefix=selected;gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;rns8_plan* export_plan=nullptr;REQUIRE(rns8_create_plan(ctx,&gd,&export_plan)==RNS8_SUCCESS);
    std::vector<uint64_t> output(side*(side+2)*3,99);REQUIRE((signed_values?rns8_export_exact_wide_signed_limbs(ctx,export_plan,c,output.data(),side+2,3):rns8_export_exact_wide_unsigned_limbs(ctx,export_plan,c,output.data(),side+2,3))==RNS8_SUCCESS);
    for(int row=0;row<side;++row)for(int col=0;col<side+2;++col)for(int limb=0;limb<3;++limb)CHECK(output[(row*(side+2)+col)*3+limb]==(col>=side?99u:limb?0u:30000u));
    REQUIRE(rns8_destroy_plan(export_plan)==RNS8_SUCCESS);for(auto* matrix:matrices)REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
