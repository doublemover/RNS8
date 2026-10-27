TEST_CASE("immutable schedule validation is amortized without hash-only equivalence", "[api][schedule]") {
  using namespace rns8::detail::api;
  auto* ctx = create_cpu_context();
  auto desc = exact_desc(RNS8_BOUNDED_U64, 16384, 16384, 1);
  desc.bound_kind = RNS8_BOUND_PER_TILE_MAX_UNSIGNED;
  desc.max_prefix = RNS8_DEFAULT_BOUNDED_PREFIX;
  desc.tile_m = desc.tile_n = 128;
  desc.flags = RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS;
  std::vector<uint64_t> bounds(128 * 128, 100000);
  std::vector<uint8_t> zero_rows(16384), zero_cols(16384);
  desc.tile_bounds = bounds.data(); desc.tile_bounds_count = bounds.size();
  desc.zero_a_rows = zero_rows.data(); desc.zero_a_rows_count = zero_rows.size();
  desc.zero_b_cols = zero_cols.data(); desc.zero_b_cols_count = zero_cols.size();
  rns8_plan *plan = nullptr, *equivalent = nullptr, *different = nullptr;
  plan_validation_work = {};
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(plan_validation_work.exhaustive_validations == 1);
  REQUIRE(plan_validation_work.fingerprint_traversals == 1);
  REQUIRE(plan->validated_contract);
  const auto& launch = plan->validated_contract->launch;
  CHECK(launch.entry_count == 128 * 128);
  CHECK(launch.max_row_blocks == 8);
  CHECK(launch.max_col_blocks == 8);
  CHECK(launch.max_elements == 128 * 128);
  CHECK(launch.i32_launch_qualified);
  CHECK_FALSE(launch.zero_output_tiles);
  CHECK(launch.uniform_zero_selected_prefix == 0);
  CHECK(plan_validation_work.backend_schedule_seals == 1);
  CHECK(plan_validation_work.backend_schedule_sealed_tile_visits == 128 * 128);
  REQUIRE(rns8_create_plan(ctx, &desc, &equivalent) == RNS8_SUCCESS);
  REQUIRE(plan->validated_contract != equivalent->validated_contract);
  rns8_workspace* workspace = nullptr;
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  plan_validation_work = {};
  for (int iteration = 0; iteration < 32; ++iteration) {
    REQUIRE(validate_plan_context_workspace(*ctx, *plan, *workspace) == RNS8_SUCCESS);
    REQUIRE(validate_plan_context_workspace(*ctx, *equivalent, *workspace) == RNS8_SUCCESS);
    REQUIRE(plan_workspace_fingerprint(*plan) == workspace->schedule_fingerprint);
  }
  REQUIRE(plan_validation_work.exhaustive_validations == 0);
  REQUIRE(plan_validation_work.fingerprint_traversals == 0);
  REQUIRE(plan_validation_work.equivalent_comparisons == 1);
  CHECK(plan_validation_work.backend_schedule_sealed_tile_visits == 0);
  // Caller-owned arrays are copied. Mutation cannot alter an existing seal.
  bounds[0] = 99999;
  REQUIRE(rns8_create_plan(ctx, &desc, &different) == RNS8_SUCCESS);
  REQUIRE(validate_plan_context_workspace(*ctx, *different, *workspace) == RNS8_INVALID_ARGUMENT);
  // Force an artificial hash collision: structural equality must still reject.
  auto collision = std::make_shared<rns8_validated_plan_contract>(*different->validated_contract);
  collision->fingerprint = plan->validated_contract->fingerprint;
  different->validated_contract = collision;
  REQUIRE(validate_plan_context_workspace(*ctx, *different, *workspace) == RNS8_INVALID_ARGUMENT);
  const auto fingerprint = workspace->schedule_fingerprint;
  workspace->schedule_fingerprint ^= 1;
  REQUIRE(validate_plan_context_workspace(*ctx, *plan, *workspace) == RNS8_INVALID_ARGUMENT);
  workspace->schedule_fingerprint = fingerprint;
  // Diagnostics check coverage and proof fields even on a sealed plan. They
  // never silently reseal changed state or certify the previous fingerprint.
  const auto entry = plan->tile_schedule[0];
  plan->tile_schedule[0].row_extent -= 1;
  REQUIRE_FALSE(revalidate_plan_schedule_contract(*plan));
  plan->tile_schedule[0] = entry;
  plan->tile_schedule[0].required_prefix += 1;
  REQUIRE_FALSE(revalidate_plan_schedule_contract(*plan));
  plan->tile_schedule[0] = entry;
  plan->zero_a_rows[0] = 1;
  REQUIRE_FALSE(revalidate_plan_schedule_contract(*plan));
  plan->zero_a_rows[0] = 0;
  REQUIRE(revalidate_plan_schedule_contract(*plan));
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  // Workspace owns the immutable contract, not a dangling original-plan pointer.
  REQUIRE(validate_plan_context_workspace(*ctx, *equivalent, *workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(equivalent) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(different) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}

TEST_CASE("ordinary HIP tiled GEMM consumes the immutable launch seal", "[api][schedule][hip]") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail::api;
  rns8_context_options options{};
  options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  rns8_context* ctx=nullptr;
  REQUIRE(rns8_create_context(0,&options,&ctx)==RNS8_SUCCESS);
  auto desc=bounded_gemm_desc(RNS8_BOUNDED_U64,128,128,1);
  desc.requested_backend=RNS8_BACKEND_HIP_DIRECT;
  desc.bound_kind=RNS8_BOUND_PER_TILE_MAX_UNSIGNED;desc.bound=0;desc.tile_m=desc.tile_n=64;
  uint64_t bounds[]{1,1000,1000,1};desc.tile_bounds=bounds;desc.tile_bounds_count=4;
  rns8_plan *plan=nullptr,*equivalent=nullptr;
  rns8_workspace* workspace=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  REQUIRE(rns8_create_plan(ctx,&desc,&equivalent)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
  auto ad=bounded_matrix_desc(128,1,RNS8_BOUNDED_U64);
  auto bd=bounded_matrix_desc(1,128,RNS8_BOUNDED_U64);
  auto cd=bounded_matrix_desc(128,128,RNS8_BOUNDED_U64);
  for(auto* matrix_desc:{&ad,&bd,&cd}) {
    matrix_desc->bound_kind=desc.bound_kind;
    matrix_desc->tile_m=desc.tile_m;matrix_desc->tile_n=desc.tile_n;
  }
  REQUIRE(rns8_create_matrix(ctx,&ad,&a)==RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx,&bd,&b)==RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx,&cd,&c)==RNS8_SUCCESS);
  std::vector<uint64_t> ones(128,1),actual(128*130,UINT64_MAX);
  REQUIRE(rns8_pack_u64(ctx,a,ones.data(),1,0)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_u64(ctx,b,ones.data(),128,0)==RNS8_SUCCESS);
  plan_validation_work={};
  for(int iteration=0;iteration<8;++iteration) {
    auto* selected=iteration%2?plan:equivalent;
    REQUIRE(rns8_gemm_rns(ctx,selected,a,b,c,workspace)==RNS8_SUCCESS);
    REQUIRE(rns8_export_u64(ctx,selected,c,actual.data(),130)==RNS8_SUCCESS);
    for(int row=0;row<128;++row) {
      for(int col=0;col<128;++col) REQUIRE(actual[row*130+col]==1);
      CHECK(actual[row*130+128]==UINT64_MAX);CHECK(actual[row*130+129]==UINT64_MAX);
    }
  }
  CHECK(plan_validation_work.backend_schedule_fast_submissions==8);
  CHECK(plan_validation_work.backend_schedule_raw_submissions==0);
  CHECK(plan_validation_work.backend_schedule_sealed_tile_visits==0);
  CHECK(plan_validation_work.exhaustive_validations==0);
  CHECK(plan_validation_work.fingerprint_traversals==0);
  CHECK(plan_validation_work.equivalent_comparisons==1);
  REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);workspace=nullptr;
  for(auto order:{RNS8_LOCALITY_ORDINARY,RNS8_LOCALITY_OUTPUT_NEIGHBOR,RNS8_LOCALITY_MODULUS,RNS8_LOCALITY_SAME_OUTPUT}) {
    rns8_locality_desc locality{};locality.struct_size=sizeof(locality);locality.abi_version=RNS8_ABI_VERSION;
    locality.order=order;locality.topology_version=1;locality.cache_domains=1;locality.concurrent_groups_per_domain=1;
    REQUIRE(rns8_create_locality_workspace(ctx,plan,&locality,&workspace)==RNS8_SUCCESS);
    rns8_locality_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_workspace_locality_info(workspace,&info)==RNS8_SUCCESS);
    CHECK(info.requested_order==order);CHECK_FALSE(info.physical_affinity_established);
    CHECK(info.k_phases==1);CHECK(info.unique_source_bytes<=info.requested_source_bytes);
    REQUIRE(rns8_gemm_rns(ctx,equivalent,a,b,c,workspace)==RNS8_SUCCESS);
    REQUIRE(rns8_export_u64(ctx,equivalent,c,actual.data(),130)==RNS8_SUCCESS);
    for(int row=0;row<128;++row)for(int col=0;col<130;++col)CHECK(actual[row*130+col]==(col<128?1:UINT64_MAX));
    REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);workspace=nullptr;
  }
  REQUIRE(rns8_destroy_matrix(c)==RNS8_SUCCESS);REQUIRE(rns8_destroy_matrix(b)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_matrix(a)==RNS8_SUCCESS);REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(equivalent)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#endif
}

#include "backend_common/locality_schedule.hpp"
#include <map>
TEST_CASE("locality lowering is a target-bound permutation of admitted adaptive work", "[api][schedule][locality]") {
  using namespace rns8::detail;
  for(uint32_t rows=1;rows<=5;++rows)for(uint32_t cols=1;cols<=7;++cols) {
    std::vector<rns8_plan_tile_schedule_entry> original;
    for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<cols;++c) {
      const uint32_t prefix=(r*7+c*3)%6;
      for(uint32_t p=0;p<prefix;++p) {
        rns8_plan_tile_schedule_entry e{};e.tile_row=r;e.tile_col=c;e.row_offset=r*16;e.col_offset=c*16;
        e.row_extent=r+1==rows?3:16;e.col_extent=c+1==cols?5:16;e.selected_prefix=prefix;e.group_index=p;
        original.push_back(e);
      }
    }
    auto keys=[](const auto& entries) {
      std::set<std::tuple<uint64_t,uint64_t,uint32_t>> out;
      for(const auto& e:entries)out.emplace(e.tile_row,e.tile_col,e.group_index);return out;
    };
    const auto expected=keys(original);
    for(auto order:{RNS8_LOCALITY_ORDINARY,RNS8_LOCALITY_OUTPUT_NEIGHBOR,RNS8_LOCALITY_MODULUS,RNS8_LOCALITY_SAME_OUTPUT})
      for(const std::string target:{"gfx942:sramecc+","gfx950","gfx1100","gfx9999"}) {
        auto entries=original;
        rns8_locality_desc desc{};desc.struct_size=sizeof(desc);desc.abi_version=RNS8_ABI_VERSION;
        desc.order=order;desc.topology_version=1;desc.cache_domains=1;desc.concurrent_groups_per_domain=4;
        rns8_locality_info info{};
        REQUIRE(lower_locality_schedule(entries,65537,desc,target,info)==RNS8_SUCCESS);
        CHECK(entries.size()==original.size());CHECK(keys(entries)==expected);CHECK(info.k_phases==2);
        CHECK(info.applied==uint32_t(order!=RNS8_LOCALITY_ORDINARY && target!="gfx9999"));
        CHECK(info.directory_bytes==entries.size()*sizeof(rns8_plan_tile_schedule_entry));
        CHECK(info.unique_source_bytes<=info.requested_source_bytes);CHECK_FALSE(info.physical_affinity_established);
        // Every launch ID has exactly one inverse, including uneven tile tails.
        std::map<std::tuple<uint64_t,uint64_t,uint32_t>,std::size_t> inverse;
        for(std::size_t i=0;i<entries.size();++i)REQUIRE(inverse.emplace(std::make_tuple(entries[i].tile_row,entries[i].tile_col,entries[i].group_index),i).second);
        for(const auto& e:original)CHECK(inverse.count(std::make_tuple(e.tile_row,e.tile_col,e.group_index))==1);
        auto again=original;rns8_locality_info second{};
        REQUIRE(lower_locality_schedule(again,65537,desc,target,second)==RNS8_SUCCESS);
        CHECK(info.permutation_fingerprint==second.permutation_fingerprint);
      }
  }
  rns8_locality_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;d.order=RNS8_LOCALITY_SAME_OUTPUT;
  d.topology_version=1;d.cache_domains=1;d.concurrent_groups_per_domain=1;d.target_id=942;
  std::vector<rns8_plan_tile_schedule_entry> empty;rns8_locality_info info{};
  REQUIRE(lower_locality_schedule(empty,1,d,"gfx950",info)==RNS8_SUCCESS);CHECK_FALSE(info.applied);
  auto saved=info;d.topology_version=2;
  CHECK(lower_locality_schedule(empty,1,d,"gfx942",info)==RNS8_INVALID_ARGUMENT);
  CHECK(info.permutation_fingerprint==saved.permutation_fingerprint);
}
