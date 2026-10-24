#include "rns8/rns8.hpp"
TEST_CASE("prepared GEMM retains handles and rejects stale bindings and completion epochs", "[api][prepared]") {
  for (bool hip : {false, true}) {
    if (hip && !rns8::detail::api::direct_hip_compiled()) continue;
    for (auto semantics : {RNS8_BOUNDED_I64, RNS8_BOUNDED_U64, RNS8_EXACT_WIDE_SIGNED,
                            RNS8_EXACT_WIDE_UNSIGNED, RNS8_FINITE_RING_U8, RNS8_FINITE_FIELD_U8,
                            RNS8_WRAP_U64_MOD_2_64}) {
      const bool wrap = semantics == RNS8_WRAP_U64_MOD_2_64;
      const bool finite = semantics == RNS8_FINITE_RING_U8 || semantics == RNS8_FINITE_FIELD_U8;
      const bool exact = rns8::detail::exact_wide_semantics(semantics);
      const bool signed_value = semantics == RNS8_BOUNDED_I64 || semantics == RNS8_EXACT_WIDE_SIGNED;
      rns8_context* ctx = nullptr;
      if (hip) {
        rns8_context_options options{};
        options.struct_size = sizeof(options); options.abi_version = RNS8_ABI_VERSION;
        options.requested_backend = RNS8_BACKEND_HIP_DIRECT;
        REQUIRE(rns8_create_context(0, &options, &ctx) == RNS8_SUCCESS);
      } else ctx = wrap ? create_wrap_context() : create_cpu_context();
      auto desc = wrap ? wrap_gemm_desc(2,2,2) : finite ? finite_gemm_desc(semantics,2,2,2,251) :
                  exact ? exact_desc(semantics,2,2,2) : bounded_gemm_desc(semantics,2,2,2);
      if (hip) desc.requested_backend = RNS8_BACKEND_HIP_DIRECT;
      if (!wrap && !finite && !exact) desc.bound = 10000;
      auto md = wrap ? wrap_matrix_desc(2,2) : finite ? finite_matrix_desc(2,2,semantics) :
                exact ? exact_matrix_desc(2,2,semantics) : bounded_matrix_desc(2,2,semantics);
      rns8_plan* plan = nullptr;
      rns8_workspace* ws = nullptr;
      rns8_matrix *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
      REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
      REQUIRE(rns8_create_workspace(ctx, plan, &ws) == RNS8_SUCCESS);
      for (auto** out : {&a, &b, &c, &d}) REQUIRE(rns8_create_matrix(ctx, &md, out) == RNS8_SUCCESS);
      auto pack = [&](rns8_matrix* matrix, int first) {
        int64_t si[4]; uint64_t ui[4]; uint8_t fi[4];
        for (int i = 0; i < 4; ++i) { si[i] = first+i; ui[i] = first+i; fi[i] = static_cast<uint8_t>(first+i); }
        return finite ? rns8_pack_finite_u8(ctx, matrix, 251, fi, 2, 0) :
            signed_value ? rns8_pack_i64(ctx, matrix, si, 2, 0) : rns8_pack_u64(ctx, matrix, ui, 2, 0);
      };
      REQUIRE(pack(a, 1) == RNS8_SUCCESS);
      REQUIRE(pack(b, 5) == RNS8_SUCCESS);
      auto check_product = [&](const std::vector<uint64_t>& expected) {
        std::vector<uint64_t> result(4);
        if (finite) {
          std::vector<uint8_t> bytes(4);
          REQUIRE(rns8_export_finite_u8(ctx, plan, 251, c, bytes.data(), 2) == RNS8_SUCCESS);
          std::copy(bytes.begin(), bytes.end(), result.begin());
        } else if (exact) {
          REQUIRE((signed_value
              ? rns8_export_exact_wide_signed_limbs(ctx, plan, c, result.data(), 2, 1)
              : rns8_export_exact_wide_unsigned_limbs(ctx, plan, c, result.data(), 2, 1)) == RNS8_SUCCESS);
        } else if (signed_value) {
          std::vector<int64_t> values(4);
          REQUIRE(rns8_export_i64(ctx, plan, c, values.data(), 2) == RNS8_SUCCESS);
          std::copy(values.begin(), values.end(), result.begin());
        } else {
          REQUIRE((wrap ? rns8_export_wrap_u64(ctx, plan, c, result.data(), 2)
                        : rns8_export_u64(ctx, plan, c, result.data(), 2)) == RNS8_SUCCESS);
        }
        REQUIRE(result == expected);
      };
      rns8_prepared_execution *first = nullptr, *second = nullptr, *bad = nullptr;
      REQUIRE(rns8_prepare_gemm(ctx, plan, a, b, a, ws, &bad) == RNS8_INVALID_ARGUMENT);
      REQUIRE(bad == nullptr);
      REQUIRE(a->prepared_references == 0);
      REQUIRE(rns8_prepare_gemm(ctx, plan, a, b, c, ws, &first) == RNS8_SUCCESS);
      REQUIRE(rns8_prepare_gemm(ctx, plan, a, a, d, ws, &second) == RNS8_SUCCESS);
      rns8_async_support_info support{};
      support.struct_size = sizeof(support); support.abi_version = RNS8_ABI_VERSION;
      REQUIRE(rns8_get_prepared_hip_support(first, &support) == RNS8_SUCCESS);
      CHECK(support.support_status == (hip && exact ? RNS8_SUCCESS : RNS8_UNSUPPORTED_BACKEND));
      CHECK(support.max_in_flight_epochs == (hip && exact ? 1u : 0u));
      CHECK(support.supports_timed_observation == support.max_in_flight_epochs);
      CHECK(support.supports_cooperative_discard == support.max_in_flight_epochs);
      CHECK(a->in_flight_readers == 0);
      CHECK_FALSE(c->in_flight_writer);

      REQUIRE(a->prepared_references == 2); // A==B is retained once.
      CHECK(rns8_destroy_matrix(a) == RNS8_BUSY);
      CHECK(rns8_destroy_matrix(c) == RNS8_BUSY);
      CHECK(rns8_destroy_workspace(ws) == RNS8_BUSY);
      CHECK(rns8_destroy_plan(plan) == RNS8_BUSY);
      CHECK(rns8_destroy_context(ctx) == RNS8_BUSY);
      rns8_completion_token one{}, two{};
      REQUIRE(rns8_execute_prepared(second, &two) == RNS8_SUCCESS);
      REQUIRE(rns8_execute_prepared(first, &one) == RNS8_SUCCESS);
      check_product({19,22,43,50});
      rns8_completion_info info{};
      info.struct_size = sizeof(info); info.abi_version = RNS8_ABI_VERSION;
      REQUIRE(rns8_get_prepared_completion(first, one, &info) == RNS8_SUCCESS);
      REQUIRE(info.completed == 1);
      REQUIRE(info.output_is_current == 1);
      REQUIRE(info.c_generation == c->content_generation);
      REQUIRE(info.a_instance == a->matrix_instance_id);
      REQUIRE(rns8_get_prepared_completion(first, two, &info) == RNS8_INVALID_ARGUMENT);
      const auto old = one;
      REQUIRE(rns8_execute_prepared(first, &one) == RNS8_SUCCESS);
      REQUIRE(one.epoch > old.epoch);
      REQUIRE(rns8_get_prepared_completion(first, old, &info) == RNS8_INVALID_ARGUMENT);
      REQUIRE(pack(a, 2) == RNS8_SUCCESS);
      const auto c_generation = c->content_generation;
      REQUIRE(rns8_execute_prepared(first, &one) == RNS8_INVALID_ARGUMENT);
      REQUIRE(c->content_generation == c_generation);
      REQUIRE(rns8_refresh_prepared_inputs(first) == RNS8_SUCCESS);
      REQUIRE(rns8_execute_prepared(first, &one) == RNS8_SUCCESS);
      check_product({31,36,55,64});
      REQUIRE(rns8_get_prepared_completion(first, one, &info) == RNS8_SUCCESS);
      REQUIRE(info.output_is_current == 1);
      REQUIRE(pack(c, 42) == RNS8_SUCCESS);
      REQUIRE(rns8_get_prepared_completion(first, one, &info) == RNS8_SUCCESS);
      REQUIRE(info.output_is_current == 0);
      REQUIRE(rns8_get_prepared_completion(second, two, &info) == RNS8_SUCCESS);
      REQUIRE(info.output_is_current == 1);
      REQUIRE(rns8_destroy_prepared_execution(first) == RNS8_SUCCESS);
      REQUIRE(a->prepared_references == 1);
      REQUIRE(rns8_destroy_matrix(c) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_prepared_execution(second) == RNS8_SUCCESS);
      REQUIRE(a->prepared_references == 0);
      for (auto* matrix : {a,b,d}) REQUIRE(rns8_destroy_matrix(matrix) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_workspace(ws) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
      REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
    }
  }
}


TEST_CASE("failed asynchronous drain quarantines context mutation until recreation", "[api][prepared][cpu]") {
  using namespace rns8::detail::api;
  auto* ctx = create_cpu_context();
  auto desc = bounded_gemm_desc(RNS8_BOUNDED_I64, 1, 1, 1);
  desc.bound = 10000;
  auto md = bounded_matrix_desc(1, 1, RNS8_BOUNDED_I64);
  rns8_plan* plan = nullptr;
  rns8_workspace* workspace = nullptr;
  rns8_matrix* a = nullptr;
  REQUIRE(rns8_create_plan(ctx, &desc, &plan) == RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx, plan, &workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_create_matrix(ctx, &md, &a) == RNS8_SUCCESS);
  int64_t value = 7;
  REQUIRE(rns8_pack_i64(ctx, a, &value, 1, 1) == RNS8_SUCCESS);
  const auto generation = a->content_generation;

  // Deterministic host-side recovery transition; no destructive GPU fault injection.
  rns8_async_failure_info failure{};
  record_async_failure(failure, RNS8_BACKEND_FAILURE, RNS8_ASYNC_FAILURE_EVENT_RECORD,
                       17, 0, 23);
  REQUIRE_FALSE(record_async_drain(*ctx, failure, 29));
  CHECK(failure.resources_retained == 1);
  record_async_failure(failure, RNS8_INTERNAL_ERROR, RNS8_ASYNC_FAILURE_EVENT_OBSERVE,
                       31, 0, 23);
  REQUIRE_FALSE(record_async_drain(*ctx, failure, 37));
  CHECK(failure.backend_error == 17);
  CHECK(failure.cleanup_error == 29);
  CHECK(failure.stage == RNS8_ASYNC_FAILURE_EVENT_RECORD);
  value = 9;
  CHECK(rns8_pack_i64(ctx, a, &value, 1, 2) == RNS8_BACKEND_FAILURE);
  CHECK(a->content_generation == generation);
  rns8_matrix* other = nullptr;
  CHECK(rns8_create_matrix(ctx, &md, &other) == RNS8_BACKEND_FAILURE);
  CHECK(other == nullptr);
  rns8_plan* other_plan = nullptr;
  CHECK(rns8_create_plan(ctx, &desc, &other_plan) == RNS8_BACKEND_FAILURE);
  CHECK(other_plan == nullptr);
  rns8_workspace* other_workspace = nullptr;
  CHECK(rns8_create_workspace(ctx, plan, &other_workspace) == RNS8_BACKEND_FAILURE);
  CHECK(other_workspace == nullptr);
  REQUIRE(record_async_drain(*ctx, failure, 0));
  CHECK(failure.resources_retained == 0);
  CHECK(failure.cleanup_error == 29);
  CHECK(ctx->hip_failure_poisoned);
  CHECK(rns8_pack_i64(ctx, a, &value, 1, 2) == RNS8_BACKEND_FAILURE);
  CHECK(a->content_generation == generation);
  REQUIRE(rns8_destroy_matrix(a) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(workspace) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan) == RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx) == RNS8_SUCCESS);
}


TEST_CASE("async deadline distinguishes pending ownership from terminal operation", "[api][async][cpu]") {
  using rns8::detail::api::poll_async_until;
  unsigned polls = 0;
  const auto pending = [&] { ++polls; return RNS8_BUSY; };
  REQUIRE(poll_async_until(0, pending) == RNS8_TIMEOUT);
  CHECK(polls == 1);
  REQUIRE(poll_async_until(0, [] { return RNS8_SUCCESS; }) == RNS8_SUCCESS);
  CHECK(poll_async_until(UINT64_MAX, [] { return RNS8_BACKEND_FAILURE; }) == RNS8_BACKEND_FAILURE);
  CHECK(poll_async_until(0, [] { return RNS8_INVALID_ARGUMENT; }) == RNS8_INVALID_ARGUMENT);
}

TEST_CASE("prepared dependencies retain exact producer epochs and refresh only produced inputs", "[api][prepared]") {
  auto* ctx=create_cpu_context();
  auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,1);
  auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);
  rns8_plan* plan=nullptr;
  rns8_workspace *w1=nullptr,*w2=nullptr;
  rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr,*d=nullptr;
  REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,plan,&w1)==RNS8_SUCCESS);
  REQUIRE(rns8_create_workspace(ctx,plan,&w2)==RNS8_SUCCESS);
  for (auto** matrix : {&a,&b,&c,&d}) REQUIRE(rns8_create_matrix(ctx,&md,matrix)==RNS8_SUCCESS);
  int64_t three=3,zero=0;
  REQUIRE(rns8_pack_i64(ctx,a,&three,1,0)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx,b,&three,1,0)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx,c,&zero,1,0)==RNS8_SUCCESS);
  rns8_prepared_execution *producer=nullptr,*consumer=nullptr;
  REQUIRE(rns8_prepare_gemm(ctx,plan,a,a,c,w1,&producer)==RNS8_SUCCESS);
  REQUIRE(rns8_prepare_gemm(ctx,plan,c,b,d,w2,&consumer)==RNS8_SUCCESS);
  rns8_completion_token source{},output{77,88};
  REQUIRE(rns8_execute_prepared(producer,&source)==RNS8_SUCCESS);
  rns8_prepared_group* producer_group=nullptr;
  REQUIRE(rns8_create_prepared_group(&producer,1,&producer_group)==RNS8_SUCCESS);
  auto stale=source; ++stale.epoch;
  CHECK(rns8_bind_prepared_dependency(consumer,producer,stale)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_bind_prepared_dependency(consumer,producer,source)==RNS8_SUCCESS);
  CHECK(rns8_destroy_prepared_execution(producer)==RNS8_BUSY);
  CHECK(rns8_execute_prepared(producer,&output)==RNS8_BUSY);
  CHECK(output.prepared_id==77);
  CHECK(rns8_refresh_prepared_inputs(consumer)==RNS8_BUSY);
  // A group created before binding cannot silently replace the retained epoch.
  uint32_t executed=99;
  CHECK(rns8_execute_prepared_group(producer_group,&output,1,&executed)==RNS8_BUSY);
  CHECK(executed==0);
  CHECK(output.prepared_id==77);
  CHECK(output.epoch==88);
  CHECK(rns8_refresh_prepared_group_inputs(producer_group)==RNS8_BUSY);
  REQUIRE(rns8_destroy_prepared_group(producer_group)==RNS8_SUCCESS);
  REQUIRE(rns8_execute_prepared(consumer,&output)==RNS8_SUCCESS);
  uint64_t result=0;
  REQUIRE(rns8_export_exact_wide_signed_limbs(ctx,plan,d,&result,1,1)==RNS8_SUCCESS);
  CHECK(result==27);
  REQUIRE(rns8_execute_prepared(producer,&source)==RNS8_SUCCESS);
  REQUIRE(rns8_bind_prepared_dependency(consumer,producer,source)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx,b,&three,1,0)==RNS8_SUCCESS); // unrelated generation change
  output={77,88};
  const auto generation=d->content_generation;
  CHECK(rns8_execute_prepared(consumer,&output)==RNS8_INVALID_ARGUMENT);
  CHECK(output.epoch==88);
  CHECK(d->content_generation==generation);
  REQUIRE(rns8_bind_prepared_dependency(consumer,nullptr,{})==RNS8_SUCCESS);
  REQUIRE(rns8_refresh_prepared_inputs(consumer)==RNS8_SUCCESS);
  REQUIRE(rns8_bind_prepared_dependency(consumer,producer,source)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(ctx,c,&zero,1,0)==RNS8_SUCCESS); // producer receipt no longer current
  CHECK(rns8_execute_prepared(consumer,&output)==RNS8_INVALID_ARGUMENT);
  REQUIRE(rns8_destroy_prepared_execution(consumer)==RNS8_SUCCESS);
  rns8_prepared_execution* cycle_consumer=nullptr;
  REQUIRE(rns8_prepare_gemm(ctx,plan,c,b,a,w2,&cycle_consumer)==RNS8_SUCCESS);
  rns8_completion_token cycle_token{};
  REQUIRE(rns8_execute_prepared(cycle_consumer,&cycle_token)==RNS8_SUCCESS);
  REQUIRE(rns8_bind_prepared_dependency(producer,cycle_consumer,cycle_token)==RNS8_SUCCESS);
  CHECK(rns8_bind_prepared_dependency(cycle_consumer,producer,source)==RNS8_INVALID_ARGUMENT);
  CHECK(rns8_destroy_prepared_execution(cycle_consumer)==RNS8_BUSY);
  // Rebinding the same token must retain exactly one reference, not leak one.
  REQUIRE(rns8_bind_prepared_dependency(producer,cycle_consumer,cycle_token)==RNS8_SUCCESS);
  REQUIRE(rns8_bind_prepared_dependency(producer,nullptr,{})==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_prepared_execution(cycle_consumer)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_prepared_execution(producer)==RNS8_SUCCESS);
  for (auto* matrix : {a,b,c,d}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(w2)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_workspace(w1)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("C++ busy move assignment preserves both owning handles", "[api][prepared][cpp]") {
  rns8::Context context;
  const auto desc=exact_desc(RNS8_EXACT_WIDE_SIGNED,1,1,1);
  const auto md=exact_matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED);
  rns8::Plan plan(context,desc);
  rns8::Workspace workspace(context,plan);
  rns8::Matrix a(context,md),b(context,md),c(context,md),replacement(context,md);
  int64_t value=3;
  REQUIRE(rns8_pack_i64(context.get(),a.get(),&value,1,0)==RNS8_SUCCESS);
  REQUIRE(rns8_pack_i64(context.get(),b.get(),&value,1,0)==RNS8_SUCCESS);
  rns8_prepared_execution* execution=nullptr;
  REQUIRE(rns8_prepare_gemm(context.get(),plan.get(),a.get(),b.get(),c.get(),workspace.get(),&execution)==RNS8_SUCCESS);
  auto* old=a.get(); auto* incoming=replacement.get();
  CHECK(a.close()==RNS8_BUSY);
  bool busy=false;
  try { a=std::move(replacement); }
  catch (const rns8::Error& error) { busy=error.status()==RNS8_BUSY; }
  REQUIRE(busy);
  CHECK(a.get()==old); CHECK(replacement.get()==incoming);
  CHECK(workspace.close()==RNS8_BUSY);
  CHECK(plan.close()==RNS8_BUSY);
  CHECK(context.close()==RNS8_BUSY);
  REQUIRE(rns8_destroy_prepared_execution(execution)==RNS8_SUCCESS);
  a=std::move(replacement);
  CHECK(a.get()==incoming); CHECK(replacement.get()==nullptr);
  REQUIRE(a.close()==RNS8_SUCCESS);
  REQUIRE(a.close()==RNS8_SUCCESS);
}
