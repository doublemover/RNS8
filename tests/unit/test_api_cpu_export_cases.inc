#include <atomic>
namespace {
struct ExportWorkerFailureFixture {
  int64_t fail_cell=-1;
  std::atomic<unsigned> workers{0},failed_worker{UINT32_MAX};
  static void cell(void* opaque,int64_t index,uint32_t worker) {
    auto& state=*static_cast<ExportWorkerFailureFixture*>(opaque);
    state.workers.fetch_or(1u<<worker,std::memory_order_relaxed);
    if(index==state.fail_cell){state.failed_worker.store(worker);throw std::bad_alloc{};}
  }
};
}
TEST_CASE("public CPU integer exports catch worker allocation failures before caller publication", "[api][export][openmp]") {
  using namespace rns8::detail;
  struct Reset {
    uint32_t requested=cpu_parallel_requested_threads(),maximum=cpu_parallel_max_threads();
    uint64_t threshold=cpu_parallel_threshold_ops();bool progress=cpu_parallel_progress_enabled();
    ~Reset(){configure_cpu_parallel(maximum,threshold,progress);configure_cpu_parallel(requested,threshold,progress);}
  } reset;
  configure_cpu_parallel(2,0,false);
  for(bool wide:{false,true})for(bool sign:{false,true}) {
    auto* ctx=create_cpu_context();constexpr int64_t rows=65,cols=67,ld=71;
    const auto semantics=wide?(sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED):(sign?RNS8_BOUNDED_I64:RNS8_BOUNDED_U64);
    auto desc=wide?exact_desc(semantics,rows,cols,1):bounded_gemm_desc(semantics,rows,cols,1);
    auto md=wide?exact_matrix_desc(rows,cols,semantics):bounded_matrix_desc(rows,cols,semantics);
    rns8_plan* plan=nullptr;rns8_matrix* matrix=nullptr;
    REQUIRE(rns8_create_plan(ctx,&desc,&plan)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    std::vector<int64_t> input(size_t(rows*cols),sign?-1:1);
    REQUIRE((sign?rns8_pack_i64(ctx,matrix,input.data(),cols,0):rns8_pack_u64(ctx,matrix,reinterpret_cast<const uint64_t*>(input.data()),cols,0))==RNS8_SUCCESS);
    ExportWorkerFailureFixture state;
    const rns8_cpu_export_test_hooks hooks{&state,&ExportWorkerFailureFixture::cell};ctx->cpu_export_test_hooks=&hooks;
    const uint32_t limbs=wide?3:1;
    std::vector<uint64_t> output(size_t(rows*ld)*limbs,99);
    auto execute=[&]() {
      if(wide)return sign?rns8_export_exact_wide_signed_limbs(ctx,plan,matrix,output.data(),ld,limbs):
                           rns8_export_exact_wide_unsigned_limbs(ctx,plan,matrix,output.data(),ld,limbs);
      return sign?rns8_export_i64(ctx,plan,matrix,reinterpret_cast<int64_t*>(output.data()),ld):rns8_export_u64(ctx,plan,matrix,output.data(),ld);
    };
    REQUIRE(execute()==RNS8_SUCCESS);
    CHECK((state.workers.load()&1u)!=0);
    if(cpu_parallel_runtime_available() && cpu_parallel_effective_threads()>1)CHECK((state.workers.load()&2u)!=0);
    for(int64_t row=0;row<rows;++row)for(int64_t col=0;col<ld;++col)for(uint32_t limb=0;limb<limbs;++limb)
      CHECK(output[size_t(row*ld+col)*limbs+limb]==(col>=cols?99:sign?UINT64_MAX:limb?0:1));
    std::fill(output.begin(),output.end(),99);state.fail_cell=rows*cols-1;
    CHECK(execute()==RNS8_INTERNAL_ERROR);CHECK(output==std::vector<uint64_t>(output.size(),99));
    if(cpu_parallel_runtime_available() && cpu_parallel_effective_threads()>1)CHECK(state.failed_worker.load()==1);
    state.fail_cell=-1;REQUIRE(execute()==RNS8_SUCCESS);
    ctx->cpu_export_test_hooks=nullptr;
    REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
  }
}
