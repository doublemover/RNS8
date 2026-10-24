#include <rns8/rns8.h>
#include <hip/hip_runtime_api.h>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
extern "C" int native_produce(void*,void*,void*);
extern "C" int native_consume(const void*,void*);
namespace {
void check(rns8_status s) {if(s!=RNS8_SUCCESS) throw std::runtime_error(rns8_status_string(s));}
void hipcheck(hipError_t s) {if(s!=hipSuccess) throw std::runtime_error(hipGetErrorString(s));}
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
struct Owner {unsigned refs=0,acquires=0,releases=0;};
rns8_status acquire(void* value) {auto& owner=*static_cast<Owner*>(value);++owner.refs;++owner.acquires;return RNS8_SUCCESS;}
void release(void* value) {auto& owner=*static_cast<Owner*>(value);--owner.refs;++owner.releases;}
rns8_native_device_desc view(void* base,uint64_t bytes,int64_t rows,int64_t cols,int64_t ld,
                             rns8_native_device_access access,Owner& owner,uint64_t identity) {
  rns8_native_device_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
  d.element_type=RNS8_NATIVE_DEVICE_I64;d.access=access;d.allocation_base=base;d.allocation_bytes=bytes;
  d.byte_offset=8;d.rows=rows;d.cols=cols;d.ld=ld;d.device_id=0;d.producer_complete=1;
  d.policy=RNS8_NATIVE_DEVICE_BORROW;d.owner_identity=identity;d.generation=1;
  d.owner=&owner;d.acquire=acquire;d.release=release;return d;
}
}
int main() {
  // Failure exits preserve allocations/leases until process teardown. Never
  // release an external allocation merely because a failed operation returned.
  try {
    hipcheck(hipSetDevice(0));void *a=nullptr,*b=nullptr,*c=nullptr,*flag=nullptr;
    hipcheck(hipMalloc(&a,22*8));hipcheck(hipMalloc(&b,21*8));hipcheck(hipMalloc(&c,13*8));hipcheck(hipMalloc(&flag,4));
    hipcheck(static_cast<hipError_t>(native_produce(a,b,c)));hipcheck(hipDeviceSynchronize());
    rns8_context_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    rns8_context* ctx=nullptr;check(rns8_create_context(0,&options,&ctx));
    std::array<Owner,3> owners{};
    const std::array<rns8_native_device_desc,3> views={view(a,22*8,3,5,7,RNS8_NATIVE_DEVICE_READ,owners[0],101),
      view(b,21*8,5,2,4,RNS8_NATIVE_DEVICE_READ,owners[1],202),view(c,13*8,3,2,4,RNS8_NATIVE_DEVICE_WRITE,owners[2],303)};
    std::array<rns8_native_device_lease*,3> leases{};std::array<rns8_matrix*,3> matrices{};
    for(std::size_t i=0;i<3;++i) {
      check(rns8_create_native_device_lease(ctx,&views[i],&leases[i]));
      rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;
      md.rows=views[i].rows;md.cols=views[i].cols;md.logical_ld=md.cols;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
      md.semantics=RNS8_EXACT_WIDE_SIGNED;md.bound_kind=RNS8_BOUND_NONE;md.max_prefix=20;
      check(rns8_create_matrix(ctx,&md,&matrices[i]));
    }
    check(rns8_import_native_device_verified(ctx,matrices[0],leases[0],8));
    check(rns8_import_native_device_verified(ctx,matrices[1],leases[1],8));
    rns8_gemm_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;
    gd.semantics=RNS8_EXACT_WIDE_SIGNED;gd.bound_kind=RNS8_BOUND_NONE;gd.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    gd.m=3;gd.n=2;gd.k=5;gd.max_prefix=20;
    // Device-native export consumes the complete matrix basis, not a reduced
    // plan prefix. Materialize all twenty planes before crossing that boundary.
    gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* plan=nullptr;rns8_workspace* workspace=nullptr;
    check(rns8_create_plan(ctx,&gd,&plan));check(rns8_create_workspace(ctx,plan,&workspace));
    check(rns8_gemm_rns(ctx,plan,matrices[0],matrices[1],matrices[2],workspace));
    rns8_native_device_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
    check(rns8_query_native_device(&views[2],&resources));
    check(rns8_export_native_device(ctx,matrices[2],leases[2],resources.export_scratch_bytes));
    // Success is host-observed before this external consumer is launched.
    hipcheck(hipMemset(flag,0,4));hipcheck(static_cast<hipError_t>(native_consume(c,flag)));hipcheck(hipDeviceSynchronize());
    int failure=0;hipcheck(hipMemcpy(&failure,flag,4,hipMemcpyDeviceToHost));require(failure==0,"external consumer rejected native output");
    // Explicit validation-only full readback, outside the device-resident pipeline.
    std::array<int64_t,13> result{};hipcheck(hipMemcpy(result.data(),c,sizeof(result),hipMemcpyDeviceToHost));
    for(std::size_t i=0;i<result.size();++i) {
      int64_t expected=99;
      if(i>=1 && i<11 && (i-1)%4<2) {
        const int row=int((i-1)/4),col=int((i-1)%4);expected=0;
        for(int q=0;q<5;++q) expected+=int64_t((row+q)%11-5)*int64_t((q+col)%7-3);
      }
      require(result[i]==expected,"independent host product or padding mismatch");
    }
    for(auto* lease:leases) check(rns8_destroy_native_device_lease(lease));
    for(const auto& owner:owners) require(owner.refs==0 && owner.acquires==1 && owner.releases==1,"owner release mismatch");
    for(auto* matrix:matrices) check(rns8_destroy_matrix(matrix));
    check(rns8_destroy_workspace(workspace));check(rns8_destroy_plan(plan));check(rns8_destroy_context(ctx));
    hipcheck(hipFree(flag));hipcheck(hipFree(c));hipcheck(hipFree(b));hipcheck(hipFree(a));
    std::cout<<"device_producer_exact_gemm_device_consumer=passed full_host_oracle=passed padding=passed pipeline_host_payload_bytes=0 conversion_summary_bytes=20 validation_payload_bytes=104\n";
    return EXIT_SUCCESS;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return EXIT_FAILURE;}
}
