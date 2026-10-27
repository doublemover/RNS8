#include "backend_common/exact_lift_workspace.hpp"
#include <catch2/catch_test_macros.hpp>
using namespace rns8::detail;
TEST_CASE("exact lift retained status-first storage admits every suffix and odd cell count") {
  auto* scratch=reinterpret_cast<void*>(uintptr_t{0x100000});auto* residues=reinterpret_cast<void*>(uintptr_t{0x200000});
  for(int64_t cells:{1,7,1024})for(uint32_t source=1;source<20;++source)for(uint32_t target=source+1;target<=20;++target){
    ExactLiftWorkspaceView v;const uint64_t required=uint64_t(cells)*(target-source)+sizeof(int);
    REQUIRE(exact_lift_workspace_view(scratch,required,residues,cells,source,target,v)==RNS8_SUCCESS);
    CHECK(v.status==scratch);CHECK(v.suffix==static_cast<int8_t*>(scratch)+sizeof(int));CHECK(v.required_bytes==required);
    CHECK(exact_lift_workspace_view(scratch,required-1,residues,cells,source,target,v)==RNS8_WORKSPACE_TOO_SMALL);
    CHECK(v.required_bytes==required); // failed preflight does not publish a replacement view
  }
}
TEST_CASE("exact lift retained storage rejects alignment overlap and invalid prefixes without publication") {
  auto* scratch=reinterpret_cast<void*>(uintptr_t{0x100000});auto* residues=reinterpret_cast<void*>(uintptr_t{0x200000});ExactLiftWorkspaceView v;
  CHECK(exact_lift_workspace_view(static_cast<char*>(scratch)+1,100,residues,1,1,20,v)==RNS8_INVALID_ARGUMENT);
  CHECK(exact_lift_workspace_view(residues,100,residues,1,1,20,v)==RNS8_INVALID_ARGUMENT);
  CHECK(exact_lift_workspace_view(nullptr,100,residues,1,1,20,v)==RNS8_INVALID_ARGUMENT);
  CHECK(exact_lift_workspace_view(scratch,100,residues,1,1,1,v)==RNS8_INVALID_ARGUMENT);
  CHECK(v.status==nullptr);CHECK(v.suffix==nullptr);
}
