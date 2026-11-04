#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include "backend_common/packed_gemm_layout.hpp"
#include "backend_common/export_launch.hpp"

TEST_CASE("generated CK and rocWMMA packing domains reject narrow intermediate overflows") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  PackedGemmLayout out;
  for(uint64_t alignment:{64u,128u}) {
    REQUIRE(ck_packed_gemm_layout(1,1,1,alignment,out));
    CHECK(out.a_bytes==alignment*64);CHECK(out.c_bytes==alignment*alignment);
    for(uint64_t k:{1u,63u,64u,65u,32767u,32768u,32769u}) {
      REQUIRE(ck_packed_gemm_layout(73,37,k,alignment,out));
      CHECK(cpp_int(out.a_bytes)==cpp_int(out.rows)*out.k);
      CHECK(cpp_int(out.b_bytes)==cpp_int(out.cols)*out.k);
      CHECK(cpp_int(out.c_bytes)==cpp_int(out.rows)*out.cols);
      CHECK(out.k==std::max(uint64_t(64),std::min(k,uint64_t(32768))));
    }
    CHECK_FALSE(ck_packed_gemm_layout(UINT64_C(33554432),1,1,alignment,out));
    CHECK_FALSE(ck_packed_gemm_layout(1,UINT64_C(33554432),1,alignment,out));
  }
  for(uint64_t k:{1u,15u,16u}) {
    CHECK(rocwmma_packed_gemm_layout((UINT64_C(1)<<27)-16,1,k,out));
    for(uint64_t m:{(UINT64_C(1)<<27)-1,UINT64_C(1)<<27,(UINT64_C(1)<<27)+1}) {
      CHECK_FALSE(rocwmma_packed_gemm_layout(m,1,k,out));
      CHECK_FALSE(rocwmma_packed_gemm_layout(1,m,k,out));
    }
  }
  REQUIRE(rocwmma_packed_gemm_layout(17,19,65537,out));CHECK(out.k==65536);
  const auto before=out;
  CHECK_FALSE(rocwmma_packed_gemm_layout(UINT64_MAX,1,1,out));
  CHECK(out.total_bytes==before.total_bytes);CHECK(out.rows==before.rows);
}

TEST_CASE("export geometry admits signed32 endpoints with wide ceil and checked scheduled grids") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  ExportLaunch out;
  for(uint64_t tail:{257u,256u,255u,3u,2u,1u,0u}) {
    const uint64_t cells=uint64_t(INT32_MAX)-tail;
    REQUIRE(checked_export_launch(1,cells,INT32_MAX,out));
    CHECK(cpp_int(out.blocks)==(cpp_int(cells)+255)/256);
    CHECK(cpp_int(out.quad_blocks)==(cpp_int(cells)+1023)/1024);
    CHECK(uint64_t(out.blocks)*256-1<=uint64_t(INT32_MAX));
    CHECK(uint64_t(out.quad_blocks)*1024-1<=uint64_t(INT32_MAX));
    const auto previous=out;
    CHECK_FALSE(checked_export_launch(1,cells,uint64_t(out.blocks)-1,out));
    CHECK(out.elements==previous.elements);
  }
  REQUIRE(checked_export_launch(32769,65534,INT32_MAX,out));
  CHECK(out.elements==INT32_MAX-1);
  CHECK_FALSE(checked_export_launch(1,uint64_t(INT32_MAX)+1,INT32_MAX,out));
  CHECK_FALSE(checked_export_launch(UINT64_MAX,2,INT32_MAX,out));
  REQUIRE(checked_export_launch(1,257,7,out));
  int blocks=-1;
  REQUIRE(checked_export_schedule(3,out,blocks));CHECK(blocks==6);
  CHECK_FALSE(checked_export_schedule(4,out,blocks));CHECK(blocks==6);
  CHECK_FALSE(checked_export_schedule(UINT64_MAX,out,blocks));CHECK(blocks==6);
}

#include "backend_common/host_upload.hpp"
#include <cstring>
#include <vector>

TEST_CASE("host uploads describe only accessible logical rows and compact device storage") {
  using namespace rns8::detail;
  for(std::size_t element_bytes:{1u,8u})for(int64_t rows:{1,2,5})for(int64_t ld:{3,8,4096}) {
    HostUploadPlan plan;
    REQUIRE(make_host_upload_plan(rows,3,ld,element_bytes,plan));
    CHECK(plan.source_span_bytes==((rows-1)*ld+3)*element_bytes);
    CHECK(plan.device_bytes==rows*3*element_bytes);
    CHECK(plan.device_ld==3);
    std::vector<unsigned char> src(plan.source_span_bytes,0xa5),dst(plan.device_bytes,0);
    // Model the actual 1D/2D boundary; reject oversized requests before reading.
    auto copy=[&](std::size_t span) {
      if(plan.source_span_bytes>span)return false;
      for(std::size_t row=0;row<plan.rows;++row) {
        const auto offset=row*plan.source_pitch_bytes;
        if(offset>span || plan.row_bytes>span-offset)return false;
        std::memcpy(dst.data()+row*plan.row_bytes,src.data()+offset,plan.row_bytes);
      }
      return true;
    };
    CHECK_FALSE(copy(src.size()-1));
    REQUIRE(copy(src.size()));
    CHECK(dst==std::vector<unsigned char>(plan.device_bytes,0xa5));
  }
  HostUploadPlan out;
  REQUIRE(make_host_upload_plan(1,1,INT64_MAX,8,out));
  CHECK(out.contiguous());CHECK(out.source_pitch_bytes==8);CHECK(out.device_bytes==8);
  const auto before=out;
  CHECK_FALSE(make_host_upload_plan(2,1,INT64_MAX,8,out));
  CHECK_FALSE(make_host_upload_plan(INT64_MAX,INT64_MAX,INT64_MAX,8,out));
  CHECK_FALSE(make_host_upload_plan(2,3,2,8,out));
  CHECK_FALSE(make_host_upload_plan(0,1,1,8,out));
  CHECK_FALSE(make_host_upload_plan(1,1,1,0,out));
  CHECK(out.device_bytes==before.device_bytes);
}
