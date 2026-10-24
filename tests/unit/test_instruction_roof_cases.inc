#include "backend_common/instruction_roof_plan.hpp"
#include <cstdlib>
TEST_CASE("instruction roof admission proves every signed byte accumulator") {
  using namespace rns8::detail;
  for(uint32_t chains:{1u,2u,4u,8u}) for(uint32_t iterations:{1u,4095u,4096u})
    for(uint32_t waves:{1u,5u,4096u}) for(bool reduce:{false,true}) {
      InstructionRoofProblem p{1,chains,iterations,waves,reduce};
      const auto support=query_instruction_roof(p);
      REQUIRE(support.status==RNS8_SUCCESS);
      CHECK(support.output_bytes==uint64_t(waves)*chains*256*4);
      CHECK(support.instructions==uint64_t(waves)*chains*iterations);
      CHECK(support.integer_operations==support.instructions*8192);
      CHECK(support.maximum_accumulator_magnitude<=INT32_MAX);
      for(int a:{-128,-127,-1,0,1,127}) for(int b:{-128,-1,0,1,127}) {
        const int64_t product=int64_t(16)*a*b;
        const int64_t bound=reduce?std::abs(product)+125:std::abs(product)*iterations;
        CHECK(uint64_t(bound)<=support.maximum_accumulator_magnitude);
      }
    }
  for(uint32_t chains:{0u,3u,16u,UINT32_MAX})
    CHECK(query_instruction_roof({1,chains,1,1,false}).status==RNS8_INVALID_ARGUMENT);
  for(uint32_t iterations:{0u,4097u,UINT32_MAX})
    CHECK(query_instruction_roof({1,1,iterations,1,false}).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_instruction_roof({2,1,1,1,false}).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_instruction_roof({1,1,1,0,false}).status==RNS8_INVALID_ARGUMENT);
  CHECK(query_instruction_roof({1,1,1,4097,false}).status==RNS8_INVALID_ARGUMENT);
}
