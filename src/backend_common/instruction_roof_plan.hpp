#pragma once
#include "rns8/status.h"
#include <cstdint>
namespace rns8::detail {
struct InstructionRoofProblem {
  uint32_t version=1,chains=1,iterations=1,waves=1;
  bool reduce_each_iteration=false;
};
struct InstructionRoofResources {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t input_bytes=0,output_bytes=0,instructions=0,integer_operations=0;
  uint64_t maximum_accumulator_magnitude=0;
};
inline InstructionRoofResources query_instruction_roof(const InstructionRoofProblem& p) {
  InstructionRoofResources out;
  if(p.version!=1 || (p.chains!=1 && p.chains!=2 && p.chains!=4 && p.chains!=8) ||
     !p.iterations || p.iterations>4096 || !p.waves || p.waves>4096) return out;
  // Every signed-byte product is bounded by 128^2. Raw chains span at most
  // 65536 products; reduced chains carry a centered modulus-251 residue.
  out.maximum_accumulator_magnitude=p.reduce_each_iteration?16*128*128+125:
      uint64_t(p.iterations)*16*128*128;
  out.input_bytes=p.chains; // each of two input arrays
  out.output_bytes=uint64_t(p.waves)*p.chains*256*sizeof(int32_t);
  out.instructions=uint64_t(p.waves)*p.chains*p.iterations;
  out.integer_operations=out.instructions*2*16*16*16;
  out.status=RNS8_SUCCESS;return out;
}
}
