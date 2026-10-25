#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#define RNS8_BINARY_HD __host__ __device__
#else
#define RNS8_BINARY_HD
#endif
namespace rns8::detail::binary_layout {
// A small linear map over GF(2). Columns name the logical image of each
// physical coordinate bit. Zero columns are explicit replicas, never stores.
struct Map {
  uint32_t input_bits=0,output_bits=0,columns[16]{};
  RNS8_BINARY_HD constexpr uint32_t apply(uint32_t input) const {
    uint32_t output=0;
    for(uint32_t i=0;i<input_bits;++i)if(input&(1u<<i))output^=columns[i];
    return output;
  }
};
constexpr bool valid(const Map& map) {
  if(!map.input_bits || map.input_bits>16 || !map.output_bits || map.output_bits>16)return false;
  for(unsigned i=0;i<16;++i)
    if(i<map.input_bits ? map.columns[i]>=(1u<<map.output_bits) : map.columns[i]!=0)return false;
  return true;
}
struct Elimination {
  uint32_t image[16]{},preimage[16]{},kernel[16]{},rank=0,nullity=0;
};
constexpr Elimination eliminate(const Map& map,uint32_t fixed=0) {
  Elimination out;
  if(!valid(map))return out;
  for(uint32_t i=0;i<map.input_bits;++i)if(!(fixed&(1u<<i))) {
    uint32_t image=map.columns[i],preimage=1u<<i;
    for(int bit=int(map.output_bits)-1;bit>=0;--bit)if(image&(1u<<bit)) {
      if(out.image[bit]){image^=out.image[bit];preimage^=out.preimage[bit];}
      else {out.image[bit]=image;out.preimage[bit]=preimage;++out.rank;break;}
    }
    if(!image)out.kernel[out.nullity++]=preimage;
  }
  return out;
}
struct Preimage {bool valid=false;uint32_t value=0;};
constexpr Preimage solve(const Map& map,uint32_t logical,uint32_t fixed_mask=0,uint32_t fixed_value=0) {
  if(!valid(map) || logical>=(1u<<map.output_bits) || fixed_mask>=(1u<<map.input_bits) ||
      (fixed_value&~fixed_mask))return {};
  const auto pivots=eliminate(map,fixed_mask);uint32_t image=logical^map.apply(fixed_value),physical=fixed_value;
  for(int bit=int(map.output_bits)-1;bit>=0;--bit)if(image&(1u<<bit)) {
    if(!pivots.image[bit])return {};
    image^=pivots.image[bit];physical^=pivots.preimage[bit];
  }
  return {true,physical};
}
struct Result {bool valid=false;Map map;};
constexpr Result right_inverse(const Map& map) {
  if(!valid(map) || eliminate(map).rank!=map.output_bits)return {};
  Result result;result.map.input_bits=map.output_bits;result.map.output_bits=map.input_bits;
  for(uint32_t i=0;i<map.output_bits;++i)result.map.columns[i]=solve(map,1u<<i).value;
  result.valid=true;return result;
}
// compose(outer,inner)(x) = outer(inner(x)).
constexpr Result compose(const Map& outer,const Map& inner) {
  if(!valid(outer) || !valid(inner) || inner.output_bits!=outer.input_bits)return {};
  Result result;result.map.input_bits=inner.input_bits;result.map.output_bits=outer.output_bits;
  for(uint32_t i=0;i<inner.input_bits;++i)result.map.columns[i]=outer.apply(inner.columns[i]);
  result.valid=true;return result;
}
enum class Conversion:uint32_t {register_only,within_wave,cross_wave_lds,unsupported};
struct Physical {
  Map map;uint32_t lane_bits=5,register_bits=0,wave_bits=0,element_bits=8;
};
constexpr bool valid(const Physical& p) {
  return valid(p.map) && (p.lane_bits==5 || p.lane_bits==6) && p.wave_bits<=4 &&
      p.lane_bits+p.register_bits+p.wave_bits==p.map.input_bits &&
      (p.element_bits==8 || p.element_bits==16 || p.element_bits==32 || p.element_bits==64);
}
inline Conversion conversion(const Physical& source,const Physical& destination) {
  if(!valid(source) || !valid(destination) || source.lane_bits!=destination.lane_bits ||
      source.wave_bits!=destination.wave_bits || source.element_bits!=destination.element_bits ||
      source.map.output_bits!=destination.map.output_bits)return Conversion::unsupported;
  const auto lane_mask=(1u<<source.lane_bits)-1;
  const auto wave_mask=((1u<<source.wave_bits)-1)<<(source.lane_bits+source.register_bits);
  Conversion result=Conversion::register_only;
  for(uint32_t slot=0;slot<(1u<<destination.map.input_bits);++slot) {
    const uint32_t wave=(slot>>(destination.lane_bits+destination.register_bits))<<(source.lane_bits+source.register_bits);
    const uint32_t logical=destination.map.apply(slot);
    if(solve(source.map,logical,lane_mask|wave_mask,(slot&lane_mask)|wave).valid)continue;
    if(solve(source.map,logical,wave_mask,wave).valid) {
      if(result==Conversion::register_only)result=Conversion::within_wave;
    } else if(solve(source.map,logical).valid)result=Conversion::cross_wave_lds;
    else return Conversion::unsupported;
  }
  return result;
}
constexpr uint64_t fingerprint(const Map& map,uint32_t target,uint32_t wave,uint32_t opcode,uint32_t element_bits) {
  uint64_t hash=UINT64_C(14695981039346656037);
  const auto add=[&](uint32_t value){for(unsigned i=0;i<4;++i)hash=(hash^uint8_t(value>>(8*i)))*UINT64_C(1099511628211);};
  add(1);add(target);add(wave);add(opcode);add(element_bits);add(map.input_bits);add(map.output_bits);
  for(unsigned i=0;i<16;++i)add(map.columns[i]);return hash;
}
// The executable paired-column contract: four lane axis bits choose even
// columns, the pair bit chooses the adjacent odd column. All 32 stores unique.
inline constexpr Map paired_columns{5,5,{2,4,8,16,1}};
RNS8_BINARY_HD constexpr uint32_t paired_column(uint32_t lane,uint32_t pair) {
  constexpr Map map=paired_columns;return map.apply((lane&15u)|(pair<<4));
}
static_assert(right_inverse(paired_columns).valid);

enum class LdsOpcode:uint32_t {unknown,read_b32,write_b32};
struct LdsContract {
  LdsOpcode opcode=LdsOpcode::unknown;
  uint32_t banks=32,bank_bytes=4,lanes_per_phase=32,wave_lanes=32;
};
struct BankModel {bool supported=false,measured=false;uint32_t phases=0,max_bank_words=0;};
inline BankModel analyze_banks(const uint64_t* addresses,uint32_t lanes,const LdsContract& c) {
  BankModel out;
  if(!addresses || lanes!=c.wave_lanes || (lanes!=32 && lanes!=64) || c.banks!=32 || c.bank_bytes!=4 ||
      c.lanes_per_phase!=32 || (c.opcode!=LdsOpcode::read_b32 && c.opcode!=LdsOpcode::write_b32))return out;
  for(uint32_t i=0;i<lanes;++i)if(addresses[i]%4)return out;
  for(uint32_t phase=0;phase<lanes;phase+=32) {
    uint32_t bank_words[32]{};
    for(uint32_t i=phase;i<phase+32;++i) {
      bool duplicate=false;
      for(uint32_t j=phase;j<i;++j)if(addresses[j]==addresses[i])duplicate=true;
      if(duplicate && c.opcode==LdsOpcode::write_b32)return {};
      if(!duplicate)++bank_words[(addresses[i]/4)%32];
    }
    for(auto count:bank_words)if(count>out.max_bank_words)out.max_bank_words=count;
    ++out.phases;
  }
  out.supported=true;return out;
}
}
#undef RNS8_BINARY_HD
