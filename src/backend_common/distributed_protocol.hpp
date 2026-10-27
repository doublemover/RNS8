#pragma once
#include "distributed_inputs.hpp"
#include "sha256.hpp"
#include <array>
#include <cstring>
namespace rns8::detail::distributed {
// Explicit little-endian wire fields. No sizeof(Work), pointer, C++ object or
// host-endian scalar is a protocol. The digest covers the complete retained
// plan, input snapshots, operation and epoch before accepting any fragments.
using Envelope=std::array<uint8_t,112>;
inline void wire_u64(uint8_t* p,uint64_t value){for(unsigned i=0;i<8;++i)p[i]=uint8_t(value>>(8*i));}
inline uint64_t wire_u64(const uint8_t* p){uint64_t v=0;for(unsigned i=0;i<8;++i)v|=uint64_t(p[i])<<(8*i);return v;}
inline std::string process_binding(const Plan& plan,const Inputs& inputs,uint64_t epoch) {
  std::string key=plan.identity()+";epoch="+std::to_string(epoch)+";native64-le;";
  // Encode native bits in a stable byte order, including the owned padding.
  // Hash the snapshots separately to keep temporary storage bounded.
  const auto hash=[](const std::vector<uint64_t>& words){
    std::string state=sha256("native64-le-chunks-v1;words="+std::to_string(words.size()));
    std::array<uint8_t,4096> bytes{};
    for(size_t start=0;start<words.size();start+=512) {
      const size_t count=std::min(size_t(512),words.size()-start);
      for(size_t i=0;i<count;++i)wire_u64(bytes.data()+8*i,words[start+i]);
      state=sha256(state+std::string(reinterpret_cast<const char*>(bytes.data()),count*8));
    }
    return state;
  };
  return sha256(key+hash(inputs.lhs())+hash(inputs.rhs()));
}
inline Envelope fragment_envelope(const Plan& plan,const std::string& binding,uint64_t epoch,const Work& work) {
  Envelope out{};if(!digest(binding))return out;std::memcpy(out.data(),"RNS8NET1",8);wire_u64(out.data()+8,epoch);
  wire_u64(out.data()+16,work.id);wire_u64(out.data()+24,work.rank);wire_u64(out.data()+32,plan.request().owner);
  wire_u64(out.data()+40,work.bytes);std::memcpy(out.data()+48,binding.data(),64);return out;
}
inline Admission validate_envelope(const Plan& plan,const std::string& binding,uint64_t epoch,
    uint32_t source,const Envelope& envelope,uint64_t& id) {
  if(!digest(binding) || !epoch || epoch==UINT64_MAX || std::memcmp(envelope.data(),"RNS8NET1",8)!=0 ||
      wire_u64(envelope.data()+8)!=epoch || wire_u64(envelope.data()+24)!=source ||
      wire_u64(envelope.data()+32)!=plan.request().owner || std::memcmp(envelope.data()+48,binding.data(),64)!=0)
    return reject(RNS8_INVALID_ARGUMENT,"foreign_or_stale_network_fragment",source);
  const auto work_id=wire_u64(envelope.data()+16);
  if(work_id>=plan.work().size())return reject(RNS8_INVALID_ARGUMENT,"unknown_network_work",source,work_id);
  const auto& expected=plan.work()[work_id];
  if(expected.rank!=source || wire_u64(envelope.data()+40)!=expected.bytes)
    return reject(RNS8_INVALID_ARGUMENT,"network_membership_or_extent",source,work_id);
  id=work_id;return {};
}
class CoverageLedger {
 public:
  explicit CoverageLedger(const Plan& plan):plan_(plan),seen_(plan.work().size(),false),ranks_(plan.request().ranks.size(),false){}
  Admission accept(uint64_t id,uint32_t rank) {
    if(id>=seen_.size() || plan_.work()[id].rank!=rank)return reject(RNS8_INVALID_ARGUMENT,"unexpected_network_fragment",rank,id);
    if(seen_[id])return reject(RNS8_INVALID_ARGUMENT,"duplicate_network_fragment",rank,id);
    seen_[id]=true;++completed_;return {};
  }
  Admission acknowledge(uint32_t rank) {
    if(rank>=ranks_.size() || ranks_[rank])return reject(RNS8_INVALID_ARGUMENT,"duplicate_or_foreign_rank_ack",rank);
    for(const auto& work:plan_.work())if(work.rank==rank && !seen_[work.id])return reject(RNS8_BUSY,"rank_coverage_incomplete",rank,work.id);
    ranks_[rank]=true;return {};
  }
  bool complete()const{return completed_==seen_.size() && std::all_of(ranks_.begin(),ranks_.end(),[](bool value){return value;});}
  uint64_t completed()const{return completed_;}
 private:
  const Plan& plan_;std::vector<bool> seen_,ranks_;uint64_t completed_=0;
};
}
