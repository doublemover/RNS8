#include "backend_common/group_crt.hpp"
#include "backend_common/garner_constants.hpp"
#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <stdexcept>

namespace rns8::detail::group_crt {
namespace {
using boost::multiprecision::cpp_int;
balanced_crt::Wide wide(cpp_int value) {
  if(value<0 || value>=(cpp_int(1)<<160))throw std::logic_error("group CRT root width");
  balanced_crt::Wide out;
  for(auto& limb:out.limb){limb=static_cast<uint64_t>(value&UINT64_MAX);value>>=64;}
  return out;
}
}
Plan make_plan(uint32_t prefix,const uint8_t* partition,uint32_t count) {
  if(!prefix || prefix>20 || !partition || !count || count>4 || count>prefix)
    throw std::invalid_argument("group CRT partition");
  Plan out;out.prefix=prefix;out.group_count=count;
  std::array<cpp_int,4> products;products.fill(cpp_int(1));
  cpp_int root=1;
  for(uint32_t channel=0;channel<prefix;++channel) {
    const uint32_t id=partition[channel];
    if(id>=count || out.groups[id].count>=8)throw std::invalid_argument("group CRT leaf width");
    auto& group=out.groups[id];const auto slot=group.count++;
    const auto modulus=garner::kConstants.moduli[channel];
    group.channels[slot]=static_cast<uint8_t>(channel);group.moduli[slot]=static_cast<uint16_t>(modulus);
    group.inverses[slot]=static_cast<uint16_t>(garner::inverse(static_cast<uint32_t>(products[id]%modulus),modulus));
    if((products[id]*group.inverses[slot])%modulus!=1)throw std::logic_error("group CRT leaf inverse");
    group.channel_mask|=uint32_t(1)<<channel;products[id]*=modulus;root*=modulus;
    if(products[id]>UINT64_MAX)throw std::logic_error("group CRT leaf product");
    group.product=static_cast<uint64_t>(products[id]);
  }
  std::array<uint32_t,4> slots{},masks{};
  for(uint32_t g=0;g<count;++g) {
    if(!out.groups[g].count)throw std::invalid_argument("empty group CRT leaf");
    slots[g]=g;masks[g]=out.groups[g].channel_mask;
  }
  for(uint32_t remaining=count;remaining>1;) {
    uint32_t next=0;
    for(uint32_t i=0;i<remaining;i+=2) {
      const auto left=slots[i];
      if(i+1<remaining) {
        const auto right=slots[i+1];cpp_int inverse=0,product=1;
        // Reuse Garner's small-modulus inverse owner, reconstructing A^-1
        // over exactly the right subtree's channels, including irregular groups.
        for(uint32_t p=0;p<prefix;++p)if(masks[right]&(uint32_t(1)<<p)) {
          const auto m=garner::kConstants.moduli[p];
          const auto target=garner::inverse(static_cast<uint32_t>(products[left]%m),m);
          const auto current=static_cast<uint32_t>(inverse%m);
          const auto local=garner::inverse(static_cast<uint32_t>(product%m),m);
          inverse+=product*(((target+m-current)%m)*local%m);product*=m;
        }
        if(product!=products[right] || (products[left]*inverse)%product!=1)
          throw std::logic_error("group CRT merge inverse");
        out.merges[out.merge_count++]={wide(products[left]),wide(products[right]),wide(inverse),left,right};
        products[left]*=products[right];masks[left]|=masks[right];
        (void)wide(products[left]);
      }
      slots[next++]=left;
    }
    remaining=next;
  }
  if(products[0]!=root || masks[0]!=(uint32_t(1)<<prefix)-1 || out.merge_count+1!=count)
    throw std::logic_error("group CRT complete root");
  return out;
}
const Plan& plan(uint32_t prefix) {
  if(!prefix || prefix>20)throw std::invalid_argument("group CRT prefix");
  static const auto plans=[] {
    std::array<Plan,21> result{};
    for(uint32_t p=1;p<=20;++p) {
      const auto count=(p+7)/8;std::array<uint8_t,20> partition{};
      for(uint32_t c=0;c<p;++c)partition[c]=static_cast<uint8_t>(c*count/p);
      result[p]=make_plan(p,partition.data(),count);
    }
    return result;
  }();
  return plans[prefix];
}
} // namespace rns8::detail::group_crt
