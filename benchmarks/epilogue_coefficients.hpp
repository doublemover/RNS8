#pragma once
#include <cstdint>
#include <string>
namespace rns8::bench {
inline bool epilogue_coefficients(const std::string& profile,bool unsigned_inputs,int64_t& alpha,int64_t& beta){
  if(profile=="extreme"){alpha=unsigned_inputs?INT64_MAX:INT64_MIN;beta=INT64_MAX;}
  else if(profile=="zero"){alpha=beta=0;}
  else if(profile=="copy"){alpha=0;beta=1;}
  else if(profile=="unit"){alpha=beta=1;}
  else if(profile=="general"){alpha=unsigned_inputs?3:-3;beta=5;}
  else if(profile=="channel-zero"){alpha=255;beta=256;}
  else return false;
  return true;
}
}
