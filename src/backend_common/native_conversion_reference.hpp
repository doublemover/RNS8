#pragma once
// Independent host fragment interpreter for authored logical regressions. It
// uses exact native remainder, never the device reciprocal reducer or WMMA.
#include <array>
#include <cstdint>
#include <vector>
#include <stdexcept>
namespace rns8::detail::native_conversion_reference {
using Words=std::array<std::array<uint32_t,4>,32>;
struct Fragments { Words a{},b{}; uint64_t conversions=0; };
inline int8_t center(uint64_t bits,bool unsign,uint32_t modulus) {
  uint64_t value=bits%modulus;
  if(!unsign && (bits>>63)) {const auto magnitude=(~bits+1)%modulus;value=magnitude ? modulus-magnitude : 0;}
  return static_cast<int8_t>(value>=(modulus+1)/2 ? int(value)-int(modulus) : int(value));
}
inline uint8_t byte(uint32_t word,uint32_t index) {return static_cast<uint8_t>(word>>(index*8));}
struct Mutation {uint32_t partner_xor=16;bool swap_role=false,uninitialized_tail=false,wrong_stride=false,premature_reuse=false;};
inline Fragments cooperative(const std::vector<uint64_t>& a,const std::vector<uint64_t>& b,
    uint32_t m,uint32_t n,uint32_t k,uint32_t lda,uint32_t ldb,uint32_t row0,uint32_t col0,uint32_t base,
    uint32_t modulus,bool unsign,Mutation mutation={}) {
  Words owned{};Fragments out;
  for(uint32_t lane=0;lane<32;++lane) for(uint32_t word=0;word<4;++word)for(uint32_t cell=0;cell<4;++cell) {
    const bool is_a=(lane<16)!=mutation.swap_role;
    const auto axis=lane%16,q=base+word*4+cell;
    const bool active=q<k && (is_a ? row0+axis<m : col0+axis<n);
    const auto raw=active ? (is_a ? a.at((row0+axis)*lda+q) : b.at(q*ldb+col0+axis)) : uint64_t{};
    const auto encoded=!active && mutation.uninitialized_tail ? uint8_t(77) : uint8_t(center(raw,unsign,modulus));
    owned[lane][word]|=uint32_t(encoded)<<(cell*8);++out.conversions;
  }
  for(uint32_t lane=0;lane<32;++lane)for(uint32_t word=0;word<4;++word) {
    const auto partner=lane^mutation.partner_xor;
    if(partner>=32)throw std::invalid_argument("partner outside wave32");
    out.a[lane][word]=lane<16 ? owned[lane][word] : owned[partner][word];
    out.b[lane][word]=lane<16 ? owned[partner][word] : owned[lane][word];
  }
  return out;
}
inline std::array<Fragments,4> encoded_lds(const std::vector<uint64_t>& a,const std::vector<uint64_t>& b,
    uint32_t m,uint32_t n,uint32_t k,uint32_t lda,uint32_t ldb,uint32_t row0,uint32_t col0,uint32_t base,
    uint32_t depth,uint32_t sub,uint32_t modulus,bool unsign,Mutation mutation={}) {
  if((depth!=16 && depth!=32) || sub%16 || sub>=depth)throw std::invalid_argument("invalid encoded tile");
  const auto words=depth/4,stride=words+1;
  std::array<std::vector<uint32_t>,2> shared{std::vector<uint32_t>(32*stride,0),std::vector<uint32_t>(32*stride,0)};
  // Disjoint word ownership, independent of device thread linearization.
  for(uint32_t role=0;role<2;++role)for(uint32_t axis=0;axis<32;++axis)for(uint32_t word=0;word<words;++word) {
    uint32_t packed=0;
    for(uint32_t cell=0;cell<4;++cell) {
      const auto q=base+word*4+cell;
      const bool active=q<k && (role==0 ? row0+axis<m : col0+axis<n);
      const auto raw=active ? (role==0 ? a.at((row0+axis)*lda+q) : b.at(q*ldb+col0+axis)) : uint64_t{};
      const auto encoded=!active && mutation.uninitialized_tail ? uint8_t(77) : uint8_t(center(raw,unsign,modulus));
      packed|=uint32_t(encoded)<<(cell*8);
    }
    shared[role][axis*stride+word]=packed;
  }
  // Models a producer reusing storage before all consumers completed.
  if(mutation.premature_reuse)for(auto& role:shared)for(auto& word:role)word=0;
  std::array<Fragments,4> out{};
  for(uint32_t wave=0;wave<4;++wave)for(uint32_t lane=0;lane<32;++lane)for(uint32_t word=0;word<4;++word) {
    const auto read_stride=mutation.wrong_stride ? words : stride;
    out[wave].a[lane][word]=shared[0][((wave/2)*16+lane%16)*read_stride+sub/4+word];
    out[wave].b[lane][word]=shared[1][((wave%2)*16+lane%16)*read_stride+sub/4+word];
  }
  out[0].conversions=uint64_t(2)*32*depth;
  return out;
}
}
