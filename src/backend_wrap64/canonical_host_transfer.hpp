#pragma once
#include "backend_common/host_upload.hpp"
#include <cstring>

namespace rns8::detail {
inline bool native_u64_matches_canonical_bytes(const uint8_t* bytes) noexcept {
  for(unsigned i=0;i<8;++i)if(bytes[i]!=i+1)return false;
  return true;
}
inline bool host_u64_has_canonical_bytes() noexcept {
  static const bool identity=[] {
    const uint64_t word=UINT64_C(0x0807060504030201);
    uint8_t bytes[8];std::memcpy(bytes,&word,8);
    return native_u64_matches_canonical_bytes(bytes);
  }();
  return identity;
}
struct CanonicalU64HostTransfer {
  HostUploadPlan logical;
  bool native_byte_identity=false;
};
inline bool canonical_u64_host_transfer(int64_t rows,int64_t cols,int64_t ld,CanonicalU64HostTransfer& out) noexcept {
  HostUploadPlan logical;
  if(sizeof(uint64_t)!=8 || !make_host_upload_plan(rows,cols,ld,8,logical))return false;
  // The AMDGPU byte-limb code object independently compile-checks its 8-byte,
  // little-endian object layout. Big-endian hosts use explicit conversion.
  out={logical,host_u64_has_canonical_bytes()};return true;
}
inline void store_canonical_u64(uint8_t* bytes,uint64_t word) noexcept {
  for(unsigned i=0;i<8;++i)bytes[i]=static_cast<uint8_t>(word>>(8*i));
}
inline uint64_t load_canonical_u64(const uint8_t* bytes) noexcept {
  uint64_t word=0;for(unsigned i=0;i<8;++i)word|=uint64_t(bytes[i])<<(8*i);return word;
}
} // namespace rns8::detail
