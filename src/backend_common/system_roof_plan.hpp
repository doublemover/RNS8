#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rns8::detail::system_roof {
enum class Mode : uint32_t { global_copy, lds_read, lds_read_write, launch, event, descriptor, graph };
struct Problem {
  Mode mode=Mode::global_copy;
  uint64_t words=128,displacement_words=0,budget_bytes=uint64_t{1}<<30;
  uint32_t blocks=1,iterations=1,stride=1;
};
struct Resources {
  bool admitted=false;
  uint64_t device_bytes=0,pinned_bytes=0,total_bytes=0;
  uint64_t unique_source_bytes=0,issued_global_read_bytes=0,issued_global_write_bytes=0;
  uint64_t issued_lds_read_bytes=0,issued_lds_write_bytes=0,displacement_bytes=0;
  uint32_t lds_bytes_per_block=0;
};
inline const char* name(Mode mode) {
  switch(mode) {
    case Mode::global_copy:return "global-copy";case Mode::lds_read:return "lds-read";
    case Mode::lds_read_write:return "lds-read-write";case Mode::launch:return "launch";
    case Mode::event:return "event";case Mode::descriptor:return "descriptor";case Mode::graph:return "graph";
  } return "invalid";
}
inline bool parse(const std::string& text,Mode& mode) {
  for(uint32_t i=0;i<=uint32_t(Mode::graph);++i)if(text==name(Mode(i))){mode=Mode(i);return true;}return false;
}
inline uint32_t seed(uint64_t index) {
  // Explicit finite-ring arithmetic, independent of any exact-GEMM semantics.
  return uint32_t(index)*UINT32_C(747796405)+UINT32_C(2891336453);
}
inline Resources query(const Problem& p) {
  Resources r;
  if(uint32_t(p.mode)>uint32_t(Mode::graph) || !p.words || p.words>(uint64_t{1}<<29) ||
      p.displacement_words>(uint64_t{1}<<29) || !p.blocks || p.blocks>4096 ||
      !p.iterations || p.iterations>4096 || !p.stride || p.stride>32 ||
      (p.stride&(p.stride-1)))return r;
  const bool lds=p.mode==Mode::lds_read || p.mode==Mode::lds_read_write;
  if(lds) {
    if(p.words!=uint64_t(p.blocks)*128 || uint64_t(p.blocks)*128*p.iterations>(uint64_t{1}<<28))return r;
    r.lds_bytes_per_block=128*p.stride*4;
    r.issued_lds_read_bytes=p.words*4*p.iterations;
    r.issued_lds_write_bytes=p.words*4*(p.mode==Mode::lds_read_write?p.iterations+1:1);
  } else if(p.iterations!=1 || p.stride!=1)return r;
  if((p.mode==Mode::launch || p.mode==Mode::event || p.mode==Mode::graph) && (p.words!=1 || p.blocks!=1))return r;
  // Include two bounded pinned copies, all device payloads, displacement audit
  // scratch, and an explicit metadata reserve. Runtime allocator overhead is unknown.
  const auto audit_words=p.displacement_words?UINT64_C(65536):0;
  r.device_bytes=(2*p.words+p.displacement_words+16)*4;
  r.pinned_bytes=(p.words+(p.words+16>audit_words?p.words+16:audit_words))*4;
  r.total_bytes=r.device_bytes+r.pinned_bytes+65536;
  r.displacement_bytes=p.displacement_words*8;
  if(p.mode!=Mode::event) {
    r.unique_source_bytes=p.words*4;
    if(p.mode!=Mode::descriptor)r.issued_global_read_bytes=p.words*4;
    r.issued_global_write_bytes=p.words*4;
  }
  r.admitted=r.total_bytes<=p.budget_bytes;return r;
}
// Separate scalar iteration over logical lane values; it does not use the LDS
// physical stride/indexing implementation or its shared-memory addresses.
inline std::vector<uint32_t> reference_block(const Problem& p,uint32_t block) {
  std::vector<uint32_t> values(128),next(128),sum(128);
  for(uint32_t lane=0;lane<128;++lane)values[lane]=seed(uint64_t(block)*128+lane);
  for(uint32_t iteration=0;iteration<p.iterations;++iteration) {
    for(uint32_t lane=0;lane<128;++lane) {
      const auto observed=values[(lane+iteration)%128];sum[lane]+=observed;
      next[lane]=observed+iteration+lane;
    }
    if(p.mode==Mode::lds_read_write)values.swap(next);
  }
  return sum;
}
} // namespace rns8::detail::system_roof
