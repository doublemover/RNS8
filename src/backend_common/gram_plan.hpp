#pragma once
#include "address_domain.hpp"
#include "rns8/rns8.h"
#include <vector>
namespace rns8::detail {
enum class GramOrientation : uint32_t { transpose_times_self=0,self_times_transpose=1 };
struct GramTile { uint32_t row,col; };
struct GramProblem {
  uint32_t version=1;
  GramOrientation orientation=GramOrientation::transpose_times_self;
  rns8_semantics semantics=RNS8_EXACT_WIDE_SIGNED;
  uint64_t source_instance=0,source_generation=0;
  int64_t rows=0,cols=0,source_ld=0,output_ld=0;
  uint32_t prefix=0,modulus=0;
};
struct GramSupport {
  rns8_status status=RNS8_INVALID_ARGUMENT;
  uint64_t order=0,k=0,tile_count=0,job_bytes=0,source_bytes=0,output_bytes=0;
};
inline GramSupport query_gram_support(const GramProblem& p,uint64_t budget,bool gpu,bool wmma) {
  GramSupport s;
  if(p.version!=1 || !p.source_instance || p.rows<=0 || p.cols<=0 ||
     uint32_t(p.orientation)>1 || p.source_ld<p.cols) return s;
  const bool finite=p.semantics==RNS8_FINITE_RING_U8 || p.semantics==RNS8_FINITE_FIELD_U8;
  if(p.semantics!=RNS8_EXACT_WIDE_SIGNED && !finite) {s.status=RNS8_UNSUPPORTED_BACKEND;return s;}
  if((finite && (p.prefix!=1 || p.modulus<2 || p.modulus>256)) ||
     (!finite && (!p.prefix || p.prefix>20 || p.modulus))) return s;
  if(p.semantics==RNS8_FINITE_FIELD_U8)
    for(uint32_t d=2;d*d<=p.modulus;++d) if(p.modulus%d==0) return s;
  s.order=p.orientation==GramOrientation::transpose_times_self?p.cols:p.rows;
  s.k=p.orientation==GramOrientation::transpose_times_self?p.rows:p.cols;
  AddressDomain src,dst;
  if(p.output_ld<0 || !checked_host_plane_address(p.rows,p.cols,p.source_ld,p.prefix,1,src) ||
     !checked_host_plane_address(s.order,s.order,p.output_ld,p.prefix,1,dst)) {
    s.status=RNS8_RANGE_ERROR;return s;
  }
  s.source_bytes=src.byte_extent;s.output_bytes=dst.byte_extent;
  const uint64_t q=(s.order+15)/16;
  if(q>UINT32_MAX || q>UINT64_MAX/(q+1)) {s.status=RNS8_RANGE_ERROR;return s;}
  s.tile_count=q*(q+1)/2;
  if(s.tile_count>uint64_t(PTRDIFF_MAX)/sizeof(GramTile)) {s.status=RNS8_RANGE_ERROR;return s;}
  s.job_bytes=s.tile_count*sizeof(GramTile);
  if(gpu && (!qualified_i32_matrix_address(p.rows,p.cols,p.source_ld,p.prefix) ||
      !qualified_i32_matrix_address(s.order,s.order,p.output_ld,p.prefix) || s.tile_count>INT32_MAX)) {
    s.status=RNS8_UNSUPPORTED_BACKEND;return s;
  }
  // GPU keeps both host job list and device copy; capability word is explicit.
  const uint64_t multiplier=gpu?2:1;
  if(s.job_bytes>(UINT64_MAX-(wmma?4:0))/multiplier ||
     s.job_bytes*multiplier+(wmma?4:0)>budget) {s.status=RNS8_WORKSPACE_TOO_SMALL;return s;}
  if(wmma && !gpu) {s.status=RNS8_UNSUPPORTED_BACKEND;return s;}
  s.status=RNS8_SUCCESS;return s;
}
inline std::vector<GramTile> gram_tiles(const GramSupport& s) {
  std::vector<GramTile> jobs;
  jobs.reserve(static_cast<size_t>(s.tile_count));
  const uint32_t q=uint32_t((s.order+15)/16);
  for(uint32_t row=0;row<q;++row) for(uint32_t col=row;col<q;++col) jobs.push_back({row,col});
  return jobs;
}
// CPU reference deliberately accumulates one product at a time modulo q.
// Caller supplies the basis admitted by the problem and certified input proof.
inline void gram_residues_cpu(const int8_t* x,int8_t* c,const GramProblem& p,
                              const std::vector<GramTile>& jobs,const uint16_t* moduli) {
  const int64_t n=p.orientation==GramOrientation::transpose_times_self?p.cols:p.rows;
  const int64_t k=p.orientation==GramOrientation::transpose_times_self?p.rows:p.cols;
  for(uint32_t plane=0;plane<p.prefix;++plane) for(auto job:jobs)
    for(int r=0;r<16;++r) for(int t=0;t<16;++t) {
      const int64_t row=int64_t(job.row)*16+r,col=int64_t(job.col)*16+t;
      if(row>=n || col>=n || row>col) continue;
      const int modulus=p.modulus?int(p.modulus):moduli[plane];
      int sum=0;
      for(int64_t q=0;q<k;++q) {
        const auto at=[&](int64_t index) {return p.orientation==GramOrientation::transpose_times_self
          ? (int64_t(plane)*p.rows+q)*p.source_ld+index
          : (int64_t(plane)*p.rows+index)*p.source_ld+q;};
        sum=(sum+int(x[at(row)])*int(x[at(col)]))%modulus;
      }
      if(sum<0) sum+=modulus;
      if(sum>=(modulus+1)/2) sum-=modulus;
      c[(int64_t(plane)*n+row)*p.output_ld+col]=int8_t(sum);
      if(row!=col) c[(int64_t(plane)*n+col)*p.output_ld+row]=int8_t(sum);
    }
}
}
