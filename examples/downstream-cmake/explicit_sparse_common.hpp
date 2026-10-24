#pragma once
#include <rns8/rns8.h>
#include <rns8/moduli.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "noninteractive_errors.hpp"

namespace sparse_example {
using std::size_t;
inline bool cleanup_failed = false;
inline void require(bool condition,const char* message) {
  if(!condition) throw std::runtime_error(message);
}
inline void check(rns8_status status,const char* operation) {
  if(status!=RNS8_SUCCESS) throw std::runtime_error(std::string(operation)+": "+rns8_status_string(status));
}
template<class T,rns8_status(*Destroy)(T*)> struct Deleter {
  void operator()(T* value) const noexcept {
    if(value) {
      const auto status=Destroy(value);
      if(status!=RNS8_SUCCESS) {
        cleanup_failed=true;
        std::cerr<<"CPU example cleanup: "<<rns8_status_string(status)<<'\n';
      }
    }
  }
};
template<class T,rns8_status(*Destroy)(T*)> using Owner=std::unique_ptr<T,Deleter<T,Destroy>>;
using Context=Owner<rns8_context,rns8_destroy_context>;
using Plan=Owner<rns8_plan,rns8_destroy_plan>;
using Workspace=Owner<rns8_workspace,rns8_destroy_workspace>;
using Matrix=Owner<rns8_matrix,rns8_destroy_matrix>;
using Sparse=Owner<rns8_sparse_matrix,rns8_destroy_sparse_matrix>;
inline Context context() {
  rns8_context_options options{};
  options.struct_size=sizeof(options); options.abi_version=RNS8_ABI_VERSION;
  options.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
  rns8_context* result=nullptr;
  check(rns8_create_context(-1,&options,&result),"create CPU context");
  return Context(result);
}
inline Plan plan(rns8_context* ctx,rns8_semantics semantics,rns8_bound_kind bound_kind,
                 int64_t m,int64_t n,int64_t k,uint64_t bound,uint16_t modulus=0) {
  rns8_gemm_desc desc{};
  desc.struct_size=sizeof(desc); desc.abi_version=RNS8_ABI_VERSION;
  desc.semantics=semantics; desc.bound_kind=bound_kind; desc.bound=bound;
  desc.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
  desc.m=m; desc.n=n; desc.k=k; desc.finite_modulus=modulus;
  rns8_plan* result=nullptr;
  check(rns8_create_plan(ctx,&desc,&result),"create sparse-compatible plan");
  return Plan(result);
}
inline Workspace workspace(rns8_context* ctx,const rns8_plan* selected) {
  rns8_workspace* result=nullptr;
  check(rns8_create_workspace(ctx,selected,&result),"create workspace");
  return Workspace(result);
}
inline Matrix matrix(rns8_context* ctx,int64_t rows,int64_t cols,rns8_semantics semantics,
                     rns8_bound_kind bound_kind,uint32_t prefix) {
  rns8_matrix_desc desc{};
  desc.struct_size=sizeof(desc); desc.abi_version=RNS8_ABI_VERSION;
  desc.rows=rows; desc.cols=cols; desc.logical_ld=cols; desc.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
  desc.semantics=semantics; desc.bound_kind=bound_kind; desc.max_prefix=prefix;
  rns8_matrix* result=nullptr;
  check(rns8_create_matrix(ctx,&desc,&result),"create dense matrix");
  return Matrix(result);
}
inline rns8_sparse_matrix_desc descriptor(int64_t rows,int64_t k,rns8_semantics semantics,
    rns8_bound_kind bound_kind,uint32_t prefix,uint16_t modulus=0) {
  rns8_sparse_matrix_desc desc{};
  desc.struct_size=sizeof(desc); desc.abi_version=RNS8_ABI_VERSION;
  desc.semantics=semantics; desc.bound_kind=bound_kind;
  desc.contract=RNS8_SPARSE_A_4_TO_2_STRUCTURED_K; desc.sparse_operand=RNS8_SPARSE_OPERAND_A;
  desc.index_layout=RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness=prefix?RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8:RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8;
  desc.rows=rows; desc.expanded_k=k; desc.group_size=4; desc.nonzeros_per_group=2;
  desc.max_prefix=prefix; desc.finite_modulus=modulus;
  return desc;
}
inline Sparse sparse(rns8_context* ctx,const rns8_sparse_matrix_desc& desc) {
  rns8_sparse_matrix* result=nullptr;
  check(rns8_create_sparse_matrix(ctx,&desc,&result),"create explicit sparse A");
  return Sparse(result);
}
inline rns8_sparse_matrix_storage_info storage(const rns8_sparse_matrix* value) {
  rns8_sparse_matrix_storage_info info{};
  info.struct_size=sizeof(info); info.abi_version=RNS8_ABI_VERSION;
  check(rns8_get_sparse_matrix_storage_info(value,&info),"inspect sparse storage");
  return info;
}
inline constexpr unsigned pairs[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

// Independent scalar compression oracle: plane, row, ascending K group/lane.
// The public expansion helper also imports externally supplied packed arrays;
// there is currently no public direct packed-array import into a resident handle.
inline std::vector<uint8_t> compress_import(const rns8_sparse_matrix_desc& desc,
    const std::vector<uint8_t>& dense,int64_t ld) {
  uint64_t groups=0,values=0;
  check(rns8_sparse_a_4_to_2_layout_counts(&desc,&groups,&values),"query explicit sparse counts");
  const uint32_t planes=desc.max_prefix?desc.max_prefix:1;
  require(groups==uint64_t(planes)*uint64_t(desc.rows)*uint64_t(desc.expanded_k/4) && values==2*groups,
          "unexpected explicit per-plane layout counts");
  auto tail=desc; --tail.expanded_k;
  uint64_t untouched_groups=99,untouched_values=99;
  require(rns8_sparse_a_4_to_2_layout_counts(&tail,&untouched_groups,&untouched_values)==RNS8_INVALID_ARGUMENT &&
          untouched_groups==99 && untouched_values==99,"non-four-aligned K changed layout-count outputs");
  std::vector<uint8_t> expected_values,expected_indices;
  for(uint32_t p=0;p<planes;++p) for(int64_t row=0;row<desc.rows;++row)
    for(int64_t group=0;group<desc.expanded_k/4;++group) {
      uint8_t indices[2]{}; unsigned count=0;
      for(uint8_t lane=0;lane<4;++lane) {
        const auto byte=dense[(uint64_t(p)*uint64_t(desc.rows)+uint64_t(row))*uint64_t(ld)+uint64_t(group*4)+lane];
        if(byte) {
          require(count<2,"source is not exactly 2:4 in every plane");
          indices[count++]=lane; expected_values.push_back(byte);
        }
      }
      require(count==2,"source has fewer than two nonzeros in a residue group");
      expected_indices.push_back(uint8_t(indices[0]|(indices[1]<<2)));
    }
  std::vector<uint8_t> packed_values(static_cast<size_t>(values),0xCD),packed_indices(static_cast<size_t>(groups),0xCD);
  check(rns8_pack_sparse_a_4_to_2_u8(&desc,dense.data(),ld,packed_values.data(),packed_indices.data()),"compress explicit planes");
  require(packed_values==expected_values && packed_indices==expected_indices,"compressed bytes differ from independent lane oracle");
  std::vector<uint8_t> imported(dense.size(),0xA5);
  check(rns8_expand_sparse_a_4_to_2_u8(&desc,expected_values.data(),expected_indices.data(),imported.data(),ld),"import packed arrays through expansion");
  require(imported==dense,"packed import differs from dense planes or changed padding");
  for(unsigned count:{0u,1u,3u,4u}) {
    auto invalid=dense;
    for(unsigned lane=0;lane<4;++lane) invalid[lane]=lane<count?uint8_t(lane+1):uint8_t(0);
    require(rns8_pack_sparse_a_4_to_2_u8(&desc,invalid.data(),ld,packed_values.data(),packed_indices.data())==RNS8_INVALID_ARGUMENT,
            "invalid V1 nonzero count was accepted");
  }
  for(uint8_t invalid:{uint8_t(0),uint8_t(1),uint8_t(0xF4)}) {
    auto indices=expected_indices; indices[0]=invalid;
    std::vector<uint8_t> untouched(dense.size(),0xA5);
    require(rns8_expand_sparse_a_4_to_2_u8(&desc,expected_values.data(),indices.data(),untouched.data(),ld)==RNS8_INVALID_ARGUMENT,
            "duplicate/reversed/reserved sparse index was accepted");
    require(std::all_of(untouched.begin(),untouched.end(),[](uint8_t x){return x==0xA5;}),"invalid first group changed import output");
  }
  return imported;
}
inline void verify_resident_rollback(rns8_context* ctx,rns8_sparse_matrix* a,
    const std::vector<uint8_t>& original,int64_t ld) {
  const auto before=storage(a);
  auto invalid=original;
  // Put a third nonzero in the final group, after earlier groups could be packed.
  const size_t logical_end=(size_t(before.max_prefix?before.max_prefix:1)*size_t(before.rows)-1)*size_t(ld)+size_t(before.expanded_k);
  for(size_t i=logical_end-4;i<logical_end;++i) invalid[i]=1;
  require(rns8_pack_sparse_a_4_to_2_matrix_u8(ctx,a,invalid.data(),ld,before.source_version+1)==RNS8_INVALID_ARGUMENT,
          "invalid resident replacement was accepted");
  const auto after=storage(a);
  require(after.source_version==before.source_version && after.matrix_instance_id==before.matrix_instance_id &&
          after.host_current==before.host_current && std::string(after.cache_key)==before.cache_key,
          "invalid resident replacement changed identity/currentness");
  std::vector<uint8_t> expanded(original.size(),0xA5);
  check(rns8_expand_sparse_a_4_to_2_matrix_u8(a,expanded.data(),ld),"expand preserved resident A");
  require(expanded==original,"invalid resident replacement changed stored values");
}
template<class T> inline void print_result(const char* label,const std::vector<T>& values,size_t rows,size_t cols,size_t ld) {
  uint64_t checksum=14695981039346656037ull;
  std::cout<<label<<" route=cpu_reference_sparse_expansion (no GPU acceleration)\n";
  for(size_t row=0;row<rows;++row) {
    for(size_t col=0;col<cols;++col) {
      const uint64_t bits=static_cast<uint64_t>(values[row*ld+col]);
      std::cout<<static_cast<int64_t>(values[row*ld+col])<<(col+1==cols?'\n':' ');
      for(unsigned byte=0;byte<8;++byte) {checksum^=(bits>>(8*byte))&255;checksum*=1099511628211ull;}
    }
  }
  std::cout<<"FNV1a64_of_logical_values="<<std::hex<<checksum<<std::dec<<'\n';
}
} // namespace sparse_example
