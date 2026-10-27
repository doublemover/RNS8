#pragma once
#include "core/api_internal.hpp"
#include <memory>
namespace rns8::detail::api {
// The resident graph owns matrix reservations and publication. This binding
// owns only the existing native lease reference and external-reader lifetime;
// scratch and pinned status are supplied by the graph's retained arena.
class ResidentNativeImport {
 public:
  virtual ~ResidentNativeImport()=default;
  virtual rns8_status validate(bool check_generation,boost::multiprecision::cpp_int& bound) const=0;
  virtual rns8_status prepare_retained_native()=0;
  virtual rns8_status before_enqueue(void* stream)=0;
  virtual rns8_status enqueue(void* scratch,uint64_t bytes,void* pinned_status,void* stream)=0;
  virtual void reserve_external() noexcept=0;
  virtual void release_external(bool success) noexcept=0;
  virtual void commit_refresh() noexcept=0;
  virtual const void* owner_key() const noexcept=0;
  virtual const void* source() const noexcept=0;
  virtual uint64_t source_bytes() const noexcept=0;
  virtual void* retained_native() const noexcept=0;
  virtual uint64_t retained_native_bytes() const noexcept=0;
  virtual uint64_t required_native_bytes() const noexcept=0;
};
rns8_status create_resident_native_import(rns8_context* ctx,const rns8_resident_node& node,
    std::unique_ptr<ResidentNativeImport>& out);
}
