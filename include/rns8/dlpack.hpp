#ifndef RNS8_DLPACK_HPP
#define RNS8_DLPACK_HPP
#include "rns8/rns8.h"
#include <dlpack/dlpack.h>
#include <atomic>
#include <cstdint>
#include <new>

// Optional DLPack 1.2 C++ consumer. Include this header only when the official
// dlpack/dlpack.h is supplied by the application; the core has no DLPack dependency.
static_assert(DLPACK_MAJOR_VERSION == 1 && DLPACK_MINOR_VERSION == 2,
              "RNS8's DLPack consumer requires the pinned 1.2 header");
namespace rns8::dlpack {
struct Binding {
  void* allocation_base = nullptr;
  uint64_t allocation_bytes = 0, owner_identity = 0, generation = 0;
  rns8_native_device_access access = RNS8_NATIVE_DEVICE_READ;
  // The producer must honor the negotiated DLPack stream before this is true.
  // Neither pointer import nor this adapter synchronizes an external producer.
  bool producer_complete = false;
};
namespace detail {
struct Owner {
  std::atomic<uint64_t> references{1};
  DLManagedTensorVersioned* tensor = nullptr;
  bool consumed = false;
  static rns8_status acquire(void* value) noexcept {
    auto& count = static_cast<Owner*>(value)->references;
    auto n = count.load(std::memory_order_relaxed);
    while(n != UINT64_MAX) {
      if(count.compare_exchange_weak(n,n+1,std::memory_order_relaxed))return RNS8_SUCCESS;
    }
    return RNS8_RANGE_ERROR;
  }
  static void release(void* value) noexcept {
    auto* owner = static_cast<Owner*>(value);
    if(owner->references.fetch_sub(1,std::memory_order_acq_rel) == 1) {
      if(owner->consumed && owner->tensor->deleter)owner->tensor->deleter(owner->tensor);
      delete owner;
    }
  }
};
}
// Pure descriptor validation, including the allocation span not encoded by
// DLPack itself. The caller supplies the real allocation base/size, never an
// invented byte span starting at a possibly interior tensor data pointer.
inline rns8_status describe(const DLManagedTensorVersioned& managed,const Binding& b,
                           rns8_native_device_desc& output) noexcept {
  const auto& t = managed.dl_tensor;
  if(managed.version.major != 1 || managed.version.minor > 2 ||
      (managed.flags & ~(uint64_t(DLPACK_FLAG_BITMASK_READ_ONLY)|DLPACK_FLAG_BITMASK_IS_COPIED)) ||
      (b.access == RNS8_NATIVE_DEVICE_WRITE && (managed.flags&DLPACK_FLAG_BITMASK_READ_ONLY)) ||
      t.device.device_type != kDLROCM || t.device.device_id < 0 || t.ndim != 2 || !t.shape ||
      t.shape[0] <= 0 || t.shape[1] <= 0 || !t.data || !b.allocation_base ||
      t.dtype.bits != 64 || t.dtype.lanes != 1 || (t.dtype.code != kDLInt && t.dtype.code != kDLUInt) ||
      !b.producer_complete || !b.owner_identity || !b.generation || b.generation == UINT64_MAX ||
      (b.access != RNS8_NATIVE_DEVICE_READ && b.access != RNS8_NATIVE_DEVICE_WRITE))return RNS8_INVALID_ARGUMENT;
  const auto base = reinterpret_cast<uintptr_t>(b.allocation_base);
  const auto data = reinterpret_cast<uintptr_t>(t.data);
  if(data < base || data-base > b.allocation_bytes || t.byte_offset > b.allocation_bytes-(data-base))
    return RNS8_INVALID_ARGUMENT;
  const auto ld = t.strides ? t.strides[0] : t.shape[1];
  if(t.strides && t.strides[1] != 1)return RNS8_INVALID_ARGUMENT;
  rns8_native_device_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
  d.element_type=t.dtype.code==kDLInt?RNS8_NATIVE_DEVICE_I64:RNS8_NATIVE_DEVICE_U64;
  d.access=b.access;d.allocation_base=b.allocation_base;d.allocation_bytes=b.allocation_bytes;
  d.byte_offset=(data-base)+t.byte_offset;d.rows=t.shape[0];d.cols=t.shape[1];d.ld=ld;
  d.device_id=t.device.device_id;d.producer_complete=1;d.policy=RNS8_NATIVE_DEVICE_BORROW;
  d.owner_identity=b.owner_identity;d.generation=b.generation;
  d.acquire=detail::Owner::acquire;d.release=detail::Owner::release;
  rns8_native_device_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
  const auto status=rns8_query_native_device(&d,&resources);
  if(status==RNS8_SUCCESS)output=d;
  return status;
}
// Consumes the managed tensor exactly on SUCCESS. Afterwards its shape, strides,
// data and manager remain owned by the native lease and all prepared borrowers.
// Failed creation leaves the managed tensor with the caller. No payload copy.
inline rns8_status consume(rns8_context* ctx,DLManagedTensorVersioned* tensor,const Binding& binding,
                          rns8_native_device_lease** output) noexcept {
  if(!output)return RNS8_INVALID_ARGUMENT;*output=nullptr;
  if(!ctx || !tensor)return RNS8_INVALID_ARGUMENT;
  rns8_native_device_desc desc{};const auto status=describe(*tensor,binding,desc);
  if(status!=RNS8_SUCCESS)return status;
  auto* owner=new(std::nothrow) detail::Owner;
  if(!owner)return RNS8_INTERNAL_ERROR;
  owner->tensor=tensor;desc.owner=owner;
  const auto created=rns8_create_native_device_lease(ctx,&desc,output);
  owner->consumed=created==RNS8_SUCCESS;
  detail::Owner::release(owner);
  return created;
}
}
#endif
