#pragma once
#include "rns8/rns8.h"
namespace native_device_fixture {
struct Owner {unsigned acquired=0,released=0;rns8_status status=RNS8_SUCCESS;};
inline rns8_status acquire(void* p) {auto& o=*static_cast<Owner*>(p);if(o.status==RNS8_SUCCESS) ++o.acquired;return o.status;}
inline void release(void* p) {++static_cast<Owner*>(p)->released;}
inline rns8_native_device_desc descriptor(void* base,uint64_t bytes,Owner& owner) {
  rns8_native_device_desc d{};d.struct_size=sizeof(d);d.abi_version=RNS8_ABI_VERSION;
  d.element_type=RNS8_NATIVE_DEVICE_I64;d.access=RNS8_NATIVE_DEVICE_READ;d.allocation_base=base;d.allocation_bytes=bytes;
  d.byte_offset=8;d.rows=3;d.cols=5;d.ld=7;d.device_id=0;d.producer_complete=1;d.owner_identity=101;d.generation=1;
  d.policy=RNS8_NATIVE_DEVICE_BORROW;d.owner=&owner;d.acquire=acquire;d.release=release;return d;
}
}
