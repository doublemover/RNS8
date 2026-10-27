#ifndef RNS8_CORE_PREPARED_TRANSFER_HPP
#define RNS8_CORE_PREPARED_TRANSFER_HPP
#include "core/api_internal.hpp"
#include "core/hip_resources.hpp"
#include <memory>
namespace rns8::detail::api {
// A transfer is a variant of the existing prepared execution, not a second
// completion/token system. All fallible proof/storage work precedes reserve().
class PreparedTransfer {
 public:
  rns8_context* ctx=nullptr;
  rns8_matrix* matrix=nullptr;
  virtual const rns8_matrix* input_a() const noexcept {return matrix;}
  virtual const rns8_matrix* input_b() const noexcept {return matrix;}
  virtual ~PreparedTransfer()=default;
  virtual bool writes_matrix() const noexcept=0;
  virtual bool reads_matrix() const noexcept=0;
  virtual bool produces_resident() const noexcept=0;
  virtual uint32_t output_prefix() const noexcept=0;
  virtual rns8_status validate(const boost::multiprecision::cpp_int* pending_bound,
      uint32_t pending_prefix,boost::multiprecision::cpp_int& output_bound)=0;
  virtual rns8_status before_enqueue(void*) {return RNS8_SUCCESS;}
  virtual rns8_status enqueue(void* stream)=0;
  virtual void reserve() noexcept=0;
  virtual rns8_status finish(rns8_status operation,boost::multiprecision::cpp_int& bound) noexcept=0;
  virtual bool output_current() const noexcept=0;
  virtual rns8_status close()=0;
  virtual const char* operation_name() const noexcept=0;
  virtual void* sync_stream() const noexcept=0;
  virtual rns8_status refresh() {boost::multiprecision::cpp_int bound;return validate(nullptr,0,bound);}
  virtual void commit_refresh() noexcept {}
  virtual void describe(rns8_prepared_transfer_info&) const noexcept {}
  struct ExternalAccess {const void* pointer=nullptr;uint64_t bytes=0;bool writer=false;const void* reservation_owner=nullptr;};
  virtual ExternalAccess external_access() const noexcept {return {};}

};
rns8_status create_transfer_prepared(std::unique_ptr<PreparedTransfer> transfer,rns8_prepared_execution** out);
}
#endif
