#pragma once
#include "core/internal.hpp"

namespace rns8::detail::api {
// A failed GPU drain deliberately retains the read reference with the owner.
// Ordinary returns and CPU exceptions release it without fallible cleanup.
class PreparedExportReadLease {
 public:
  PreparedExportReadLease(const rns8_matrix& source,const bool& pending) noexcept
      :source_(source),pending_(pending) {++source_.in_flight_readers;}
  PreparedExportReadLease(const PreparedExportReadLease&)=delete;
  PreparedExportReadLease& operator=(const PreparedExportReadLease&)=delete;
  ~PreparedExportReadLease() {if(!pending_)--source_.in_flight_readers;}
 private:
  const rns8_matrix& source_;
  const bool& pending_;
};
} // namespace rns8::detail::api
