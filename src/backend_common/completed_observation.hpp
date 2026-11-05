#pragma once

namespace rns8::detail {
// A failed completion observation cannot authorize timing or result evidence.
// HIP and the backend status domains both reserve zero for success.
template <class Drain, class Publish>
auto publish_after_completion(Drain&& drain, Publish&& publish) -> decltype(drain()) {
  const auto status = drain();
  if (status == decltype(status){})
    publish();
  return status;
}
} // namespace rns8::detail
