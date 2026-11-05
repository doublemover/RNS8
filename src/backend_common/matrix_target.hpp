#pragma once

#include <string_view>

// The body guards and host admission describe the same finite implementation
// set. A loadable empty specialization must never certify arithmetic support.
#if defined(__gfx942__)
#define RNS8_BUILTIN_BODY_CDNA3 1
#else
#define RNS8_BUILTIN_BODY_CDNA3 0
#endif
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
#define RNS8_BUILTIN_BODY_RDNA3 1
#else
#define RNS8_BUILTIN_BODY_RDNA3 0
#endif
#if defined(__gfx1200__) || defined(__gfx1201__)
#define RNS8_BUILTIN_BODY_RDNA4 1
#else
#define RNS8_BUILTIN_BODY_RDNA4 0
#endif

namespace rns8::detail {

enum class BuiltinArch { unsupported, cdna3, rdna3, rdna4 };

inline std::string_view normalized_amdgpu_target(std::string_view target) noexcept {
  const auto separator = target.find(':');
  const auto base = target.substr(0, separator);
  if (separator == std::string_view::npos)
    return base;
  bool xnack = false, sramecc = false;
  auto rest = target.substr(separator + 1);
  while (!rest.empty()) {
    const auto next = rest.find(':');
    const auto feature = rest.substr(0, next);
    if (feature == "xnack+" || feature == "xnack-") {
      if (xnack)
        return {};
      xnack = true;
    } else if (feature == "sramecc+" || feature == "sramecc-") {
      if (sramecc)
        return {};
      sramecc = true;
    } else {
      return {};
    }
    if (next == std::string_view::npos)
      return base;
    rest.remove_prefix(next + 1);
  }
  return {}; // Empty/trailing feature tokens are malformed.
}

inline BuiltinArch builtin_arch(std::string_view target) noexcept {
  const auto base = normalized_amdgpu_target(target);
  if (base == "gfx942")
    return BuiltinArch::cdna3;
  if (base == "gfx1100" || base == "gfx1101" || base == "gfx1102")
    return BuiltinArch::rdna3;
  if (base == "gfx1200" || base == "gfx1201")
    return BuiltinArch::rdna4;
  return BuiltinArch::unsupported;
}

inline bool builtin_sparse_target(std::string_view target) noexcept {
  const auto arch = builtin_arch(target);
  return arch == BuiltinArch::cdna3 || arch == BuiltinArch::rdna4;
}

} // namespace rns8::detail
