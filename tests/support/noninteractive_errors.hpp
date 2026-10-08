#ifndef RNS8_TESTS_NONINTERACTIVE_ERRORS_HPP
#define RNS8_TESTS_NONINTERACTIVE_ERRORS_HPP

#include <cstdio>
#include <initializer_list>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <crtdbg.h>
#include <windows.h>

#include <cstdlib>
#endif

namespace rns8::test {
inline void configure_noninteractive_errors() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
#if defined(_WIN32)
  // Process-local settings: keep assertions active and report failures through
  // stderr. Automated tests must not open dialogs, debuggers, or WER UI.
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
  _set_error_mode(_OUT_TO_STDERR);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#if defined(_DEBUG)
  for (int kind : {_CRT_WARN, _CRT_ERROR, _CRT_ASSERT}) {
    _CrtSetReportMode(kind, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(kind, _CRTDBG_FILE_STDERR);
  }
#endif
#endif
}
}  // namespace rns8::test

#endif
