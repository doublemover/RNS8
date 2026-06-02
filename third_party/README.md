# Third-Party Policy

RNS8 uses vcpkg-managed C++ dependencies declared in `vcpkg.json`.

Required Phase 0 dependencies are Catch2 for tests and Boost.Multiprecision for
exact CRT and wide-integer reference behavior. GMP and FLINT are optional
differential references on Windows and must not be required for core
correctness. hipBLASLt, CK, rocWMMA, and AMDGPU builtins remain
feature-detected accelerator paths and are not correctness requirements.

