# Third-Party Policy

RNS8 uses vcpkg-managed C++ dependencies declared in `vcpkg.json`.

Required Phase 0 dependencies are Catch2 for tests and Boost.Multiprecision for
exact CRT and wide-integer reference behavior. GMP and FLINT are optional
differential references on Windows and must not be required for core
correctness. hipBLASLt, CK, rocWMMA, and AMDGPU builtins remain
feature-detected accelerator paths and are not correctness requirements.

CK and rocWMMA source is repo-local only through pinned Git submodules:

- `third_party/rocm/composable_kernel`
- `third_party/rocm/rocWMMA`

Initialize and probe them with:

```powershell
python tools\bootstrap_rocm_accelerators.py --init --probe --target gfx1100
```

Do not clone these repositories under `C:\` or commit upstream build products.
Generated dependency artifacts belong under ignored `temp/`, `out/`, or
`build/` paths.
