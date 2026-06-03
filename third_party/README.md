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

Record pin state with `git submodule status --recursive`. The durable format is
the command's plain text output:

```text
 d9272218c4c59a58e41d3d346362cdaa707c30ce third_party/rocm/composable_kernel (rocm-7.1.1)
 1ab208f49945c38626b79e3f0c284d65ac44a781 third_party/rocm/rocWMMA (mock-tag-test-147-g1ab208f4)
```

Initialize and probe them with:

```powershell
python tools\bootstrap_rocm_accelerators.py --init --probe --target gfx1100
python tools\bootstrap_rocm_accelerators.py --dry-run --init --probe --target gfx1100 --json
```

The bootstrap command compile-probes CK and rocWMMA headers and object-compiles
RNS8-owned int8 matrix-engine primitive probes for `gfx1100`. That output is
dependency readiness evidence only; it is not backend execution, differential
correctness validation, ISA evidence, or performance evidence.

Do not clone these repositories under `C:\` or commit upstream build products.
Generated dependency artifacts belong under ignored `temp/`, `out/`, or
`build/` paths.
