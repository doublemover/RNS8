# Contributing

RNS8 is an exact-arithmetic GPU library. Correctness, explicit semantics, and
measured hardware evidence matter more than broad claims.

## Development Expectations

- Keep semantic contracts explicit. Do not infer exactness, signedness,
  wraparound, or finite-ring behavior from a C++ type alone.
- Add CPU reference coverage before a GPU path counts as implemented.
- Keep unsupported accelerators fail-fast and clearly reported.
- Put scratch captures, smoke binaries, downloaded installers, and raw evidence
  under ignored `temp/`, `build/`, or `out/` paths.
- Use focused PRs with validation output in the description.

## Local Checks

For CPU-only work:

```powershell
python tools\test_check_dependencies.py
python tools\test_benchmark_schema.py
python tools\test_result_compare.py
python tools\test_benchmark_sweep.py
python tools\check_release_tree.py
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

For Windows HIP work, also run the Windows preset and smoke test documented in
`README.md` and `docs/platform-windows.md`.

## Pull Requests

- State the semantic behavior changed or validated.
- List the exact commands run.
- Call out any hardware or dependency that was unavailable.
- Do not commit raw benchmark captures or generated ISA dumps.
- Update docs and tests in the same PR when public APIs, emitted JSON, or
  backend names change.
