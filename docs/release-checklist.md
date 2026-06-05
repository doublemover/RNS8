# Release Checklist

Use this checklist before publishing a pre-1.0 release tag or release-candidate
PR.

## Metadata

- `LICENSE`, `NOTICE`, `vcpkg.json`, README, and docs identify first-party RNS8
  code as MIT.
- `CHANGELOG.md` has an unreleased entry with public API and behavior changes.
- `SECURITY.md`, `CONTRIBUTING.md`, and GitHub templates are current.
- GitHub vulnerability alerts are enabled, and private vulnerability reporting
  is enabled and verified after the repository visibility changes to public.
- Third-party submodule pins document path, branch, URL, and commit SHA when
  used, and `git submodule status --recursive` matches those SHAs.

## CPU And Package Gate

```powershell
git diff --check
python tools\test_check_dependencies.py
python tools\test_benchmark_schema.py
python tools\test_result_compare.py
python tools\test_bound_discovery_report.py
python tools\test_host_api_batch_report.py
python tools\test_many_small_grouped_report.py
python tools\test_rns_chain_report.py
python tools\test_benchmark_sweep.py
python tools\check_release_tree.py
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
cmake --install build/cpu-debug --prefix temp/install-rns8/Debug
```

The CPU CTest preset also runs the downstream CMake smoke when examples and
package export support are enabled.

## Windows HIP Gate

```powershell
python tools\check_dependencies.py
python tools\windows_dev.py cmake --build --preset windows-debug
python tools\windows_dev.py ctest --preset windows-debug --output-on-failure
build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke
```

Optional accelerator presets must be validated separately before any
accelerator-specific claim is promoted.

## Evidence Gate

- Benchmark captures are schema v4.
- Captures include command line, commit, seed, warmups, repeats, compiler,
  HIP/ROCm, device target, backend, semantic contract, prefix metadata, and
  timing source.
- Performance claims have same-contract baselines and reviewed status.
- Raw captures and ISA reports are not committed.

## Documentation Gate

- README limitations, hardware scope, and semantic table are current.
- Public status-code documentation matches `include/rns8/status.h` and
  `docs/RNS8_RESEARCH_SPEC.md`.
- Environment-variable behavior is current in the spec and platform docs.
- `docs/RNS8_RESEARCH_SPEC.md` remains the source of truth.
- `docs/performance-model.md` and `docs/performance-wins.md` distinguish
  reviewed evidence from promotion.
