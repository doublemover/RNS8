# CPU fuzz discovery and arithmetic corpus

`RNS8_BUILD_FUZZERS=ON` creates `rns8_fuzz_library`, a separate static copy of
the configured first-party CPU sources. Its objects receive libFuzzer feedback
and ASan; native Unix Clang also receives UBSan with nonrecovering checks.
Fuzz entrypoints alone link the fuzzer runtime. Ordinary libraries and installed
packages do not acquire fuzz instrumentation. HIP must be OFF in this build;
GPU replay is separate from discovery. Windows uses the existing
`windows-clang-asan-debug` preset and static CRT. The configured source inventory
is `rns8-fuzz-instrumentation-<CONFIG>.json`; this is not execution evidence.

The instrumentation control authors a real branch-containing library object,
checks actual compile commands and inspects the archive with the matching
`llvm-nm`. Missing native tools produce an explicit unavailable control.
Third-party dependencies are outside the library instrumentation claim.
See LLVM's [separate fuzzer instrumentation](https://llvm.org/docs/LibFuzzer.html#fuzzer-usage).

`rns8_fuzz_export_contract` consumes arithmetic-case-v1 bytes. Its primary
oracle uses scalar Boost integers without production reduction, CRT, prefix
selection or cached results. Finite results use mathematical modulo q;
wrap64 uses modulo 2^64. Signed/unsigned bounded values must fit their declared
destination and prefix. Wide cases must fit a conservative input/accumulation
proof within their explicit prefix; three exported limbs include required
sign extension. Out-of-envelope inputs are not counted as valid arithmetic
executions. Wrong successful values, unexpected statuses, modified padding,
invalid-LD writes and stale prepared execution cause a crash/reproducer.

The binary format is little-endian and contains no pointers or implicit RNG:

| Bytes | Contract |
|---|---|
| 0–3 | ASCII `R8F1` |
| 4 | 0 bounded i64; 1 bounded u64; 2 finite ring u8; 3 wrap64; 4 signed wide; 5 unsigned wide |
| 5 | Bits: 1 repeat GEMM; 2 invalid-LD/no-write control; 4 prepared/repack/stale rejection/refresh. Prepared bit is RNS-only. |
| 6–7 | Literal M and N, each 1–8 |
| 8–11 | Literal K, 1–65537 |
| 12–14 | A/B/output row padding, each 0–7; layout is explicitly row-major |
| 15 | RNS prefix 1–20; zero for finite/wrap |
| 16–17 | Finite modulus 2–256; zero for other semantics |
| 18–23 | Reserved zero bytes |
| 24 onward | M×K A values then K×N B values, each uint64 little-endian; signed domains use two's-complement bits; finite values must fit u8 |

The decoder checks the full byte count, 262144 input-value/operation budgets
and semantic fields before allocation. Maximum input is 2097176 bytes.
Source buffers and output use minimal final-row extents plus guards.

`python tools/fuzz_cases.py --out temp/fuzz-corpus/arithmetic` expands named
seeds for all six domains, every supported prefix, every finite byte modulus,
native boundaries, cancellation and padded rectangular storage. Add `--large-k`
for skinny K=65535/65536/65537 cases. Raw corpus files remain ignored; a sibling
manifest records names, sizes and hashes without becoming a fuzzer input.
The Python integer oracle and codec are independent cross-language controls.

Implementation is in progress under #101. These new controls and harness have
not run in the deferred validation batch. Production-library instrumentation
inspection, bounded discovery/replay CI, structured failure reports/shrinking,
stronger descriptor-state controls, production Python metadata fuzzing and
explicit GPU/backend replay remain open. Neither source registration nor a
compiled harness establishes exercised library branches or GPU coverage.
