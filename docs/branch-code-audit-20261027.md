# Branch code audit: October 27, 2026

The audit starts at `4ca143d` on `codex/runtime-cache-transactions-20261011`.
It covers the whole first-party source inventory and concentrates manual review
on exact arithmetic, address admission, resource ownership, asynchronous
publication, graph capture, backend dispatch, evidence tooling and build wiring.
The inventory is approximately 1,100 source files / 243,000 lines, including
tests and benchmark tools. Inventory and static scans are whole-tree work;
manual review is risk-directed, not a claim that every line is proven correct.

## Findings and repairs

| Priority | Finding | Repair and regression coverage |
| --- | --- | --- |
| High | Event/stream wrappers and the duplicated CK/rocWMMA timers discarded handles when HIP destruction failed. Move replacement could lose the old resource too. | Shared typed owners preallocate intrusive retry records before acquisition, retain failed explicit destruction and queue failed destructor cleanup on the original device. CPU fault-injection cases cover reset, replacement, partial acquisition and device identity. |
| High | Prepared graph recapture could overwrite an old executable after failed destruction; temporary graph/executable cleanup ignored errors in several capture paths. | Prepared, resident, native-group and caller-owned graph replacements use typed temporary owners. Old handles are cleared only after their checked destruction succeeds; failed temporary cleanup stays retryable. |
| Medium | A host exception between owned stream capture begin/end could leave the stream capturing. | A capture scope ends capture during unwinding without allocating. A real HIP regression throws after recording a memset, then instantiates/replays the retained graph and checks every output byte. |
| Performance | Builtin arbitrary-byte-modulus output reduction calculated a reciprocal in device code per output; sparse finite input centering also used runtime remainder. | A constexpr constant-memory reciprocal table covers q=2..256. Fixed-ladder specializations retain their existing route. CPU endpoint/reference tests cover every byte modulus; actual accelerator differential coverage adds q=3 beside q=9. No timing gain is claimed. |
| Low | Sweep command/report annotations referred to unimported names; postponed annotations hid the error until reflective inspection. | Import the actual types and resolve every locally defined function's annotations in the sweep self-test. |
| Low | Current backend documentation named cache implementation revision 2 while code used 7. | Reconcile documentation and advance code/tests to revision 8 so earlier measured routing is not reused across these implementation changes. |

The overlap planner and collector charge the eleven new stream/event retry
records as known ordinary-host payload: 520 bytes of control storage and a
2,568-byte minimum host payload on the supported 64-bit ABI. Driver-internal
allocation remains outside that payload model. Correctness-only completion
events disable timing; timing pairs keep elapsed-time support.

## Shared helpers and comments

- `src/core/hip_handle_owner.hpp` owns checked replacement and retry identity;
  `hip_runtime_handles.hpp` shares runtime event/stream/graph ownership and the
  existing accelerator timing retry policy. CMake tracks these headers for both
  custom-compiled accelerator objects.
- `src/backend_common/checked_arithmetic.hpp` replaces duplicated checked u64
  size addition/multiplication in integer splitting and distributed planning.
  Aliasing and unchanged-on-overflow behavior have explicit unit coverage.
- `tools/report_inputs.py` replaces six copies of strict capture loading and
  the identical directory expansion in two reports. Report-specific filtering,
  timing aggregation and admission policies remain explicit at their callers.
- The accelerator reducer delegates to the common exact reducer rather than
  carrying an identical implementation. Comments explain retry allocation,
  capture unwinding, graph publication and reciprocal exactness where those
  invariants would otherwise be easy to miss.

## Review and validation boundaries

Review included native and residue currentness, reader/writer reservations,
partial enqueue/drain behavior, graph replacement, signed/unsigned byte
semantics, full-width integer splitting, CRT/range admission, checked address
arithmetic, optional backend discovery and report evidence admission. Existing
CPU/GPU differential cases remain the arithmetic oracle; the audit does not
weaken exactness or infer wraparound semantics from odd-modulus CRT.

Targeted Python checks cover undefined names, mutable defaults and delayed-loop
binding. Dynamic test-fragment namespaces produce many whole-tree undefined-name
reports, and immediate-use closures need manual triage. LLVM 23 clang-tidy
core/C++ analyzer and selected bug-prone checks cover 17 distinct CPU-configured
translation units, including their headers. The pinned-host null report assumes
an unpublished private owner has `published=true`; construction/publication
does not admit that state. Boost template diagnostics were inspected separately
and did not establish a reachable project defect. A CPU compilation database
does not analyze the active HIP-only branches.

Changed host ownership paths and all three changed HIP translation units pass
syntax compilation with the local SDK. Consolidated CPU-ASan, HIP correctness,
tooling and installed-consumer results will be recorded after the source commit.
Prior stopped residue-dot campaigns remain stopped; the unchanged symbolic
checker retains its separately identified result. No new throughput results,
AUTO promotion or transfer of frozen gains to the repaired source is implied.

## Opportunities requiring evidence

The new retry metadata adds host allocations during event creation. Pooling
timing events could amortize that cost, but needs bounded lifetime/device
ownership and clean measured comparisons before adoption. Timing instrumentation
is opt-in; ordinary untimed launches do not allocate timing pairs.

Large sweep/report modules contain more similar code, but different cohort,
median, exclusion and promotion rules are meaningful. Further extraction should
start with explicit policy contracts rather than merging superficially identical
names. Matrix-engine tile choices, launch fusion and alternative CRT schedules
need profiler evidence; no broad kernel rewrite follows from source similarity.

Linux ROCm, other Radeon/Instinct targets, genuine multi-GPU/network transports,
and unsupported CPU instruction families still require their qualification
hosts. The audit does not turn source readiness into hardware qualification.
