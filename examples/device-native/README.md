# Installed native HIP producer/consumer

This standalone consumer uses only installed RNS8 public headers/targets and
HIP's public runtime API. Its HIP producer initializes padded device-native A/B
and C canaries. Borrowed read leases certify A/B magnitudes on device, import
into exact-wide signed RNS matrices, run public exact GEMM, and reconstruct into
a borrowed device-native C view. The GEMM uses a fixed 20-plane prefix matching
the matrices: native-device export requires a complete basis rather than unused
capacity above a reduced-prefix result. Only after successful host-observed export
does a custom HIP consumer compare every result and padding cell.

The application path transfers no host payload: two eight-byte magnitude
summaries and one four-byte CRT status are read back. The consumer additionally
reads a four-byte failure flag. A separate 104-byte full-output readback checks
an independent host scalar reference, whose partial sums are bounded by 75.
It is validation traffic, not an omitted pipeline dependency. No performance
or observed memory-peak claim follows from this tiny integration example.

Build the main RNS8 project with direct HIP and install its development package.
Configure this installed example directory with a supported C++ toolchain:

```powershell
cmake -S <prefix>/share/RNS8/examples/device-native -B build/native-consumer -G Ninja -DCMAKE_PREFIX_PATH=<prefix> -DRNS8_NATIVE_HIP_ROOT="C:/Program Files/AMD/ROCm/7.1" -DRNS8_NATIVE_GPU_TARGET=gfx1100
cmake --build build/native-consumer
ctest --test-dir build/native-consumer --output-on-failure
```

Shared linkage is the default; `RNS8_NATIVE_USE_SHARED=OFF` selects the exported
static target. Put the installed RNS8 and HIP runtime DLL directories on PATH
when required. Match the library's build configuration/runtime. Linux uses the
same explicit HIP compiler invocation with its actual supported target. This
example does not call CMake enable_language(HIP). If the active HIP/toolchain
combination requires the project's STL compatibility overlay, pass its include
flag through the CMake list `RNS8_NATIVE_HIP_EXTRA_FLAGS`; no SDK header is edited.

The process keeps owner callbacks alive for each lease, releases references
exactly once after real completion, and frees external allocations only after
all leases are destroyed. On a failure it exits without attempting speculative
free/release of potentially in-flight resources. A long-running application
must retain and retry quarantined owners through the public destruction rules.
Producer readiness is explicitly synchronized before leases are created; this
is not a prepared asynchronous/graph integration demonstration.

The installed static and shared variants are checked with the public HIP
producer/consumer test, including the full host oracle and padding canaries.


## Prepared graph pipeline

Pass `--prepared` to run an additional replay after the explicitly synchronous
setup. The example retains two verified-bound import owners, GEMM and native
export, captures all four bodies and uses five nondefault streams. A custom
external producer ends three generation-2 mutation windows with real readiness
events. Import graphs wait on those current events outside capture; GEMM waits
on both import tokens; export waits on GEMM and its external destination's
readiness. One final host observation checks the complete dependency chain before
the external consumer runs. Three four-byte status words are read back in this
replay; setup retains its separately reported twenty summary bytes. No host
payload transfer enters either application path. The independent final output
readback remains validation-only.

The [October 25 local validation](../../docs/local-validation-20261025.md)
compiled this installed example with HIP 7.1 / VS2026 and passed both modes
with static and shared linkage on the RX 7900 XTX (`gfx1100`). The record binds
the result to its installed library artifact. This tiny example checks
lifecycle and ordering; matched performance and other targets remain separate.

To exercise the optional tensor boundary, supply the official DLPack1.2 header
directory as `-DRNS8_DLPACK_INCLUDE_DIR=...` and run `--dlpack` or
`--dlpack-prepared`. The same GPU producer, padded inputs, exact product and
GPU consumer then use `DLManagedTensorVersioned` leases. Each tensor deleter
must run once after all prepared owners retire. These modes retain the complete
independent host oracle and require no Python framework. Existing native-only
execution receipts do not establish these new modes' qualification.

On Windows, CTest binds its process to the SDK selected for the producer's
compiler and headers, including when the calling shell still names an older
HIP installation. This changes no user or system environment variables.
