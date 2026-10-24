# RNS8

RNS8 multiplies matrices of integers on AMD GPUs while preserving the requested
integer result. Each output is a sum of products, computed exactly within a
declared range or under an explicit modular-arithmetic rule. It provides a C API,
a small C++ wrapper, and intermediate values that can stay on the GPU for
repeated operations.

Exact integer products are useful in computer algebra, Gram matrices and
cross-products of integer data, batched finite-field linear transforms, and
repeated computations where rounding or fixed-width overflow would change the
answer. These are uses of the arithmetic; RNS8 supplies the matrix kernel that
an application can build around.

## Hardware tested and hardware wanted

| GPU / family | Target | OS + tested stack | Implementation availability | Actual RNS8 validation evidence | Next needed check |
|---|---|---|---|---|---|
| **RX 7900 XTX / RDNA3** | `gfx1100` | Windows, VS2026, HIP 7.1 | Public direct HIP, vector ALU, wrap64; opt-in hipBLASLt, CK, rocWMMA, builtins | [All-feature correctness at `c7f7e05`](docs/local-all-features-20261023.md); [matched 512³/1024³ measurements](docs/7900xtx-large-throughput-2026-10-23.md) | Requalify changed kernels/configurations; Linux GPU run |
| Radeon PRO W6800 / RDNA2 | `gfx1030` | No RNS8 GPU run; Linux ROCm 7.1 candidate | Direct HIP source; optional backends subject to target support | **Validation wanted / unvalidated** | Supported Linux stack, compile and CPU differentials |
| RX 9060 / 9060 XT / RDNA4 | `gfx1200` | No RNS8 GPU run; AMD lists Linux 7.1 and Windows HIP SDK 7.1.1 | Direct HIP and target-specific builtin source | **Validation wanted / unvalidated** | Exact SKU/OS/compiler check and device differentials |
| RX 9070 / 9070 XT / 9070 GRE / RDNA4 | `gfx1201` | No RNS8 GPU run; same AMD release matrices | Direct HIP and target-specific builtin source | **Validation wanted / unvalidated** | Per-device compilation, arithmetic and ISA checks |
| MI210 / MI250 / MI250X / CDNA2 | `gfx90a` | No RNS8 GPU run; Linux ROCm target | Direct HIP; optional libraries depend on target | **Validation wanted / unvalidated** | Real Instinct host and supported OS/ROCm pairing |
| MI300X / MI325X / CDNA3 | `gfx942` | No RNS8 GPU run; Linux ROCm target | Direct HIP, MFMA and optional accelerator source | **Validation wanted / unvalidated** | Compile, CPU differentials and ISA on each device |
| MI350X / MI355X / CDNA4 | `gfx950` | No RNS8 GPU run; Linux ROCm target | Direct HIP and target-dependent accelerator source | **Validation wanted / unvalidated** | Supported compiler/components and device execution |

[Offer hardware validation, temporary compute access, or cloud credits](https://github.com/doublemover/RNS8/issues/new?template=hardware_validation_offer.yml).
You can volunteer to run a bounded set of checks yourself. **Do not post passwords,
SSH keys, tokens, account credentials, private connection endpoints, or billing
data.** Offers start a coordination conversation; no access or spending is authorized.

GPU mappings follow [AMD's architecture reference](https://rocm.docs.amd.com/en/docs-7.0.0/reference/gpu-arch-specs.html)
and the release-specific [Linux ROCm 7.1](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/reference/system-requirements.html)
and [Windows HIP SDK 7.1.1](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.1.1/reference/system-requirements.html)
matrices. PRO W6800 is the Linux RDNA2 candidate; that Windows matrix excludes
RDNA2. A target name alone says nothing about a supported installation or an
RNS8 validation result. [Platform readiness](docs/platform-readiness.md) explains
the evidence classes and per-OS checks.

## Measured RX 7900 XTX results

| Shape and arithmetic/output contract | Matched baseline | Candidate | Improvement | Evidence |
|---|---|---|---|---|
| 1024³ bounded signed i64, repack + native host output | Direct HIP: 8.127 ms | Public AMDGPU builtins: 6.736 ms | 1.21× | [Public pipelines](docs/7900xtx-large-throughput-2026-10-23.md#public-bounded-i64-pipelines-and-reuse) |
| 1024³ signed byte values stored as i64, exact i64 device output | Minimal four-plane RNS pipeline: 2.783 ms | Internal checked native-byte WMMA: 1.250 ms | 2.23× | [Native output](docs/7900xtx-large-throughput-2026-10-23.md#native-bounded-integer-device-output) |
| 512³ `alpha*AB + beta*D`, nine resident residue planes | Direct GEMM + separate combine: 760.91 µs | rocWMMA fused: 459.58 µs | 1.66× | [Residue epilogue](docs/7900xtx-large-throughput-2026-10-23.md#resident-residue-epilogue) |

These October 23 results use resident inputs on the recorded Windows/HIP stack.
The public pipeline includes repacking and host export after common allocation;
its statistic is the median of three process medians. The other rows use five
process burst means with timing hooks disabled; checked calls include allocation,
admission and synchronization. Initial uploads and verification are outside timing.
The residue epilogue has no native integer export or proof of unique reconstruction
of the full expression. Compare rows only within their stated contract. The
[full report](docs/7900xtx-large-throughput-2026-10-23.md) preserves ranges, commits,
hashes, commands and exact checks; [performance methodology](docs/performance.md)
owns the remaining measurement and promotion gates.

## What it computes

| Explicit contract | Result |
|---|---|
| Bounded signed / unsigned 64-bit | Every integer bit of the result, within the caller's declared bound |
| Exact-wide signed / unsigned | A range-proven integer in residue storage, exported as fixed-width limbs |
| Finite ring / prime field | Products modulo one selected integer 2–256, or prime up to 251 |
| Strict wrap64 | The low 64 bits, with arithmetic explicitly modulo 2⁶⁴ |

Execution supports prefixes of 1–20 default moduli, providing up to about 154.84
bits of represented CRT range. Exact-wide is finite-capacity storage; exporting
1–32 limbs does not increase that range. Bounded plans check range coverage,
while callers remain responsible for the truth of supplied bounds. See
[correctness and export rules](docs/correctness.md) before choosing a contract.
All matrices use row-major layout.

## How it works

A **modulus** specifies the divisor; a **residue** is the remainder. For an
illustrative pair of moduli, 23 has residues 2 modulo 3 and 3 modulo 5. Those
remainders identify 8 modulo 15, so they need an appropriate range bound to
identify a particular integer. RNS8 uses its own larger, pairwise-coprime ladder.

1. Convert each input integer into several small residues.
2. Compute independent small-integer matrix products, using AMD integer matrix
   hardware where available. Reduce often enough to keep accumulation exact.
3. Select enough moduli to cover the proven result range.
4. **Reconstruct** the integer from its residues when the caller needs an integer
   output, or keep the residues **resident** in GPU memory for another operation.
5. Account for conversion, reconstruction and operand reuse when choosing a path.

Finite arithmetic uses its single explicit modulus. Strict wrap64 uses a separate
byte-limb implementation. The [architecture guide](docs/design.md) connects these
representations to matrix ownership, reuse and execution.

## Build and run on an AMD GPU

Required: a C++17 toolchain, a supported AMD driver and HIP/ROCm stack, CMake
3.22+, Ninja, Git, Python, Boost.Multiprecision and nlohmann-json. Catch2 3.x is
needed when building tests. Optional accelerator libraries, OpenMP and GMP/FLINT
serve selected paths. [Dependency setup](third_party/README.md) owns prerequisites,
current pins and installer links. The dependency report inspects installed components; the vcpkg toolchain may
install manifest packages during configuration.

### Windows PowerShell

Install the [Windows prerequisites](docs/platform-windows.md#prerequisites), then
set paths to your actual installations; spaces and nondefault roots are supported.
The current preset includes workstation defaults, so pass the overrides explicitly.

```powershell
$env:VCPKG_ROOT = 'D:/Development Tools/vcpkg'
$rns8HipRoot = 'D:/AMD SDK/ROCm/7.1'
python tools/check_dependencies.py
python tools/windows_dev.py cmake --preset windows-msvc-hip-debug "-DRNS8_HIP_ROOT=$rns8HipRoot" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DRNS8_AMDGPU_TARGETS=gfx1100
python tools/windows_dev.py cmake --build --preset windows-debug --parallel 4
.\build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-i64 --m 32 --n 32 --k 64 --warmups 1 --repeats 3 --seed 3135
```

### Native Linux

Install the [native Linux prerequisites](docs/platform-linux.md#prerequisites)
for your GPU and distribution. Choose the real ROCm root and target explicitly:

```sh
rns8_rocm_root=/opt/rocm
python3 tools/check_dependencies.py
cmake --preset linux-rocm-debug "-DRNS8_HIP_ROOT=$rns8_rocm_root" -DRNS8_AMDGPU_TARGETS=gfx1100
cmake --build --preset linux-debug --parallel 4
./build/linux-rocm-debug/rns8-bench --backend hip-direct --semantics bounded-i64 --m 32 --n 32 --k 64 --warmups 1 --repeats 3 --seed 3135
```

For Instinct, select `linux-cdna-debug` and the device target, such as `gfx942`;
use the matching build preset and directory. Native Linux GPU commands remain
unexecuted on this Windows host. WSL has a separate compatibility boundary in
the [Linux guide](docs/platform-linux.md#wsl). Automatic dependency-root discovery,
locked upgrades and the bounded remote-session bundle remain work tracked by
[#57](https://github.com/doublemover/RNS8/issues/57),
[#60](https://github.com/doublemover/RNS8/issues/60) and
[#58](https://github.com/doublemover/RNS8/issues/58).

## Use it

The [public API](include/rns8/rns8.h) exposes checked one-shot calls and persistent
`context → plan → matrix/workspace → pack → GEMM → export` workflows. Initialize
`struct_size` and `abi_version`, choose semantics explicitly, check returned
statuses, and release handles after use. Input versions assert unchanged values.
Ordinary calls are synchronous. Public [prepared submissions](docs/prepared-execution.md)
and [resident DAGs](docs/resident-graphs.md) additionally support explicit
fixed-prefix direct-HIP stream/capture subsets with completion tokens and retained
resource lifetimes.

Start with the [example index](examples/README.md) for complete checked API
programs and their execution boundaries. Its current examples select the CPU
reference; the GPU command above exercises the existing benchmark driver.
Practical GPU application examples are tracked by
[#62](https://github.com/doublemover/RNS8/issues/62).

Install a configured build with `cmake --install <build-dir> --prefix <install-dir>`.
Consumers use `find_package(RNS8 CONFIG REQUIRED)` and link `rns8::rns8_static`,
or `rns8::rns8` from a shared build. The installed
[consumer guide](examples/downstream-cmake/README.md) and sources live under
`share/RNS8/examples/downstream-cmake`; consumers need the installed public
package and a C++17 compiler.

## Backends

| Path | Availability / selection | Computes / constraints | Evidence |
|---|---|---|---|
| Direct HIP | Available public path; `hip-direct`, `RNS8_ENABLE_HIP` | Bounded, exact-wide, finite; separate wrap64 storage | [Local validation](docs/local-all-features-20261023.md) |
| Native vector ALU | Available public HIP path; `hip-vector-alu-int64` | Bounded i64/u64 in native storage | [Matched public calls](docs/7900xtx-large-throughput-2026-10-23.md#public-bounded-i64-pipelines-and-reuse) |
| hipBLASLt | Opt-in public path; `RNS8_ENABLE_HIPBLASLT` | Fixed-prefix bounded, wide, finite; separate INT32 reduction | [Details](src/backend_hipblaslt/README.md) |
| CK | Opt-in public path; `RNS8_ENABLE_CK` | Bounded schedules, wide, finite; headers/target constraints | [Details](src/backend_ck/README.md) |
| rocWMMA | Opt-in public path; `RNS8_ENABLE_ROCWMMA` | Bounded schedules, wide, finite; matrix-instruction support | [Details](src/backend_rocwmma/README.md) |
| AMDGPU builtins | Opt-in public path; `RNS8_ENABLE_AMDGPU_BUILTINS` | Target-specific dense RNS/finite and explicit Sparse-A | [Capabilities](docs/backend-notes.md) |
| Wrap64 byte limbs | Available public route; direct HIP or `wrap64-byte-limb` CPU context | Strict modulo 2⁶⁴ only | [Details](src/backend_wrap64/README.md) |

Optional accelerators require HIP plus their compiler/library support.
Compilation and device validation are recorded separately. `AUTO` uses the
baseline context unless an eligible exact-match reviewed cache entry selects
another backend; it performs no runtime tuning. The
[backend reference](docs/backend-notes.md) covers limits and selection.
[Research topics](docs/research/README.md) distinguish private Strassen/rank-23,
low-bit and integrity prototypes from reserved public entrypoints and broader
async/multi-GPU work.

[Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) ·
[Roadmap](docs/roadmap.md) · [Security](SECURITY.md) · [MIT license](LICENSE) ·
[Financial sponsorship](https://github.com/sponsors/doublemover)
