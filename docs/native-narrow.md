# Proof-admitted native narrow candidates

The private direct-HIP signed-i64 candidate scans actual device magnitudes and
admits only `K*max_abs(A)*max_abs(B) <= INT64_MAX`. The common host admission
helper checks division before multiplication, including INT64_MIN magnitudes,
zero operands and UINT64_MAX inputs. The envelope bounds every intermediate
partial, including a tree reduction; cancellation in the final output alone
does not establish safety.

The private raw-byte WMMA candidate uses native i64/u64 storage with explicit
signedness. Its device scan requires every signed operand in [-128,127], or
every unsigned operand in [0,255], and an absolute product-sum envelope no larger
than INT32_MAX. It reads only logical elements; stride padding is excluded.
Unsupported device instructions, domain violations and excessive accumulation
reject before output writes. Even multiplying by zero does not waive the byte
domain requirement. Every call rescans inputs, so reusing an address cannot
reuse a stale magnitude proof. Callers must keep inputs stable during the call.

One RDNA3 wave computes a 16x16 tile using the actual signed/unsigned byte WMMA
instruction, with saturation disabled. Masked native loads are packed directly
into byte fragments. There is no residue conversion, CRT, packed-input global
buffer, INT32 output buffer or modular reduction. The exact INT32 accumulator
is widened directly into the requested i64/u64 output. The launch supports
gfx1100/1101/1102 only after a device-compiled capability check. The unsigned
route deliberately uses the positive INT32 cap rather than assuming wrapping
unsigned accumulator behavior.

These are explicit research candidates. They do not change public dispatch or
cache admission and do not replace full-width/RNS fallbacks. Public generation
binding, planner integration, epilogues, grouped specialization and measured
one-shot/resident break-even remain open under #53. Output transaction owners
must invalidate a destination if a launched backend operation fails.

Authored CPU/Boost and HIP fixtures cover partial-sum cap boundaries, signed and
unsigned byte endpoints, invalid full-width values, zero final products with
invalid inputs, 16-wide tile tails, padded strides, long K and changed inputs
at the same addresses. Registered applicable tests are covered by the
[dated validation](local-all-features-20261023.md). The
[larger-workload report](7900xtx-large-throughput-2026-10-23.md#native-bounded-integer-device-output)
records scoped native-output timing. Broader target/ISA qualification and public
dispatch integration remain separate gates.
