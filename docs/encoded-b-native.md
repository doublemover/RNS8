# Generation-bound encoded B

`rns8_prepare_encoded_b` and `rns8_gemm_encoded_b_exact` are explicit experimental
synchronous direct-HIP APIs for exact-wide signed/unsigned inputs. Unlike the
prepared native queue's cached descriptors, this handle stores actual B residues.
Preparation requires a current `RNS8_MATRIX_RETAIN_NATIVE_INPUT` B and encodes its
full default-basis prefix on the device. No host value download is used.

The layout is `[plane][column][K-word]`: four adjacent centered residues occupy
one aligned 32-bit word; K is zero-padded to a multiple of 16. The grouped WMMA
consumer converts A inline and loads each B fragment word directly. Explicit
channel groups 1, 2 and 4 share the existing exact modular accumulation/reduction
rules, including the 65536-term safety boundary. The compiled device capability
is checked during preparation. There is no unsupported-target fallback.

The handle retains source/context lifetimes and binds the source's engine instance
and generation. Repacking B is legal, but makes the old handle unusable. A fresh
handle is needed for the new generation. Each call checks current A native storage,
B's generation, shared semantic/prefix/shape/device identities, public alias and
reader/writer reservations, and the ordinary full-expression exact-range proof.
Failure before launch preserves C; failure after launch may invalidate it. Success
publishes all RNS output planes and retires C's old native materialization.

Retained allocation is `prefix*N*round_up(K,16)+4` bytes, including a capability
word. The explicit preparation budget covers this allocation; source/output
matrices, existing handles during replacement and allocator overhead are separate.
Warm execution allocates no additional device scratch and does not encode B again.
CPU proof allocation and synchronization remain part of the public call.

Authored fixtures cover all prefixes, groups, signedness, full-width values,
padded source/output pitches, tails, K=65535/65536/65537, source retention and stale
rejection. These are unexecuted in the current frontload. Calibration must compare
identical shapes/prefixes/native semantics and separate cold encoding+allocation
from warm reuse, including a setup-inclusive break-even. No measured win is claimed.

## Synchronous encoded queue

`rns8_gemm_encoded_b_grouped` admits up to 4096 tasks and 1..1024 persistent workers.
It stages every exact proof and checks every cache generation before enqueue. It
rejects duplicate outputs and cross-task output/input matrix aliases, including
native/RNS representations with different allocation pointers. Read-only A/B
sharing is legal. One successful queue completion publishes all outputs; a possible
write failure invalidates all outputs. It has no per-task completion or async API.

Private queue descriptors also accept encoded B alongside raw tasks; their builder
checks encoded layout, full allocation spans and cross-task write hazards. Encoded
queues use the heterogeneous kernel; original all-raw uniform specializations stay
separate. The prepared ordinary-grid control consumes the same encoded storage.
The queue caches descriptors; the public encoded-B handle owns the value payload.
New fixtures cover mixed raw/encoded tasks, groups/signedness, K-slab tails, shared B,
logical aliases and changed source rejection. They remain unexecuted.

Async completion slots and automatic materialization selection remain separate
work. Existing descriptor-only queue reports do not measure this representation.
