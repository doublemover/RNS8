# Independent native-product integrity sentinel

The opt-in HIP checker recomputes each dot product from original native
i64/u64 inputs modulo 257 and 263. It separately reconstructs the candidate's
default-basis value, applies explicit signed/unsigned centering and a supplied
exact magnitude bound, then compares the two check residues. Check moduli are
coprime to all current computational channels and each other; they contribute
nothing to the claimed integer range. Growing a default prefix through 20 does
not consume them. A future basis change requires a new qualified check contract.

The native scalar check path does not reuse candidate reconstruction as its
input. It reduces every native operand and every product accumulation, bounding
the accumulator by 262^2+262 independently of K. Full-width signed values,
including INT64_MIN, use signed remainder without signed negation. Host
admission requires a matching operation/epoch, input instances/generations,
basis version and prefix, plus strict computational-basis uniqueness for the
bound. Caller-supplied identity metadata does not authenticate mislabeled memory.

`RNS8_VERIFICATION_FAILED` reports malformed centered residues, a candidate
outside the bound, or mismatches modulo either check prime. A failure bit mask
distinguishes those observations. Other failures remain invalid-argument,
range, unsupported or backend errors; callers must inspect status rather than
treating a zero mask as successful verification. The standalone private checker is read-only. The public
`rns8_set_prepared_native_integrity` adapter instead attaches the independent
check to the prepared producer transaction. It requires uniform direct-HIP
exact-wide GEMM, retained current original native A/B, and the ordinary independent
product-range proof. It does not accept epilogues or infer native inputs from C.

Synchronous execution, borrowed streams, owned producer graphs and native
prepared groups enqueue the check after the product and before their completion
event. Four pinned host bytes receive the four-byte device failure word on that
same stream. Observation interprets the result before publishing any current
output or releasing its writer. A mismatch leaves C unexportable and propagates
through prepared DAG dependencies; a downstream device kernel may already have
run, but its result is never published after its producer fails. Cancellation and
backend/dependency failure stay distinct from consistency. Undrainable failures
retain operand/workspace leases and both flag allocations. Check setup is retained;
replay does not allocate flags or synchronize the whole device.

`rns8_get_prepared_native_integrity_report` binds only the latest actual operation
epoch, original input instances/generations, basis/prefix, unsigned interpretation,
independent bound, checking moduli and output generation. The report records
consistent, mismatch, not-checked, canceled and execution-error separately. Report
reads are busy until observation completes. The shared operation-provider query
exposes this same envelope and the eight-byte known scratch lower bound; opaque
HIP event/stream storage is additional. No basis growth through the default 20
channels consumes either check modulus.

Success means consistency under these checks, not arbitrary correctness.
Candidate errors divisible by 257*263 are invisible. Shared erroneous native
inputs or common faults in both paths can agree. No probabilistic 1/q claim,
Freivalds claim, correction or replacement for deterministic exactness is made.
The path adds a complete scalar check dot product plus reconstruction and
synchronization; no low-overhead claim is supported yet.

CPU identity/coprimality fixtures and GPU exact/corrupted-value fixtures are
authored. They include isolated plane perturbations, stale identity, malformed
residues, insufficient bounds, full-width native operands, and an explicitly
undetectable 257*263 error. These are authored data changes, not device fault
injection. Public transaction fixtures additionally author mismatching native/residue
representations, sync/stream/graph publication failures, stale epochs and native
prepared-group composition. Host syntax is checked; HIP compilation, execution,
fault-model qualification and cost measurement remain deferred under #49.


## Matched cost experiments (not executed)

`rns8-native-integrity-calibrate` and `collect_native_integrity.py` provide
counterbalanced independent-process experiments for unverified exact compute,
producer-attached compute+sentinel, compute+full multiprecision CPU comparison,
and a separately labeled standalone scalar check control. Every route includes
padded three-limb output export; cold and repack cohorts are separate from resident
replay. The standalone control starts with a resident candidate and includes its
private per-call flag allocation/device synchronization; it is not mislabeled as
the retained public transaction's cost. Full CPU comparison uses original host
inputs, not a hidden input D2H transfer. Complete-output oracle work is outside
production intervals except in the explicit CPU-comparison route.

Fixtures cover square, odd rectangular, skinny, K=65537 and signed/unsigned
full-width inputs. Captures bind artifact/build/environment, basis, target,
actual producer epoch and generations, report compute/export-reference/setup/
cleanup costs, known logical payload and sustained durations. The collector
rejects unsupported probability claims, check channels counted as computational
range, stale/missing report identity, incomplete timing intervals and repeated
process identities. Only matched compute+check routes form cost ratios; standalone
checks retain their different boundary. The existing finite verifier experiment
separately measures Freivalds projections with its own ring/randomness theorem.
No captures, cost advantage or target qualification have been produced here.
