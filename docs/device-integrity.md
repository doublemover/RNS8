# Independent native-product integrity sentinel

The private opt-in HIP checker recomputes each dot product from original native
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
treating a zero mask as successful verification. This private checker is
read-only. Its future public transaction owner must run it before publication
and prevent cache reuse on failure; resident/DAG integration remains open.

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
injection. Compilation, execution, fault-model coverage expansion and cost
measurement remain pending under #49.
