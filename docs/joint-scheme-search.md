# Joint basis and schedule oracle

`tools/joint_scheme_optimizer.py` extends the additive form search with an exact,
bounded oracle for up to eight supplied forms. It enumerates coprime subsets,
logical basis permutations, physical channel partitions, and explicitly supplied
K slabs. It never changes the runtime basis or authorizes a hardware instruction.

Each form retains its arithmetic certificate and calibration-cohort identity.
Both optimizers share a strict whole-domain admission pass. Reports retain each
form's certificate digest, checker version, complete separate proof verdicts and
failed obligations. `FormAdmissionError`, a `ValueError`, carries that structured
report when any form is inadmissible; the ordinary file CLI writes the rejection
report and exits nonzero. A cheaper invalid form is never silently selected or
filtered out of the declared domain. Mathematical proof is checked once per
form, and the resulting identities are reused in the search report. This does
not cache proofs across changed inputs or manufacture target qualification.
Separate interval calculations report the dot-accumulation cap and the ordered
weighted-epilogue cap. The allowed slab is also bounded by the certificate's
declared safe K. Each group uses the minimum cap of all members. Final CRT
admission independently requires the complete product to exceed B (unsigned)
or 2B (symmetric signed); more slabs cannot repair insufficient output range.

Group compatibility requires matching instruction contract, fragment layout,
physical plane count, and denominator. Register and LDS requirements are summed
against explicit budgets. These supplied resource values are model inputs,
not compiler resource evidence. Logical plane IDs remain independent of group
execution order, and every chosen channel occurs exactly once in the schedule.

The declared cost objective charges conversion and final reconstruction per
channel, work and epilogue per slab, one launch per group/slab, and a supplied
cost per cumulative prefix bit. It sums channel work inside a group; it assumes
no throughput gain from fusion. Prefix order and physical grouping are separable
only under this explicit objective. More complex overlap and order-dependent
conversion costs require a different model. Costs must be known nonnegative
integer picoseconds from one named cohort.

The existing `modulus_scheme_optimizer.py` CLI accepts an optional
`joint_schedule` object with `slabs`, `max_group_planes`, `register_budget`,
`lds_budget`, `group_launch_ps`, and `prefix_bit_ps`. Each form then also supplies
`resources` containing `layout`, `registers`, and `lds_bytes`. The original
additive request format remains available. Reports retain certificate hashes,
costs, resources, legal caps, exact prefix products, physical plane IDs, and the
declared domain. Search-budget exhaustion returns only an incumbent and a zero
lower bound, never an optimality claim.

The joint report also retains latency/physical-plane-count Pareto candidates.
One lowest-cost schedule per plane count is sufficient for this two-objective
frontier, so retained candidate storage is bounded by the nonempty subsets of
at most eight supplied forms. Schedule dictionaries are constructed only when
that incumbent improves. Equivalent objective points keep one representative;
this is not a list of every equivalent schedule or a measured hardware frontier.
The shared two-objective helper preserves equal-point alternatives for its
ordinary latency/workspace caller and uses a stable sorted sweep instead of
pairwise comparisons. An exhausted search labels its frontier incomplete.

An optional `allowed_form_ids` shortlist can be compared with unrestricted
enumeration to quantify a staged search's lost optimum. The authored fixture
demonstrates a cost of 202 for an isolated-cost shortlist versus 103 for the
joint optimum: a slightly more expensive form shares a launch with the other
channel. These are synthetic picoseconds, not timings. A separate labelled-bin
brute-force fixture checks all toy basis orders, groups, and slabs without using
the optimizer's partition generator. Runtime execution of these tests remains
pending in the consolidated validation batch.

Set `joint_schedule.compare_unrestricted=true` with explicit `allowed_form_ids`
to retain both the unrestricted and staged searches, their separate identical
node budgets and excluded form identities. The comparison reports a latency
penalty only when both searches complete. An empty feasible shortlist and a
budget-limited comparison have separate dispositions. Caller exclusions are
labelled as unproved pruning; no default dominance justification is invented.
The file CLI exits nonzero for inconclusive or infeasible selections. The same
comparison is available as `compare_staged` for bounded research consumers.

This finishes a concrete offline oracle slice of #32. It does not establish a
globally optimal hardware pipeline, measured Pareto frontier or production
qualification. The separate experimental runtime basis has its own
[explicit ABI and ownership](../include/rns8/experimental_basis.h); neither
offline search selects it implicitly nor its existence qualifies these costs.
