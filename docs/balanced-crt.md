# Balanced CRT device baseline

The private canonical export candidate builds a pairwise tree for each default
basis prefix 1 through 20. Successive levels merge adjacent subtrees, carrying
an unmatched final subtree forward. Its dependency depth is ceil(log2(prefix)),
unlike a sequential recurrence over precomputed pairs. Each immutable plan
records original channel slots, subtree products and modular inverses; a merge
overwrites only the left child's slot. Host plans are initialized once with
Boost, using the existing Garner small-modulus inverse owner to reconstruct
each wide inverse. The plan is supplied by value as kernel arguments.

For canonical children a in [0,A), b in [0,B), the kernel computes
t = ((b - a) mod B) * inverse(A mod B) mod B and x = a + A*t.
All subtree products are below 2^160. Bitwise exact reduction doubles a value
below B before adding a bit, requiring less than 161 bits. Modular multiplication
reduces after each doubling and addition, also requiring less than 161 bits.
The final nonnegative multiply A*t and its ordered Horner intermediates are
below A*B; adding a yields x below A*B. Three 64-bit limbs therefore suffice
throughout, without forming an unchecked double-width inverse product.

This per-thread candidate uses 20 reusable wide-value slots and exact bit-serial
arithmetic. Those choices can create register pressure, spills and expensive
integer work. No throughput advantage is claimed. It is private, synchronous,
and returns three little-endian limbs representing [0,P); root signed centering
and destination-width admission are not inferred from the limb type. The caller
must supply valid default-basis residues and own publication on device failure.
The maintained public Garner route is unchanged.

CPU fixtures verify every merge with Boost, all prefix products and inverse
identities, tree depth, random values, zero, P-1 and centered thresholds. GPU
fixtures compare direct and balanced candidates with the same exact values and
tail canaries. Compilation and execution are pending. Quotient-assisted CRT,
subgroup and tile-batched implementations, group-arrival scheduling, full export
contracts and measured target/prefix selection remain open under issue #39.
