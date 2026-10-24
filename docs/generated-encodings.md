# Executable scalar encoding certificates

`encoding_catalogue.py --header ...` now emits a distinct C++ type containing
its certificate hash, byte lookup tables, coefficients, modulus, certified K
cap, and signed/unsigned 64-bit `dot` implementation. The optional `--symbol`
selects a name under `rns8_encoding_reference`, prefixed with `rns8_`; otherwise
the certificate hash determines the name. Multiple constructions can coexist.
The JSON artifact binds the exact generated LF-normalized source with SHA-256.

Each native operand is reduced to its canonical residue before table lookup.
Every product plane has its own signed 64-bit accumulator. The generated loop
uses certified K slabs, performs the ordered weighted sum, checks and performs
exact denominator division, and only then reduces modulo the channel modulus.
Canonical slab residues are combined without a wide full-K accumulator. Null
nonempty input fails without changing the output; empty input returns zero.

Generation requires the independently verified product identity and interval
proofs, zero prior accumulation, byte alphabets, representable coefficients and
signed 64-bit intermediate limits. A valid arbitrary-precision certificate with
wider intermediates is rejected by this implementation. This explicitly binds
the arithmetic proof to the generated scalar storage and operation widths.

CMake generates and compiles AWE6 and base-13 fixtures at moduli 251 and 256,
plus a denominator-two variant that requires division before reduction. The
authored C++ tests exhaust residue pairs and compare full-width signed/unsigned
dots with Boost at K=0,1,17,65535,65536,65537. Python fixtures cover deterministic
generation, invalid symbols, and certificates exceeding the generated widths.
The [dated all-feature validation](local-all-features-20261023.md) records
execution of the registered applicable generated-reference tests.

This is an executable scalar integer model for #44/#29, not an FP4 kernel.
Actual floating encodings, scale bits, target instructions, resource usage and
performance require separate hardware qualification. Default dispatch is unchanged.
