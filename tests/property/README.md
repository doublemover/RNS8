# Property Tests

Reserved for future property and randomized correctness tests.

Current deterministic public-API sweeps and fixed-seed random checks live under
`tests/unit/` while the scaffold is still in Phase 0/1 bring-up.

When property tests are added, include semantic-contract generators that keep
these cases distinct:

- bounded signed and unsigned exact results with valid recovery bounds,
- exact-wide signed and unsigned RNS-output contracts,
- strict `mod 2^64` wraparound through byte limbs,
- invalid attempts to satisfy exact-wide or wraparound semantics by attaching
  bounded CRT metadata.
