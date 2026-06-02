# Strict Wraparound Backend

Reserved for strict `mod 2^64` byte-limb GEMM.

This path must use unsigned byte-limb semantics. It must not reuse odd-modulus
CRT as a wraparound substitute unless an explicit exact bound is supplied by a
bounded semantic contract.
