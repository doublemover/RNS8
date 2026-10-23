"""Polynomial certificate plus exact composite/prime ring and tail oracles."""
from dataclasses import replace
import random

from block_gemm_research import STRASSEN, centered, one_level, verify_program


def main() -> int:
    assert verify_program(STRASSEN)["polynomial_identity_verified"]
    bad = replace(STRASSEN[0], c=(1, 0, 0, -1))
    assert not verify_program((bad,) + STRASSEN[1:])["polynomial_identity_verified"]
    rng = random.Random(144)
    for modulus in range(2, 257):
        for m, n, k in ((1, 1, 1), (3, 5, 7), (4, 4, 4), (1, 8, 3)):
            a = [[rng.choice((-128, 127, 0, modulus, -modulus, rng.randrange(-1000, 1000))) for _ in range(k)] for _ in range(m)]
            b = [[rng.randrange(-1000, 1000) for _ in range(n)] for _ in range(k)]
            actual, report = one_level(a, b, modulus)
            expected = [[centered(sum(a[i][q] * b[q][j] for q in range(k)), modulus)
                         for j in range(n)] for i in range(m)]
            assert actual == expected, (modulus, m, n, k)
            assert report["packed_operand_abs_bound"] <= 128
            assert report["stored_product_matrices"] == 0
            assert not report["runtime_dispatch_eligible"]
    print("block GEMM research self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
