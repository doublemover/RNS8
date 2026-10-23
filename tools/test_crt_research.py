"""Exhaustive small-basis and full-ladder alternative CRT differentials."""
import itertools
import random

from crt_research import balanced, direct, make_plan, width_certificate
from modulus_set_search import DEFAULT_MODULI


def main() -> int:
    small = make_plan((2, 251))
    certificate = width_certificate(small)
    assert certificate["direct_sum_accumulator_unsigned_bits"] == 10
    assert certificate["direct_stages"][1]["inverse_multiply_max"] == 31500
    assert certificate["direct_stages"][1]["inverse_multiply_unsigned_bits"] == 15
    assert direct(small, (1, 250), accumulator_bits=10, temporary_bits=15) == 501
    try:
        direct(small, (1, 250), accumulator_bits=10, temporary_bits=10)
    except ValueError:
        pass
    else:
        raise AssertionError("uniform 10-bit CRT admitted despite wide inverse product")
    for basis in ((2,), (2, 3), (4, 9, 5), (8, 3, 5, 7)):
        plan = make_plan(basis)
        for residues in itertools.product(*(range(m) for m in basis)):
            a, b = direct(plan, residues), balanced(plan, residues)
            assert a == b
            assert all(a % m == r for m, r in zip(basis, residues))
    rng = random.Random(390039)
    for prefix in range(1, 21):
        plan = make_plan(tuple(DEFAULT_MODULI[:prefix]))
        certificate = width_certificate(plan)
        assert certificate["direct_sum_max"] < prefix * plan.product
        temporary = max(max(stage["inverse_multiply_unsigned_bits"], stage["weighted_multiply_unsigned_bits"])
                        for stage in certificate["direct_stages"])
        values = [0, 1, plan.product - 1, plan.product // 2, (plan.product + 1) // 2,
                  (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
        values.extend(rng.randrange(plan.product) for _ in range(100))
        for value in values:
            residues = tuple(value % m for m in plan.moduli)
            assert direct(plan, residues) == value % plan.product
            assert direct(plan, residues, plan.direct_accumulator_bits, temporary) == value % plan.product
            assert balanced(plan, residues) == value % plan.product
        try:
            direct(plan, tuple(0 for _ in plan.moduli), plan.direct_accumulator_bits - 1)
        except ValueError:
            pass
        else:
            raise AssertionError("short accumulator accepted")
    print("CRT research prototype self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
