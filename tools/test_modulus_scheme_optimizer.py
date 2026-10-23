"""Brute-force cross-check of bounded weighted scheme search."""
import itertools
import math

from arithmetic_certificate import centered_int8
from modulus_scheme_optimizer import Form, optimize


def main() -> int:
    forms = [Form(centered_int8(m), i + 1, 2 * i + 1, 1, 3, "synthetic-fixture-only")
             for i, m in enumerate((4, 5, 7, 9, 11))]
    for bound in (0, 8, 100, 1000, 10000):
        for signed in (False, True):
            possible = []
            for count in range(1, len(forms) + 1):
                for subset in itertools.combinations(forms, count):
                    moduli = [form.certificate.modulus for form in subset]
                    if all(math.gcd(a, b) == 1 for a, b in itertools.combinations(moduli, 2)):
                        if math.prod(moduli) > bound * (2 if signed else 1):
                            possible.append(sum(f.cost(65537) for f in subset))
            report = optimize(forms, bound, signed, 65537)
            assert report["search_complete"]
            assert report["feasible_upper_bound_ps"] == (min(possible) if possible else None)
            assert report["feasible"] == bool(possible)
    limited = optimize(forms, 100, False, 1, max_nodes=1)
    assert not limited["search_complete"]
    assert limited["lower_bound_ps"] == 0
    assert not limited["promotion_eligible"]
    print("modulus scheme optimizer self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
