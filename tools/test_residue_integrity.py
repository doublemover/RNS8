"""Authored corruption fixtures, not device/OS fault injection."""
from dataclasses import replace

from residue_integrity import Identity, check_candidate, independent_dot_check


def main() -> int:
    basis = (5, 7)
    identity = Identity("dot-0", (("a", 1), ("b", 2)), 3, basis)
    for expected in range(-17, 18):
        check = independent_dot_check(identity, (expected,), (1,), (11,))
        for candidate in range(-17, 18):
            report = check_candidate(identity, tuple(candidate % m for m in basis), check,
                                     signed=True, magnitude_bound=17)
            assert (report["status"] == "consistent_under_declared_checks") == ((candidate - expected) % 11 == 0)
        original = tuple(expected % m for m in basis)
        for plane, modulus in enumerate(basis):
            for replacement in range(modulus):
                damaged = list(original)
                damaged[plane] = replacement
                report = check_candidate(identity, tuple(damaged), check, signed=True, magnitude_bound=17)
                assert (report["status"] == "consistent_under_declared_checks") == ((report["candidate"] - expected) % 11 == 0)
    check = independent_dot_check(identity, (2, 3), (4, 1), (11,))
    residues = (1, 4)
    for bad in (replace(check, identity=replace(identity, epoch=2)),
                replace(check, identity=replace(identity, inputs=(("a", 0), ("b", 2)))),
                replace(check, input_path="candidate_base_extension")):
        assert check_candidate(identity, residues, bad, signed=True, magnitude_bound=17)["status"] == "failed_integrity"
    grown = replace(identity, basis=(5, 7, 11))
    assert "check_basis_not_independent" in check_candidate(grown, (1, 4, 0), check, signed=True, magnitude_bound=17)["failures"]
    refreshed = independent_dot_check(grown, (2, 3), (4, 1), (13,))
    assert check_candidate(grown, (1, 4, 0), refreshed, signed=True, magnitude_bound=17)["status"] == "consistent_under_declared_checks"
    # Shared wrong input in both paths is intentionally undetectable.
    shared_wrong = independent_dot_check(identity, (12,), (1,), (11,))
    assert check_candidate(identity, (2, 5), shared_wrong, signed=True, magnitude_bound=17)["status"] == "consistent_under_declared_checks"
    print("residue integrity self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
