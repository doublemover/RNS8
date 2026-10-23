"""Exhaustive byte oracle; generated data and alphabet rejection tests."""
from dataclasses import replace

from arithmetic_certificate import verify
from encoding_catalogue import (AWE_COEFFICIENTS, AWE_WEIGHTS, awe_certificate, awe_encode,
                               decode, encode, generate_header, schoolbook_certificate)
from modulus_set_search import DEFAULT_MODULI


def main() -> int:
    for value in range(-128, 128):
        assert decode(encode(value)) == value
        faces = awe_encode(value)
        assert sum(w * x for w, x in zip(AWE_WEIGHTS, faces[:3])) == value
        assert all(-4 <= x <= 4 for x in faces)
        for other in range(-128, 128):
            assert sum(w * x * y for w, x, y in zip(AWE_COEFFICIENTS, faces, awe_encode(other))) == value * other
    for modulus in (*DEFAULT_MODULI, 2, 4, 128, 256):
        certificate = awe_certificate(modulus, 1 << 20)
        result = verify(certificate)
        assert result["algebraically_verified"], result
        assert not result["runtime_dispatch_eligible"]
    cap = awe_certificate(256, (1 << 20) + 1)
    assert not verify(cap)["algebraically_verified"]
    base_awe = awe_certificate(256)
    changed = replace(base_awe.planes[0], weight=base_awe.planes[0].weight + 1)
    assert not verify(replace(base_awe, planes=(changed,) + base_awe.planes[1:]))["algebraically_verified"]
    for modulus in DEFAULT_MODULI:
        certificate = schoolbook_certificate(modulus)
        result = verify(certificate)
        assert result["algebraically_verified"], result
        assert generate_header(certificate) == generate_header(certificate)
        assert not result["instruction_measured"]
    base = schoolbook_certificate(256)
    generated = generate_header(base, "base13_256")
    assert "struct rns8_base13_256" in generated
    assert "numerator / 1LL" in generated and "offset += length" in generated
    assert "certificate_sha256" in generated and "static bool dot" in generated
    for symbol in ("bad-name", "x;int injected", "double__underscore", ""):
        if symbol == "":
            continue  # Empty optional symbol selects the certificate hash.
        try:
            generate_header(base, symbol)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid C++ symbol accepted")
    # Algebraically valid arbitrary precision certificates are not necessarily
    # executable with this generator's fixed byte/int64 implementation.
    wide_intermediate = replace(base, recombination_max=1 << 80)
    assert verify(wide_intermediate)["algebraically_verified"]
    try:
        generate_header(wide_intermediate)
    except ValueError:
        pass
    else:
        raise AssertionError("unqualified int64 implementation emitted")
    corrupted = replace(base.planes[0], lhs=(5,) + base.planes[0].lhs[1:])
    result = verify(replace(base, planes=(corrupted,) + base.planes[1:]))
    assert "plane_0_stored_alphabet" in result["proof_failures"]
    missing = replace(base.planes[0], lhs=base.planes[0].lhs[:128] + base.planes[0].lhs[129:])
    assert "encoding_table_length" in verify(replace(base, planes=(missing,) + base.planes[1:]))["proof_failures"]
    assert not verify(replace(base, denominator=2))["algebraically_verified"]
    print("encoding catalogue self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
