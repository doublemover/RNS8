"""Exhaustive byte oracle; generated data and alphabet rejection tests."""
from dataclasses import replace

from arithmetic_certificate import verify
from encoding_catalogue import decode, encode, generate_header, schoolbook_certificate
from modulus_set_search import DEFAULT_MODULI


def main() -> int:
    for value in range(-128, 128):
        assert decode(encode(value)) == value
    for modulus in DEFAULT_MODULI:
        certificate = schoolbook_certificate(modulus)
        result = verify(certificate)
        assert result["algebraically_verified"], result
        assert generate_header(certificate) == generate_header(certificate)
        assert not result["instruction_measured"]
    base = schoolbook_certificate(256)
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
