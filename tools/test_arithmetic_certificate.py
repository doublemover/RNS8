"""Independent positive and adversarial arithmetic-certificate checks."""
from dataclasses import asdict, replace
import json

from arithmetic_certificate import centered_int8, certificate_hash, from_json, verify


def main() -> int:
    for modulus in range(2, 257):
        certificate = centered_int8(modulus)
        result = verify(certificate)
        assert result["algebraically_verified"], (modulus, result)
        assert not result["runtime_dispatch_eligible"]
        assert certificate_hash(from_json(json.loads(json.dumps(asdict(certificate))))) == certificate_hash(certificate)
    base = centered_int8(256, 131071)
    assert verify(base)["algebraically_verified"]
    broken = [
        (replace(base, safe_k=131072), "plane_0_accumulator_overflow"),
        (replace(base, instruction=replace(base.instruction, operand_min=0, operand_max=255)), "plane_0_operand_alphabet"),
        (replace(base, basis=(255, 15)), "basis_not_pairwise_coprime_byte_moduli"),
        (replace(base, semantics="bounded_signed", basis=(256, 255), output_bound=32640), "insufficient_crt_range"),
        (replace(base, denominator=2), "nonintegral_recombination"),
        (replace(base, recombination_max=100), "plane_0_coefficient_overflow"),
        (replace(base, semantics="finite_field"), "field_modulus_not_prime"),
        (replace(base, semantics="wrap64"), "wrap64_requires_byte_limb_composition_certificate"),
        (replace(base, instruction=replace(base.instruction, exact_integer_products=False)), "instruction_exactness_unproven"),
    ]
    for certificate, obligation in broken:
        result = verify(certificate)
        assert not result["algebraically_verified"]
        assert obligation in result["proof_failures"], result
    altered = replace(base.planes[0], lhs=(0,) * 256)
    assert "encoding_product_identity" in verify(replace(base, planes=(altered,)))["proof_failures"]
    # Algebraic cancellation does not permit overflowing ordered intermediates.
    planes = (replace(base.planes[0], weight=2), replace(base.planes[0], weight=-1))
    ordered = replace(base, planes=planes, recombination_min=-(1 << 31), recombination_max=(1 << 31) - 1)
    assert "plane_0_coefficient_overflow" in verify(ordered)["proof_failures"]
    print("arithmetic certificate self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
