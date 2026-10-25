"""Independent positive and adversarial arithmetic-certificate checks."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from arithmetic_certificate import centered_int8, certificate_hash, from_json, verify, verify_wrap64_recode, verify_wrap64_mfma_recode


def main() -> int:
    for modulus in range(2, 257):
        certificate = centered_int8(modulus)
        result = verify(certificate)
        assert result["algebraically_verified"], (modulus, result)
        assert not result["runtime_dispatch_eligible"]
        assert certificate_hash(from_json(json.loads(json.dumps(asdict(certificate))))) == certificate_hash(certificate)
    base = centered_int8(256, 131071)
    assert verify(base)["algebraically_verified"]
    for value in (True, 1.0, math.nan, math.inf, -math.inf, "1", None):
        for field in ("prior_min", "prior_max", "recombination_min", "recombination_max"):
            malformed = replace(base, **{field: value})
            result = verify(malformed)
            assert not result["algebraically_verified"] and result["sha256"] is None
            json.dumps(result, allow_nan=False)
            try:
                certificate_hash(malformed)
            except ValueError:
                pass
            else:
                raise AssertionError("malformed proof received a canonical identity")
        for field in ("operand_min", "operand_max", "accumulator_min", "accumulator_max"):
            malformed = replace(base, instruction=replace(base.instruction, **{field: value}))
            assert not verify(malformed)["algebraically_verified"]
    for instruction in (replace(base.instruction, exact_integer_products=1),
                        replace(base.instruction, target=False),
                        replace(base.instruction, measured_evidence_sha256="not-a-digest")):
        assert not verify(replace(base, instruction=instruction))["algebraically_verified"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "certificate.json"
        for value in (math.nan, math.inf, True, "1", None):
            data = asdict(base)
            data["instruction"]["accumulator_max"] = value
            path.write_text(json.dumps(data), encoding="utf-8")
            run = subprocess.run([sys.executable, str(Path(__file__).with_name("arithmetic_certificate.py")),
                                  str(path)], capture_output=True, text=True, check=False)
            assert run.returncode == 1, run.stderr
            result = json.loads(run.stdout)
            assert not result["algebraically_verified"] and result["sha256"] is None
    doubled = replace(base.planes[0], weight=2)
    assert verify(replace(base, planes=(doubled,), denominator=2))["algebraically_verified"]
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
    recode=json.loads((Path(__file__).parent / "data/wrap64_signed_radix_certificate.json").read_text())
    assert verify_wrap64_recode(recode)["algebraically_verified"]
    assert not verify_wrap64_recode(recode)["runtime_dispatch_eligible"]
    for field,value in (("bias",0),("carry_before_xor",False),("signed_operands",False),
                        ("clamp",True),("prior_accumulator",1),("pair_count",35),
                        ("fold","int32_diagonal_sum"),("shift","i_plus_j"),
                        ("instruction_padded_k",131072),("instruction_padded_k",True)):
        assert not verify_wrap64_recode(dict(recode,**{field:value}))["algebraically_verified"]
    assert verify_wrap64_recode(dict(recode,instruction_padded_k=131071))["algebraically_verified"]
    for target in ("gfx90a", "gfx942", "gfx950"):
        mfma=json.loads((Path(__file__).parent / ("data/wrap64_signed_mfma_"+target+".json")).read_text())
        result=verify_wrap64_mfma_recode(mfma)
        assert result["algebraically_verified"] and not result["runtime_dispatch_eligible"]
        assert result["target_execution"] == "not_run"
        for field,value in (("target","gfx1100"),("opcode","wmma_i32_16x16x16_iu8"),
                            ("wave_size",32),("input_bytes_per_lane",1),("c_map","row=lane%16"),
                            ("instruction_padded_k",131071),("carry_before_xor",False),("extra",1)):
            assert not verify_wrap64_mfma_recode(dict(mfma,**{field:value}))["algebraically_verified"]
    print("arithmetic certificate self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
