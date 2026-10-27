"""Source-authored receipt/identity/scenario regressions; no GPU required."""
from benchmark_cases import case_root
import copy
import math
from pathlib import Path
from benchmark_schema.wide_magnitude import MODULI, validate_magnitude
from benchmark_sweep_lib.scenarios import load_scenario_data_family
from result_compare import contract_value


def main():
    for signed in (True, False):
        for exponent, profile in ((40, "wide-two-limb-constant"), (62, "wide-three-limb-constant")):
            bound = 512 << (2 * exponent)
            data = dict(input_profile=profile, semantics="exact_wide_signed" if signed else "exact_wide_unsigned",
                        m=2,n=3,k=512,prefix=20,exact_wide_limb_count=3)
            data["exact_wide_magnitude"] = dict(schema="exact_wide_magnitude_v1", oracle_complete=True,
                oracle="boost_cpp_int_full_integer_gemm_all_limbs", a_max_abs=str(1<<exponent), b_max_abs=str(1<<exponent),
                admitted_abs_bound=str(bound), basis_product=str(math.prod(MODULI)), a_max_magnitude_bits=exponent+1,
                b_max_magnitude_bits=exponent+1,output_min_magnitude_bits=bound.bit_length(),output_max_magnitude_bits=bound.bit_length(),
                output_max_magnitude_limbs=(bound.bit_length()+63)//64,storage_bits=192,checked_outputs=6,checked_limbs=18,
                negative_outputs=0,admission_us=1.0,oracle_us=10.0,timing_scope="outside_all_repeated_public_call_samples")
            assert not validate_magnitude(data)
            for key, value in (("checked_limbs",6),("output_max_magnitude_bits",64),("basis_product",str(2*bound)),
                               ("admitted_abs_bound",str(math.prod(MODULI))),("oracle_complete",False)):
                bad=copy.deepcopy(data);bad["exact_wide_magnitude"][key]=value
                assert validate_magnitude(bad),key
            bad=copy.deepcopy(data);del bad["exact_wide_magnitude"]
            assert validate_magnitude(bad)
            assert contract_value(data,"input_profile") != contract_value({},"input_profile")
    items=load_scenario_data_family(case_root() / 'scenarios/exact_wide_magnitude.json')
    assert len(items)==20
    assert {item.case.input_profile for item in items} == {"uniform-small","wide-two-limb-constant","wide-three-limb-constant","wide-two-limb-mixed","wide-three-limb-mixed"}
    assert all(item.backends==("cpu","hip-direct") and item.max_prefix==20 for item in items)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
