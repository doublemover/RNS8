"""Independent exact quotient and matched public/private harness fixtures."""
from benchmark_cases import case_root
from copy import deepcopy
from fractions import Fraction
import unittest
from crt_research import make_plan,quarter_certificate,quarter_reconstruct
from modulus_set_search import DEFAULT_MODULI
from collect_crt_calibration import QUARTER_CANDIDATES,dataset,validate_capture,load_cases
from test_collect_crt_calibration import fixture
from pathlib import Path

class QuarterCrt(unittest.TestCase):
    def test_small_bases_and_every_prefix(self):
        bases=[(2,3),(3,5),(2,251)]+[tuple(DEFAULT_MODULI[:p]) for p in range(1,21)]
        for basis in bases:
            plan=make_plan(basis);bound=plan.product//4
            c=quarter_certificate(plan,bound,192)
            self.assertLess(Fraction(*c["error_bound"]),Fraction(1,64))
            values=range(-bound,bound+1) if bound<256 else (-bound,-bound+1,-1,0,1,bound-1,bound)
            for value in values:
                residues=tuple(value%m for m in basis)
                self.assertEqual(quarter_reconstruct(plan,residues,c),value)
            with self.assertRaises(ValueError):quarter_certificate(plan,bound+1,192)
            bad=deepcopy(c);bad["fraction_bits"][0]^=1
            with self.assertRaises(ValueError):quarter_reconstruct(plan,tuple(0 for _ in basis),bad)

    def test_matched_cohorts_are_not_canonical_random_or_signedness_aliases(self):
        for prefix in range(1,21):
            for sign in ("signed","unsigned"):
                base=dict(candidate="quarter_"+sign,cells=17,prefix=prefix,scope="resident")
                expected=dataset(base)
                for kind in ("garner_quarter_","direct_quarter_","balanced_quarter_"):
                    self.assertEqual(dataset(dict(base,candidate=kind+sign)),expected)
                narrow=dataset(dict(base,candidate="quarter_narrow_"+sign))
                for kind in ("garner_narrow_","public_quarter_","public_garner_"):
                    self.assertEqual(dataset(dict(base,candidate=kind+sign)),narrow)
                self.assertNotEqual(expected,dataset(dict(base,candidate="direct")))
                for candidate in QUARTER_CANDIDATES:
                    case=dict(base,candidate=candidate);capture=fixture(case)
                    validate_capture(capture,case,"fixture-run",2,1)
                    if not candidate.startswith("public_"):
                        capture["quarter_certificate"]["error_numerator"]*=4
                        with self.assertRaises(ValueError):validate_capture(capture,case,"fixture-run",2,1)

    def test_tracked_quarter_manifests(self):
        root=case_root()/"conversion-calibration"
        cases=load_cases(root/"quarter-every-prefix.json")
        self.assertEqual({c["prefix"] for c in cases},set(range(1,21)))
        self.assertEqual({c["candidate"] for c in cases},set(QUARTER_CANDIDATES))
        self.assertEqual({c["scope"] for c in cases},{"resident","cold"})
        self.assertTrue(any(c["cells"]==1 for c in load_cases(root/"quarter-tails.json")))

if __name__=="__main__":unittest.main()
