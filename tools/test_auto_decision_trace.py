"""Authored actual-decision receipt checks, not synthetic performance evidence."""
from copy import deepcopy
import unittest
from benchmark_schema import validate_capture
from test_benchmark_schema import expect_valid,add_helper_lane_fields
from metadata_registry_constants import SELECTOR_REJECTION_REASONS


def fixture():
    c=add_helper_lane_fields(deepcopy(expect_valid("v4_bounded_i64_ck.json")))
    records=[]
    reasons={0:"baseline policy",4:"backend not compiled",10:"slower than selected",11:"reviewed exact entry selected"}
    for backend,code,median in (("hip-direct",0,None),("hip-vector-alu-int64",4,None),("hipblaslt",10,5.),("ck",11,2.),("rocwmma",4,None),("amdgpu-builtins",4,None)):
        records.append(dict(backend=backend,disposition=2 if code==0 else 1 if code==11 else 0,reason_code=code,reason=reasons[code],
            exact_entry_present=code in (10,11),validated_entry_present=code in (10,11),required_workspace_bytes=0,
            reviewed_median_us=median,key_sha256="a"*64 if code in (10,11) else ""))
    c["auto_selector"].update(source="immutable_plan_selection_trace_v1",trace_recorded=True,selection_performed=True,
        cache_exists=True,cache_loaded=True,cache_entry_count=2,exact_hit=True,validated_hit=True,
        fallback_reason="reviewed exact entry selected",candidate_decisions=records,
        rejection_reason_vocabulary=sorted(SELECTOR_REJECTION_REASONS),
        rejected_candidates=[dict(backend=d["backend"],reason=d["reason"]) for d in records if d["disposition"]==0])
    return c

class AutoTrace(unittest.TestCase):
    def test_two_eligible_hits_are_not_reported_absent(self):
        c=fixture();validate_capture(c)
        d=c["auto_selector"]["candidate_decisions"][2]
        self.assertTrue(d["exact_entry_present"]);self.assertEqual(d["reason"],"slower than selected")
        self.assertIn("amdgpu-builtins",[d["backend"] for d in c["auto_selector"]["candidate_decisions"]])

    def test_forged_missing_hit_projection_order_and_inventory_fail(self):
        for mutation in ("false_absence","projection","wrong_order","missing_builtins","not_recorded"):
            c=fixture();a=c["auto_selector"]
            if mutation=="false_absence":a["candidate_decisions"][2]["exact_entry_present"]=False
            elif mutation=="projection":a["rejected_candidates"][1]["reason"]="no exact entry"
            elif mutation=="wrong_order":a["candidate_decisions"][2]["reviewed_median_us"]=1.
            elif mutation=="missing_builtins":a["candidate_decisions"].pop()
            else:a["trace_recorded"]=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_capture(c)

    def test_tie_is_not_falsely_slower(self):
        c=fixture();a=c["auto_selector"];d=a["candidate_decisions"][2]
        d.update(reviewed_median_us=2.,reason_code=13,reason="tied with selected")
        a["rejected_candidates"][1]["reason"]="tied with selected";validate_capture(c)
        d["reviewed_median_us"]=3.
        with self.assertRaises(ValueError):validate_capture(c)

if __name__=="__main__":unittest.main()
