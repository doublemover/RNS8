"""Authored CPU proof fixtures. Execution is deferred by implementation policy."""
from dataclasses import asdict, replace
from fractions import Fraction
import json
import unittest
from arithmetic_certificate import ArithmeticCertificate, InstructionContract, ProductPlane, certificate_hash, from_json, verify
from floating_contract import FloatingBlock, FloatingContract, decode_element, decode_scale, bind_artifacts
from floating_rewrite_oracle import evaluate_sequence, INF


def fixture(format_name="ocp_e2m1", count=32):
    # Canonical residues modulo 4 are exactly represented in every supported format.
    codes=tuple(next(code for code in range(1 << (4 if format_name=="ocp_e2m1" else 6 if format_name in ("ocp_e2m3","ocp_e3m2") else 8))
                     if decode_element(format_name,code)==value) for value in range(4))
    f=FloatingContract(format_name,(FloatingBlock(count,127,127,0),),block_k=max(32,count),
                       lowering_sha256="1"*64,proof_artifact_sha256="2"*64)
    i=InstructionContract("host-only","model-fp-product",0,3,-(1<<31),(1<<31)-1,"fixture",floating=f)
    p=ProductPlane(tuple(range(4)),tuple(range(4)),1,codes,codes)
    return bind_artifacts(ArithmeticCertificate(4,(p,),1,count,0,0,-(1<<63),(1<<63)-1,i))


class FloatingContracts(unittest.TestCase):
    def test_decoding_and_distinct_families(self):
        self.assertEqual(decode_element("ocp_e2m1",7),6)
        self.assertEqual(decode_element("ocp_e2m3",31),Fraction(15,2))
        self.assertEqual(decode_element("ocp_e3m2",31),28)
        self.assertEqual(decode_element("ocp_e4m3",126),448)
        self.assertEqual(decode_element("ocp_e5m2",123),57344)
        self.assertEqual(decode_scale(254),1<<127)
        for name,code in (("ocp_e4m3",127),("ocp_e5m2",124),("fnuz_e4m3",0)):
            with self.assertRaises(ValueError):decode_element(name,code)
        with self.assertRaises(ValueError):decode_scale(255)

    def test_all_formats_and_roundtrip(self):
        for name in ("ocp_e2m1","ocp_e2m3","ocp_e3m2","ocp_e4m3","ocp_e5m2"):
            c=fixture(name);r=verify(c)
            self.assertEqual(r["verdicts"]["finite_exact_recoverability"],"proven",r)
            self.assertEqual(r["verdicts"]["target_execution"],"not_run")
            self.assertFalse(r["runtime_dispatch_eligible"])
            self.assertEqual(certificate_hash(c),certificate_hash(from_json(json.loads(json.dumps(asdict(c))))))

    def test_explicit_outer_scaling_and_prior(self):
        c=fixture(count=64);f=c.instruction.floating
        f=replace(f,blocks=(FloatingBlock(32,126,127,1),FloatingBlock(32,128,127,-1)))
        c=bind_artifacts(replace(c,instruction=replace(c.instruction,floating=f)))
        f=c.instruction.floating
        self.assertEqual(verify(c)["verdicts"]["finite_exact_recoverability"],"proven")
        bad=replace(f,blocks=(FloatingBlock(32,126,127,0),FloatingBlock(32,128,127,-1)))
        self.assertIn("floating_plane_0_block_0_outer_grid_alignment",verify(replace(c,instruction=replace(c.instruction,floating=bad)))["proof_failures"])
        for field,value in (("outer_accumulation","rescale_old_accumulator"),("initial_accumulator",(1,1)),
                            ("rounding","unknown"),("reduction","unspecified_tree"),("proof_artifact_sha256","stale")):
            r=verify(replace(c,instruction=replace(c.instruction,floating=replace(f,**{field:value}))))
            self.assertNotEqual(r["verdicts"]["finite_exact_recoverability"],"proven",r)

    def test_every_partial_not_just_final_cancellation(self):
        c=fixture(count=32);f=c.instruction.floating
        for changed in (replace(f,accumulator_precision=3),replace(f,accumulator_max_exponent=2,accumulator_min_exponent=-2),
                        replace(f,blocks=(FloatingBlock(32,0,0,254),)),replace(f,destination_bits=8)):
            r=verify(replace(c,instruction=replace(c.instruction,floating=changed)))
            self.assertEqual(r["verdicts"]["algebraic_value_equality"],"proven")
            self.assertNotEqual(r["verdicts"]["finite_exact_recoverability"],"proven")
        p=replace(c.planes[0],lhs_codes=(0,1,2,3))
        self.assertIn("floating_plane_0_decoded_grid_mismatch",verify(replace(c,planes=(p,)))["proof_failures"])

    def test_identity_changes_on_all_assumptions(self):
        c=fixture();f=c.instruction.floating
        for changed in (replace(f,reduction="fold"),replace(f,denormals="flush_to_zero"),
                        replace(f,product_placement="raw_product_then_block_scale"),replace(f,lowering_sha256="3"*64),
                        replace(f,blocks=(FloatingBlock(32,126,128,0),))):
            self.assertNotEqual(certificate_hash(c),certificate_hash(replace(c,instruction=replace(c.instruction,floating=changed))))

    def test_outer_oracle_distinguishes_float_equivalence_from_integer_recovery(self):
        blocks=[([2]*32,[2]*32,127,127,"balanced",0)]*2
        r=evaluate_sequence(blocks)
        self.assertTrue(r["evaluators_agree"]);self.assertEqual(r["verdicts"]["finite_exact_recoverability"],"proven")
        r=evaluate_sequence(blocks,initial_bits=INF)
        self.assertEqual(r["verdicts"]["floating_rewrite_equivalence"],"proven")
        self.assertEqual(r["verdicts"]["finite_exact_recoverability"],"refuted")
        overflow=[([7]*32,[7]*32,252,252,"balanced",0)]*2
        self.assertEqual(evaluate_sequence(overflow)["verdicts"]["finite_exact_recoverability"],"refuted")


if __name__=="__main__":unittest.main()
