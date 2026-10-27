from dataclasses import replace
from fractions import Fraction
import unittest
from arithmetic_certificate import verify
from fp8_modular_certificate import certificate,encode
from floating_contract import decode_element

class Fp8Models(unittest.TestCase):
    def test_named_fnuz_encoding_is_distinct_from_ocp(self):
        for value in range(-8,9):self.assertEqual(decode_element('amd_e4m3_fnuz',encode(value)),Fraction(value))
        self.assertEqual(decode_element('amd_e4m3_fnuz',64),1)
        self.assertEqual(decode_element('ocp_e4m3',64),2)
        with self.assertRaises(ValueError):decode_element('amd_e4m3_fnuz',128)
    def test_four_plane_conditional_proof_and_mutations(self):
        for modulus in (2,16,127,251,255,256):
            c=certificate(modulus,'synthetic-compiler','a'*64);r=verify(c)
            self.assertEqual(r['proof_failures'],[]);self.assertEqual(r['verdicts']['target_execution'],'not_run')
            wrong=replace(c,planes=(replace(c.planes[0],weight=2),)+c.planes[1:])
            self.assertTrue(verify(wrong)['proof_failures'])
            wrong_format=replace(c,instruction=replace(c.instruction,floating=replace(c.instruction.floating,element_format='ocp_e4m3')))
            self.assertTrue(verify(wrong_format)['proof_failures'])
    def test_three_product_formats_and_corrupted_weight(self):
        for format, name in ((1,'amd_e4m3_fnuz'),(2,'ocp_e4m3')):
            for value in range(-16,17):
                self.assertEqual(decode_element(name,encode(value,format==2)),Fraction(value))
            for method in (0,1,2,3):
                c=certificate(256,'synthetic-compiler','a'*64,format=format,method=method)
                self.assertEqual(verify(c)['proof_failures'],[])
                self.assertEqual(len(c.planes),3 if method>=2 else 4)
                wrong=replace(c,planes=(replace(c.planes[0],weight=c.planes[0].weight+1),)+c.planes[1:])
                self.assertTrue(verify(wrong)['proof_failures'])
        for value in (17,-17,1.5):
            with self.assertRaises(ValueError):encode(value)

if __name__=='__main__':unittest.main()
