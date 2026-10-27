import copy
import unittest
from fractions import Fraction
from rank48_certificate import ROOT,header,load,verify
from rank48_slp import programs,header as slp_header,compile_stage

class Rank48Certificate(unittest.TestCase):
    def test_canonical_slp_maps_counts_and_live_schedule(self):
        stages=programs()
        self.assertEqual(sum(len(stage['steps']) for stage in stages),204)
        self.assertEqual((ROOT/'src/backend_common/rank48_slp.hpp').read_text(encoding='utf-8'),slp_header(stages))
        for text in ('o0:=i0*i1;','o0:=i0/3;','o0:=unknown;','o0:=__import__("os");'):
            with self.subTest(text=text),self.assertRaises(ValueError):compile_stage(text,2,1)
    def test_exact_rational_tensor_and_reproducible_header(self):
        left,right,output=load();verify(left,right,output)
        self.assertEqual((ROOT/'src/backend_common/rank48_program.hpp').read_text(encoding='utf-8'),header(left,right,output))

    def test_coefficient_and_product_order_mutations_are_detected(self):
        original=load()
        for matrix,row,col in ((0,0,0),(1,0,0),(2,0,0)):
            values=copy.deepcopy(original);values[matrix][row][col]+=Fraction(1,2)
            with self.assertRaises(ValueError):verify(*values)
        values=copy.deepcopy(original);values[1][0],values[1][1]=values[1][1],values[1][0]
        with self.assertRaises(ValueError):verify(*values)

if __name__=='__main__':unittest.main()
