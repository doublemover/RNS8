import copy
import unittest
from fractions import Fraction
from rank48_certificate import ROOT,header,load,verify

class Rank48Certificate(unittest.TestCase):
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
