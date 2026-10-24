"""Authored exact import/schedule fixtures; no GPU or performance qualification."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
import unittest

from rank23_slp import (BLOB, CERTIFICATE, compile_stage, evaluate, expansion,
                       full_plans, load_pinned, multiply, report, verify_certificate)


class Rank23SLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_pinned()
        cls.plans = full_plans(cls.data)

    def test_pinned_artifact_line_endings_and_mutation(self):
        raw = CERTIFICATE.read_bytes().replace(b"\r\n",b"\n").rstrip(b"\n")
        upstream = raw.replace(b"\n",b"\r\n")
        self.assertEqual(hashlib.sha1(b"blob "+str(len(upstream)).encode()+b"\0"+upstream).hexdigest(),BLOB)
        self.assertEqual(len(upstream),8649)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"certificate.json"
            for content in (raw,raw+b"\n",upstream):
                path.write_bytes(content)
                self.assertEqual(load_pinned(path),self.data)
            path.write_bytes(raw.replace(b'"rank": 23',b'"rank": 22'))
            with self.assertRaisesRegex(ValueError,"pinned upstream"):
                load_pinned(path)

    def test_independent_integer_expansion_and_basis_additions(self):
        stages = verify_certificate(self.data)
        self.assertEqual([stages[k].source_additions for k in ("U","V","W")],[13,12,26])
        self.assertEqual([self.plans[k].source_additions for k in ("U","V","W")],[14,15,27])
        for key in ("U","V","W"):
            self.assertEqual(expansion(self.plans[key]),self.data["original_matrices"][key])
            self.assertLessEqual(self.plans[key].slots,64)

    def test_original_mutable_names_and_cse(self):
        stage = compile_stage(["u0:=x0+x1;","u1:=x1+x0;","u0:=u0-x1;"],
                              ["x0","x1"],["u0","u1"])
        self.assertEqual(stage.source_additions,3)
        self.assertEqual(len(stage.nodes),2)
        self.assertEqual(evaluate(stage,[11,7]),[11,18])
        self.assertEqual(stage,compile_stage(["u0:=x0+x1;","u1:=x1+x0;","u0:=u0-x1;"],
                                             ["x0","x1"],["u0","u1"]))
        # Two distinct values of u22/v12 must survive their exact upstream order.
        for key, target in (("U","u22"),("V","v12")):
            changed = copy.deepcopy(self.data)
            lines = changed["slp"][key]
            at = [i for i,line in enumerate(lines) if line.startswith(target+":=")]
            self.assertEqual(len(at),2)
            lines[at[0]],lines[at[1]] = lines[at[1]],lines[at[0]]
            with self.assertRaises(ValueError): verify_certificate(changed)

    def test_no_early_slot_reuse_for_pinned_outputs(self):
        stage = compile_stage(["u0:=x0;","u1:=x0+x1;","u2:=u1-x1;","u3:=u1+u2;"],
                              ["x0","x1"],["u0","u1","u2","u3"])
        self.assertEqual(evaluate(stage,[3,5]),[3,8,3,11])
        self.assertEqual(len(set(stage.output_slots)),4)
        # Long chain reuses a last-read operand, rather than allocating per node.
        lines = ["u0:=x0+x1;"]+[f"u{i}:=u{i-1}-x1;" for i in range(1,50)]
        chain = compile_stage(lines,["x0","x1"],["u49"])
        self.assertEqual(chain.slots,2)
        self.assertEqual(evaluate(chain,[3,5]),[3+5-49*5])

    def test_malformed_grammar_and_missing_evidence(self):
        for lines in (["u0:=x0*x1;"],["u0:=x0//x1;"],["u0:=__import__(x0);"],
                      ["u0:=x0+1;"],["u0:=x0+-x1;"],["u0:=x0"],
                      ["u0:=u1+x0;"],[True],["u1:=x0;"]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                compile_stage(lines,["x0","x1"],["u0"])
        for key in ("basis","original_matrices","changed_matrices","slp"):
            data=copy.deepcopy(self.data); del data[key]
            with self.assertRaises(ValueError): verify_certificate(data)
        for shape in ([],[3,3], [3,3,4],[3.0,3,3]):
            data=copy.deepcopy(self.data); data["dimension"]=shape
            with self.assertRaises(ValueError): verify_certificate(data)
        for bad in (True,23.0,"23",None):
            data=copy.deepcopy(self.data); data["rank"]=bad
            with self.assertRaises(ValueError): verify_certificate(data)
        for modulus in (1,257,True,2.0):
            with self.assertRaises(ValueError): evaluate(self.plans["U"],[0]*9,modulus)
        with self.assertRaises(ValueError): evaluate(self.plans["U"],[0]*8)

    def test_individual_coefficient_and_basis_mutations_rejected(self):
        for group in ("original_matrices","changed_matrices","basis"):
            for key, rows in self.data[group].items():
                for i,row in enumerate(rows):
                    for j,value in enumerate(row):
                        changed=copy.deepcopy(self.data)
                        changed[group][key][i][j] = 1 if value == 0 else 0
                        with self.subTest(group=group,key=key,i=i,j=j), self.assertRaises(ValueError):
                            verify_certificate(changed)
        for key in ("U","V","W"):
            for i,line in enumerate(self.data["slp"][key]):
                if "+" not in line and "-" not in line: continue
                changed=copy.deepcopy(self.data)
                changed["slp"][key][i]=line.replace("+","-",1) if "+" in line else line.replace("-","+",1)
                with self.subTest(key=key,i=i),self.assertRaises(ValueError):
                    verify_certificate(changed)

    def test_cpp_lowered_data_matches_upstream_and_basis_exactly(self):
        text = (CERTIFICATE.parents[2]/"src/backend_common/rank23_slp_data.hpp").read_text()
        for key,prefix,out,ni,no,before,after in (
            ("U","x","u",9,23,["x8:=x8-x6;"],[]),
            ("V","y","v",9,23,["y2:=y2+y0;","y6:=y6-y7;","y8:=y8+y5;"],[]),
            ("W","p","z",23,9,[],["z8:=z8+z2;"])):
            block = re.search(r"raw_"+key.lower()+r"\(\).*?return \{(.*?)\};",text,re.S).group(1)
            declared_inputs,declared_outputs,count=map(int,re.match(r"\s*(\d+),\s*(\d+),\s*(\d+)",block).groups())
            instructions=[(int(a),int(b),int(c),d=="true") for a,b,c,d in
                          re.findall(r"\{(\d+), (\d+), (-?\d+), (true|false)\}",block)]
            def reg(name):
                letter,i=name[0],int(name[1:])
                return i if letter==prefix else (23+i if letter=="t" else 30+i) if key=="W" else 9+i
            expected=[]
            for line in before+self.data["slp"][key]+after:
                destination,expression=line[:-1].split(":=")
                tokens=re.findall(r"[a-z][0-9]+|[+-]",expression)
                target,left=reg(destination),reg(tokens[0])
                if len(tokens)==1: expected.append((target,left,-1,False))
                for i in range(1,len(tokens),2):
                    expected.append((target,left,reg(tokens[i+1]),tokens[i]=="-"));left=target
            self.assertEqual((declared_inputs,declared_outputs,count),(ni,no,len(expected)))
            self.assertEqual(instructions,expected)
            results = list(map(int,re.findall(r"\d+",block.rsplit("}, {",1)[1])))
            self.assertEqual(results,[reg(out+str(i)) for i in range(no)])

    def test_exact_basis_tensors_and_all_byte_rings(self):
        def classical(a,b):
            return [sum(a[r*3+k]*b[k*3+c] for k in range(3)) for r in range(3) for c in range(3)]
        for i in range(9):
            for j in range(9):
                a,b=[int(k==i) for k in range(9)],[int(k==j) for k in range(9)]
                self.assertEqual(multiply(self.plans,a,b),classical(a,b))
        rng=random.Random(234851)
        cases=[([-128]*9,[127,-128,127]*3),([127]*9,[127]*9),([0]*9,[0]*9)]
        cases += [([rng.randrange(-128,128) for _ in range(9)],
                   [rng.randrange(-128,128) for _ in range(9)]) for _ in range(16)]
        cases += [([(-1)**i*((1<<64)-1) for i in range(9)],[(1<<64)-1]*9)]
        for a,b in cases:
            expected=classical(a,b)
            self.assertEqual(multiply(self.plans,a,b),expected)
            for modulus in range(2,257):
                canonical=[x%modulus-modulus if x%modulus >= (modulus+1)//2 else x%modulus for x in expected]
                self.assertEqual(multiply(self.plans,a,b,modulus),canonical)

    def test_report_stable_proof_not_qualification(self):
        first=report(self.data)
        self.assertEqual(first,report(copy.deepcopy(self.data)))
        identity=first.pop("sha256")
        self.assertEqual(identity,hashlib.sha256(json.dumps(first,sort_keys=True,separators=(",",":")).encode()).hexdigest())
        self.assertEqual(first["device_execution"],"not_run")
        self.assertIsNone(first["performance"])
        self.assertFalse(first["runtime_dispatch_eligible"])
        changed=copy.deepcopy(self.data);changed["complexity"]["changed"]=50
        with self.assertRaisesRegex(ValueError,"pinned certificate"): report(changed)


if __name__ == "__main__":
    unittest.main()
