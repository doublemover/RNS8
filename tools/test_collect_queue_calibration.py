import copy
import json
from pathlib import Path
import tempfile
import unittest
from collect_queue_calibration import MODES, commands, validate_case

ROOT=Path(__file__).resolve().parents[1]
class QueueCollectionTests(unittest.TestCase):
    def test_repository_controls_and_independent_order(self):
        cases=sorted((ROOT/"benchmarks/native-queue").glob("*.json"))
        self.assertGreaterEqual(len(cases),7)
        for path in cases:validate_case(json.loads(path.read_text()))
        plan=commands(Path("binary"),cases,Path("output"),4,38,"collection")
        self.assertEqual(len(plan),len(cases)*16)
        self.assertEqual(len({e["run_id"] for e in plan}),len(plan))
        for path in cases:
            own=[e for e in plan if e["command"][1]==str(path)]
            for run in range(4):self.assertEqual({e["mode"] for e in own if e["run"]==run},set(MODES))
            self.assertGreater(len({tuple(e["mode"] for e in own if e["run"]==run) for run in range(4)}),1)
    def test_invalid_abi_limits_and_bool_integer_aliases(self):
        good=json.loads((ROOT/"benchmarks/native-queue/many-tiny.json").read_text())
        for key,value in (("schema","v0"),("workers",True),("workers",1025),("chunk",33),("reuse",0),("tasks",[]),("mutate_b",1)):
            bad=copy.deepcopy(good);bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_case(bad)
        for key,value in (("m",True),("prefix",21),("group",3),("unsigned",1)):
            bad=copy.deepcopy(good);bad["tasks"][0][key]=value
            with self.assertRaises(ValueError):validate_case(bad)
        bad=copy.deepcopy(good);bad["unexpected"]=1
        with self.assertRaises(ValueError):validate_case(bad)
    def test_single_process_is_not_independent_cohort(self):
        with self.assertRaises(ValueError):commands(Path("b"),[],Path("o"),1,0,"id")

if __name__=="__main__":unittest.main()
