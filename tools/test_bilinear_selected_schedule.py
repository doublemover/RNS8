from fractions import Fraction
import unittest

from bilinear_selected_schedule import identity_and_work, lower, schedules


class SelectedSchedule(unittest.TestCase):
    def test_every_selected_map_matches_the_independent_pinned_coefficients(self):
        from rank48_certificate import load
        from rank23_october55 import ARTIFACT, expand, parse

        rational = load()[:2]
        integer = parse(ARTIFACT.read_bytes())
        for rank48 in (False, True):
            for side, plans in enumerate(schedules(rank48)):
                expected = rational[side] if rank48 else expand(integer["UV"[side]])[0]
                width = 16 if rank48 else 9
                for output, plan in enumerate(plans):
                    slots = [[Fraction(0)] * width for _ in range(plan["slots"])]
                    for index, source in enumerate(plan["input_ids"]):
                        slots[index][source] = Fraction(1)
                    for target, left, right, sa, sb, halve in plan["steps"]:
                        slots[target] = [
                            (sa * a + sb * b) / (2 if halve else 1)
                            for a, b in zip(slots[left], slots[right])
                        ]
                    with self.subTest(rank48=rank48, side=side, output=output):
                        self.assertEqual(slots[plan["result"]], expected[output])
            digest, work = identity_and_work(rank48)
            self.assertEqual(len(digest), 64)
            self.assertEqual((digest, work), identity_and_work(rank48))
            self.assertLess(sum(work["binary"]), 48 * 114 if rank48 else 23 * 27)

    def test_pruning_retains_earlier_definitions_and_can_return_an_input(self):
        nodes = [(0, 1, 1, 1, False), (2, 0, 1, -1, False), (0, 1, 1, -1, False)]
        plan = lower(2, nodes, 3)
        self.assertEqual(len(plan["steps"]), 2)
        self.assertEqual(lower(2, nodes, 0), dict(input_ids=[0], steps=[], slots=1, result=0))
        for bad in ([(5, 0, 1, 1, False)], [(0, -2, 1, 1, False)], [(0, 1, 1, 1, True)]):
            with self.assertRaises(ValueError):
                lower(2, bad, 0)


if __name__ == "__main__":
    unittest.main()
