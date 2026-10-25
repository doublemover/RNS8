"""Synthetic mapping/arithmetic/model checks; no topology or hardware evidence."""
from dataclasses import replace
import random
import unittest

from locality_schedule import Descriptor, Schedule, Work, simulate


def descriptor(**changes):
    return replace(Descriptor(5, 7, 9, 2, 3, 4, 2, (1, 3, 0, 5, 2, 4, 1, 3, 5),
                              "native64", "same-output", "gfx1100", "single-device-control"),
                   **changes)


class LocalityTests(unittest.TestCase):
    def test_split_fragments_and_modulus_groups_preserve_shuffled_full_products(self):
        for fragments in (1,2,3,7):
            for group in (1,2,4):
                for order in ("output-neighbor","modulus","same-output"):
                    d=descriptor(k=9,slab_k=4,fragments=fragments,group_size=group,order=order)
                    schedule=Schedule(d);seen=set();partials={}
                    for phase in schedule.phases:
                        shuffled=list(phase);random.Random(3637).shuffle(shuffled)
                        for work in shuffled:
                            self.assertNotIn(work,seen);seen.add(work)
                            begin=d.k*work.fragment//fragments+work.slab*d.slab_k
                            extent=schedule.extents(work)
                            for r in range(work.row*d.tile_m,work.row*d.tile_m+extent[0]):
                                for c in range(work.col*d.tile_n,work.col*d.tile_n+extent[1]):
                                    for p in range(work.first_plane,work.first_plane+work.planes):
                                        key=(work.fragment,p,r,c)
                                        partials[key]=(partials.get(key,0)+sum((r-q)*(q+c) for q in range(begin,begin+extent[2])))%(p+2)
                    for r in range(d.m):
                        for c in range(d.n):
                            for p in range(d.prefixes[(r//d.tile_m)*schedule.cols+c//d.tile_n]):
                                self.assertEqual(sum(partials.get((f,p,r,c),0) for f in range(fragments))%(p+2),sum((r-q)*(q+c) for q in range(d.k))%(p+2))
                    report=simulate(schedule)
                    self.assertFalse(report['physical_affinity_verified']);self.assertEqual(report['fragments'],fragments)

    def test_bijection(self):
        for m in range(1, 5):
            for n in range(1, 6):
                rows, cols = (m + 1) // 2, (n + 1) // 2
                for group in (1, 2, 3, 4, 8):
                    prefixes = tuple((i * 7) % 21 for i in range(rows * cols))
                    expected = {Work(s, r, c, p, min(group, prefixes[r * cols + c] - p))
                                for s in range(3) for r in range(rows) for c in range(cols)
                                for p in range(0, prefixes[r * cols + c], group)}
                    for order in ("output-neighbor", "modulus", "same-output"):
                        schedule = Schedule(descriptor(m=m, n=n, k=5, tile_n=2, slab_k=2,
                                                       group_size=group, prefixes=prefixes, order=order))
                        seen = []
                        for phase, items in enumerate(schedule.phases):
                            for launch, work in enumerate(items):
                                self.assertEqual(schedule.decode(phase, launch), work)
                                self.assertEqual(schedule.encode(work), (phase, launch))
                                seen.append(work)
                        self.assertEqual(set(seen), expected)
                        self.assertEqual(len(seen), len(expected))
                        with self.assertRaises(ValueError):
                            schedule.decode(-1, 0)
                        with self.assertRaises(ValueError):
                            schedule.decode(0, len(schedule.phases[0]))

    def test_shuffled_arithmetic(self):
        d = descriptor()
        a = [[(r * 17 + k * 3) % 23 - 11 for k in range(d.k)] for r in range(d.m)]
        b = [[(k * 13 - c * 7) % 29 - 14 for c in range(d.n)] for k in range(d.k)]
        reference = {}
        for r in range(d.m):
            for c in range(d.n):
                prefix = d.prefixes[(r // d.tile_m) * 3 + c // d.tile_n]
                for p in range(prefix):
                    reference[p, r, c] = sum(a[r][k] * b[k][c] for k in range(d.k)) % (p + 2)
        for order in ("output-neighbor", "modulus", "same-output"):
            schedule = Schedule(replace(d, order=order))
            for seed in range(5):
                result = {}
                rng = random.Random(seed)
                # Permute execution within a phase only. Completion of the whole
                # phase is the explicit dependency before the next K slab.
                for phase in schedule.phases:
                    shuffled = list(phase)
                    rng.shuffle(shuffled)
                    for item in shuffled:
                        m, n, k = schedule.extents(item)
                        for r in range(item.row * d.tile_m, item.row * d.tile_m + m):
                            for c in range(item.col * d.tile_n, item.col * d.tile_n + n):
                                part = sum(a[r][kk] * b[kk][c]
                                           for kk in range(item.slab * d.slab_k, item.slab * d.slab_k + k))
                                for p in range(item.first_plane, item.first_plane + item.planes):
                                    key = p, r, c
                                    result[key] = (result.get(key, 0) + part) % (p + 2)
                self.assertEqual(result, reference)

    def test_traffic_and_warm_capacity(self):
        d = descriptor(m=2, n=2, k=2, tile_m=2, tile_n=2, slab_k=2,
                       group_size=1, prefixes=(2,), cache_bytes_per_domain=1000)
        cold = simulate(Schedule(d))
        self.assertEqual(cold["requested_source_bytes"], 128)
        self.assertEqual(cold["unique_source_bytes"], 64)
        self.assertEqual(cold["unique_bytes"], 72)
        self.assertEqual(cold["modeled_cache_miss_bytes"], 72)
        self.assertEqual(cold["modeled_cache_hit_bytes"], 64)
        self.assertEqual(simulate(Schedule(d), hot=True)["modeled_cache_miss_bytes"], 0)
        self.assertEqual(simulate(Schedule(replace(d, cache_bytes_per_domain=0)), hot=True)
                         ["modeled_cache_hit_bytes"], 0)
        packed = simulate(Schedule(replace(d, representation="residue8")))
        self.assertEqual(packed["requested_source_bytes"], 16)
        self.assertEqual(packed["unique_source_bytes"], 16)
        self.assertEqual(packed["modeled_cache_hit_bytes"], 0)
        self.assertFalse(packed["physical_affinity_verified"])
        self.assertFalse(packed["promotion_eligible"])
        self.assertIsNone(packed["actual_hbm_bytes"])
        parallel = simulate(Schedule(replace(d, cache_domains=3, concurrent_groups_per_domain=2)))
        self.assertEqual(parallel["empty_wave_slots"], 4)
        self.assertEqual(parallel["scheduling_waves"], 1)
        self.assertEqual(parallel["domain_work_items"], [1, 1, 0])

    def test_invalid_and_empty(self):
        for changes in ({"m": 0}, {"group_size": True}, {"prefixes": (1,)},
                        {"prefixes": (21,) * 9}, {"cache_domains": 0},
                        {"representation": "unknown"}, {"order": "xcd-pinned"},
                        {"cache_bytes_per_domain": -1}, {"version": 2}):
            with self.assertRaises(ValueError):
                Schedule(descriptor(**changes))
        schedule = Schedule(descriptor(prefixes=(0,) * 9))
        self.assertEqual(simulate(schedule)["scheduling_waves"], 0)
        with self.assertRaises(ValueError):
            schedule.encode(Work(0, 0, 0, 0, 1))



class RuntimeLocalityMetadataTests(unittest.TestCase):
    def capture(self):
        return {'k':65537,'locality_schedule':dict(version=1,requested_order=3,applied_order=3,target_id=942,topology_version=1,
            modulus_group_size=1,representation='resident_residue8',applied=True,physical_affinity_established=False,
            cache_domains_model=1,concurrent_groups_per_domain_model=4,cache_bytes_per_domain_model=0,
            active_entries=9,directory_bytes=720,k_phases=2,requested_source_bytes=900,unique_source_bytes=600,
            output_read_write_bytes=300,permutation_fingerprint=123,traffic_scope='active_arithmetic_only_excludes_pack_zero_fill_export',actual_hbm_bytes=None,measured_cache_hit_rate=None,promotion_eligible=False)}
    def test_schema_and_nonpromotion(self):
        import copy
        from benchmark_schema.locality import validate_locality
        from benchmark_sweep_lib.review import scenario_promotion_blockers
        capture=self.capture();self.assertEqual(validate_locality(capture),[])
        self.assertEqual(scenario_promotion_blockers(capture),['experimental_locality_workspace_not_runtime_auto_route'])
        for key,value in [('physical_affinity_established',True),('actual_hbm_bytes',900),('measured_cache_hit_rate',0.9),
                          ('promotion_eligible',True),('unique_source_bytes',901),('target_id',9999),('k_phases',1),('applied_order',0),
                          ('active_entries',True),('representation','native64')]:
            bad=copy.deepcopy(capture);bad['locality_schedule'][key]=value
            with self.subTest(key=key):self.assertTrue(validate_locality(bad))
        capture['locality_schedule'].update(applied=False,applied_order=0,target_id=0)
        self.assertEqual(validate_locality(capture),[])

if __name__ == "__main__":
    unittest.main()
