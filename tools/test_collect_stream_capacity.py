import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from collect_stream_capacity import M,N,K,P,BASE,MAXIMUM,BOUND,MINIMUM_PREFIX,TILES,PANELS,expected_checksums,main,validate
from benchmark_schema.wide_magnitude import MODULI
import math


def fixture(mode='serial'):
    pinned=0 if mode=='serial' else 16*2**20
    caller=8*(M*(K+3)+K*(N+3)+M*(N+3)+M*N+15*(K+3))
    phases=[]
    for i,checksum in enumerate(expected_checksums()):
        phases.append(dict(source_a_generation=2 if i==2 else 1,epoch=i+1,checked_outputs=M*N,
            all_output_and_padding_exact=True,weighted_checksum=checksum,completed_tiles=TILES,
            completed_panels=TILES*PANELS,h2d_payload_bytes=8*(M*K+TILES*K*N),d2h_payload_bytes=8*M*N+4*TILES,
            h2d_transfers=2*TILES*PANELS,d2h_transfers=2*TILES,completed_host_call_ns=1000,
            observed_requested_device_peak_delta=32*2**20,observed_requested_pinned_peak_delta=pinned))
    return dict(schema='rns8_stream_physical_capacity_v1',mode=mode,run_id='fixture',m=M,n=N,k=K,prefix=P,
        semantic_contract='bounded_i64_explicit_k_amax_bmax',max_abs_a=MAXIMUM,max_abs_b=MAXIMUM,
        full_expression_abs_bound=BOUND,minimum_proven_prefix=MINIMUM_PREFIX,
        all_owned_device_and_pinned_payload_retired=True,
        independent_oracle_inside_timer=False,promotion_eligible=False,process_rss_measured=False,device_id=0,
        ordered_moduli=list(MODULI[:P]),basis_product_decimal=str(math.prod(MODULI[:P])),
        actual_total_device_bytes=24*2**30,caller_payload_bytes=caller,actual_caller_capacity_bytes=caller,
        caller_budget_bytes=40*2**30,available_host_bytes_before=90*2**30,host_reserve_bytes=8*2**30,
        a_native_allocation_bytes=M*(K+3)*8,full_a_residue_bytes=M*K*P,device_budget_bytes=64*2**20,
        minimum_full_a_residue_bytes=M*K*MINIMUM_PREFIX,
        library_host_budget_bytes=64*2**20,build_manifest=dict(version=2,source_dirty=False,
            source_commit='a'*40,source_digest='b'*64,dependency_digest='c'*64,linkage='static-rns8',
            identity_scope='source-config-and-external-headers',configuration='hip=ON',external_dependencies={}),
        target='gfx1100',device='synthetic-test-not-hardware',hip_runtime_version=70100000,hip_driver_version=70100000,
        queried_resources=dict(output_tiles=TILES,k_panels_per_tile=PANELS,device_bytes=32*2**20,
            host_bytes=32*2**20,pinned_host_bytes=pinned,ordinary_host_bytes=32*2**20-pinned),phases=phases)


class CapacityEvidence(unittest.TestCase):
    def test_complete_capacity_contract_for_both_schedules(self):
        for mode in ('serial','overlap'):validate(fixture(mode),mode,'fixture')
        a,b,c=map(int,expected_checksums());self.assertEqual(a,b);self.assertEqual(c-a,-2*BASE-12)
        self.assertEqual(MINIMUM_PREFIX,8)

    def test_numerical_coverage_budget_and_physical_capacity_mutations(self):
        mutations=[lambda d:d.update(actual_total_device_bytes=128*2**30),
            lambda d:d.update(caller_payload_bytes=1),lambda d:d.update(actual_caller_capacity_bytes=1),
            lambda d:d.update(available_host_bytes_before=1),lambda d:d.update(all_owned_device_and_pinned_payload_retired=False),
            lambda d:d['build_manifest'].update(source_dirty=True),lambda d:d['build_manifest'].update(configuration='hip=OFF'),
            lambda d:d['phases'].pop(),lambda d:d['phases'][2].update(weighted_checksum=d['phases'][0]['weighted_checksum']),
            lambda d:d['phases'][0].update(completed_panels=TILES*PANELS-1),
            lambda d:d['phases'][0].update(h2d_payload_bytes=8*M*K),
            lambda d:d['phases'][0].update(observed_requested_device_peak_delta=0),
            lambda d:d['phases'][0].update(observed_requested_device_peak_delta=65*2**20),
            lambda d:d['phases'][0].update(completed_host_call_ns=float('nan'))]
        good=fixture()
        for mutate in mutations:
            bad=copy.deepcopy(good);mutate(bad)
            with self.assertRaises(ValueError):validate(bad,'serial','fixture')
        with self.assertRaises(ValueError):validate(good,'serial','wrong-run')
        with self.assertRaises(ValueError):validate(good,'serial','fixture',1)

    def test_skip_or_timeout_retains_failure_and_stops_without_retry(self):
        for code,expected in ((77,'unrun'),(None,'failed')):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);binary=root/'binary';binary.write_bytes(b'fixture')
                output=root/'output'
                def run(command,path,timeout):
                    path.with_suffix('.failed.json').write_text(json.dumps(dict(returncode=code,timed_out=code is None,command=command)))
                    return False
                argv=['collect_stream_capacity.py','--binary',str(binary),'--output',str(output),
                    '--caller-budget-bytes','1','--device-budget-bytes','1','--host-budget-bytes','1']
                with patch('sys.argv',argv),patch('collect_stream_capacity.run_command',side_effect=run) as launch:
                    self.assertEqual(main(),77 if code==77 else 1);self.assertEqual(launch.call_count,1)
                receipt=json.loads((output/'receipt.json').read_text())
                self.assertEqual(receipt['status'],expected);self.assertEqual(len(receipt['records']),1)


if __name__=='__main__':unittest.main()
