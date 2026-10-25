"""Synthetic receipts validate collection contracts; timings are not evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_host_transfers import MODES,envelope,validate_case,validate_capture,summarize


def case(mode='pageable-sync'):
    return dict(schema='host_transfer_case_v1',device=0,m=2,n=3,k=5,unsigned_inputs=False,dataset='small',mode=mode,
        repeats=3,warmups=1,seed=43,max_payload_bytes=67108864)


def fixture(mode='pageable-sync',run=0):
    c=case(mode);m,n,k=(c[key] for key in ('m','n','k'));ac,bc,cc=m*k,k*n,m*n
    pinned=mode!='pageable-sync';stream=mode in ('pinned-stream','pinned-graph');graph=mode=='pinned-graph'
    sample=dict(enqueue_ns=8000,observation_ns=2000,total_ns=10000,phase_censored=False)
    matrices=[dict(host_bytes=0,device_bytes=20*cells) for cells in (ac,bc,cc)];transfers=[]
    if pinned:
        for cells,padded,importing in ((ac,m*(k+3)*8,True),(bc,k*(n+3)*8,True),(cc,m*(n+3)*24,False)):
            transfers.append(dict(operation=21 if importing else 22,device_bytes=28*cells+16 if importing else 48*cells+4,
                pinned_bytes=padded+4,retained_native_bytes=0,verified_input_bound=0))
    phases=[]
    for generation in (1,2):
        phases.append(dict(generation=generation,binding_action='initial_inputs' if generation==1 else 'changed_A0_same_handles',
            rebind_ns=0,first_call=deepcopy(sample),samples=[deepcopy(sample) for _ in range(3)],
            sustained=[dict(calls=16,elapsed_ns=320000000,mean_call_ns=20000000.,final_exact_reference_passed=True) for _ in range(3)],
            production_calls=53,independent_audits=8,full_output_and_padding_passed=True))
    capture=dict(schema='rns8_host_transfer_microcase_v1',case=c,run_id=f'fixture-{mode}-{run}',build_manifest={},
        target='gfx1100',device='synthetic mocked GPU',device_actual=0,runtime_version=70100000,
        output_boundary='readable_host_padded_three_limb_result',prefix=20,input_bound_policy='full_native64_envelope' if pinned else 'engine_host_value_scan',
        verify_input_bound=False,graph_requested=graph,graph_executed=graph,prepared_owners=4 if pinned else 0,
        pinned_buffers=3 if pinned else 0,borrowed_streams=4 if stream else 0,setup_bootstrap_import_calls=2 if pinned else 0,
        setup_bootstrap_gemm_calls=2 if pinned else 1,setup_scope='construction_initial_proof_materialization_and_optional_capture',
        setup_ns=10000,capture_ns=500 if graph else 0,cleanup_ns=1000,timing_source='steady_clock_complete_host_observed_call_ns',
        phase_scope='submission_then_terminal_observation_sync_submission_includes_completion',production_timing_hooks_enabled=False,
        oracle_inside_interval=False,caller_initialization_inside_interval=False,observed_overlap=None,
        ordinary_export_pinned_staging='disabled_explicit_control',matrix_allocations=matrices,transfer_allocations=transfers,
        plan_workspace_bytes=0,ordinary_host_payload_bytes=8*(m*(k+3)+k*(n+3)+m*(n+3)*3),
        caller_host_payload_capacity_bytes=8*(m*(k+3)+k*(n+3)+m*(n+3)*3),
        pinned_host_payload_bytes=sum(t['pinned_bytes'] for t in transfers),
        device_payload_bytes=sum(t['device_bytes'] for t in transfers)+sum(t['device_bytes'] for t in matrices),
        reference_reservation_bytes=128*cc,preflight_payload_envelope_bytes=envelope(c),opaque_runtime_setup_unknown=True,
        capacity_scope='queried_retained_payloads_not_measured_peak_or_driver_allocations',production_calls=106,phases=phases,
        exact_reference_passed=True,promotion_eligible=False,execution_id=f'process-{mode}-{run}',
        artifact_capture=dict(unchanged_during_run=True,executable_sha256='a'*64))
    entry=dict(case=c,run_id=capture['run_id'],independent_run=run,output=f'fixture-{mode}-{run}.json')
    return entry,capture


class HostTransfers(unittest.TestCase):
    def test_case_admission_and_manifests(self):
        for mode in MODES:self.assertEqual(validate_case(case(mode)),case(mode))
        for key,value in [('device',True),('m',0),('k',65538),('max_payload_bytes',8),('unsigned_inputs',1),('mode','auto')]:
            bad=case();bad[key]=value
            with self.assertRaises(ValueError):validate_case(bad)
        root=Path(__file__).resolve().parents[1]/'benchmarks/host-transfers'
        groups={}
        for path in root.glob('*.json'):
            c=validate_case(json.loads(path.read_text()));key=tuple(c[k] for k in ('m','n','k','dataset','unsigned_inputs'))
            groups.setdefault(key,set()).add(c['mode'])
        self.assertTrue(groups);self.assertTrue(all(modes==set(MODES) for modes in groups.values()))

    @patch('collect_host_transfers.manifest_provenance_complete',return_value=True)
    def test_full_generation_and_resource_receipts(self,_manifest):
        for mode in MODES:
            entry,capture=fixture(mode);validate_capture(capture,entry)
            for field,value in [('prefix',9),('oracle_inside_interval',True),('observed_overlap',True),('graph_executed',not capture['graph_executed']),
                    ('ordinary_host_payload_bytes',0),('pinned_host_payload_bytes',1),('production_calls',1),('preflight_payload_envelope_bytes',1)]:
                bad=deepcopy(capture);bad[field]=value
                with self.assertRaises(ValueError):validate_capture(bad,entry)
            bad=deepcopy(capture);bad['phases'][1]['generation']=1
            with self.assertRaises(ValueError):validate_capture(bad,entry)
            bad=deepcopy(capture);bad['phases'][0]['independent_audits']=1
            with self.assertRaises(ValueError):validate_capture(bad,entry)
            bad=deepcopy(capture);bad['phases'][0]['samples'][0]['phase_censored']=True
            with self.assertRaises(ValueError):validate_capture(bad,entry)

    @patch('collect_host_transfers.manifest_provenance_complete',return_value=True)
    def test_process_cohorts_preserve_generations_and_reject_duplicates(self,_manifest):
        rows=[fixture(mode,run) for mode in MODES for run in range(2)]
        summary=summarize(rows);self.assertEqual(len(summary),2)
        self.assertTrue(all(g['complete_modes'] for g in summary));self.assertTrue(all(len(g['observations'])==8 for g in summary))
        self.assertTrue(all(not g['beneficial_speedups_over_pageable_sync'] for g in summary))
        with self.assertRaises(ValueError):summarize(rows+[rows[0]])
        with self.assertRaises(ValueError):summarize([fixture()])


@unittest.skipUnless(os.environ.get('RNS8_HOST_TRANSFER_EXE'),'compiled CLI supplied by final build')
class CompiledCli(unittest.TestCase):
    def test_help_and_pre_context_rejections(self):
        exe=os.environ['RNS8_HOST_TRANSFER_EXE']
        self.assertEqual(subprocess.run([exe,'--help'],capture_output=True,timeout=10).returncode,0)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'case.json'
            for key,value in [('device',64),('m',0),('unsigned_inputs',1),('mode','automatic'),('max_payload_bytes',8)]:
                c=case();c[key]=value;path.write_text(json.dumps(c))
                self.assertEqual(subprocess.run([exe,str(path),'fixture-run'],capture_output=True,timeout=10).returncode,1)


if __name__=='__main__':unittest.main()
