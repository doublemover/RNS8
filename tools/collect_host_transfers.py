"""Matched public host-transfer calls; no route promotion or inferred overlap."""
import argparse
import json
import math
from pathlib import Path
import statistics
import uuid
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from collect_native_calibration import strict_json
from collect_int4_calibration import collection_plan
from collect_native_interop import quantile
from collect_finite_verify import demand,integer,same

MODES=('pageable-sync','pinned-sync','pinned-stream','pinned-graph')
KEYS={'schema','device','m','n','k','unsigned_inputs','dataset','mode','repeats','warmups','seed','max_payload_bytes'}


def envelope(c):
    m,n,k=(c[key] for key in ('m','n','k'));ac,bc,cc=m*k,k*n,m*n
    host=8*(m*(k+3)+k*(n+3)+m*(n+3)*3)
    return 2*host+128*cc+20*(ac+bc+cc)+16*(ac+bc)+48*cc+20*(ac+bc)+65536


def validate_case(c):
    demand(type(c) is dict and set(c)==KEYS and c['schema']=='host_transfer_case_v1','case ABI')
    for key,low,high in (('device',0,63),('m',1,512),('n',1,512),('k',1,65537),('repeats',3,100),
            ('warmups',1,100),('seed',0,2**32-1),('max_payload_bytes',8,512*1024*1024)):
        demand(integer(c[key],low,high),'invalid '+key)
    demand(type(c['unsigned_inputs']) is bool and c['mode'] in MODES and c['dataset'] in ('small','fullwidth'),'typed route/dataset')
    demand(c['m']*c['n']*c['k']<=134217728 and envelope(c)<=c['max_payload_bytes'],'work/payload admission')
    return c


def validate_sample(s):
    demand(type(s) is dict and set(s)=={'enqueue_ns','observation_ns','total_ns','phase_censored'},'sample fields')
    demand(integer(s['enqueue_ns'],0,2**63-1) and integer(s['observation_ns'],0,2**63-1) and integer(s['total_ns'],1,2**63-1),'sample units')
    demand(same(s['phase_censored'],s['enqueue_ns']==0 or s['observation_ns']==0),'phase censoring')
    demand(abs(s['enqueue_ns']+s['observation_ns']-s['total_ns'])<=2,'complete call accounting')


def validate_capture(c,entry):
    case=validate_case(entry['case']);mode=case['mode'];pinned=mode!='pageable-sync';stream=mode in ('pinned-stream','pinned-graph');graph=mode=='pinned-graph'
    m,n,k=(case[key] for key in ('m','n','k'));ac,bc,cc=m*k,k*n,m*n
    fixed=dict(schema='rns8_host_transfer_microcase_v1',case=case,run_id=entry['run_id'],device_actual=case['device'],
        output_boundary='readable_host_padded_three_limb_result',prefix=20,input_bound_policy='full_native64_envelope' if pinned else 'engine_host_value_scan',
        verify_input_bound=False,graph_requested=graph,graph_executed=graph,prepared_owners=4 if pinned else 0,
        pinned_buffers=3 if pinned else 0,borrowed_streams=4 if stream else 0,setup_bootstrap_import_calls=2 if pinned else 0,
        setup_bootstrap_gemm_calls=2 if pinned else 1,setup_scope='construction_initial_proof_materialization_and_optional_capture',
        timing_source='steady_clock_complete_host_observed_call_ns',phase_scope='submission_then_terminal_observation_sync_submission_includes_completion',
        production_timing_hooks_enabled=False,oracle_inside_interval=False,caller_initialization_inside_interval=False,observed_overlap=None,
        ordinary_export_pinned_staging='disabled_explicit_control',reference_reservation_bytes=128*cc,
        opaque_runtime_setup_unknown=True,capacity_scope='queried_retained_payloads_not_measured_peak_or_driver_allocations',
        exact_reference_passed=True,promotion_eligible=False)
    for key,value in fixed.items():demand(same(c.get(key),value),'capture '+key)
    demand(manifest_provenance_complete(c.get('build_manifest')) and c.get('target') and c.get('device') and c.get('execution_id'),'build/device/process identity')
    artifact=c.get('artifact_capture') or {}
    demand(artifact.get('unchanged_during_run') is True and type(artifact.get('executable_sha256')) is str and len(artifact['executable_sha256'])==64,'artifact identity')
    demand(integer(c.get('runtime_version'),1,2**63-1),'runtime identity')
    for key in ('setup_ns','cleanup_ns'):demand(integer(c.get(key),1,2**63-1),'lifecycle '+key)
    demand(integer(c.get('capture_ns'),1 if graph else 0,2**63-1 if graph else 0),'actual capture cost')
    matrices=c.get('matrix_allocations');transfers=c.get('transfer_allocations')
    demand(type(matrices) is list and len(matrices)==3 and type(transfers) is list and len(transfers)==(3 if pinned else 0),'allocation boundaries')
    for row,cells in zip(matrices,(ac,bc,cc)):
        demand(type(row) is dict and set(row)=={'host_bytes','device_bytes'} and integer(row['host_bytes'],0,case['max_payload_bytes']) and
            integer(row['device_bytes'],20*cells,case['max_payload_bytes']),'matrix payload query')
    if pinned:
        expected=[]
        for cells,padded,importing in ((ac,m*(k+3)*8,True),(bc,k*(n+3)*8,True),(cc,m*(n+3)*24,False)):
            expected.append(dict(operation=21 if importing else 22,device_bytes=28*cells+16 if importing else 48*cells+4,
                pinned_bytes=padded+4,retained_native_bytes=0,verified_input_bound=0))
        demand(same(transfers,expected),'complete queried transfer storage')
    workspace=c.get('plan_workspace_bytes');demand(integer(workspace,0,case['max_payload_bytes']),'workspace')
    logical_host=8*(m*(k+3)+k*(n+3)+m*(n+3)*3);caller=c.get('caller_host_payload_capacity_bytes')
    demand(integer(caller,logical_host,case['max_payload_bytes']),'caller capacity')
    ordinary=caller+sum(row['host_bytes'] for row in matrices)
    device=workspace+sum(row['device_bytes'] for row in matrices)+sum(row['device_bytes'] for row in transfers)
    pins=sum(row['pinned_bytes'] for row in transfers)
    demand(same(c.get('ordinary_host_payload_bytes'),ordinary) and same(c.get('device_payload_bytes'),device) and
        same(c.get('pinned_host_payload_bytes'),pins),'payload totals')
    demand(ordinary+device+pins+128*cc<=case['max_payload_bytes'] and
        same(c.get('preflight_payload_envelope_bytes'),envelope(case)+workspace+caller-logical_host) and
        envelope(case)+workspace+caller-logical_host<=case['max_payload_bytes'],'hard payload envelope')
    phases=c.get('phases');demand(type(phases) is list and len(phases)==2,'two source generations')
    calls=0
    for generation,phase in enumerate(phases,1):
        demand(same(phase.get('generation'),generation) and phase.get('binding_action')==('initial_inputs' if generation==1 else 'changed_A0_same_handles'),'generation binding')
        demand(integer(phase.get('rebind_ns'),0,2**63-1) and phase.get('full_output_and_padding_passed') is True and
            same(phase.get('independent_audits'),case['repeats']+5),'outside-timer full audits')
        validate_sample(phase.get('first_call'))
        samples=phase.get('samples');bursts=phase.get('sustained')
        demand(type(samples) is list and len(samples)==case['repeats'] and type(bursts) is list and len(bursts)==3,'sampling contract')
        for sample in samples:validate_sample(sample)
        executed=1+case['warmups']+case['repeats']
        for burst in bursts:
            demand(type(burst) is dict and set(burst)=={'calls','elapsed_ns','mean_call_ns','final_exact_reference_passed'},'sustained fields')
            count,time=burst['calls'],burst['elapsed_ns'];mean=burst['mean_call_ns']
            demand(integer(count,16,4096) and integer(time,1,2**63-1) and (time>=250000000 or count==4096) and
                type(mean) in (int,float) and math.isfinite(mean) and mean>0 and math.isclose(mean*count,time,rel_tol=1e-12) and
                burst['final_exact_reference_passed'] is True,'sustained completion/measurement')
            executed+=count
        demand(same(phase.get('production_calls'),executed),'executed calls per generation');calls+=executed
    demand(same(c.get('production_calls'),calls),'complete production calls');return c


def summarize(observations):
    cohorts={}
    for entry,capture in observations:
        c=validate_capture(capture,entry)
        for phase in c['phases']:
            identity=dict(case={key:value for key,value in entry['case'].items() if key!='mode'},generation=phase['generation'],
                build=c['build_manifest'],target=c['target'],device=c['device'],runtime=c['runtime_version'],
                artifact={k:c['artifact_capture'].get(k) for k in ('executable_sha256','rns8_linkage','shared_library_sha256')},
                environment=(c.get('request_identity') or {}).get('environment'))
            cohorts.setdefault(json.dumps(identity,sort_keys=True),[]).append(dict(mode=entry['case']['mode'],
                execution_id=c['execution_id'],independent_run=entry['independent_run'],path=entry['output'],setup_ns=c['setup_ns'],
                capture_ns=c['capture_ns'],p50_ns=quantile([s['total_ns'] for s in phase['samples']],.5),
                p95_ns=quantile([s['total_ns'] for s in phase['samples']],.95),
                sustained_ns=statistics.median(s['mean_call_ns'] for s in phase['sustained'])))
    output=[]
    for identity,rows in sorted(cohorts.items()):
        demand(len({r['execution_id'] for r in rows})==len(rows),'duplicate process observation')
        metrics={}
        for mode in MODES:
            selected=[r for r in rows if r['mode']==mode]
            if selected:
                demand(len({r['independent_run'] for r in selected})>=2,'at least two independent processes per mode')
                metrics[mode]={key:statistics.median(r[key] for r in selected) for key in ('setup_ns','capture_ns','p50_ns','p95_ns','sustained_ns')}
        gains={}
        if 'pageable-sync' in metrics:
            for mode in MODES[1:]:
                if mode in metrics:
                    ratios={key:metrics['pageable-sync'][key]/metrics[mode][key] for key in ('p50_ns','p95_ns','sustained_ns')}
                    if all(value>1 for value in ratios.values()):gains[mode]=ratios
        output.append(dict(identity=json.loads(identity),observations=rows,process_medians=metrics,
            beneficial_speedups_over_pageable_sync=gains,complete_modes=set(MODES)<=set(metrics),promotion_eligible=False,
            qualification='not_inferred_from_api_submission_or_stream_count'))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--cases',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=int,default=2);p.add_argument('--seed',type=int,default=4343)
    p.add_argument('--timeout-seconds',type=float,default=1200);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    plan=collection_plan(a.binary,a.cases,a.output,a.runs,a.seed,str(uuid.uuid4()),case_validator=validate_case)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'plan.json').open('x') as f:json.dump(dict(schema='host_transfer_collection_v1',observations=plan,promotion_eligible=False),f,indent=2)
    if a.plan_only:return 0
    captures=[]
    for entry in plan:
        path=Path(entry['output']);demand(not path.exists(),'immutable capture exists')
        demand(run_command(entry['command'],path,a.timeout_seconds),'capture failed: '+str(path))
        captures.append((entry,strict_json(path.read_text())))
    with (a.output/'summary.json').open('x') as f:json.dump(dict(schema='host_transfer_summary_v1',cohorts=summarize(captures),promotion_eligible=False),f,indent=2)
    return 0


if __name__=='__main__':raise SystemExit(main())
