"""Matched complete device-native and explicit host-roundtrip application costs.

The unchanged-input resident control is a labeled input-binding-excluded cost
floor, never a candidate speedup. All reported ratios require independent runs
of the same build/artifact/environment and retain complete oracle validation.
"""
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
from collect_native_integrity import MODULI, positive

MODES = ('device-stream', 'device-graph', 'host-roundtrip', 'host-scalar-first', 'resident-control')


def validate_case(c):
    keys = {'schema','m','n','k','prefix','unsigned_inputs','mode','dataset','repeats','warmups','seed','max_payload_bytes'}
    if type(c) is not dict or set(c) != keys or c['schema'] != 'native_interop_case_v1' or type(c['unsigned_inputs']) is not bool:
        raise ValueError('case ABI')
    for key,lo,hi in [('m',1,512),('n',1,512),('k',1,65537),('prefix',1,20),('repeats',3,100),('warmups',1,100),('seed',0,2**32-1),('max_payload_bytes',8,2**40)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError(key)
    if c['mode'] not in MODES or c['dataset'] not in ('small','fullwidth'):
        raise ValueError('mode/dataset')
    m,n,k,p = (c[x] for x in ('m','n','k','prefix'))
    ac,bc,cc = m*k,k*n,m*(n+3)*3
    if m*n*k>134217728 or max((ac+bc)*8,cc*8,ac*p,bc*p)>2**31-1:
        raise ValueError('address/work')
    device = (ac+bc+m*n)*p+(ac+bc)*8+cc*8+8+(ac+bc)*p+32+m*n*24+4
    host = (ac+bc)*8+2*cc*8+12
    if device+host>c['max_payload_bytes']:
        raise ValueError('logical payload budget')
    bound = (2**64-1 if c['unsigned_inputs'] else 2**63) if c['dataset']=='fullwidth' else (14 if c['unsigned_inputs'] else 7)
    if math.prod(MODULI[:p]) <= k*bound*bound*(1 if c['unsigned_inputs'] else 2):
        raise ValueError('strict independent product range')
    return c


def validate_capture(c, entry):
    case=validate_case(entry['case']);m,n,k,p=(case[x] for x in ('m','n','k','prefix'));mode=case['mode'];device=mode.startswith('device-');resident=mode=='resident-control'
    ac,bc,cc=m*k,k*n,m*(n+3)*3
    fixed=dict(schema='rns8_native_interop_microcase_v4',case=case,run_id=entry['run_id'],
        exact_proof_policy='default_eager' if mode=='host-roundtrip' else 'scalar_first',
        compute_provider='public_prepared_direct_HIP_grouped_prefix',compute_execution_mode='owned_graph' if mode=='device-graph' else 'borrowed_stream',
        compute_provider_scope='source_bound_enqueue_prepared_to_hip_direct_enqueue_rns_no_publish_to_launch_rns_grouped_prefix_gemm',dynamic_compute_symbol=None,
        output_boundary='external_producer_to_external_consumer_padded_three_limbs',dataset_replayed_unchanged=True,external_generations_advanced=True,resident_control_excludes_input_binding=resident,
        host_payload_bytes_per_call=0 if device or resident else 2*(ac+bc)*8+m*n*24+cc*8,compact_status_bytes_per_call=12 if device else 4,
        consumer_validation_bytes=8,validation_payload_bytes=cc*8,preflight_device_payload_estimate_bytes=(ac+bc+m*n)*p+(ac+bc)*8+cc*8+8+(ac+bc)*p+32+m*n*24+4,
        preflight_host_payload_estimate_bytes=(ac+bc)*8+2*cc*8+12,memory_scope='conservative_all_route_logical_preflight_excludes_Boost_library_heap_driver_opaque_events_graphs_workspace_separate',
        participating_external_owners=3 if device else 1 if resident else 0,
        owner_acquire_callbacks=3 if device else 1 if resident else 0,owner_release_callbacks=3 if device else 1 if resident else 0,
        launch_count=None,launch_count_scope='not_inferred_from_api_calls',timing_source='steady_clock_complete_host_observed_transaction',
        phase_scope='enqueue_and_observation_not_disjoint_device_kernel_durations',exact_reference_passed=True,owner_callbacks_exactly_once=True,promotion_eligible=False,
        observed_allocation_scope='RNS8_owned_requested_device_and_pinned_payload_excludes_app_allocations_host_heap_allocator_and_driver',
        all_owned_payload_retired=True,external_application_device_requested_bytes=(ac+bc)*8+cc*8+8,
        ordinary_host_capacity_scope='four_native_vectors_only_excludes_Boost_objects_and_dynamic_limbs_library_heap_and_RSS')
    for key,value in fixed.items():
        if type(c.get(key)) is not type(value) or c.get(key)!=value:
            raise ValueError('capture '+key)
    capacity=c.get('ordinary_host_native_vector_capacity_bytes')
    if type(capacity) is not int or capacity < (ac+bc+2*cc)*8 or capacity%8:
        raise ValueError('ordinary native-vector capacity')
    validate_owned_observations(c)
    if type(c.get('plan_workspace_bytes')) is not int or c['plan_workspace_bytes']<0 or not manifest_provenance_complete(c.get('build_manifest')) or not c.get('target') or not c.get('device') or not c.get('execution_id') or not isinstance(c.get('plan_default_compute_kernel'),str) or not c['plan_default_compute_kernel']:
        raise ValueError('build/device/workspace/process identity')
    artifact=c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256',''))!=64:
        raise ValueError('artifact identity')
    if not positive(c.get('setup_us')) or len(c.get('samples',[]))!=case['repeats']:
        raise ValueError('setup/sample count')
    for s in [c.get('first_call',{}),*c['samples']]:
        phases=[s.get(key) for key in ('producer_enqueue_us','input_compute_us','export_observation_us','consumer_us')]
        if not all(positive(x,True) for x in phases) or not positive(s.get('total_us')) or not math.isclose(sum(phases),s['total_us'],rel_tol=1e-7,abs_tol=1e-6):
            raise ValueError('complete wall phases')
    if len(c.get('sustained',[]))!=3:
        raise ValueError('sustained observations')
    for s in c['sustained']:
        count=s.get('calls');duration=s.get('total_us')
        if type(count) is not int or not 16<=count<=4096 or not positive(duration) or (duration<250000 and count!=4096) or not math.isclose(s.get('per_call_us',-1),duration/count,rel_tol=1e-7):
            raise ValueError('sustained boundary')
    return c


def validate_owned_observations(c):
    keys={'live_bytes','peak_bytes','live_allocations','peak_allocations','allocation_calls','free_calls','allocated_bytes'}
    snapshots=[]
    for name in ('owned_before','owned_after_setup','owned_after_calls','owned_after_cleanup'):
        value=c.get(name)
        if type(value) is not dict or set(value)!={'epoch','device','pinned_host'} or type(value['epoch']) is not int or not 1<=value['epoch']<=2**64-1:
            raise ValueError('owned allocation epoch')
        for domain in ('device','pinned_host'):
            d=value[domain]
            if type(d) is not dict or set(d)!=keys or any(type(x) is not int or not 0<=x<=2**64-1 for x in d.values()):
                raise ValueError('owned allocation domain')
            if d['peak_bytes']<d['live_bytes'] or d['peak_allocations']<d['live_allocations'] or d['allocated_bytes']<d['live_bytes'] or d['allocation_calls']-d['free_calls']!=d['live_allocations']:
                raise ValueError('owned allocation accounting')
        snapshots.append(value)
    if len({s['epoch'] for s in snapshots})!=1:
        raise ValueError('reset allocation epoch during measurement')
    for domain in ('device','pinned_host'):
        baseline=snapshots[0][domain];retired=snapshots[-1][domain]
        if any(retired[key]!=baseline[key] for key in ('live_bytes','live_allocations')):
            raise ValueError('unretired owned payload')
        for previous,current in zip(snapshots,snapshots[1:]):
            if any(current[domain][key]<previous[domain][key] for key in ('peak_bytes','peak_allocations','allocation_calls','free_calls','allocated_bytes')):
                raise ValueError('nonmonotone allocation counters')


def quantile(values, fraction):
    values=sorted(values);position=(len(values)-1)*fraction;lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def summarize(observations):
    cohorts={}
    for entry,c in observations:
        validate_capture(c,entry)
        identity=dict(case={k:v for k,v in entry['case'].items() if k!='mode'},build=c['build_manifest'],target=c['target'],device=c['device'],runtime=c.get('runtime_version'),
            artifact={k:c['artifact_capture'].get(k) for k in ('executable_sha256','rns8_linkage','shared_library_sha256')},environment=(c.get('request_identity') or {}).get('environment'))
        cohorts.setdefault(json.dumps(identity,sort_keys=True),[]).append(dict(mode=entry['case']['mode'],execution_id=c['execution_id'],independent_run=entry['independent_run'],path=entry['output'],
            setup_us=c['setup_us'],p50_us=quantile([s['total_us'] for s in c['samples']],.5),p95_us=quantile([s['total_us'] for s in c['samples']],.95),
            sustained_us=sum(s['total_us'] for s in c['sustained'])/sum(s['calls'] for s in c['sustained']),
            observed_owned_device_peak_bytes=c['owned_after_calls']['device']['peak_bytes'],
            observed_owned_pinned_peak_bytes=c['owned_after_calls']['pinned_host']['peak_bytes']))
    result=[]
    for key,rows in cohorts.items():
        if len({r['execution_id'] for r in rows})!=len(rows):
            raise ValueError('duplicate process')
        medians={}
        for mode in MODES:
            selected=[r for r in rows if r['mode']==mode]
            if selected and len({r['independent_run'] for r in selected})<2:
                raise ValueError('independent process evidence')
            if selected:medians[mode]={metric:statistics.median(r[metric] for r in selected) for metric in ('setup_us','p50_us','p95_us','sustained_us')}
        def ratios(control):
            return {mode:{metric:medians[mode][metric]/medians[control][metric] for metric in ('p50_us','p95_us','sustained_us')} for mode in ('device-stream','device-graph') if mode in medians and control in medians}
        hosts=[mode for mode in ('host-roundtrip','host-scalar-first') if mode in medians]
        strongest=min(hosts,key=lambda mode:(medians[mode]['sustained_us'],mode)) if hosts else None
        result.append(dict(identity=json.loads(key),observations=rows,median_process_metrics=medians,
            device_to_host_roundtrip_cost_ratios=ratios('host-roundtrip'),device_to_scalar_first_host_cost_ratios=ratios('host-scalar-first'),
            strongest_matched_host_control=strongest,device_to_strongest_host_cost_ratios=ratios(strongest),
            eager_control_can_include_extra_proof_work=True,resident_control_is_input_binding_excluded_floor=True,promotion_eligible=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runs',type=int,default=2);p.add_argument('--seed',type=int,default=6767);p.add_argument('--timeout-seconds',type=float,default=1200);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    plan=collection_plan(a.binary,a.cases,a.output,a.runs,a.seed,str(uuid.uuid4()),case_validator=validate_case);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'plan.json').open('x') as f:json.dump(dict(schema='native_interop_collection_v1',observations=plan,promotion_eligible=False),f,indent=2)
    if a.plan_only:return 0
    observations=[]
    for entry in plan:
        path=Path(entry['output'])
        if path.exists():raise ValueError('immutable capture exists')
        if not run_command(entry['command'],path,a.timeout_seconds):raise RuntimeError('capture failed '+str(path))
        observations.append((entry,strict_json(path.read_text())))
    with (a.output/'summary.json').open('x') as f:json.dump(dict(schema='native_interop_summary_v1',cohorts=summarize(observations),promotion_eligible=False),f,indent=2)
    return 0


if __name__=='__main__':raise SystemExit(main())
