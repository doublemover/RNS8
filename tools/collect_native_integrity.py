"""Counterbalanced native-integrity costs, with distinct nonpromoting fault model."""
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

MODES = ('compute', 'sentinel-only', 'compute-sentinel', 'compute-cpu-reference')
MODULI = (256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167)


def validate_case(c):
    keys = {'schema','m','n','k','prefix','unsigned_inputs','mode','residency','repeats','warmups','seed','max_payload_bytes','dataset'}
    if type(c) is not dict or set(c) != keys or c['schema'] != 'native_integrity_case_v1' or type(c['unsigned_inputs']) is not bool:
        raise ValueError('case ABI')
    for key, lo, hi in [('m',1,512),('n',1,512),('k',1,65537),('prefix',1,20),('repeats',3,100),('warmups',1,100),('seed',0,2**32-1),('max_payload_bytes',8,2**40)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError(key)
    if c['mode'] not in MODES or c['residency'] not in ('cold','resident','repack') or c['dataset'] not in ('small','fullwidth'):
        raise ValueError('route')
    if c['mode'] == 'sentinel-only' and c['residency'] != 'resident':
        raise ValueError('standalone requires a pre-existing candidate')
    m,n,k,p = (c[key] for key in ('m','n','k','prefix'))
    if m*n*k > 134217728 or max(m*k*8,k*n*8,p*m*k,p*k*n,p*m*n) > 2**31-1:
        raise ValueError('work/address')
    payload = (2*p+24)*(m*k+k*n+m*n)+48*m*(n+3)+8
    if payload > c['max_payload_bytes']:
        raise ValueError('payload budget')
    magnitude = (2**64-1 if c['unsigned_inputs'] else 2**63) if c['dataset']=='fullwidth' else (14 if c['unsigned_inputs'] else 7)
    bound = k*magnitude*magnitude
    if math.prod(MODULI[:p]) <= bound*(1 if c['unsigned_inputs'] else 2):
        raise ValueError('strict independent product bound')
    return c


def positive(x, zero=False):
    return type(x) in (int,float) and math.isfinite(x) and (x >= 0 if zero else x > 0)


def validate_capture(c, entry):
    case = validate_case(entry['case']);m,n,k,p = (case[x] for x in ('m','n','k','prefix'))
    fixed = dict(schema='rns8_native_integrity_microcase_v1',case=case,run_id=entry['run_id'],
        host_instrumentation_state='hooks_disabled',output_boundary='public_prepared_product_and_padded_three_limb_export',
        standalone_control=case['mode']=='sentinel-only',standalone_control_scope='private_read_only_checker_per_call_flag_allocation_device_sync_and_export',
        known_payload_envelope_bytes=(2*p+24)*(m*k+k*n+m*n)+48*m*(n+3)+8,
        memory_scope='logical_payload_excludes_cpp_int_heap_capacity_driver_allocator_and_opaque_runtime',
        guarantee='fixed_redundant_residue_consistency_not_arbitrary_correctness',ideal_failure_probability=None,
        check_moduli_in_computational_range=False,cpu_reference_inputs='original_host_dataset_no_input_D2H',
        exact_reference_passed=True,full_cpu_reference_inside_interval=case['mode']=='compute-cpu-reference',promotion_eligible=False)
    for key, value in fixed.items():
        if type(c.get(key)) is not type(value) or c.get(key) != value:
            raise ValueError('capture '+key)
    if not manifest_provenance_complete(c.get('build_manifest')) or not c.get('execution_id') or not c.get('target') or not c.get('device'):
        raise ValueError('compiled/process/target provenance')
    artifact=c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256','')) != 64:
        raise ValueError('artifact identity')
    report=c.get('verification')
    if case['mode']=='compute-sentinel':
        if type(report) is not dict or any(report.get(key)!=value for key,value in dict(outcome=1,failure_mask=0,basis_version=1,prefix=p,check_moduli=[257,263],device_flag_bytes=4,pinned_flag_bytes=4).items()):
            raise ValueError('sentinel report')
        if any(type(report.get(key)) is not int or report[key] <= 0 for key in ('operation','epoch','a_generation','b_generation')):
            raise ValueError('actual operation binding')
    elif report is not None:
        raise ValueError('unrequested sentinel report')
    if not positive(c.get('setup_us')) or len(c.get('samples',[])) != case['repeats']:
        raise ValueError('sample count/setup')
    for sample in [c.get('first_call',{}),*c['samples']]:
        phases=[sample.get(key) for key in ('prepare_us','execute_us','export_reference_us','cleanup_us')]
        if not all(positive(x,True) for x in phases) or not positive(sample.get('total_us')) or not math.isclose(sum(phases),sample['total_us'],rel_tol=1e-7,abs_tol=1e-6):
            raise ValueError('complete timing phases')
    if len(c.get('sustained',[])) != 3:
        raise ValueError('sustained observations')
    for burst in c['sustained']:
        count=burst.get('calls');duration=burst.get('total_us')
        if type(count) is not int or not 16 <= count <= 4096 or not positive(duration) or (duration<250000 and count!=4096) or not math.isclose(burst.get('per_call_us',-1),duration/count,rel_tol=1e-7):
            raise ValueError('sustained work/time')
    return c


def summarize(observations):
    cohorts={}
    for entry,c in observations:
        validate_capture(c,entry)
        identity=dict(case={k:v for k,v in entry['case'].items() if k!='mode'},build=c['build_manifest'],target=c['target'],device=c['device'],runtime=c.get('runtime_version'),
            artifact={k:c['artifact_capture'].get(k) for k in ('executable_sha256','rns8_linkage','shared_library_sha256')},environment=(c.get('request_identity') or {}).get('environment'))
        key=json.dumps(identity,sort_keys=True);cohorts.setdefault(key,[]).append(dict(mode=entry['case']['mode'],execution_id=c['execution_id'],independent_run=entry['independent_run'],path=entry['output'],median_us=statistics.median(s['total_us'] for s in c['samples'])))
    result=[]
    for key, rows in cohorts.items():
        if len({row['execution_id'] for row in rows})!=len(rows):
            raise ValueError('duplicate process execution')
        medians={}
        for mode in MODES:
            selected=[row for row in rows if row['mode']==mode]
            if selected and len({row['independent_run'] for row in selected})<2:
                raise ValueError('independent process evidence')
            if selected:medians[mode]=statistics.median(row['median_us'] for row in selected)
        # Standalone read-only checks have a different input/output boundary and
        # allocation policy: report cost, never call their ratio a GEMM speedup.
        overhead={mode:medians[mode]/medians['compute'] for mode in ('compute-sentinel','compute-cpu-reference') if mode in medians and 'compute' in medians}
        result.append(dict(identity=json.loads(key),observations=rows,median_wall_us=medians,complete_route_cost_ratio=overhead,guarantee='fixed_redundant_residue_consistency_only',promotion_eligible=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--cases',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=int,default=2);p.add_argument('--seed',type=int,default=4949);p.add_argument('--timeout-seconds',type=float,default=1200);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    plan=collection_plan(a.binary,a.cases,a.output,a.runs,a.seed,str(uuid.uuid4()),case_validator=validate_case)
    a.output.mkdir(parents=True,exist_ok=True);target=a.output/'plan.json'
    with target.open('x') as f:json.dump(dict(schema='native_integrity_collection_v1',observations=plan,promotion_eligible=False),f,indent=2)
    if a.plan_only:return 0
    observations=[]
    for entry in plan:
        path=Path(entry['output'])
        if path.exists():raise ValueError('immutable capture exists')
        if not run_command(entry['command'],path,a.timeout_seconds):raise RuntimeError('capture failed: '+str(path))
        observations.append((entry,strict_json(path.read_text())))
    with (a.output/'summary.json').open('x') as f:json.dump(dict(schema='native_integrity_summary_v1',cohorts=summarize(observations),promotion_eligible=False),f,indent=2)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
