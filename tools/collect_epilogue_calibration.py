"""Independent complete-public-expression controls; report only, no promotion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import uuid
from benchmark_sweep_lib.execution import run_command
from collect_native_calibration import strict_json,DEFAULT_MODULI
from build_manifest import manifest_provenance_complete
PROFILES={'extreme':(-(1<<63),(1<<63)-1),'zero':(0,0),'copy':(0,1),'unit':(1,1),'general':(-3,5),'channel-zero':(255,256)}


def validate_case(c):
    if not isinstance(c,dict) or set(c)!={'schema','m','n','k','repeats','warmups','backend','coefficient_profile','residency'} or c['schema']!='epilogue_public_case_v1':raise ValueError('case ABI')
    for key,lo,hi in (('m',1,512),('n',1,512),('k',1,65537),('repeats',3,1000),('warmups',1,1000)):
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError(key)
    if c['m']*c['n']*c['k']>134217728 or c['backend'] not in ('cpu','hip-direct') or c['coefficient_profile'] not in PROFILES or c['residency'] not in ('resident','repack'):raise ValueError('case work/contract')
    return c


def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0


def validate_capture(c,e):
    case=e['case'];alpha,beta=PROFILES[case['coefficient_profile']]
    fixed=dict(schema='rns8_public_epilogue_microcase_v1',case=case,run_id=e['run_id'],dataset='signed_small_mt19937_64_seed4747_v1',seed=4747,
        alpha=alpha,beta=beta,prefix=20,limbs=3,output_ld=case['n']+2,host_instrumentation_state='hooks_disabled',
        output_boundary='complete_public_expression_and_padded_signed_limb_export',promotion_eligible=False,exact_reference_passed=True,
        oracle_inside_production_interval=False,source_counts_are_measured_instructions=False,
        active_dot_channels=sum(alpha%p!=0 for p in DEFAULT_MODULI[:20]),active_addend_channels=sum(beta%p!=0 for p in DEFAULT_MODULI[:20]))
    fixed['source_scalar_dot_terms_per_call']=case['m']*case['n']*case['k']*fixed['active_dot_channels']
    for key,value in fixed.items():
        if type(c.get(key)) is not type(value) or c[key]!=value:raise ValueError('capture '+key)
    if not manifest_provenance_complete(c.get('build_manifest')):raise ValueError('compiled provenance')
    artifact=c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256',''))!=64 or not c.get('execution_id'):raise ValueError('artifact/process receipt')
    if not positive(c.get('setup_us')) or not positive(c.get('proof_diagnostic_us')):raise ValueError('setup/proof diagnostics')
    samples=c.get('samples',[])
    if len(samples)!=case['repeats']:raise ValueError('actual latency observations')
    for sample in [c.get('first_call',{}),*samples]:
        phases=[sample.get(key) for key in ('pack_us','expression_us','export_us')]
        if not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in phases) or not positive(sample.get('total_us')) or not math.isclose(sum(phases),sample['total_us'],rel_tol=1e-9,abs_tol=1e-9):raise ValueError('disjoint complete public phases')
    bursts=c.get('sustained',[])
    if len(bursts)!=3:raise ValueError('three bounded intervals required')
    for b in bursts:
        if type(b.get('calls')) is not int or not 16<=b['calls']<=4096 or not positive(b.get('duration_us')) or b.get('exact_reference_passed') is not True:raise ValueError('sustained receipt')
        if b['duration_us']<250000 and b['calls']!=4096:raise ValueError('sustained stopped before time or call cap')
        for key in ('first_input_generation','last_input_generation'):
            if type(b.get(key)) is not int or b[key]<1:raise ValueError('generation receipt')
        if b['last_input_generation']-b['first_input_generation']!=(b['calls'] if case['residency']=='repack' else 0):raise ValueError('generation interval')
    return c


def collection_plan(binary,paths,output,runs,seed,collection_id):
    if not paths:raise ValueError('at least one case required')
    if not 2<=runs<=100:raise ValueError('2..100 independent processes')
    inputs=[(p,validate_case(strict_json(p.read_text())),hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    if len({h for _,_,h in inputs})!=len(inputs):raise ValueError('duplicate cases')
    result=[]
    for run in range(runs):
        order=list(inputs);shift=(seed+run)%len(order);order=order[shift:]+order[:shift]
        if run%2:order.reverse()
        for position,(path,case,digest) in enumerate(order):
            run_id=f'{collection_id}-{digest[:12]}-{run}'
            result.append(dict(command=[str(binary),str(path),run_id],case=case,case_sha256=digest,run_id=run_id,independent_run=run,position=position,output=str(output/f'{run_id}.json')))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--cases',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=int,default=2);p.add_argument('--seed',type=int,default=4747);p.add_argument('--timeout-seconds',type=float,default=600);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    plan=collection_plan(a.binary,a.cases,a.output,a.runs,a.seed,str(uuid.uuid4()));a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'plan.json').write_text(json.dumps(dict(schema='public_epilogue_collection_v1',order='rotating_reversed_case_order_per_independent_run',commands=plan,promotion_eligible=False),indent=2)+'\n')
    if a.plan_only:return 0
    groups={}
    for e in plan:
        output=Path(e['output'])
        if output.exists():raise ValueError('refusing observation overwrite')
        if not run_command(e['command'],output,a.timeout_seconds):raise RuntimeError('capture failed: '+str(output))
        c=validate_capture(strict_json(output.read_text()),e)
        identity=dict(case=e['case'],artifact=c['artifact_capture'],build=c['build_manifest'],device=c.get('device'),target=c.get('target'),runtime=c.get('runtime_version'))
        # Artifact contains per-execution collection fields: bind stable binary hash,
        # not timestamps, while retaining all raw immutable receipts separately.
        identity['artifact']={k:c['artifact_capture'].get(k) for k in ('executable_sha256','rns8_linkage','shared_library_sha256')}
        key=json.dumps(identity,sort_keys=True);groups.setdefault(key,[]).append(dict(run_id=e['run_id'],execution_id=c['execution_id'],path=str(output),latency_us=statistics.median(s['total_us'] for s in c['samples']),sustained_us=statistics.median(b['duration_us']/b['calls'] for b in c['sustained']),setup_us=c['setup_us']))
    rows=[]
    for key,observations in sorted(groups.items()):
        if len({o['execution_id'] for o in observations})!=len(observations):raise ValueError('duplicate process receipt')
        rows.append(dict(identity=json.loads(key),observations=observations,independent_processes=len(observations),latency_median_us=statistics.median(o['latency_us'] for o in observations),latency_range_us=[min(o['latency_us'] for o in observations),max(o['latency_us'] for o in observations)],sustained_process_median_us=statistics.median(o['sustained_us'] for o in observations)))
    (a.output/'report.json').write_text(json.dumps(dict(schema='public_epilogue_observations_v1',cohorts=rows,promotion_eligible=False,interpretation='named_expression_workloads_not_interchangeable_no_causal_speedup_inference'),indent=2)+'\n');return 0

if __name__=='__main__':raise SystemExit(main())
