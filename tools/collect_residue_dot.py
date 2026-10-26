"""Arithmetic-only paired builds; scheduler costs are a separate comparison axis."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import uuid
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from collect_int4_calibration import collection_plan,positive
from collect_native_calibration import strict_json
from collect_native_interop import quantile

MODES=('independent','grouped','scheduled','region')

def validate_case(c):
    keys={'schema','m','n','k','prefix','modulus','tasks','repeats','warmups','mode','domain','dataset'}
    if type(c) is not dict or set(c)!=keys or c['schema']!='residue_dot_case_v1':raise ValueError('case ABI')
    for key,lo,hi in [('m',2,512),('n',2,512),('k',1,65537),('prefix',1,20),('modulus',0,256),('tasks',1,8),('repeats',3,100),('warmups',1,100)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError(key)
    if c['mode'] not in MODES or c['domain'] not in ('finite','wide','bounded') or c['dataset'] not in ('zero','sparse','centered','fullwidth'):raise ValueError('route/domain/dataset')
    finite=c['domain']=='finite';wide=c['domain']=='wide'
    if (finite and (c['modulus']<2 or c['prefix']!=1)) or (not finite and c['modulus']!=0) or (c['mode']=='scheduled' and c['domain']!='bounded') or (c['mode']=='region' and wide):raise ValueError('scheduler domain')
    if c['dataset']=='fullwidth' and (not wide or c['prefix']!=20):raise ValueError('fullwidth')
    if c['m']*c['n']*c['k']*c['tasks']>134217728 or (c['m']*c['k']+c['k']*c['n']+c['m']*c['n'])*c['tasks']*256>=512*1024*1024:raise ValueError('work/payload')
    return c

def paired_manifest(manifest,arithmetic):
    if not manifest_provenance_complete(manifest):raise ValueError('manifest provenance')
    lines=manifest['configuration'].splitlines();options={}
    for line in lines:
        key,sep,value=line.partition('=')
        if key in ('residue_dot4','residue_scalar_control'):
            if not sep or key in options:raise ValueError('duplicate/missing arithmetic option')
            options[key]=value
    if options!={'residue_dot4':'ON' if arithmetic=='dot4' else 'OFF','residue_scalar_control':'OFF' if arithmetic=='dot4' else 'ON'}:raise ValueError('legacy or mismatched arithmetic build')
    normalized=copy.deepcopy(manifest);normalized['configuration']='\n'.join(line for line in lines if not line.startswith(('residue_dot4=','residue_scalar_control=')))
    return normalized

def validate_capture(c,e,diagnostic=False):
    case=validate_case(e['case']);region=case['mode']=='region';finite=case['domain']=='finite';arithmetic=e['arithmetic']
    if arithmetic not in ('scalar','dot4'):raise ValueError('arithmetic')
    cells=(case['m']-int(region))*(case['n']-int(region))*case['tasks']*(1 if finite else case['prefix'])
    fixed=dict(schema='rns8_residue_dot_microcase_v2',case=case,run_id=e['run_id'],arithmetic=arithmetic,output_boundary='public_incremental_nonzero_rectangle_with_full_padded_export' if region else 'public_complete_products_with_full_padded_export',dataset_seed=7171,residency='unchanged_packed_inputs',hooks_enabled=diagnostic,oracle_inside_interval=False,all_residue_planes_checked=True,exact_reference_passed=True,promotion_eligible=False,
        source_logical_scalar_updates=cells*case['k'] if arithmetic=='scalar' else 0,source_logical_dot4_updates=cells*((case['k']+3)//4) if arithmetic=='dot4' else 0,
        source_count_scope='active_output_accumulator_updates_not_issued_instructions_inactive_lanes_or_total_gather_traffic',selected_kernel_scope='plan_identity_plus_explicit_public_scheduler_not_dynamic_profiler_evidence',actual_launches=None,isa_and_resources=None,sustained_body='compute_export_only_no_oracle_or_json')
    for k,v in fixed.items():
        if type(c.get(k)) is not type(v) or c.get(k)!=v:raise ValueError('capture '+k)
    if c.get('diagnostic_event_scope')!='separate_final_hook_enabled_invocation_group_spans_not_dynamic_launch_or_instruction_counts':raise ValueError('diagnostic scope')
    if diagnostic:
        events=c.get('diagnostic_events')
        if type(events) is not list or not events or any(not isinstance(x.get('label'),str) or not x['label'] or not positive(x.get('microseconds'),True) for x in events):raise ValueError('diagnostic spans')
    elif c.get('diagnostic_events') is not None:raise ValueError('production instrumentation')
    contract=c.get('plan_contract');scheduled=case['mode']=='scheduled';bounded=case['domain']=='bounded'
    expected_tiles=((case['m']+63)//64)*((case['n']+63)//64) if scheduled else 0
    expected_prefix=0 if finite else case['prefix']
    if type(contract) is not dict or set(contract)!={'global_bound','per_tile_bound_count','schedule_entries','min_selected_prefix','max_selected_prefix'}:raise ValueError('plan contract ABI')
    if any(type(v) is not int or v<0 for v in contract.values()) or contract['global_bound']>1<<63:raise ValueError('plan contract values')
    if contract['per_tile_bound_count']!=expected_tiles or contract['schedule_entries']!=expected_tiles or contract['min_selected_prefix']!=expected_prefix or contract['max_selected_prefix']!=expected_prefix:raise ValueError('plan schedule ownership')
    if (scheduled or not bounded) and contract['global_bound']!=0:raise ValueError('plan bound placement')
    paired_manifest(c.get('build_manifest'),arithmetic)
    if not c.get('execution_id') or not c.get('target') or not c.get('device') or not c.get('selected_kernel'):raise ValueError('selected identity')
    artifact=c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256',''))!=64:raise ValueError('artifact')
    if not positive(c.get('setup_us')) or len(c.get('samples',[]))!=case['repeats']:raise ValueError('samples/setup')
    for s in [c.get('first_call',{}),*c['samples']]:
        if not positive(s.get('compute_us')) or not positive(s.get('export_us')) or not positive(s.get('total_us')) or not math.isclose(s['compute_us']+s['export_us'],s['total_us'],rel_tol=1e-7,abs_tol=1e-6):raise ValueError('complete phases')
    if len(c.get('sustained',[]))!=3:raise ValueError('sustained')
    for b in c['sustained']:
        if type(b.get('calls')) is not int or not 16<=b['calls']<=4096 or not positive(b.get('total_us')) or (b['total_us']<250000 and b['calls']!=4096) or not math.isclose(b.get('per_call_us',-1),b['total_us']/b['calls'],rel_tol=1e-7):raise ValueError('sustained boundary')
    return c

def summarize(observations):
    cohorts={};seen=set()
    for e,c in observations:
        validate_capture(c,e)
        if c['execution_id'] in seen:raise ValueError('reused process')
        seen.add(c['execution_id']);identity=dict(case={k:v for k,v in e['case'].items() if k!='mode'},manifest=paired_manifest(c['build_manifest'],e['arithmetic']),target=c['target'],device=c['device'],runtime=c.get('runtime_version'),environment=(c.get('request_identity') or {}).get('environment'))
        cohorts.setdefault(json.dumps(identity,sort_keys=True),[]).append(dict(arithmetic=e['arithmetic'],mode=e['case']['mode'],run=e['independent_run'],execution_id=c['execution_id'],path=e['output'],artifact=c['artifact_capture'],selected_kernel=c['selected_kernel'],p50_us=quantile([s['total_us'] for s in c['samples']],.5),p95_us=quantile([s['total_us'] for s in c['samples']],.95),compute_p50_us=quantile([s['compute_us'] for s in c['samples']],.5),sustained_us=statistics.median(b['per_call_us'] for b in c['sustained'])))
    result=[]
    for key,rows in cohorts.items():
        cells={}
        for arithmetic in ('scalar','dot4'):
            # Paths/timestamps may differ; compare immutable content identity only.
            hashes={tuple(r['artifact'].get(k) for k in ('executable_sha256','rns8_linkage','shared_library_sha256')) for r in rows if r['arithmetic']==arithmetic}
            if len(hashes)>1:raise ValueError('arithmetic artifact changed within cohort')
            for mode in MODES:
                selected=[r for r in rows if (r['arithmetic'],r['mode'])==(arithmetic,mode)]
                if not selected:continue
                if len({r['run'] for r in selected})<2 or len({r['selected_kernel'] for r in selected})!=1:raise ValueError('independent repeats/selected route')
                cells[arithmetic+':'+mode]={metric:statistics.median(r[metric] for r in selected) for metric in ('p50_us','p95_us','compute_p50_us','sustained_us')}
        paired={};scheduler={}
        for mode in MODES:
            a=cells.get('scalar:'+mode);b=cells.get('dot4:'+mode)
            if a and b:paired[mode]={metric:b[metric]/a[metric] for metric in a}
        for arithmetic in ('scalar','dot4'):
            base=cells.get(arithmetic+':independent')
            if base:
                # Region computes fewer cells: never report its ratio as batching speedup.
                for mode in ('grouped','scheduled'):
                    candidate=cells.get(arithmetic+':'+mode)
                    if candidate:scheduler[arithmetic+':'+mode]={metric:candidate[metric]/base[metric] for metric in base}
        result.append(dict(identity=json.loads(key),observations=rows,median_process_metrics=cells,same_scheduler_arithmetic_cost_ratios=paired,same_arithmetic_scheduler_cost_ratios=scheduler,promotion_eligible=False,isa_qualification='requires selected binary diagnostic companion; source update counts are not instructions'))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scalar',type=Path,required=True);p.add_argument('--dot4',type=Path,required=True);p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=int,default=2);p.add_argument('--seed',type=int,default=7171);p.add_argument('--timeout-seconds',type=float,default=1200);p.add_argument('--plan-only',action='store_true');p.add_argument('--diagnostics',action='store_true');a=p.parse_args()
    if a.scalar.resolve()==a.dot4.resolve():raise ValueError('two separately compiled artifacts required')
    plans=[];identity=str(uuid.uuid4())
    for arithmetic,binary in [('scalar',a.scalar),('dot4',a.dot4)]:
        for e in collection_plan(binary,a.cases,a.output,a.runs,a.seed,identity+'-'+arithmetic,case_validator=validate_case):e['arithmetic']=arithmetic;plans.append(e)
    plans.sort(key=lambda e:(e['independent_run'],e['position'],(0 if e['arithmetic']=='scalar' else 1)^((e['independent_run']+e['position'])%2)))
    if a.diagnostics:
        for e in plans:e['diagnostic']=dict(command=[*e['command'][:2],e['run_id']+'-diagnostic','--diagnostic'],output=str(a.output/(e['run_id']+'-diagnostic.json')))
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'plan.json').open('x') as f:json.dump(dict(schema='residue_dot_collection_v1',observations=plans,promotion_eligible=False),f,indent=2)
    if a.plan_only:return 0
    captures=[]
    for e in plans:
        path=Path(e['output'])
        if path.exists():raise ValueError('immutable capture exists')
        if not run_command(e['command'],path,a.timeout_seconds):raise RuntimeError('capture failed '+str(path))
        captured=strict_json(path.read_text());captures.append((e,captured))
        if a.diagnostics:
            d=dict(e,run_id=e['run_id']+'-diagnostic');dpath=a.output/(d['run_id']+'.json')
            if dpath.exists():raise ValueError('immutable diagnostic capture exists')
            command=[*e['command'][:2],d['run_id'],'--diagnostic']
            if not run_command(command,dpath,a.timeout_seconds):raise RuntimeError('diagnostic failed')
            diagnostic=strict_json(dpath.read_text());validate_capture(diagnostic,d,True)
            for identity_key in ('case','build_manifest','arithmetic','target','device','runtime_version','selected_kernel'):
                if diagnostic.get(identity_key)!=captured.get(identity_key):raise ValueError('diagnostic identity drift')
            for artifact_key in ('executable_sha256','rns8_linkage','shared_library_sha256'):
                if diagnostic['artifact_capture'].get(artifact_key)!=captured['artifact_capture'].get(artifact_key):raise ValueError('diagnostic artifact drift')
            if diagnostic['execution_id']==captured['execution_id']:raise ValueError('diagnostic not independent process')
    with (a.output/'summary.json').open('x') as f:json.dump(dict(schema='residue_dot_summary_v1',cohorts=summarize(captures),promotion_eligible=False),f,indent=2)
    return 0
if __name__=='__main__':raise SystemExit(main())
