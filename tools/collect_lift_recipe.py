"""Boundary-matched exact lift/recipe costs; no scientific qualification inferred."""
import argparse
import json
import math
from pathlib import Path
import statistics
import uuid
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from collect_int4_calibration import collection_plan
from collect_native_calibration import strict_json
from collect_native_integrity import MODULI, positive
from collect_native_interop import quantile

MATERIALIZED_MODES=('recipe-packed-full','recipe-packed-minimum','recipe-layout-model')
MODES=(*MATERIALIZED_MODES,'recipe-ragged-full','recipe-ragged-minimum','recipe-full','recipe-minimum','recipe-cost-model','manual-full','manual-scalar-first','manual-auto','native-roundtrip','eager-observe','lift-wide','lift-mixed','lift-graph','lift-cpu-roundtrip','wider-producer')


def graph(c):
    if c['topology']=='diamond':return [(0,1),(2,1),(0,2),(3,4)]
    return [(i+1 if i else 0,1) for i in range(c['depth'])]


def bounds(c):
    result=[1,1]
    for a,b in graph(c):result.append(c['side']*result[a]*result[b])
    return result


def native_payload(c):
    """Logical native64 payload; actual HIP copies can transfer RNS/staging data."""
    if c['mode']!='native-roundtrip':return (0,0,0)
    cells=c['side']**2;nodes=graph(c)
    return (sum(1 if a==b else 2 for a,b in nodes)*cells*8,
            (len(nodes)-1)*cells*8,(len(nodes)+1)*cells*8)


def payload_bytes(c):
    cells=c['side']**2;nodes=len(graph(c))
    return (nodes+3)*cells*c['prefix']+cells*48*(nodes+1)+cells*16+native_payload(c)[2]


def validate_case(c):
    keys={'schema','side','depth','prefix','source_prefix','repeats','warmups','model_launch_ns','max_payload_bytes','unsigned_inputs','mode','topology'}
    if type(c) is not dict or set(c)!=keys or c['schema']!='lift_recipe_case_v1' or type(c['unsigned_inputs']) is not bool:
        raise ValueError('case ABI')
    for key,lo,hi in [('side',1,256),('depth',1,12),('prefix',1,20),('source_prefix',1,20),('repeats',3,100),('warmups',1,100),('model_launch_ns',0,10**9),('max_payload_bytes',64,2**40)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError(key)
    if c['source_prefix']>c['prefix'] or c['mode'] not in MODES or c['topology'] not in ('chain','diamond') or (c['topology']=='diamond' and c['depth']!=4):
        raise ValueError('route/topology')
    lift=c['mode'].startswith('lift-');single=lift or c['mode']=='wider-producer'
    if lift and c['topology']!='chain':raise ValueError('lift topology')
    b=c['side'] if single else bounds(c)[-1];factor=1 if c['unsigned_inputs'] else 2
    if math.prod(MODULI[:c['prefix']])<=b*factor or (lift and math.prod(MODULI[:c['source_prefix']])<=c['side']*factor):raise ValueError('strict range')
    if c['mode']=='native-roundtrip':
        lo,hi=(0,2**64-1) if c['unsigned_inputs'] else (-2**63,2**63-1)
        if any(not lo<=v<=hi for v in bounds(c)[2:-1]):raise ValueError('native roundtrip intermediate exceeds native64')
    cells=c['side']**2;nodes=len(graph(c));payload=payload_bytes(c)
    if payload>=c['max_payload_bytes'] or cells*c['prefix']>2**31-1 or c['side']**3*nodes>134217728:raise ValueError('payload/work')
    return c


def validate_capture(c,entry):
    case=validate_case(entry['case']);mode=case['mode'];lift=mode.startswith('lift-');single=lift or mode=='wider-producer';cells=case['side']**2;nodes=len(graph(case));p=case['prefix']
    native_h2d,native_d2h,native_host=native_payload(case)
    fixed=dict(schema='rns8_lift_recipe_microcase_v3',case=case,run_id=entry['run_id'],output_boundary='one_product_full_prefix_final_three_limb_export' if single else 'complete_dag_final_three_limb_export',
        extra_intermediate_observations=mode=='eager-observe',private_residue_transfer_control=mode=='lift-cpu-roundtrip',native_roundtrip_control=mode=='native-roundtrip',
        fixed_plan_exact_proof_policy='scalar_first' if mode=='manual-scalar-first' else 'eager_axis',logical_payload_estimate_bytes=payload_bytes(case),
        host_payload_bytes_before_final_export=cells*p if mode=='lift-cpu-roundtrip' else native_h2d+native_d2h,
        native_roundtrip_h2d_payload_bytes_per_replay=native_h2d,native_roundtrip_d2h_payload_bytes_per_replay=native_d2h,native_roundtrip_retained_host_bytes=native_host,
        replay_hip_counter_samples=case['repeats'],final_export_payload_bytes=cells*24,exact_reference_passed=True,promotion_eligible=False)
    for key,value in fixed.items():
        if type(c.get(key)) is not type(value) or c.get(key)!=value:raise ValueError('capture '+key)
    if not manifest_provenance_complete(c.get('build_manifest')) or not c.get('execution_id') or not c.get('target') or not c.get('device'):raise ValueError('provenance')
    artifact=c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256',''))!=64:raise ValueError('artifact')
    for key in ('ragged_retained_bytes','ragged_materialized_input_bytes','ragged_binding_peak_bytes','recipe_arena_bytes','recipe_retained_lift_bytes','auto_retained_device_bytes','replay_hip_allocations','replay_hip_frees','replay_hip_h2d_bytes','replay_hip_d2h_bytes','replay_hip_h2d_calls','replay_hip_d2h_calls'):
        if type(c.get(key)) is not int or c[key]<0:raise ValueError(key)
    if mode=='native-roundtrip':
        imports=sum(1 if a==b else 2 for a,b in graph(case))*case['repeats']
        if c['replay_hip_h2d_calls']<imports or c['replay_hip_d2h_calls']<nodes*case['repeats'] or not c['replay_hip_h2d_bytes'] or not c['replay_hip_d2h_bytes']:
            raise ValueError('native roundtrip actual transfer receipt')
    if mode.startswith('recipe-ragged-'):
        if c['ragged_retained_bytes']<=0 or c['ragged_materialized_input_bytes']<cells*p:raise ValueError('ragged conversion footprint omitted')
    elif any(c[k] for k in ('ragged_retained_bytes','ragged_materialized_input_bytes','ragged_binding_peak_bytes')):raise ValueError('unrequested ragged adapter')
    if mode.startswith('recipe-') and (c['replay_hip_allocations'] or c['replay_hip_frees']):raise ValueError('retained recipe unexpectedly allocates on replay')
    selected=c.get('selected_compute_prefixes')
    if type(selected) is not list or len(selected)!=(0 if single else nodes):raise ValueError('selected prefix trace')
    if mode in ('manual-full','manual-scalar-first','native-roundtrip','eager-observe','recipe-full','recipe-ragged-full') and selected!=[p]*nodes:
        raise ValueError('full compute prefix control')
    factor=1 if case['unsigned_inputs'] else 2
    for i,selected_prefix in enumerate(selected):
        if type(selected_prefix) is not int or not 1<=selected_prefix<=p or math.prod(MODULI[:selected_prefix])<=bounds(case)[i+2]*factor:raise ValueError('independent selected range')
    model=c.get('cost_model')
    expected_model=dict(kind='explicit-unqualified-analytical-work-model-v2',launch_ns=case['model_launch_ns'],gemm_units='one_ns_per_scalar_modular_multiply',qualified=False) if mode=='recipe-cost-model' else None
    if mode in MATERIALIZED_MODES:expected_model=dict(kind='explicit-unqualified-layout-work-model-v2',launch_ns=case['model_launch_ns'],gemm_units='one_ns_per_scalar_modular_multiply',packed_gemm_units='one_ns_per_four_term_group',pack_units='one_ns_per_logical_byte_read_or_write',cost_schema_version=2,packed_cost_capacity='physical_shared_b_prefix',qualified=False)
    validate_materializations(c,case,selected)
    if type(model) is not type(expected_model) or model!=expected_model or (expected_model is not None and any(type(model[k]) is not type(v) for k,v in expected_model.items())):raise ValueError('unqualified cost model identity')
    if not positive(c.get('setup_us')) or len(c.get('samples',[]))!=case['repeats']:raise ValueError('samples/setup')
    for s in [c.get('first_call',{}),*c['samples']]:
        phases=[s.get(k) for k in ('producer_us','execute_lift_us','final_export_us')]
        if not all(positive(x,True) for x in phases) or not positive(s.get('total_us')) or not math.isclose(sum(phases),s['total_us'],rel_tol=1e-7,abs_tol=1e-6):raise ValueError('complete phases')
    if len(c.get('sustained',[]))!=3:raise ValueError('sustained')
    for s in c['sustained']:
        count=s.get('calls');duration=s.get('total_us')
        if type(count) is not int or not 16<=count<=4096 or not positive(duration) or (duration<250000 and count!=4096) or not math.isclose(s.get('per_call_us',-1),duration/count,rel_tol=1e-7):raise ValueError('sustained boundary')
    return c


def validate_materializations(c,case,selected):
    if 'materialization_plan' not in c:raise ValueError('materialization receipt missing')
    plan=c['materialization_plan']
    if case['mode'] not in MATERIALIZED_MODES:
        if plan is not None:raise ValueError('unrequested materialization receipt')
        return
    if type(plan) is not dict or set(plan)!={'packed_b','packed_values','evaluated','exhaustive','chain_dynamic_programming','cost_optimal','estimated_ns','completed_packing_calls'}:raise ValueError('materialization ABI')
    nodes=graph(case);flags=plan['packed_b'];views=plan['packed_values']
    if type(flags) is not list or len(flags)!=len(nodes) or any(type(x) is not int or x not in (0,1) for x in flags):raise ValueError('packed choice')
    if case['mode']!='recipe-layout-model' and flags!=[1]*len(nodes):raise ValueError('forced packed control')
    if case['mode']=='recipe-packed-full' and selected!=[case['prefix']]*len(nodes):raise ValueError('full packed prefix')
    for key in ('exhaustive','chain_dynamic_programming','cost_optimal'):
        if type(plan[key]) is not bool:raise ValueError('search scope')
    if type(plan['evaluated']) is not int or not 1<=plan['evaluated']<=4096 or type(plan['estimated_ns']) is not int or not 0<=plan['estimated_ns']<2**64-1:raise ValueError('search cost/evaluations')
    if type(views) is not list:raise ValueError('packed views')
    expected={};current={};calls=0;raw=[case['prefix'],case['prefix']]+[0]*len(nodes);cost=0
    cells=case['side']**2;launch=case['model_launch_ns']
    def demand(value,p):
        nonlocal cost
        if value>=2 and raw[value]<p:cost+=3*launch+cells*raw[value]*p;raw[value]=p
    for i,((_,b),p,flag) in enumerate(zip(nodes,selected,flags)):
        demand(nodes[i][0],p)
        if not flag:
            demand(b,p);cost+=launch+cells*case['side']*p;raw[i+2]=p;continue
        if current.get(b,0)<p:
            demand(b,p);cost+=launch+2*cells*(p-current.get(b,0))
        cost+=launch+cells*((case['side']+3)//4)*p;raw[i+2]=p
        if b not in expected:expected[b]=dict(value_id=b,capacity_prefix=p,first_reader=i,last_reader=i)
        expected[b]['capacity_prefix']=max(expected[b]['capacity_prefix'],p);expected[b]['last_reader']=i
        if current.get(b,0)<p:current[b]=p;calls+=1
    demand(len(nodes)+1,case['prefix'])
    if plan['estimated_ns']!=cost:raise ValueError('materialization cost trace')
    if type(plan['completed_packing_calls']) is not int or plan['completed_packing_calls']!=calls or len(views)!=len(expected):raise ValueError('packing reuse/growth')
    seen=set();tile_cells=((case['side']+15)//16*16)**2;arena=c['recipe_arena_bytes']
    for view in views:
        if type(view) is not dict or set(view)!={'value_id','capacity_prefix','first_reader','last_reader','bytes','arena_offset'} or any(type(v) is not int for v in view.values()):raise ValueError('view ABI')
        value=view['value_id']
        if value in seen or value not in expected or any(view[k]!=v for k,v in expected[value].items()):raise ValueError('view demand/lifetime')
        seen.add(value)
        if view['bytes']!=tile_cells*view['capacity_prefix'] or view['arena_offset']<0 or view['arena_offset']%256 or view['arena_offset']+view['bytes']>arena:raise ValueError('packed capacity/budget')
    for i,a in enumerate(views):
        for b in views[i+1:]:
            simultaneous=a['first_reader']<=b['last_reader'] and b['first_reader']<=a['last_reader']
            overlaps=a['arena_offset']<b['arena_offset']+b['bytes'] and b['arena_offset']<a['arena_offset']+a['bytes']
            if simultaneous and overlaps:raise ValueError('packed reader interference')
    if c['recipe_arena_bytes']+c['recipe_retained_lift_bytes']>case['max_payload_bytes']-c['logical_payload_estimate_bytes']:raise ValueError('retained recipe budget')


def summarize(observations):
    cohorts={}
    for entry,c in observations:
        validate_capture(c,entry)
        identity=dict(case={k:v for k,v in entry['case'].items() if k!='mode'},boundary=c['output_boundary'],extra_intermediate_observations=c['extra_intermediate_observations'],build=c['build_manifest'],target=c['target'],device=c['device'],runtime=c.get('runtime_version'),
            artifact={k:c['artifact_capture'].get(k) for k in ('executable_sha256','rns8_linkage','shared_library_sha256')},environment=(c.get('request_identity') or {}).get('environment'))
        cohorts.setdefault(json.dumps(identity,sort_keys=True),[]).append(dict(mode=entry['case']['mode'],execution_id=c['execution_id'],independent_run=entry['independent_run'],path=entry['output'],setup_us=c['setup_us'],
            p50_us=quantile([s['total_us'] for s in c['samples']],.5),p95_us=quantile([s['total_us'] for s in c['samples']],.95),sustained_us=statistics.median(s['per_call_us'] for s in c['sustained'])))
    result=[]
    for key,rows in cohorts.items():
        if len({r['execution_id'] for r in rows})!=len(rows):raise ValueError('duplicate process')
        modes={}
        for mode in MODES:
            selected=[r for r in rows if r['mode']==mode]
            if selected and len({r['independent_run'] for r in selected})<2:raise ValueError('independent process evidence')
            if selected:modes[mode]={metric:statistics.median(r[metric] for r in selected) for metric in ('setup_us','p50_us','p95_us','sustained_us')}
        baseline='manual-scalar-first' if 'manual-scalar-first' in modes else 'manual-full' if 'manual-full' in modes else 'wider-producer' if 'wider-producer' in modes else None
        pairs={}
        if baseline:
            for mode,metrics in modes.items():
                if mode==baseline:continue
                saving=modes[baseline]['sustained_us']-metrics['sustained_us'];extra_setup=max(0,metrics['setup_us']-modes[baseline]['setup_us'])
                pairs[mode]=dict(cost_ratio=metrics['sustained_us']/modes[baseline]['sustained_us'],break_even_replays=math.ceil(extra_setup/saving) if saving>0 else None,profitable_measured_replay=saving>0)
        result.append(dict(identity=json.loads(key),observations=rows,median_process_metrics=modes,baseline=baseline,matched_comparisons=pairs,promotion_eligible=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=int,default=2);p.add_argument('--seed',type=int,default=4142);p.add_argument('--timeout-seconds',type=float,default=1200);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    plan=collection_plan(a.binary,a.cases,a.output,a.runs,a.seed,str(uuid.uuid4()),case_validator=validate_case);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'plan.json').open('x') as f:json.dump(dict(schema='lift_recipe_collection_v1',observations=plan,promotion_eligible=False),f,indent=2)
    if a.plan_only:return 0
    captures=[]
    for entry in plan:
        path=Path(entry['output'])
        if path.exists():raise ValueError('immutable capture exists')
        if not run_command(entry['command'],path,a.timeout_seconds):raise RuntimeError('capture failed '+str(path))
        captures.append((entry,strict_json(path.read_text())))
    with (a.output/'summary.json').open('x') as f:json.dump(dict(schema='lift_recipe_summary_v1',cohorts=summarize(captures),promotion_eligible=False),f,indent=2)
    return 0


if __name__=='__main__':raise SystemExit(main())
