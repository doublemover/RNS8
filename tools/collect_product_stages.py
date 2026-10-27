"""Collect completed resident stages and export scoped, non-promoting prices."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import uuid
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from calibration_manifest import independent_run_interval
from collect_int4_calibration import collection_plan
from collect_lift_recipe import graph, bounds, validate_capture as validate_recipe
from collect_native_calibration import strict_json
from collect_native_integrity import MODULI, positive
from product_stage_contract import price_cells,price_identity
from product_dataset import inputs as dataset_inputs,identity as dataset_identity


def digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_case(c):
    fields={'schema','side','depth','prefix','topology','unsigned_inputs','prefixes','capacities','warmups','repeats','max_payload_bytes'}
    if type(c) is not dict or not fields<=set(c) or set(c)-fields-{'input_values'} or c['schema']!='product_stage_case_v1' or type(c['unsigned_inputs']) is not bool:
        raise ValueError('stage case ABI')
    for key,lo,hi in [('side',1,256),('depth',1,12),('prefix',1,20),('warmups',1,100),('repeats',3,100),('max_payload_bytes',64,2**32-1)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError(key)
    dataset_inputs(c)
    if c['topology'] not in ('chain','diamond') or (c['topology']=='diamond' and c['depth']!=4):raise ValueError('stage topology')
    for key in ('prefixes','capacities'):
        values=c[key]
        if type(values) is not list or not values or any(type(p) is not int or not 1<=p<=c['prefix'] for p in values) or values!=sorted(set(values)):
            raise ValueError('stage domain '+key)
    factor=1 if c['unsigned_inputs'] else 2
    if factor*bounds(c)[-1]>=math.prod(MODULI[:c['prefix']]):raise ValueError('stage strict range')
    cells=c['side']**2;p=c['prefix'];tile_cells=((c['side']+15)//16*16)**2
    payload=5*cells*p+tile_cells*c['capacities'][-1]+4+cells*(p-1)
    if payload>=c['max_payload_bytes'] or cells*c['side']*len(graph(c))>134217728:raise ValueError('stage payload/work')
    return c


def minimum_prefixes(c):
    factor=1 if c['unsigned_inputs'] else 2
    return [next(p for p in range(1,c['prefix']+1) if math.prod(MODULI[:p])>factor*b) for b in bounds(c)[2:]]


def stage_keys(c):
    result=set();ps=c['prefixes'];caps=c['capacities']
    for node,minimum in enumerate(minimum_prefixes(c)):
        for p in ps:
            if p>=minimum:
                result.add((node,'uniform_gemm',0,0,p))
                result.update((node,'packed_gemm',cap,0,p) for cap in caps if cap>=p)
                result.update((node,'lift',0,p,q) for q in ps if q>p)
            for cap in caps:
                if cap>=p:result.update((node,'pack',cap,old,p) for old in [0,*ps] if old<p)
    result.add((len(graph(c))-1,'publish',0,0,c['prefix']))
    return result


def cell_key(cell):
    return tuple(cell[k] for k in ('node','kind','capacity','old_prefix','prefix'))


def validate_capture(c,entry):
    case=validate_case(entry['case'])
    fixed=dict(schema='rns8_product_stage_capture_v1',case=case,run_id=entry['run_id'],cost_schema_version=2,
        dataset=dataset_identity(case),cache_state='hot_repeated_stage',timing_source='host_monotonic_completed_backend_call',
        host_instrumentation_state='hooks_disabled',exact_reference_passed=True,promotion_eligible=False)
    for key,value in fixed.items():
        if type(c.get(key)) is not type(value) or c.get(key)!=value:raise ValueError('stage capture '+key)
    if not manifest_provenance_complete(c.get('build_manifest')) or not c.get('execution_id') or not c.get('device') or type(c.get('runtime_version')) is not int:
        raise ValueError('stage compiled/process provenance')
    artifact=c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or not digest(artifact.get('executable_sha256')):raise ValueError('stage artifact')
    identity=c.get('identity') or {}
    if set(identity)!={'graph_sha256','build_sha256','target','node_count','minimum_prefixes'} or not all(digest(identity[k]) for k in ('graph_sha256','build_sha256')) or not identity.get('target'):
        raise ValueError('stage graph/build identity')
    if type(identity['node_count']) is not int or identity['node_count']!=len(graph(case)) or identity['minimum_prefixes']!=minimum_prefixes(case) or any(type(p) is not int for p in identity['minimum_prefixes']):
        raise ValueError('stage independent exact prefixes')
    cells=case['side']**2;p=case['prefix'];tile_cells=((case['side']+15)//16*16)**2
    payload=5*cells*p+tile_cells*case['capacities'][-1]+4+cells*(p-1)
    if type(c.get('logical_device_payload_bytes')) is not int or c['logical_device_payload_bytes']!=payload or not positive(c.get('setup_us')) or not positive(c.get('calibration_total_us')) or c['calibration_total_us']<c['setup_us'] or not positive(c.get('nominal_clock_tick_ns')):
        raise ValueError('stage setup/payload/clock')
    stages=c.get('stages');seen=set();expected_keys=stage_keys(case)
    if type(stages) is not list:raise ValueError('stage records')
    for stage in stages:
        fields={'node','kind','capacity','old_prefix','prefix','elapsed_ns','sample_h2d_bytes','sample_d2h_bytes','sample_d2d_bytes','exact_reference_passed'}
        if type(stage) is not dict or set(stage)!=fields or stage['exact_reference_passed'] is not True:raise ValueError('stage cell ABI/reference')
        if any(type(stage[k]) is not int or stage[k]<0 for k in ('node','capacity','old_prefix','prefix','sample_h2d_bytes','sample_d2h_bytes','sample_d2d_bytes')):raise ValueError('stage physical axes/counters')
        if type(stage['kind']) is not str:raise ValueError('stage operation kind')
        key=cell_key(stage)
        if key in seen or key not in expected_keys:raise ValueError('duplicate/unrequested stage')
        seen.add(key)
        times=stage['elapsed_ns']
        if type(times) is not list or len(times)!=case['repeats'] or any(not positive(t) or t<c['nominal_clock_tick_ns'] for t in times):raise ValueError('stage samples')
    if seen!=expected_keys:raise ValueError('missing stage measurements')
    return c


def price_bundle(observations):
    if not 5<=len(observations)<=100:raise ValueError('five independent training processes required')
    captures=[validate_capture(c,e) for e,c in observations]
    for key in ('run_id','execution_id'):
        if len({c[key] for c in captures})!=len(captures):raise ValueError('duplicate training process')
    def cohort(c):
        return {k:c[k] for k in ('case','identity','build_manifest','device','runtime_version','dataset','cache_state','timing_source','host_instrumentation_state')} | {
            'artifact_sha256':c['artifact_capture']['executable_sha256'],'environment':(c.get('request_identity') or {}).get('environment')}
    expected=cohort(captures[0])
    if any(cohort(c)!=expected for c in captures):raise ValueError('mixed stage cohorts')
    by_key=[{cell_key(s):s for s in c['stages']} for c in captures];cells=[];reports=[]
    for key in sorted(by_key[0]):
        interval=independent_run_interval([table[key]['elapsed_ns'] for table in by_key],max(c['nominal_clock_tick_ns'] for c in captures))
        reports.append(dict(key=key,interval=interval))
        # No zero/censored/unstable price substitution. Unknown cells remain
        # absent and the runtime keeps their UINT64_MAX sentinel.
        if interval['lower_ns']<=0:continue
        row=dict(zip(('node','kind','capacity','old_prefix','prefix'),key))
        row.update(cost_ns=math.ceil(interval['mean_ns']),interval_ns=[interval['lower_ns'],interval['upper_ns']]);cells.append(row)
    if not cells:raise ValueError('no stable stage prices')
    bundle=dict(schema='rns8_product_stage_prices_v1',cost_schema_version=2,identity=captures[0]['identity'],
        training_run_ids=[c['run_id'] for c in captures],training_execution_ids=[c['execution_id'] for c in captures],
        source_artifacts=[dict(executable_sha256=c['artifact_capture']['executable_sha256'],capture_sha256=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()) for c in captures],
        dataset=captures[0]['dataset'],cache_state=captures[0]['cache_state'],timing_source=captures[0]['timing_source'],device=captures[0]['device'],runtime_version=captures[0]['runtime_version'],
        training_charge_ns=math.ceil(sum(c['calibration_total_us'] for c in captures)*1000),cells=cells,qualified=False,promotion_eligible=False)
    price_cells(bundle)
    return bundle,dict(schema='rns8_product_stage_training_report_v1',cohort=expected,intervals=reports,
        charged_training_us=sum(c['calibration_total_us'] for c in captures),training_cost_scope='context_setup_warmups_samples_fixture_transfers_and_oracles_excluding_process_start_and_output_serialization',
        resolution_scope='nominal_clock_tick_only_not_empirically_qualified',actual_hardware_traffic_bytes=None,isa_qualified=False,promotion_eligible=False)


def held_out_report(bundle,observations,relative_tolerance):
    if not positive(relative_tolerance,True) or relative_tolerance>=1:raise ValueError('explicit tolerance in [0,1) required')
    if len(observations)<5:raise ValueError('five held-out processes required')
    training=set(bundle['training_run_ids'])|set(bundle['training_execution_ids']);seen=set();costs=price_cells(bundle);rows=[]
    for entry,c in observations:
        validate_recipe(c,entry)
        ids={c['run_id'],c['execution_id']}
        if len(ids)!=2 or ids & (training|seen):raise ValueError('held-out/training process overlap')
        seen.update(ids)
        if entry['case']['mode']!='recipe-measured' or c.get('stage_price_identity')!=price_identity(bundle):
            raise ValueError('held-out frozen price identity')
        receipt=c['materialization_plan'];ps=c['selected_compute_prefixes'];flags=receipt['packed_b'];nodes=graph(entry['case']);p=entry['case']['prefix']
        capacities={v['value_id']:v['capacity_prefix'] for v in receipt['packed_values']};raw=[p,p]+[0]*len(nodes);packed={};lower=upper=0
        policy=receipt['replay_policy']
        if policy['retain_external_b']:packed={v:cap for v,cap in capacities.items() if v<2}
        def add(key):
            nonlocal lower,upper
            if key not in costs:raise ValueError('unmeasured selected stage')
            lo,hi=costs[key]['interval_ns'];lower+=lo;upper+=hi
        def demand(value,target):
            if value>=2 and raw[value]<target:add((value-2,'lift',0,raw[value],target));raw[value]=target
        for i,((a,b),q,flag) in enumerate(zip(nodes,ps,flags)):
            demand(a,q)
            if flag:
                cap=capacities[b];old=packed.get(b,0)
                if old<q:demand(b,q);add((i,'pack',cap,old,q));packed[b]=q
                add((i,'packed_gemm',cap,0,q))
            else:demand(b,q);add((i,'uniform_gemm',0,0,q))
            raw[i+2]=q
        demand(len(nodes)+1,p);add((len(nodes)-1,'publish',0,0,p))
        overhead=policy['orchestration_ns']+max(0,policy['publication_ns']-costs[(len(nodes)-1,'publish',0,0,p)]['cost_ns'])
        lower+=overhead;upper+=overhead
        # "publish" prices the final resident D2D copy, already inside execute.
        # Host CRT export has no cell in this price schema and stays separate.
        measured_ns=statistics.mean(s['execute_lift_us']*1000 for s in c['samples'])
        rows.append(dict(execution_id=c['execution_id'],predicted_lower_ns=lower,predicted_upper_ns=upper,measured_mean_ns=measured_ns,
            within_tolerance=lower*(1-relative_tolerance)<=measured_ns<=upper*(1+relative_tolerance)))
    cohorts={json.dumps({**{k:c[k] for k in ('case','build_manifest','target','device','runtime_version','selected_compute_prefixes')},
        'materialization_plan':{k:v for k,v in c['materialization_plan'].items() if k not in ('completed_packing_calls','completed_replays','reused_external_packing_calls')}},sort_keys=True) for _,c in observations}
    if len(cohorts)!=1:raise ValueError('held-out prediction cohort changed')
    measured=independent_run_interval([[s['execute_lift_us']*1000 for s in c['samples']] for _,c in observations],max(c['nominal_clock_tick_ns'] for _,c in observations))
    lower,upper=rows[0]['predicted_lower_ns'],rows[0]['predicted_upper_ns']
    interval_passed=measured['lower_ns']>=lower*(1-relative_tolerance) and measured['upper_ns']<=upper*(1+relative_tolerance)
    return dict(schema='rns8_product_stage_held_out_v1',relative_tolerance=relative_tolerance,cases=rows,measured_process_interval=measured,
        held_out_passed=interval_passed and all(row['within_tolerance'] for row in rows),scope='independent_processes_same_graph_dataset_and_build',prediction_boundary='resident_recipe_execution_including_publication',separate_boundary='final_host_crt_export',common_cost_scope='declared_orchestration_and_publication_override_are_modeled',promotion_eligible=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path);parser.add_argument('--case',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review-held-out-plan',type=Path);parser.add_argument('--prices',type=Path);parser.add_argument('--relative-tolerance',type=float)
    parser.add_argument('--runs',type=int,default=5);parser.add_argument('--seed',type=int,default=4231);parser.add_argument('--timeout-seconds',type=float,default=600);parser.add_argument('--plan-only',action='store_true')
    a=parser.parse_args()
    if a.review_held_out_plan:
        if a.prices is None or a.relative_tolerance is None or a.binary or a.case:raise ValueError('review requires frozen prices and explicit tolerance only')
        plan=strict_json(a.review_held_out_plan.read_text())['observations'];observations=[(e,strict_json(Path(e['output']).read_text())) for e in plan if e['case']['mode']=='recipe-measured']
        report=held_out_report(strict_json(a.prices.read_text()),observations,a.relative_tolerance)
        with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        return 0
    if a.binary is None or a.case is None or a.prices or a.relative_tolerance is not None:raise ValueError('collection requires binary and case')
    if not 5<=a.runs<=100:raise ValueError('five independent training processes required')
    plan=collection_plan(a.binary,[a.case],a.output,a.runs,a.seed,str(uuid.uuid4()),case_validator=validate_case);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'plan.json').open('x') as f:json.dump(dict(schema='product_stage_collection_v1',observations=plan,promotion_eligible=False),f,indent=2)
    if a.plan_only:return 0
    observations=[]
    for entry in plan:
        path=Path(entry['output'])
        if path.exists():raise ValueError('immutable stage capture exists')
        if not run_command(entry['command'],path,a.timeout_seconds):raise RuntimeError('stage capture failed '+str(path))
        observations.append((entry,strict_json(path.read_text())))
    prices,report=price_bundle(observations)
    for name,value in [('prices.json',prices),('report.json',report)]:
        with (a.output/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
