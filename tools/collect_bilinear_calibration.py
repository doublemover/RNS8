"""Counterbalanced public bilinear controls; diagnostics never become timings."""
import argparse
import json
import math
from pathlib import Path
import statistics
import uuid
from collect_int4_calibration import collection_plan, positive
from collect_native_calibration import strict_json, DEFAULT_MODULI
from build_manifest import manifest_provenance_complete
from benchmark_sweep_lib.execution import run_command

MODES=('classical','strassen','coefficient-rank23','september-slp','october55-retained','october55-recompute')


def resources(c):
    m,n,k,p=c['m'],c['n'],c['k'],c['prefix'];mode=c['mode']
    residue=p*(m*k+k*n+m*n)
    host=8*(m*(k+1)+k*(n+2)+m*n)+24*m*(n+3)+residue+p*m*n
    if mode=='classical':return 0,residue,host
    blocks=2 if mode=='strassen' else 3;hm,hn,hk=((x+blocks-1)//blocks for x in (m,n,k))
    a,b,product=hm*hk*p,hk*hn*p,hm*hn*p
    retained=mode in ('september-slp','october55-retained')
    scratch=(23 if retained else 1)*(a+b)+(23 if mode in ('september-slp','october55-retained','october55-recompute') else 1)*product+p*m*n
    return scratch,residue,host


def validate_case(c):
    fields={'schema','m','n','k','prefix','unsigned_inputs','mode','subproduct_backend','residency','instrumented','repeats','warmups','seed','max_scratch_bytes','max_host_bytes','matrix_variant'}
    if not isinstance(c,dict) or set(c)!=fields or c['schema']!='bilinear_case_v1':raise ValueError('case ABI')
    for key,lo,hi in [('matrix_variant',0,2),('m',1,32768),('n',1,32768),('k',1,196611),('prefix',1,20),('repeats',3,100),('warmups',1,100),('seed',0,2**32-1),('max_scratch_bytes',1,2**40),('max_host_bytes',1,2**40)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError(key)
    if type(c['unsigned_inputs']) is not bool or type(c['instrumented']) is not bool or c['mode'] not in MODES or c['subproduct_backend'] not in ('direct','matrix') or c['residency'] not in ('cold','resident','repack'):raise ValueError('route')
    if c['subproduct_backend']=='direct' and c['matrix_variant']:raise ValueError('matrix variant on direct route')
    if math.prod(DEFAULT_MODULI[:c['prefix']])<=c['k']*49*(1 if c['unsigned_inputs'] else 2):raise ValueError('strict CRT capacity')
    blocks=1 if c['mode']=='classical' else 2 if c['mode']=='strassen' else 3
    if c['subproduct_backend']=='matrix' and (c['k']+blocks-1)//blocks>65536:raise ValueError('matrix K cap')
    scratch,_,host=resources(c)
    if scratch>c['max_scratch_bytes'] or host>c['max_host_bytes']:raise ValueError('declared payload budget')
    # Complete stack qualification of the actual subproduct, not the padded
    # output staging buffer whose indices are independently widened.
    dims=[(c[x]+blocks-1)//blocks for x in ('m','n','k')];m,n,k=dims
    if max(m*n,m*k,k*n)*c['prefix']>2147483647 or max(c['m']*c['n'],c['m']*c['k'],c['k']*c['n'])>2147483647:raise ValueError('qualified address domain')
    return c


def validate_capture(c,entry):
    case=validate_case(entry['case']);scratch,residue,host=resources(case);mode=case['mode'];diagnostic=case['instrumented']
    expected=dict(schema='rns8_bilinear_microcase_v1',case=case,run_id=entry['run_id'],dataset='period15_native_small_v1',
        selected_prefix=case['prefix'],algorithm=mode,subproduct_backend=case['subproduct_backend'],
        matrix_family_override=('auto','cdna3_16x16x32_or_target_wmma','cdna3_32x32x16_or_target_wmma')[case['matrix_variant']] if case['subproduct_backend']=='matrix' else 'not_applicable',
        subproduct_launch_schedule='batched23' if case['subproduct_backend']=='matrix' and mode in ('september-slp','october55-retained') else 'serial',
        required_bilinear_scratch_bytes=scratch,resident_residue_payload_bytes=residue,host_payload_envelope_bytes=host,
        memory_scope='logical_live_payload_excludes_driver_allocator_and_opaque_runtime_overhead',
        transform_binary_nodes=56 if mode=='september-slp' else 55 if mode.startswith('october55') else 0,
        input_transform_evaluations_per_cell=23 if mode=='october55-recompute' else 1,
        output_boundary='complete_public_product_and_padded_three_limb_export',host_instrumentation_state='diagnostic_hooks_enabled' if diagnostic else 'hooks_disabled',
        exact_reference_passed=True,all_residue_planes_audited=True,residue_audit_scope='post_timing_public_product',oracle_inside_production_interval=False,promotion_eligible=False,native_useful_products=case['m']*case['n']*case['k'])
    for key,value in expected.items():
        if type(c.get(key)) is not type(value) or c[key]!=value:raise ValueError('capture '+key)
    if not manifest_provenance_complete(c.get('build_manifest')) or not c.get('execution_id'):raise ValueError('compiled/process provenance')
    artifact=c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256',''))!=64:raise ValueError('artifact binding')
    ordinary=c.get('ordinary_workspace_bytes')
    if type(ordinary) is not int or ordinary<0 or c.get('known_peak_device_payload_bytes')!=residue+ordinary+scratch+case['m']*case['n']*24:raise ValueError('peak payload')
    if not positive(c.get('setup_us')) or len(c.get('samples',[]))!=case['repeats']:raise ValueError('sample count')
    for s in [c.get('first_call',{}),*c['samples']]:
        phases=[s.get(k) for k in ('prepare_us','execute_us','export_us','cleanup_us')]
        if not all(positive(v,True) for v in phases) or not positive(s.get('total_us')) or not math.isclose(sum(phases),s['total_us'],rel_tol=1e-7,abs_tol=1e-6):raise ValueError('complete timing boundary')
    if len(c.get('sustained',[]))!=3:raise ValueError('sustained observations')
    for s in c['sustained']:
        count=s.get('calls');duration=s.get('total_us')
        if type(count) is not int or not 16<=count<=4096 or not positive(duration) or (duration<250000 and count!=4096) or not math.isclose(s.get('per_call_us',-1),duration/count,rel_tol=1e-7):raise ValueError('sustained work/duration')
    events=c.get('diagnostic_events')
    if not isinstance(events,list) or (not diagnostic and events):raise ValueError('production hook contamination')
    if diagnostic:
        if not events or any(not isinstance(e.get('label'),str) or not positive(e.get('microseconds'),True) for e in events):raise ValueError('diagnostic event evidence')
        if case['subproduct_backend']=='matrix' and not any(e['label'].startswith('amdgpu_builtin_') and ('wmma_' in e['label'] or 'mfma_' in e['label']) for e in events):raise ValueError('missing actual matrix-engine event')
        if mode.startswith('october55') and not {'rank23_oct55_transform_pair','rank23_oct55_recombine','rank23_oct55_commit'}<={e['label'] for e in events}:raise ValueError('wrong executed DAG')
    return c


def summarize(rows):
    if len({r['execution_id'] for r in rows})!=len(rows):raise ValueError('duplicate process')
    medians={};diagnostics={}
    for mode in MODES:
        selected=[r for r in rows if r['mode']==mode and not r['instrumented']]
        if len({r['independent_run'] for r in selected})>=2:medians[mode]=statistics.median(r['median_us'] for r in selected)
        diagnostics[mode]=[r['path'] for r in rows if r['mode']==mode and r['instrumented']]
    baseline=medians.get('classical');comparisons=[]
    for mode,latency in medians.items():
        if mode=='classical' or baseline is None:continue
        candidate=[r for r in rows if r['mode']==mode and not r['instrumented']]
        base=[r for r in rows if r['mode']=='classical' and not r['instrumented']]
        comparisons.append(dict(mode=mode,complete_route_speedup=baseline/latency,
            known_payload_ratio=max(r['peak_bytes'] for r in candidate)/max(r['peak_bytes'] for r in base),
            diagnostic_companions_present=bool(diagnostics[mode] and diagnostics['classical']),
            promotion_eligible=False,qualification='requires_reviewed_large_shape_complete_memory_ISA_execution_and_variance_evidence'))
    return dict(observations=rows,independent_process_medians_us=medians,diagnostic_companions=diagnostics,comparisons=comparisons,promotion_eligible=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--cases',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=int,default=2);p.add_argument('--seed',type=int,default=4855);p.add_argument('--timeout-seconds',type=float,default=1800);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    plan=collection_plan(a.binary,a.cases,a.output,a.runs,a.seed,str(uuid.uuid4()),validate_case);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'plan.json').exists():raise ValueError('immutable collection exists')
    (a.output/'plan.json').write_text(json.dumps(dict(schema='bilinear_collection_v1',observations=plan,promotion_eligible=False),indent=2)+'\n')
    if a.plan_only:return 0
    cohorts={}
    for entry in plan:
        path=Path(entry['output'])
        if path.exists() or not run_command(entry['command'],path,a.timeout_seconds):raise RuntimeError('capture exists or process failed: '+str(path))
        c=validate_capture(strict_json(path.read_text()),entry)
        identity=dict(case={k:v for k,v in entry['case'].items() if k not in ('mode','instrumented')},target=c.get('target'),device=c.get('device'),runtime=c.get('runtime_version'),
            build=c['build_manifest'],artifact={k:c['artifact_capture'].get(k) for k in ('executable_sha256','rns8_linkage','shared_library_sha256')},environment=(c.get('request_identity') or {}).get('environment'))
        cohorts.setdefault(json.dumps(identity,sort_keys=True),[]).append(dict(mode=entry['case']['mode'],instrumented=entry['case']['instrumented'],execution_id=c['execution_id'],independent_run=entry['independent_run'],path=str(path),median_us=statistics.median(s['total_us'] for s in c['samples']),peak_bytes=c['known_peak_device_payload_bytes']))
    result=[dict(identity=json.loads(identity),**summarize(rows)) for identity,rows in cohorts.items()]
    (a.output/'summary.json').write_text(json.dumps(dict(schema='bilinear_summary_v1',cohorts=result,promotion_eligible=False),indent=2)+'\n');return 0


if __name__=='__main__':raise SystemExit(main())
