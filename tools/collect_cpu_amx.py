"""Matched independent scalar/AMX whole-product costs, never automatic promotion."""
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

MODES = ('cpu-reference', 'cpu-amx-int8')
MODULI = (256,255,253,251,247,239,233,229,227,223,217,211,199,197,193,191,181,179,173,167)


def validate_case(c):
    keys = {'schema','m','n','k','prefix','threads','modulus','warmups','repeats','seed','mode','residency','semantics'}
    if type(c) is not dict or set(c) != keys or c['schema'] != 'cpu_amx_case_v1':
        raise ValueError('case ABI')
    for key, lo, hi in [('m',1,512),('n',1,512),('k',1,131073),('prefix',0,20),('threads',1,256),('modulus',0,256),('warmups',1,100),('repeats',3,100),('seed',0,2**32-1)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError(key)
    if c['mode'] not in MODES or c['residency'] not in ('cold','resident','repack-a','repack-b','repack-both') or c['semantics'] not in ('finite','wide-signed','wide-unsigned','bounded-signed','bounded-unsigned'):
        raise ValueError('route')
    m,n,k,p = (c[x] for x in ('m','n','k','prefix'))
    if m*n*k > 134217728:
        raise ValueError('oracle work envelope')
    if c['semantics'] == 'finite':
        if not 2 <= c['modulus'] <= 256 or p != 0:
            raise ValueError('finite modulus/prefix')
    else:
        if c['modulus'] or not p:
            raise ValueError('integer modulus/prefix')
        unsigned = c['semantics'].endswith('unsigned')
        required = k*(2**128 if unsigned else 2**127) if c['semantics'].startswith('wide') else k*(196 if unsigned else 98)
        if math.prod(MODULI[:p]) <= required:
            raise ValueError('strict public plan range')
    return c


def positive(value, zero=False):
    return type(value) in (int,float) and math.isfinite(value) and (value >= 0 if zero else value > 0)


def validate_capture(c, entry):
    case=validate_case(entry['case']);amx=case['mode']=='cpu-amx-int8';finite=case['semantics']=='finite'
    fixed=dict(schema='rns8_cpu_amx_microcase_v1',case=case,run_id=entry['run_id'],
        output_boundary='public_host_pack_dense_gemm_complete_padded_export',
        gemm_scope='public_range_dispatch_compute_reduction_publication_including_AMX_pack_when_selected',
        sustained_scope='complete_calls_cleanup_and_harness_clock_json_overhead_no_oracle',
        sample_scope='disjoint_operation_and_cleanup_phase_sum_excludes_between_phase_oracle',
        oracle_inside_interval=False,all_residue_planes_checked=True,exact_reference_passed=True,promotion_eligible=False)
    for key,value in fixed.items():
        if type(c.get(key)) is not type(value) or c.get(key)!=value:
            raise ValueError('capture '+key)
    if not manifest_provenance_complete(c.get('build_manifest')) or not c.get('execution_id'):
        raise ValueError('compiled/process provenance')
    artifact=c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256',''))!=64:
        raise ValueError('artifact identity')
    selected='cpu_amx_int8_16x16x64_centered_v1' if amx else 'cpu_reference_finite_u8_gemm_v1' if finite else 'cpu_reference_scalar_rns_gemm_v1'
    if c.get('selected_kernel') != selected:
        raise ValueError('selected implementation')
    env=c.get('environment') or {}
    if not env.get('cpu_model') or env.get('cpu_model')=='unavailable' or not env.get('os') or type(env.get('process_id')) is not int or env['process_id']<=0 or env.get('requested_threads')!=case['threads'] or env.get('parallel_runtime') not in ('serial','openmp') or env.get('affinity_policy')!='inherited_no_pinning' or env.get('page_placement_observed') is not False:
        raise ValueError('CPU OS thread provenance')
    if type(env.get('physical_core_count')) not in (int,type(None)) or (env.get('physical_core_count') is not None and env['physical_core_count']<1) or env.get('topology_scope')!='proc_cpuinfo_visible_hardware_not_affinity_filtered':
        raise ValueError('physical core topology scope')
    if env.get('allowed_cpus') in (None,'','unavailable') or env.get('allowed_memory_nodes') in (None,'','unavailable'):
        raise ValueError('affinity/NUMA envelope')
    if type(env.get('maximum_threads')) is not int or env['maximum_threads']<1 or type(env.get('hardware_threads')) is not int or env['hardware_threads']<1:
        raise ValueError('thread observation')
    if amx:
        info=c.get('admission') or {}
        for field in ('compiled','cpuid_tile','cpuid_int8','osxsave','palette_supported','os_supported','permission_enabled','xcr0_enabled','admitted'):
            if type(info.get(field)) is not int or info[field]!=1:
                raise ValueError('AMX admission '+field)
        if info.get('permission_errno') != 0:
            raise ValueError('AMX permission error')
        config=c['build_manifest']['configuration']
        if not any(line.split('=',1)[0]=='cpu_amx' and line.split('=',1)[1].upper() in ('ON','TRUE','1','YES') for line in config.splitlines() if '=' in line):
            raise ValueError('AMX build flag')
    if not positive(c.get('setup_us'),True) or len(c.get('samples',[]))!=case['repeats']:
        raise ValueError('sample count/setup')
    for sample in [c.get('first_call',{}),*c['samples']]:
        phases=[sample.get(k) for k in ('prepare_us','gemm_us','crt_export_us','cleanup_us')]
        if not all(positive(v,True) for v in phases) or not positive(sample.get('total_us')) or not math.isclose(sum(phases),sample['total_us'],rel_tol=1e-7,abs_tol=1e-6):
            raise ValueError('complete phases')
    bursts=c.get('sustained',[])
    if len(bursts)!=3:
        raise ValueError('sustained count')
    for burst in bursts:
        count=burst.get('calls');duration=burst.get('wall_us');operation=burst.get('operation_us')
        if type(count) is not int or not 16<=count<=4096 or not positive(duration) or (duration<250000 and count!=4096) or not positive(operation) or operation>duration+1e-6 or not math.isclose(burst.get('per_call_us',-1),duration/count,rel_tol=1e-7):
            raise ValueError('sustained envelope')
        energy=burst.get('energy')
        if energy is not None and (type(energy) is not dict or energy.get('scope')!='package0_shared_system_not_process' or energy.get('counter_wrap_count_known') is not False or type(energy.get('range_uj')) is not int or energy['range_uj']<=0 or type(energy.get('delta_uj_mod_counter')) is not int or not 0<=energy['delta_uj_mod_counter']<energy['range_uj']):
            raise ValueError('energy counter scope')
    calls=1+case['warmups']+case['repeats']+sum(b['calls'] for b in bursts)+3
    if type(c.get('calls')) is not int or c['calls']!=calls:
        raise ValueError('complete invocation count')
    residency=case['residency']
    for side in ('a','b'):
        expected=calls if residency=='cold' else 1+calls if residency in ('repack-both','repack-'+side) else 1
        if c.get('pack_'+side+'_calls')!=expected:
            raise ValueError('public pack count '+side)
    w=c.get('last_workspace')
    if amx:
        if type(w) is not dict or any(type(w.get(key)) is not int or w[key]<0 for key in ('prepared_threads','last_execution_threads','owned_payload_bytes','packed_b_bytes','output_staging_bytes','b_matrix_instance','b_generation','b_pack_count','b_reuse_count','layout_version','safe_k')):
            raise ValueError('workspace receipt')
        m,n,k=case['m'],case['n'],case['k'];p=1 if finite else case['prefix']
        if w['layout_version']!=1 or w['safe_k']!=65536 or not 1<=w['last_execution_threads']<=w['prepared_threads']<=case['threads'] or not w['b_matrix_instance'] or not w['b_generation']:
            raise ValueError('workspace owner/worker contract')
        if w['packed_b_bytes']<p*((n+15)//16)*((k+63)//64)*1024 or w['output_staging_bytes']<p*m*n or w['owned_payload_bytes']<w['packed_b_bytes']+w['output_staging_bytes']+w['prepared_threads']*2048:
            raise ValueError('complete owned payload')
        packs=1 if residency in ('cold','resident','repack-a') else calls
        reuses=calls-1 if residency in ('resident','repack-a') else 0
        if (w['b_pack_count'],w['b_reuse_count'])!=(packs,reuses):
            raise ValueError('generation-bound B cache counts')
    elif w is not None:
        raise ValueError('scalar must not consume optimized workspace')
    return c


def summarize(observations):
    cohorts={}
    for entry,c in observations:
        validate_capture(c,entry)
        env={k:v for k,v in c['environment'].items() if k!='process_id'}
        identity=dict(case={k:v for k,v in entry['case'].items() if k!='mode'},build=c['build_manifest'],environment=env,
            artifact={k:c['artifact_capture'].get(k) for k in ('executable_sha256','rns8_linkage','shared_library_sha256')})
        cohorts.setdefault(json.dumps(identity,sort_keys=True),[]).append(dict(mode=entry['case']['mode'],execution_id=c['execution_id'],independent_run=entry['independent_run'],
            median_us=statistics.median(s['total_us'] for s in c['samples']),sustained_us=statistics.median(s['per_call_us'] for s in c['sustained']),output=entry['output']))
    result=[]
    for identity,rows in cohorts.items():
        if len({r['execution_id'] for r in rows})!=len(rows):
            raise ValueError('duplicate process receipt')
        median={};sustained={}
        for mode in MODES:
            selected=[r for r in rows if r['mode']==mode]
            if len({r['independent_run'] for r in selected})<2:
                raise ValueError('complete independently repeated scalar/AMX pair required')
            median[mode]=statistics.median(r['median_us'] for r in selected)
            sustained[mode]=statistics.median(r['sustained_us'] for r in selected)
        result.append(dict(identity=json.loads(identity),observations=rows,median_complete_phase_sum_us=median,sustained_wall_us=sustained,
            measured_amx_over_scalar_cost_ratio=median['cpu-amx-int8']/median['cpu-reference'],
            slower_amx_preserved=median['cpu-amx-int8']>=median['cpu-reference'],promotion_eligible=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--cases',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--runs',type=int,default=2);parser.add_argument('--seed',type=int,default=5454);parser.add_argument('--timeout-seconds',type=float,default=1200);parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    plan=collection_plan(args.binary,args.cases,args.output,args.runs,args.seed,str(uuid.uuid4()),case_validator=validate_case)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'plan.json').open('x') as stream:json.dump(dict(schema='cpu_amx_collection_v1',observations=plan,promotion_eligible=False),stream,indent=2)
    if args.plan_only:return 0
    observations=[]
    for entry in plan:
        path=Path(entry['output'])
        if path.exists():raise ValueError('immutable capture exists')
        if not run_command(entry['command'],path,args.timeout_seconds):raise RuntimeError('capture failed: '+str(path))
        observations.append((entry,strict_json(path.read_text())))
    with (args.output/'summary.json').open('x') as stream:json.dump(dict(schema='cpu_amx_summary_v1',cohorts=summarize(observations),promotion_eligible=False),stream,indent=2)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
