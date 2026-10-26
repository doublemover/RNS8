"""Bounded system-roof observations; exact source counts are not hardware traffic."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import uuid
from functools import lru_cache
from benchmark_request import request_environment_identity
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from collect_int4_calibration import collection_plan
from collect_native_calibration import strict_json

MODES=('global-copy','lds-read','lds-read-write','launch','event','descriptor','graph')


def validate_case(c):
    fields={'schema','device','mode','words','blocks','iterations','stride','displacement_words','budget_bytes','warmups','repeats'}
    if type(c) is not dict or set(c)!=fields or c['schema']!='rns8_system_roof_case_v1' or c['mode'] not in MODES:
        raise ValueError('case ABI')
    for field,lo,hi in [('device',0,63),('words',1,2**29),('blocks',1,4096),('iterations',1,4096),
            ('stride',1,32),('displacement_words',0,2**29),('budget_bytes',1,2**40),('warmups',1,100),('repeats',3,100)]:
        if type(c[field]) is not int or not lo<=c[field]<=hi:raise ValueError(field)
    lds=c['mode'].startswith('lds-')
    if c['stride']&(c['stride']-1) or (lds and (c['words']!=128*c['blocks'] or c['words']*c['iterations']>2**28)):
        raise ValueError('LDS shape/work domain')
    if not lds and (c['iterations']!=1 or c['stride']!=1):raise ValueError('non-LDS schedule')
    if c['mode'] in ('launch','event','graph') and (c['words']!=1 or c['blocks']!=1):raise ValueError('tiny operation domain')
    q=resources(c)
    if q['admitted_payload_bytes']>c['budget_bytes']:raise ValueError('complete live payload budget')
    return c


def resources(c):
    n=c['words'];lds=c['mode'].startswith('lds-');iteration=c['iterations'];displace=c['displacement_words']
    device=(2*n+displace+16)*4;pinned=(n+max(n+16,65536 if displace else 0))*4
    return dict(device_payload_bytes=device,pinned_payload_bytes=pinned,admitted_payload_bytes=device+pinned+65536,
        lds_bytes_per_block=128*c['stride']*4 if lds else 0,
        modeled_unique_source_bytes=0 if c['mode']=='event' else n*4,
        modeled_global_read_bytes=0 if c['mode'] in ('event','descriptor') else n*4,
        modeled_global_write_bytes=0 if c['mode']=='event' else n*4,
        modeled_lds_read_bytes=n*4*iteration if lds else 0,
        modeled_lds_write_bytes=n*4*(iteration+1 if c['mode']=='lds-read-write' else 1) if lds else 0,
        modeled_displacement_bytes=displace*8)


def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0
def digest(x):return type(x) is str and re.fullmatch('[0-9a-f]{64}',x) is not None


def floor_sum(n,m,a,b):
    """Exact Euclidean sum floor((a*i+b)/m), without visiting huge buffers."""
    result=0
    while True:
        if a>=m:result+=(n-1)*n*(a//m)//2;a%=m
        if b>=m:result+=n*(b//m);b%=m
        top=a*n+b
        if top<m:return result
        n,b=divmod(top,m);m,a=a,m


@lru_cache(maxsize=128)
def checksums(mode,words,blocks,iterations):
    modulus=2**32;mask=modulus-1;a=747796405;b=2891336453
    source=(a*words*(words-1)//2+b*words-modulus*floor_sum(words,modulus,a,b))%(2**64)
    if mode=='event':output=None
    elif mode in ('launch','graph'):output=(b+1)&mask
    elif not mode.startswith('lds-'):output=source
    else:
        output=0
        for block in range(blocks):
            values=[(a*(block*128+lane)+b)&mask for lane in range(128)];sums=[0]*128
            for iteration in range(iterations):
                read=values[iteration%128:]+values[:iteration%128]
                sums=[(x+y)&mask for x,y in zip(sums,read)]
                if mode=='lds-read-write':values=[(value+iteration+lane)&mask for lane,value in enumerate(read)]
            output+=sum(sums)
        output%=2**64
    return source,output


def validate_capture(c,e):
    case=validate_case(e['case'])
    fixed=dict(schema='rns8_system_roof_microcase_v1',case=case,run_id=e['run_id'],device=case['device'],
        actual_hbm_bytes=None,actual_lds_transactions=None,physical_cache_eviction_proven=False,
        clock_power_policy='inherited_unobserved',lds_bank_mapping='unknown_until_named_target_instruction_review',
        timer='host_steady_clock_nanoseconds_and_same_stream_HIP_events',
        host_scope='complete_borrowed_probe_on_owned_stream_including_requested_displacement_and_stream_completion',
        event_scope='probe_only_excludes_displacement_includes_recorded_operation_and_required_LDS_barriers',
        arithmetic='uint32_mod_2_32_probe_only',oracle_inside_interval=False,exact_reference_passed=True,
        production_timing_hooks_enabled=False,promotion_eligible=False,allocator_overhead_known=False,
        input_seed=2891336453,input_multiplier=747796405,descriptor_buffer_kind='portable_pinned_owned',isa_verified=False,isa_sha256=None,
        checksum_contract='sum_of_all_canonical_uint32_values_mod_2_64_not_a_correctness_substitute',
        **resources(case))
    fixed['source_u64_sum'],fixed['output_u64_sum']=checksums(case['mode'],case['words'],case['blocks'],case['iterations'])
    if type(c) is not dict:raise ValueError('capture object')
    for key,value in fixed.items():
        if type(c.get(key)) is not type(value) or c[key]!=value:raise ValueError('capture '+key)
    if not manifest_provenance_complete(c.get('build_manifest')) or not c.get('execution_id') or not c.get('target') or not c.get('device_name'):
        raise ValueError('target/build/process identity')
    if c.get('host_platform') not in ('native-windows','native-linux','other') or type(c.get('device_uuid')) is not str or re.fullmatch('[0-9a-f]{32}',c['device_uuid']) is None or not c.get('device_bdf'):
        raise ValueError('platform/runtime device identity')
    for field in ('runtime_version','driver_version','setup_ns','first_ns'):
        if not positive(c.get(field)):raise ValueError(field)
    if case['mode']=='graph':
        if not positive(c.get('capture_instantiate_ns')):raise ValueError('graph preparation')
    elif c.get('capture_instantiate_ns') is not None:raise ValueError('unexpected capture')
    request=c.get('request_identity') or {};artifact=c.get('artifact_capture') or {}
    environment,error=request_environment_identity(request)
    if error or artifact.get('unchanged_during_run') is not True or not digest(artifact.get('executable_sha256')) or request.get('executable_sha256')!=artifact['executable_sha256'] or request.get('arguments')!=e['command'][1:]:
        raise ValueError('artifact/request identity')
    samples=c.get('samples');bursts=c.get('sustained')
    if type(samples) is not list or len(samples)!=case['repeats'] or type(bursts) is not list or len(bursts)!=3:raise ValueError('sample inventory')
    for row in samples:
        if set(row)!={'instrumented_host_ns','production_host_ns','device_event_ns','exact_reference_passed'} or row['exact_reference_passed'] is not True or not all(positive(row.get(k)) for k in ('instrumented_host_ns','production_host_ns')) or (row['device_event_ns'] is not None and not positive(row['device_event_ns'])):
            raise ValueError('sample clock/censoring/reference')
    calls=1+case['warmups']+2*case['repeats']
    for burst in bursts:
        if type(burst) is not dict or set(burst)!={'calls','elapsed_ns','duration_target_met','call_cap_reached','exact_reference_passed'}:raise ValueError('burst ABI')
        n=burst['calls'];elapsed=burst['elapsed_ns']
        if type(n) is not int or not 16<=n<=4096 or not positive(elapsed) or burst['exact_reference_passed'] is not True:raise ValueError('sustained clock/count/reference')
        for field,expected in [('duration_target_met',n>=16 and elapsed>=250000000),('call_cap_reached',n==4096)]:
            if type(burst[field]) is not bool or burst[field]!=expected:raise ValueError('burst censoring')
        if not burst['duration_target_met'] and not burst['call_cap_reached']:raise ValueError('early burst')
        calls+=n
    if type(c.get('calls')) is not int or c['calls']!=calls or type(c.get('displacement_calls')) is not int or c['displacement_calls']!=(calls if case['displacement_words'] else 0):raise ValueError('complete calls')
    return dict(mean_ns=statistics.mean(row['production_host_ns'] for row in samples),environment=environment)


def summarize(observations):
    groups={};seen=set()
    for entry,c in observations:
        checked=validate_capture(c,entry);identity=c['execution_id']
        if identity in seen:raise ValueError('duplicate independent process')
        seen.add(identity)
        key=json.dumps(dict(case=entry['case'],target=c['target'],device_name=c['device_name'],environment=checked['environment'],
            host_platform=c['host_platform'],device_uuid=c['device_uuid'],device_bdf=c['device_bdf'],runtime=c['runtime_version'],driver=c['driver_version'],
            artifact=c['artifact_capture']['executable_sha256'],build=c['build_manifest']),sort_keys=True)
        groups.setdefault(key,[]).append((entry['independent_run'],checked['mean_ns']))
    result=[]
    for key,rows in groups.items():
        if len(rows)<2 or len({r[0] for r in rows})!=len(rows):raise ValueError('independent process inventory')
        values=[r[1] for r in rows]
        result.append(dict(identity=json.loads(key),process_means_ns=values,mean_ns=statistics.mean(values),
            process_range_ns=[min(values),max(values)],promotion_eligible=False,planner_rate_admitted=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runs',type=int,default=5);p.add_argument('--seed',type=int,default=3131)
    p.add_argument('--timeout-seconds',type=float,default=1800);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if not positive(a.timeout_seconds):raise ValueError('bounded timeout')
    plan=collection_plan(a.binary,a.cases,a.output,a.runs,a.seed,str(uuid.uuid4()),case_validator=validate_case)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'plan.json').open('x') as stream:json.dump(dict(schema='rns8_system_roof_collection_v1',observations=plan),stream,indent=2)
    if a.plan_only:return 0
    observations=[]
    for entry in plan:
        if hashlib.sha256(Path(entry['command'][1]).read_bytes()).hexdigest()!=entry['case_sha256']:raise ValueError('case changed')
        path=Path(entry['output'])
        if path.exists():raise ValueError('capture exists')
        if not run_command(entry['command'],path,a.timeout_seconds):raise RuntimeError('capture failed '+str(path))
        capture=strict_json(path.read_text());validate_capture(capture,entry);observations.append((entry,capture))
    with (a.output/'summary.json').open('x') as stream:json.dump(dict(schema='rns8_system_roof_summary_v1',cohorts=summarize(observations),promotion_eligible=False),stream,indent=2)
    return 0

if __name__=='__main__':raise SystemExit(main())
