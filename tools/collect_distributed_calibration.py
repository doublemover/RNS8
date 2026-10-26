"""Bounded, counterbalanced real modulus-shard collection; no automatic promotion."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import uuid
from benchmark_request import request_environment_identity
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from collect_int4_calibration import collection_plan
from collect_native_calibration import strict_json
from modulus_set_search import DEFAULT_MODULI


def validate_case(case):
    keys = {'schema','m','n','k','prefix','tile','slots','devices','transport','scope',
            'host_budget','pinned_budget','device_budget','warmups','repeats'}
    if type(case) is not dict or set(case)-{'native_integrity'} != keys or case['schema'] != 'rns8_distributed_case_v1':
        raise ValueError('case ABI')
    if 'native_integrity' in case and type(case['native_integrity']) is not bool:raise ValueError('integrity policy')
    for key, low, high in [('m',1,8192),('n',1,8192),('k',1,131073),('prefix',1,20),('tile',1,256),
            ('slots',1,3),('host_budget',1,2**60),('pinned_budget',4,2**30),('device_budget',1,2**60),
            ('warmups',1,100),('repeats',3,100)]:
        if type(case[key]) is not int or not low <= case[key] <= high: raise ValueError(key)
    devices = case['devices']
    if type(devices) is not list or not 1 <= len(devices) <= 64 or any(type(x) is not int or not 0 <= x <= 2**31-1 for x in devices) or len(set(devices)) != len(devices):
        raise ValueError('explicit unique visible device inventory')
    if case['transport'] not in ('local','peer','pinned') or (case['transport']=='local') != (len(devices)==1):
        raise ValueError('explicit transport and single-device control')
    if case['scope'] not in ('resident-device-limbs','resident-host-limbs','cold-host-limbs'):raise ValueError('scope')
    if math.prod(DEFAULT_MODULI[:case['prefix']]) <= 12*case['k']:raise ValueError('strict signed output range')
    if math.ceil(case['m']/case['tile'])*math.ceil(case['n']/case['tile'])*case['prefix'] > 2**20:raise ValueError('descriptor cap')
    return case


def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value > 0

def digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def validate_capture(capture, entry):
    if type(capture) is not dict:raise ValueError('capture object')
    case = validate_case(entry['case'])
    fixed = dict(schema='rns8_distributed_microcase_v1',case=case,run_id=entry['run_id'],
        dataset='signed_separable_7x3x5_v1',timer='single_host_steady_clock_nanoseconds',
        completion_inside_interval=True,oracle_inside_interval=False,exact_reference_passed=True,
        promotion_eligible=False,physical_card_count_known=False,allocator_overhead_known=False,published=True,
        plan_reused=True,production_timing_hooks_enabled=False,affinity_policy='inherited_not_modified',numa_placement_observed=False)
    for key,value in fixed.items():
        if type(capture.get(key)) is not type(value) or capture[key] != value:raise ValueError('capture '+key)
    integrity=case.get('native_integrity',False)
    if capture.get('integrity_requested') is not integrity or capture.get('integrity_completed') is not integrity or type(capture.get('integrity_failure_mask')) is not int or capture['integrity_failure_mask']!=0 or capture.get('checking_moduli') != [257,263]:raise ValueError('independent integrity completion')
    if not manifest_provenance_complete(capture.get('build_manifest')) or not capture.get('execution_id'):
        raise ValueError('compiled/process provenance')
    artifact = capture.get('artifact_capture') or {}; request = capture.get('request_identity') or {}
    cohort,error = request_environment_identity(request)
    if error or artifact.get('unchanged_during_run') is not True or not digest(artifact.get('executable_sha256')) or request.get('executable_sha256') != artifact['executable_sha256'] or request.get('arguments') != entry['command'][1:]:
        raise ValueError('artifact/request/environment identity')
    expected = math.ceil(case['m']/case['tile'])*math.ceil(case['n']/case['tile'])*case['prefix']
    if any(type(capture.get(k)) is not int or capture[k] != expected for k in ('expected_work','local_completed','transfers_queued')):
        raise ValueError('complete expected work membership')
    ranks = capture.get('ranks'); certificates = capture.get('arithmetic_certificates')
    if type(ranks) is not list or len(ranks) != len(case['devices']) or type(certificates) is not list or len(certificates) != len(ranks):raise ValueError('rank/certificate inventory')
    identities=[]
    for rank,(row,ordinal,certificate) in enumerate(zip(ranks,case['devices'],certificates)):
        if type(row) is not dict:raise ValueError('rank record')
        if type(row.get('rank')) is not int or row['rank'] != rank or type(row.get('ordinal')) is not int or row['ordinal'] != ordinal or type(row.get('physical_id')) is not str or not row['physical_id'] or not row.get('target') or type(certificate) is not str or re.fullmatch('[0-9a-f]{64}',certificate) is None:
            raise ValueError('rank identity or arithmetic certificate')
        if not positive(row.get('runtime_version')) or not positive(row.get('driver_version')) or not positive(row.get('requested_device_reservation')) or row['requested_device_reservation'] > case['device_budget']:
            raise ValueError('rank runtime/resource admission')
        identities.append(row['physical_id'])
        for field in ('native_inputs','packed_inputs','local_output','gather'):
            if type(row.get(field)) is not int or row[field] < 0:raise ValueError('rank payload accounting')
        if row['gather'] != (case['m']*case['n']*case['prefix'] if rank==0 else 0):raise ValueError('owner gather extent')
        if row['native_inputs'] != (8*(case['m']*case['k']+case['k']*case['n']) if row['local_output'] or (integrity and rank==0) else 0):raise ValueError('resident source replication')
    if len(set(identities)) != len(identities):raise ValueError('duplicate observed devices')
    if sum(r['local_output'] for r in ranks) != case['m']*case['n']*case['prefix']:raise ValueError('complete residue payload')
    for field in ('caller_output_bytes','host_owner_reservation','pinned_reservation','transfer_bytes_per_call'):
        if type(capture.get(field)) is not int or capture[field] < 0:raise ValueError('global payload accounting')
    if capture['caller_output_bytes'] != case['m']*(case['n']+2)*24 or capture['host_owner_reservation']+capture['caller_output_bytes'] > case['host_budget'] or not 4 <= capture['pinned_reservation'] <= case['pinned_budget']:
        raise ValueError('complete host/pinned budget')
    if capture['host_owner_reservation'] < 8*(case['m']*case['k']+case['k']*case['n'])+case['m']*case['n']*24:raise ValueError('missing native/output owner reservation')
    if capture['transfer_bytes_per_call'] != sum(r['local_output'] for r in ranks[1:]):raise ValueError('declared gather traffic')
    if type(capture.get('plan_sha256')) is not str or re.fullmatch('[0-9a-f]{64}',capture['plan_sha256']) is None:raise ValueError('plan identity')
    samples = capture.get('samples_ns'); bursts = capture.get('sustained')
    if capture.get('host_platform') not in ('native-windows','native-linux','other') or type(capture.get('host_hardware_threads')) is not int or capture['host_hardware_threads'] < 0 or not positive(capture.get('plan_compilation_ns')):raise ValueError('host/plan metadata')
    if type(samples) is not list or len(samples) != case['repeats'] or not all(positive(v) for v in samples) or type(bursts) is not list or len(bursts) != 3 or not positive(capture.get('first_ns')) or not positive(capture.get('setup_ns')):
        raise ValueError('first/setup/sample/sustained counts')
    for burst in bursts:
        if type(burst) is not dict:raise ValueError('sustained record')
        count,ns=burst.get('calls'),burst.get('elapsed_ns')
        if type(count) is not int or not 16 <= count <= 4096 or type(ns) is not int or ns <= 0 or not positive(burst.get('mean_call_ns')) or not math.isclose(burst['mean_call_ns'],ns/count,rel_tol=1e-12):raise ValueError('sustained work normalization')
        if burst.get('duration_reached') is not (ns>=250000000) or burst.get('stop_reason') != ('duration' if ns>=250000000 else 'call_cap') or (ns<250000000 and count!=4096):raise ValueError('censored sustained interval')
    expected_calls=1+case['warmups']+case['repeats']+sum(b['calls'] for b in bursts)
    if capture.get('calls') != expected_calls:raise ValueError('one logical distributed product per call')
    if capture.get('setup_scope') != ('untimed_preparation_admission_probe_and_cleanup' if case['scope']=='cold-host-limbs' else 'retained_preparation'):raise ValueError('setup boundary')
    return dict(environment=cohort,artifact=artifact['executable_sha256'],mean_ns=statistics.median(b['mean_call_ns'] for b in bursts))


def summarize(rows):
    groups={};executions=set();members=set()
    for entry,capture in rows:
        checked=validate_capture(capture,entry)
        if capture['execution_id'] in executions:raise ValueError('duplicate process identity')
        executions.add(capture['execution_id'])
        problem={k:v for k,v in entry['case'].items() if k not in ('devices','transport')}
        key=json.dumps(dict(problem=problem,environment=checked['environment'],artifact=checked['artifact'],host_platform=capture['host_platform'],host_hardware_threads=capture['host_hardware_threads']),sort_keys=True)
        member=(key,tuple(entry['case']['devices']),entry['case']['transport'],entry['independent_run'])
        if member in members:raise ValueError('duplicate independent observation')
        members.add(member)
        groups.setdefault(key,[]).append((entry,capture,checked))
    summaries=[]
    for key,observations in groups.items():
        controls={};indices={}
        for entry,_,_ in observations:
            indices.setdefault((tuple(entry['case']['devices']),entry['case']['transport']),set()).add(entry['independent_run'])
        for route,runs in indices.items():
            if len(runs)<2:raise ValueError('at least two independent observations per route')
            if len(route[0])>1 and runs != indices.get(((route[0][0],),'local')):raise ValueError('incomplete matched process inventory')
        for entry,capture,checked in observations:
            if len(entry['case']['devices'])==1:
                identity=(entry['case']['devices'][0],entry['independent_run'])
                if identity in controls:raise ValueError('duplicate single-device control')
                controls[identity]=(capture,checked)
        comparisons=[]
        for entry,capture,checked in observations:
            if len(entry['case']['devices'])==1:continue
            control=controls.get((entry['case']['devices'][0],entry['independent_run']))
            if control is None:raise ValueError('missing matched single-device process control')
            if control[0]['ranks'][0]['physical_id'] != capture['ranks'][0]['physical_id']:raise ValueError('owner identity changed')
            comparisons.append(dict(independent_run=entry['independent_run'],devices=entry['case']['devices'],transport=entry['case']['transport'],
                single_ns=control[1]['mean_ns'],distributed_ns=checked['mean_ns'],observed_ratio=control[1]['mean_ns']/checked['mean_ns'],promotion_eligible=False))
        summaries.append(dict(identity=json.loads(key),observations=comparisons,qualification='descriptive ratios only; physical topology, native Linux, independent spread/traffic and the 1.55x release gate require review'))
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--cases',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=int,default=5);p.add_argument('--seed',type=int,default=5050)
    p.add_argument('--timeout-seconds',type=float,default=3600);p.add_argument('--plan-only',action='store_true');args=p.parse_args()
    if not positive(args.timeout_seconds):raise ValueError('bounded process timeout')
    plan=collection_plan(args.binary,args.cases,args.output,args.runs,args.seed,str(uuid.uuid4()),case_validator=validate_case)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'plan.json').open('x') as stream:json.dump(dict(schema='rns8_distributed_collection_v1',observations=plan,promotion_eligible=False),stream,indent=2)
    if args.plan_only:return 0
    observations=[]
    for entry in plan:
        if hashlib.sha256(Path(entry['command'][1]).read_bytes()).hexdigest()!=entry['case_sha256']:raise ValueError('case changed after planning')
        path=Path(entry['output'])
        if path.exists():raise ValueError('immutable capture exists')
        if not run_command(entry['command'],path,args.timeout_seconds):raise RuntimeError('capture failed: '+str(path))
        capture=strict_json(path.read_text());validate_capture(capture,entry);observations.append((entry,capture))
    with (args.output/'summary.json').open('x') as stream:json.dump(dict(schema='rns8_distributed_summary_v1',cohorts=summarize(observations),promotion_eligible=False),stream,indent=2)
    return 0

if __name__=='__main__':raise SystemExit(main())
