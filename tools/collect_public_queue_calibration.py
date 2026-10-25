"""Counterbalanced public native queue observations; never changes promotion state."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import uuid
from native_locality_contract import public_contract as locality_contract, ORDERS as LOCALITY_ORDERS
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from collect_native_calibration import strict_json, DEFAULT_MODULI


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def validate_case(c):
    fields = {'schema', 'mode', 'input_policy', 'encoded_b', 'schedule', 'chunk', 'workers', 'reuse', 'warmups', 'repeats', 'seed', 'padding', 'tasks'}
    if not isinstance(c, dict) or set(c)-{'locality_order'} != fields or c['schema'] != 'public_native_group_case_v1':
        raise ValueError('case ABI')
    if c['mode'] not in ('persistent', 'ordinary', 'graph', 'resident') or c['input_policy'] not in ('resident', 'mutated-b') or type(c['encoded_b']) is not bool:
        raise ValueError('execution/input contract')
    if c['mode'] == 'resident' and c['encoded_b']:
        raise ValueError('resident RNS does not consume encoded B')
    for key, low, high in [('schedule', 0, 2), ('chunk', 1, 32), ('workers', 1, 1024), ('reuse', 1, 64), ('warmups', 1, 1000), ('repeats', 3, 1000), ('seed', 0, 2**32-1), ('padding', 0, 64)]:
        if not integer(c[key], low, high):
            raise ValueError(key)
    if 'locality_order' in c and (c['locality_order'] not in LOCALITY_ORDERS or c['schedule'] or c['mode'] not in ('persistent','graph')):
        raise ValueError('locality requires explicit-work persistent or graph execution')
    if not isinstance(c['tasks'], list) or not 1 <= len(c['tasks']) <= 64:
        raise ValueError('task count')
    bs, work = {}, 0
    for t in c['tasks']:
        if not isinstance(t, dict) or set(t) != {'m', 'n', 'k', 'prefix', 'group', 'unsigned_inputs', 'b_slot'} or type(t['unsigned_inputs']) is not bool:
            raise ValueError('task ABI')
        for key, low, high in [('m', 1, 512), ('n', 1, 512), ('k', 1, 65537), ('prefix', 1, 20), ('b_slot', 0, 63)]:
            if not integer(t[key], low, high):
                raise ValueError('task ' + key)
        if not integer(t['group'], 1, 4) or t['group'] == 3:
            raise ValueError('modulus group')
        identity = (t['k'], t['n'], t['prefix'], t['unsigned_inputs'])
        if bs.setdefault(t['b_slot'], identity) != identity:
            raise ValueError('incompatible shared B')
        product = math.prod(DEFAULT_MODULI[:t['prefix']])
        if product <= t['k'] * 49 * (1 if t['unsigned_inputs'] else 2):
            raise ValueError('independent dataset prefix proof')
        work += t['m'] * t['n'] * t['k']
    if work > 134217728:
        raise ValueError('full oracle work budget')
    return c


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def counts(case):
    tasks = case['tasks']
    work = [((t['m']+15)//16)*((t['n']+15)//16)*((t['prefix']+t['group']-1)//t['group']) for t in tasks]
    metadata = len(tasks)*104 + 4 + (sum(work)*16 if case['schedule'] == 0 else len(tasks)*16 + (len(set(work))*16 if case['schedule'] == 2 else 0))
    bs = {t['b_slot']: t for t in tasks}
    encoded = sum(t['prefix']*t['n']*((t['k']+15)//16)*16+4 for t in bs.values()) if case['encoded_b'] else 0
    return sum(work), metadata, encoded, sum(t['m']*t['n']*24 for t in tasks)


def validate_capture(c, entry):
    case = entry['case']; work, metadata, encoded, output = counts(case)
    fixed = dict(schema='rns8_public_native_group_microcase_v1', case=case, run_id=entry['run_id'], seed=case['seed'],
                 dataset='bounded_mt19937_64_shared_b_v1', host_instrumentation_state='hooks_disabled',
                 output_boundary='each_public_group_and_all_padded_three_limb_exports',
                 exact_reference_passed=True, oracle_inside_production_interval=False,
                 oracle_scope='complete_final_output_after_each_observation_and_sustained_interval',
                 per_task_device_completion_latency_measured=False, promotion_eligible=False,
                 useful_tile_channel_groups_per_call=work, output_payload_bytes_per_call=output,
                 encoded_b_retained_device_bytes=encoded,
                 encoded_b_peak_replacement_device_bytes=encoded*(2 if case['input_policy'] == 'mutated-b' else 1))
    for key, value in fixed.items():
        if type(c.get(key)) is not type(value) or c[key] != value:
            raise ValueError('capture ' + key)
    if 'locality_order' in case:
        expected_locality=locality_contract(case,c.get('target',''))
        if type(c.get('locality')) is not dict or c['locality']!=expected_locality or any(type(c['locality'].get(k)) is not type(v) for k,v in expected_locality.items()):
            raise ValueError('public native locality receipt')
    elif 'locality' in c:raise ValueError('unrequested public locality')
    if not manifest_provenance_complete(c.get('build_manifest')):
        raise ValueError('compiled provenance')
    artifact = c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256', '')) != 64 or not c.get('execution_id'):
        raise ValueError('artifact/process receipt')
    if not finite(c.get('setup_us'), True):
        raise ValueError('setup')
    samples = c.get('samples', [])
    if len(samples) != case['repeats']:
        raise ValueError('real sample count')
    for sample in [c.get('first_call', {}), *samples]:
        phases = [sample.get(k) for k in ('prepare_us', 'execute_us', 'export_us', 'host_loop_us')]
        if not all(finite(v) for v in phases) or not finite(sample.get('total_us'), True) or not math.isclose(sum(phases), sample['total_us'], rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError('complete disjoint phases')
        if type(sample.get('complete_groups')) is not int or sample['complete_groups'] != case['reuse']:
            raise ValueError('reuse must execute real groups')
    if case['mode'] == 'resident':
        if c.get('metadata') is not None:
            raise ValueError('resident baseline cannot claim native descriptors')
    else:
        info = c.get('metadata', {})
        expected = dict(tasks=len(case['tasks']), work=work, device_bytes=metadata,
                        preparation_peak_device_bytes=metadata*(2 if case['input_policy'] == 'mutated-b' else 1), graph_ready=case['mode'] == 'graph')
        for key, value in expected.items():
            if type(info.get(key)) is not type(value) or info[key] != value:
                raise ValueError('metadata ' + key)
        if not integer(info.get('host_bytes'), 1, 2**63-1):
            raise ValueError('actual host metadata')
    bursts = c.get('sustained', [])
    if len(bursts) != 3:
        raise ValueError('three sustained intervals required')
    for b in bursts:
        calls = b.get('calls')
        if not integer(calls, 16, 4096) or not finite(b.get('duration_us'), True) or (b['duration_us'] < 250000 and calls != 4096):
            raise ValueError('bounded sustained interval')
        if type(b.get('complete_groups')) is not int or b['complete_groups'] != calls*case['reuse']:
            raise ValueError('actual sustained groups')
        changes = calls*case['reuse'] if case['input_policy'] == 'mutated-b' else 0
        if not integer(b.get('first_input_generation'), 1, 2**63-1) or not integer(b.get('last_input_generation'), 1, 2**63-1) or b['last_input_generation']-b['first_input_generation'] != changes:
            raise ValueError('generation interval')
        if type(b.get('rebuilds')) is not int or b['rebuilds'] != changes or b.get('exact_reference_passed') is not True:
            raise ValueError('rebuild/output receipt')
    return c


def collection_plan(binary, paths, output, runs, seed, collection_id):
    if not paths or not integer(runs, 2, 100):
        raise ValueError('cases and 2..100 independent processes required')
    cases = [(p, validate_case(strict_json(p.read_text())), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    if len({h for _, _, h in cases}) != len(cases):
        raise ValueError('duplicate cases')
    result = []
    for run in range(runs):
        shift = (seed+run) % len(cases); order = cases[shift:]+cases[:shift]
        if run % 2:
            order.reverse()
        for position, (path, case, digest) in enumerate(order):
            run_id = f'{collection_id}-{digest[:12]}-{run}'
            result.append(dict(command=[str(binary), str(path), run_id], case=case, run_id=run_id,
                               case_sha256=digest, independent_run=run, position=position, output=str(output/(run_id+'.json'))))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True); p.add_argument('--cases', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--runs', type=int, default=2)
    p.add_argument('--seed', type=int, default=3838); p.add_argument('--timeout-seconds', type=float, default=600); p.add_argument('--plan-only', action='store_true')
    a = p.parse_args(); plan = collection_plan(a.binary, a.cases, a.output, a.runs, a.seed, str(uuid.uuid4()))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'plan.json').write_text(json.dumps(dict(schema='public_native_group_collection_v1', commands=plan, promotion_eligible=False), indent=2)+'\n')
    if a.plan_only:
        return 0
    cohorts = {}
    for entry in plan:
        path = Path(entry['output'])
        if path.exists():
            raise ValueError('refusing immutable observation overwrite')
        if not run_command(entry['command'], path, a.timeout_seconds):
            raise RuntimeError('capture failed: '+str(path))
        capture = validate_capture(strict_json(path.read_text()), entry)
        identity = dict(case=entry['case'], artifact={k: capture['artifact_capture'].get(k) for k in ('executable_sha256', 'rns8_linkage', 'shared_library_sha256')},
                        build=capture['build_manifest'], device=capture.get('device'), target=capture.get('target'), runtime=capture.get('runtime_version'))
        reuse = entry['case']['reuse']; latency = statistics.median(s['total_us'] for s in capture['samples'])
        cohorts.setdefault(json.dumps(identity, sort_keys=True), []).append(dict(path=str(path), execution_id=capture['execution_id'], run_id=entry['run_id'],
            group_latency_us=latency/reuse, observed_batch_us=latency, setup_us=capture['setup_us'],
            setup_plus_observed_batch_us=capture['setup_us']+latency,
            sustained_group_us=statistics.median(b['duration_us']/b['complete_groups'] for b in capture['sustained'])))
    rows = []
    for key, observations in sorted(cohorts.items()):
        if len({o['execution_id'] for o in observations}) != len(observations):
            raise ValueError('duplicate process receipt')
        rows.append(dict(identity=json.loads(key), observations=observations, independent_processes=len(observations),
            group_latency_process_median_us=statistics.median(o['group_latency_us'] for o in observations),
            group_latency_range_us=[min(o['group_latency_us'] for o in observations), max(o['group_latency_us'] for o in observations)],
            sustained_group_process_median_us=statistics.median(o['sustained_group_us'] for o in observations)))
    (a.output/'report.json').write_text(json.dumps(dict(schema='public_native_group_observations_v1', cohorts=rows, promotion_eligible=False,
        interpretation='complete_public_routes; compare identical workloads/output/reuse/generation policies and independent-process spread; no automatic winner',
        per_task_fairness='not inferred from post-batch completion events; separate private diagnostic required'), indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
