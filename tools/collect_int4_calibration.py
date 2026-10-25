"""Counterbalanced independent-process finite INT4 public-route research captures."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import uuid
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from collect_native_calibration import strict_json


def validate_case(c):
    fields = {'schema', 'm', 'n', 'k', 'modulus', 'mode', 'residency', 'repeats', 'warmups', 'seed'}
    if not isinstance(c, dict) or set(c) != fields or c['schema'] != 'finite_int4_public_case_v1':
        raise ValueError('case ABI')
    for key, low, high in [('m', 1, 512), ('n', 1, 512), ('k', 1, 131073), ('modulus', 2, 16), ('repeats', 3, 100), ('warmups', 1, 100), ('seed', 0, 2**32-1)]:
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError(key)
    if c['m']*c['n']*c['k'] > 134217728 or c['mode'] not in ('int8', 'onthefly', 'packed-b') or c['residency'] not in ('cold', 'repeated-b', 'repeated-ab'):
        raise ValueError('work/mode/residency')
    return c


def positive(v, zero=False):
    return type(v) in (int, float) and math.isfinite(v) and (v >= 0 if zero else v > 0)


def validate_capture(c, entry):
    case = validate_case(entry['case'])
    fixed = dict(schema='rns8_public_int4_microcase_v1', case=case, run_id=entry['run_id'], dataset='affine_finite_bytes_v1',
                 host_instrumentation_state='hooks_disabled', output_boundary='complete_public_finite_product_and_padded_export',
                 exact_reference_passed=True, oracle_inside_production_interval=False, promotion_eligible=False,
                 native_useful_products=case['m']*case['n']*case['k'], signed_nibble_operands=case['mode'] != 'int8',
                 layout_version=1 if case['mode'] == 'packed-b' else 0,
                 retained_packed_b_device_bytes=((case['k']+15)//16)*case['n']*8 if case['mode'] == 'packed-b' else 0)
    for key, value in fixed.items():
        if type(c.get(key)) is not type(value) or c[key] != value:
            raise ValueError('capture '+key)
    if not manifest_provenance_complete(c.get('build_manifest')) or not c.get('execution_id'):
        raise ValueError('compiled/process provenance')
    artifact = c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256', '')) != 64:
        raise ValueError('artifact identity')
    if case['mode'] != 'int8' and (len(c.get('certificate_sha256', '')) != 64 or c.get('target', '').split(':')[0] not in ('gfx1100', 'gfx1101', 'gfx1102')):
        raise ValueError('INT4 concrete target/certificate')
    if not positive(c.get('setup_us')) or len(c.get('samples', [])) != case['repeats']:
        raise ValueError('setup/real sample count')
    for sample in [c.get('first_call', {}), *c['samples']]:
        phases = [sample.get(k) for k in ('prepare_us', 'execute_us', 'export_us', 'cleanup_us')]
        if not all(positive(v, True) for v in phases) or not positive(sample.get('total_us')) or not math.isclose(sum(phases), sample['total_us'], rel_tol=1e-7, abs_tol=1e-6):
            raise ValueError('complete public phases')
    if len(c.get('sustained', [])) != 3:
        raise ValueError('sustained observations')
    for burst in c['sustained']:
        calls = burst.get('calls'); duration = burst.get('total_us')
        if type(calls) is not int or not 16 <= calls <= 4096 or not positive(duration) or (duration < 250000 and calls != 4096):
            raise ValueError('sustained duration')
        if burst.get('minimum_duration_reached') is not (duration >= 250000) or not math.isclose(burst.get('per_call_us', -1), duration/calls, rel_tol=1e-7):
            raise ValueError('sustained work normalization')
    return c


def collection_plan(binary, cases, output, runs, seed, identity):
    if not 2 <= runs <= 100 or not cases:
        raise ValueError('2..100 independent runs and cases required')
    parsed = [(path, validate_case(strict_json(path.read_text())), hashlib.sha256(path.read_bytes()).hexdigest()) for path in cases]
    if len({digest for _, _, digest in parsed}) != len(parsed):
        raise ValueError('duplicate cases')
    plan = []
    for run in range(runs):
        shift = (seed+run) % len(parsed); order = parsed[shift:]+parsed[:shift]
        if run % 2:
            order.reverse()
        for position, (path, case, digest) in enumerate(order):
            run_id = f'{identity}-{digest[:12]}-{run}'
            plan.append(dict(command=[str(binary), str(path), run_id], case=case, run_id=run_id, independent_run=run,
                             case_sha256=digest, position=position, output=str(output/(run_id+'.json'))))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True); p.add_argument('--cases', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--runs', type=int, default=2)
    p.add_argument('--seed', type=int, default=4646); p.add_argument('--timeout-seconds', type=float, default=600); p.add_argument('--plan-only', action='store_true')
    a = p.parse_args(); plan = collection_plan(a.binary, a.cases, a.output, a.runs, a.seed, str(uuid.uuid4()))
    a.output.mkdir(parents=True, exist_ok=True); target = a.output/'plan.json'
    if target.exists():
        raise ValueError('immutable collection already exists')
    target.write_text(json.dumps(dict(schema='int4_public_collection_v1', observations=plan, promotion_eligible=False), indent=2)+'\n')
    if a.plan_only:
        return 0
    cohorts = {}
    for entry in plan:
        path = Path(entry['output'])
        if path.exists():
            raise ValueError('immutable observation exists')
        if not run_command(entry['command'], path, a.timeout_seconds):
            raise RuntimeError('capture failed: '+str(path))
        c = validate_capture(strict_json(path.read_text()), entry)
        identity = dict(case={k: v for k, v in entry['case'].items() if k != 'mode'}, artifact=c['artifact_capture'],
                        build=c['build_manifest'], target=c.get('target'), device=c.get('device'), runtime=c.get('runtime_version'),
                        environment=(c.get('request_identity') or {}).get('environment'))
        # Process identifiers/paths are provenance, not pair identity.
        identity['artifact'] = {k: c['artifact_capture'].get(k) for k in ('executable_sha256', 'rns8_linkage', 'shared_library_sha256')}
        cohorts.setdefault(json.dumps(identity, sort_keys=True), []).append(dict(mode=entry['case']['mode'], path=str(path),
            execution_id=c['execution_id'], independent_run=entry['independent_run'], median_us=statistics.median(s['total_us'] for s in c['samples']), setup_us=c['setup_us']))
    result = []
    for identity, observations in cohorts.items():
        modes = {}
        for observation in observations:
            modes.setdefault(observation['mode'], []).append(observation)
        if len({o['execution_id'] for o in observations}) != len(observations):
            raise ValueError('reused process identity')
        medians = {mode: statistics.median(o['median_us'] for o in values) for mode, values in modes.items() if len(values) >= 2}
        result.append(dict(identity=json.loads(identity), observations=observations, independent_process_medians_us=medians,
                           int8_over_candidate={mode: medians['int8']/value for mode, value in medians.items() if mode != 'int8' and 'int8' in medians},
                           promotion_eligible=False, qualification='not_inferred'))
    (a.output/'summary.json').write_text(json.dumps(dict(schema='int4_public_summary_v1', cohorts=result, promotion_eligible=False), indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
