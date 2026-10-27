"""Matched public narrow/RNS/UInt192 routes; no measured-cache promotion."""
import argparse
import json
from pathlib import Path
import math
import statistics
import uuid
from collect_int4_calibration import collection_plan as shared_plan, positive
from collect_native_calibration import strict_json, DEFAULT_MODULI
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete

MODES = ('scalar-staged', 'scalar-fused', 'wmma-staged', 'wmma-fused', 'rns-public', 'rns-parallel', 'vector-width8', 'vector-exact-width', 'adaptive-fresh', 'adaptive-cached',
         'adaptive-resident-fresh', 'adaptive-resident-cached', 'wmma-resident-fused', 'scalar-resident-fused', 'rns-resident-public')
RESIDENT_OUTPUT_MODES = {'adaptive-resident-fresh', 'adaptive-resident-cached', 'wmma-resident-fused', 'scalar-resident-fused', 'rns-resident-public'}


def validate_case(c):
    if not isinstance(c, dict) or set(c) != {'schema', 'm', 'n', 'k', 'prefix', 'unsigned_inputs', 'mode', 'residency', 'repeats', 'warmups', 'seed'} or c['schema'] != 'narrow_public_case_v1':
        raise ValueError('case ABI')
    for key, low, high in [('m', 1, 512), ('n', 1, 512), ('k', 1, 131073), ('prefix', 0, 20), ('repeats', 3, 100), ('warmups', 1, 100), ('seed', 0, 2**32-1)]:
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError(key)
    if type(c['unsigned_inputs']) is not bool or c['mode'] not in MODES or c['residency'] not in ('cold', 'resident', 'repack'):
        raise ValueError('mode/domain')
    if (c['mode'].startswith('adaptive-') or c['mode'] in RESIDENT_OUTPUT_MODES) and c['residency'] != 'resident':
        raise ValueError('adaptive summary comparison requires retained resident inputs')
    if (c['mode'].startswith('scalar-') and c['unsigned_inputs']) or (c['mode'] == 'rns-parallel' and c['n'] > 8) or (c['mode'] == 'vector-exact-width' and not (2 <= c['n'] <= 8 and c['k'] >= 512)):
        raise ValueError('concrete recipe')
    if c['m']*c['n']*c['k'] > 134217728:
        raise ValueError('complete oracle work')
    if c['prefix'] and math.prod(DEFAULT_MODULI[:c['prefix']]) <= c['k']*49*(1 if c['unsigned_inputs'] else 2):
        raise ValueError('strict prefix capacity')
    return c


def selected_prefix(c):
    if c['prefix']:
        return c['prefix']
    product = 1
    for index, modulus in enumerate(DEFAULT_MODULI, 1):
        product *= modulus
        if product > c['k']*49*(1 if c['unsigned_inputs'] else 2):
            return index
    raise ValueError('capacity')


def validate_capture(c, entry):
    case = validate_case(entry['case']); mode = case['mode']; narrow = mode.startswith(('scalar-', 'wmma-')); vector = mode.startswith('vector-')
    resident_output = mode in RESIDENT_OUTPUT_MODES
    scratch = ((32 if mode.startswith('wmma-') else 16)+(0 if mode.endswith('-fused') else case['m']*case['n']*8)) if narrow else 0
    fixed = dict(schema='rns8_public_narrow_microcase_v1', case=case, run_id=entry['run_id'], dataset='affine_native_small_v1',
                 selected_prefix=selected_prefix(case), narrow_producer=mode if narrow else 'not_applicable', known_narrow_scratch_bytes=scratch,
                 output_boundary='complete_public_product_resident_rns' if resident_output else 'complete_public_product_and_padded_three_limb_integer_adapter',
                 vector_export_adapter='public_native64_export_then_host_sign_or_zero_extension' if vector else 'not_applicable',
                 host_instrumentation_state='hooks_disabled', exact_reference_passed=True, oracle_inside_production_interval=False,
                 all_residue_planes_audited=not vector, residue_audit_scope='not_applicable_native_control' if vector else 'post_timing_public_product',
                 promotion_eligible=False, native_useful_products=case['m']*case['n']*case['k'])
    for key, value in fixed.items():
        if type(c.get(key)) is not type(value) or c[key] != value:
            raise ValueError('capture '+key)
    if not manifest_provenance_complete(c.get('build_manifest')) or not c.get('execution_id'):
        raise ValueError('compiled/process provenance')
    artifact = c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or len(artifact.get('executable_sha256', '')) != 64:
        raise ValueError('artifact identity')
    if not positive(c.get('setup_us')) or len(c.get('samples', [])) != case['repeats']:
        raise ValueError('sample count/setup')
    for sample in [c.get('first_call', {}), *c['samples']]:
        phases = [sample.get(k) for k in ('prepare_us', 'execute_us', 'export_us', 'cleanup_us')]
        if not all(positive(v, True) for v in phases) or not positive(sample.get('total_us')) or not math.isclose(sum(phases), sample['total_us'], rel_tol=1e-7, abs_tol=1e-6):
            raise ValueError('complete route')
        if resident_output and sample['export_us'] != 0:
            raise ValueError('resident route must not hide a timed host export')
    if len(c.get('sustained', [])) != 3:
        raise ValueError('sustained observations')
    for burst in c['sustained']:
        calls = burst.get('calls'); duration = burst.get('total_us')
        if type(calls) is not int or not 16 <= calls <= 4096 or not positive(duration) or (duration < 250000 and calls != 4096) or not math.isclose(burst.get('per_call_us', -1), duration/calls, rel_tol=1e-7):
            raise ValueError('sustained work/duration')
    kernel = c.get('ordinary_plan_kernel', '')
    if mode == 'rns-parallel' and kernel != ('direct_hip_selected_prefix_rns_gemv_n1_v1' if case['n'] == 1 else 'direct_hip_selected_prefix_rns_gemv_small_n_v1'):
        raise ValueError('public parallel route identity')
    if mode in ('rns-public', 'rns-resident-public') and kernel != ('direct_hip_persistent_small_rns_gemm_v1' if case['m']*case['n'] <= 64 else 'direct_hip_selected_prefix_grouped_rns_gemm_v1'):
        raise ValueError('public default route identity')
    if mode == 'vector-exact-width' and f'_n{case["n"]}_exact_width_' not in kernel:
        raise ValueError('exact-width vector route identity')
    if mode.startswith('adaptive-'):
        validate_adaptive_summary_capture(c, case)
    if resident_output:
        for key, value in dict(timed_host_export=False, oracle_export_outside_production_interval=True, exact_proof_policy='scalar_first').items():
            if type(c.get(key)) is not type(value) or c[key] != value:
                raise ValueError('resident output identity '+key)
    return c


def validate_adaptive_summary_capture(c, case):
    cached = case['mode'].endswith('-cached')
    expected = dict(adaptive_summary_policy='generation_bound' if cached else 'fresh_scan',
                    adaptive_method='axis_l1', adaptive_group=2, adaptive_workers=32, exact_proof_policy='scalar_first')
    for key, value in expected.items():
        if type(c.get(key)) is not type(value) or c[key] != value:
            raise ValueError('adaptive identity '+key)
    calls = 1 + case['warmups'] + case['repeats'] + sum(b['calls'] for b in c['sustained'])
    size = ((case['m']+15)//16+(case['n']+15)//16)*24
    for name, count in [('adaptive_summary_first', 1), ('adaptive_summary_final', calls)]:
        info = c.get(name)
        keys = {'retained_bytes', 'a_instance', 'a_generation', 'b_instance', 'b_generation', 'a_scans', 'b_scans', 'a_reuses', 'b_reuses'}
        if not isinstance(info, dict) or set(info) != keys or any(type(v) is not int or v < 0 for v in info.values()):
            raise ValueError('adaptive summary receipt')
        if info['retained_bytes'] != (size if cached else 0):
            raise ValueError('adaptive retained bytes')
        for side in ('a', 'b'):
            if info[side+'_scans'] != int(cached) or info[side+'_reuses'] != (count-1 if cached else 0):
                raise ValueError('adaptive scans/reuses')
            if cached and (not info[side+'_instance'] or not info[side+'_generation']):
                raise ValueError('adaptive generation identity')
            if not cached and (info[side+'_instance'] or info[side+'_generation']):
                raise ValueError('fresh route cannot report cached identity')
    if cached:
        for key in ('a_instance', 'a_generation', 'b_instance', 'b_generation'):
            if c['adaptive_summary_first'][key] != c['adaptive_summary_final'][key]:
                raise ValueError('unchanged generation required')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--binary', type=Path, required=True); p.add_argument('--cases', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--runs', type=int, default=2); p.add_argument('--seed', type=int, default=5353)
    p.add_argument('--timeout-seconds', type=float, default=600); p.add_argument('--plan-only', action='store_true'); a = p.parse_args()
    plan = shared_plan(a.binary, a.cases, a.output, a.runs, a.seed, str(uuid.uuid4()), validate_case); a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'plan.json').exists():
        raise ValueError('immutable collection exists')
    (a.output/'plan.json').write_text(json.dumps(dict(schema='narrow_public_collection_v1', observations=plan, promotion_eligible=False), indent=2)+'\n')
    if a.plan_only:
        return 0
    cohorts = {}
    for entry in plan:
        path = Path(entry['output'])
        if path.exists() or not run_command(entry['command'], path, a.timeout_seconds):
            raise RuntimeError('immutable capture exists or subprocess failed: '+str(path))
        c = validate_capture(strict_json(path.read_text()), entry)
        identity = dict(case={k: v for k, v in entry['case'].items() if k != 'mode'}, target=c.get('target'), device=c.get('device'), runtime=c.get('runtime_version'),
                        proof_policy=c.get('exact_proof_policy', 'default_eager'),
                        output_boundary=c['output_boundary'],
                        build=c['build_manifest'], artifact={k: c['artifact_capture'].get(k) for k in ('executable_sha256', 'rns8_linkage', 'shared_library_sha256')},
                        environment=(c.get('request_identity') or {}).get('environment'))
        cohorts.setdefault(json.dumps(identity, sort_keys=True), []).append(dict(mode=entry['case']['mode'], execution_id=c['execution_id'], independent_run=entry['independent_run'],
                          path=str(path), median_us=statistics.median(s['total_us'] for s in c['samples']), setup_us=c['setup_us'], scratch_bytes=c['known_narrow_scratch_bytes']))
    result = []
    for identity, rows in cohorts.items():
        if len({r['execution_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate process observation')
        medians = {}
        for mode in MODES:
            values = [r['median_us'] for r in rows if r['mode'] == mode]
            if len(values) >= 2:
                medians[mode] = statistics.median(values)
        result.append(dict(identity=json.loads(identity), observations=rows, independent_process_medians_us=medians, qualification='not_inferred', promotion_eligible=False))
    (a.output/'summary.json').write_text(json.dumps(dict(schema='narrow_public_summary_v1', cohorts=result, promotion_eligible=False), indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
