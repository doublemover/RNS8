"""Explicit, non-promoting FP8/TDM captures with matched independent processes."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import uuid
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete
from collect_native_calibration import strict_json
from collect_int4_calibration import collection_plan, positive


def validate_case(c):
    if type(c) is not dict:
        raise ValueError('case object')
    fp8 = c.get('schema') == 'finite_fp8_public_case_v1'
    fields = {'schema', 'm', 'n', 'k', 'mode', 'repeats', 'warmups', 'seed'} | (
        {'modulus', 'residency'} if fp8 else {'prefix', 'group', 'signed_input', 'compute', 'cluster_axis'})
    if set(c) != fields or c['schema'] not in ('finite_fp8_public_case_v1', 'native_tdm_case_v1'):
        raise ValueError('case ABI')
    bounds = [('m', 1, 512), ('n', 1, 512), ('k', 1, 131073), ('repeats', 3, 100), ('warmups', 1, 100), ('seed', 0, 2**32-1)]
    bounds += [('modulus', 2, 256)] if fp8 else [('prefix', 1, 20), ('group', 1, 4)]
    for key, low, high in bounds:
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError(key)
    if c['m']*c['n']*c['k'] > 134217728:
        raise ValueError('complete oracle work budget')
    if fp8:
        if c['mode'] not in ('int8', 'fp8', 'fp8-materialized', 'fp8-three', 'fp8-three-materialized', 'fp8-ocp', 'fp8-ocp-materialized', 'fp8-ocp-three', 'fp8-ocp-three-materialized') or c['residency'] not in ('cold', 'repeated-ab', 'repeated-b'):
            raise ValueError('FP8 mode/residency')
    elif c['mode'] not in ('ordinary', 'tdm', 'multicast2', 'multicast4') or c['group'] not in (1, 2, 4) or type(c['signed_input']) is not bool:
        raise ValueError('TDM mode/group/domain')
    elif c['compute'] not in ('scalar', 'wmma') or c['cluster_axis'] not in ('moduli', 'output-neighbor') or (c['cluster_axis'] == 'output-neighbor' and c['mode'] not in ('multicast2', 'multicast4')):
        raise ValueError('TDM primitive/cluster axis')
    return c


def _equal(c, fixed):
    for key, value in fixed.items():
        if type(c.get(key)) is not type(value) or c[key] != value:
            raise ValueError('capture '+key)


def validate_capture(c, entry):
    case = validate_case(entry['case']); validate_case(c.get('case')); fp8 = case['schema'] == 'finite_fp8_public_case_v1'
    _equal(c, dict(schema='rns8_public_fp8_microcase_v1' if fp8 else 'rns8_native_tdm_microcase_v1',
                   case=case, run_id=entry['run_id'], exact_reference_passed=True,
                   oracle_inside_production_interval=False, target_qualified=False,
                   performance_qualified=False, promotion_eligible=False))
    if not manifest_provenance_complete(c.get('build_manifest')) or not c.get('execution_id'):
        raise ValueError('compiled/process provenance')
    artifact = c.get('artifact_capture') or {}
    if artifact.get('unchanged_during_run') is not True or re.fullmatch('[0-9a-f]{64}', artifact.get('executable_sha256', '')) is None:
        raise ValueError('artifact identity')
    if not c.get('target') or not c.get('device') or type(c.get('runtime_version')) is not int or c['runtime_version'] <= 0:
        raise ValueError('runtime tuple')
    if not positive(c.get('setup_us')) or len(c.get('samples', [])) != case['repeats']:
        raise ValueError('setup/sample count')
    if fp8:
        candidate = case['mode'] != 'int8'
        ocp = 'ocp' in case['mode']; three = 'three' in case['mode']; materialized = 'materialized' in case['mode']
        method = 2*int(three)+int(materialized)
        _equal(c, dict(format=('ocp_e4m3' if ocp else 'amd_e4m3_fnuz') if candidate else 'signed_int8', conditional_instruction_model=candidate,
                       method_actual=method, retained_recipe_device_bytes=(4+((12 if three else 16)*case['m']*case['n'] if materialized else 0)) if candidate else 0, dataset='affine_finite_bytes_v1',
                       host_instrumentation_state='hooks_disabled', native_useful_products=case['m']*case['n']*case['k'],
                       output_boundary='complete_public_finite_product_and_padded_export', backend_actual='hip-direct'))
        for key in ('a_checksum', 'b_checksum', 'output_checksum'):
            if re.fullmatch('[0-9a-f]{64}', c.get(key, '')) is None:
                raise ValueError('complete data checksum')
        if candidate and (c['target'].split(':')[0] != ('gfx1201' if ocp else 'gfx942') or re.fullmatch('[0-9a-f]{64}', c.get('certificate_sha256', '')) is None):
            raise ValueError('specific FP8 target/format/model identity')
        for sample in [c.get('first_call', {}), *c['samples']]:
            phases = [sample.get(k) for k in ('prepare_us', 'execute_us', 'export_us', 'cleanup_us')]
            if not all(positive(v, True) for v in phases) or not positive(sample.get('total_us')) or not math.isclose(sum(phases), sample['total_us'], rel_tol=1e-7, abs_tol=1e-6):
                raise ValueError('complete FP8 public phases')
    else:
        cluster = {'ordinary': 1, 'tdm': 1, 'multicast2': 2, 'multicast4': 4}[case['mode']]
        _equal(c, dict(mode_actual=case['mode'], cluster_actual=cluster, backend_actual='hip-direct',
                       source_representation='native64_signed' if case['signed_input'] else 'native64_unsigned', signed_input=case['signed_input'],
                       compute_primitive='wmma_i32_16x16x64_iu8_wave32' if case['compute'] == 'wmma' else 'integer_alu_residue_dot64', cluster_axis_actual=case['cluster_axis'], stage_count=2, lds_bytes=32768, status_bytes=4,
                       dataset='fullwidth_native_xor_affine_v1', semantic_contract='residues_only_no_native_range_claim',
                       output_boundary='complete_resident_native64_to_centered_rns_checked_call', production_timing_hooks_enabled=False))
        flags = c.get('capability_flags')
        if type(flags) is not int or flags & ~7 or (case['mode'] != 'ordinary' and (c['target'].split(':')[0] != 'gfx1250' or not flags & 1)) or (cluster > 1 and flags & 3 != 3) or (case['compute'] == 'wmma' and (not flags & 4 or c['target'].split(':')[0] != 'gfx1250')):
            raise ValueError('actual TDM/multicast code object')
        blocks = ((case['m']+15)//16)*((case['n']+15)//16)*cluster*((((case['prefix']+case['group']-1)//case['group'])+cluster-1)//cluster)
        if case['cluster_axis'] == 'output-neighbor':
            blocks = ((case['m']+15)//16)*((((case['n']+15)//16)+cluster-1)//cluster)*cluster*((case['prefix']+case['group']-1)//case['group'])
        rows, cols, groups = (case['m']+15)//16, (case['n']+15)//16, (case['prefix']+case['group']-1)//case['group']
        no_a, no_b = case['m']*case['k']*8*cols*groups, case['k']*case['n']*8*rows*groups
        ideal_a = no_a if cluster == 1 else case['m']*case['k']*8*((cols+cluster-1)//cluster)*groups if case['cluster_axis'] == 'output-neighbor' else case['m']*case['k']*8*cols*((groups+cluster-1)//cluster)
        ideal_b = no_b if cluster == 1 or case['cluster_axis'] == 'output-neighbor' else case['k']*case['n']*8*rows*((groups+cluster-1)//cluster)
        _equal(c, dict(modeled_no_intergroup_reuse_read_bytes=no_a+no_b, modeled_ideal_multicast_source_read_bytes=ideal_a+ideal_b))
        if not positive(c.get('modeled_ideal_source_read_amplification')) or not math.isclose(c['modeled_ideal_source_read_amplification'], (ideal_a+ideal_b)/((case['m']*case['k']+case['k']*case['n'])*8), rel_tol=1e-7):
            raise ValueError('modeled raw-source amplification')
        for key in ('actual_hbm_read_bytes', 'tdm_stall_cycles', 'barrier_stall_cycles', 'achieved_occupancy', 'compiler_private_scratch_bytes'):
            if key not in c or c[key] is not None:
                raise ValueError('unmeasured profiler metric')
        if type(c.get('grid_blocks')) is not int or c['grid_blocks'] != blocks or not positive(c.get('logical_payload_envelope_bytes')) or c['logical_payload_envelope_bytes'] > c.get('payload_budget_bytes', 0):
            raise ValueError('TDM resource admission')
        for key in ('a_checksum', 'b_checksum', 'output_checksum'):
            if re.fullmatch('[0-9a-f]{64}', c.get(key, '')) is None:
                raise ValueError('complete data checksum')
        if not all(positive(v) for v in [c.get('first_call_us'), *c['samples']]):
            raise ValueError('complete TDM calls')
    if len(c.get('sustained', [])) != 3:
        raise ValueError('sustained observations')
    for burst in c['sustained']:
        calls, duration = burst.get('calls'), burst.get('total_us')
        if type(calls) is not int or not 16 <= calls <= 4096 or not positive(duration) or (duration < 250000 and calls != 4096):
            raise ValueError('censored sustained duration')
        if burst.get('minimum_duration_reached') is not (duration >= 250000) or not math.isclose(burst.get('per_call_us', -1), duration/calls, rel_tol=1e-7):
            raise ValueError('sustained normalization')
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True); parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=52645); parser.add_argument('--timeout-seconds', type=float, default=600)
    parser.add_argument('--execute', action='store_true', help='default only writes an immutable plan; opt in explicitly to device execution')
    args = parser.parse_args()
    if not positive(args.timeout_seconds):
        raise ValueError('timeout')
    plan = collection_plan(args.binary, args.cases, args.output, args.runs, args.seed, str(uuid.uuid4()), validate_case)
    if len({entry['case']['schema'] for entry in plan}) != 1:
        raise ValueError('one executable/family per collection')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'plan.json').open('x') as stream:
        json.dump(dict(schema='research_kernels_collection_v1', observations=plan, promotion_eligible=False), stream, indent=2)
    if not args.execute:
        return 0
    cohorts = {}
    for entry in plan:
        path = Path(entry['output'])
        if path.exists():
            raise ValueError('immutable observation exists')
        if not run_command(entry['command'], path, args.timeout_seconds):
            raise RuntimeError('capture failed or unsupported: '+str(path))
        capture = validate_capture(strict_json(path.read_text()), entry)
        identity = dict(case={k: v for k, v in entry['case'].items() if k not in ('mode', 'cluster_axis')},
                        artifact={k: capture['artifact_capture'].get(k) for k in ('executable_sha256', 'rns8_linkage', 'shared_library_sha256')},
                        build=capture['build_manifest'], target=capture['target'], device=capture['device'], runtime=capture['runtime_version'],
                        environment=(capture.get('request_identity') or {}).get('environment'))
        if 'a_checksum' in capture:
            identity.update({k: capture[k] for k in ('a_checksum', 'b_checksum', 'output_checksum')})
        samples = [v['total_us'] if type(v) is dict else v for v in capture['samples']]
        cohorts.setdefault(json.dumps(identity, sort_keys=True), []).append(dict(mode=entry['case']['mode']+('-output-neighbor' if entry['case'].get('cluster_axis') == 'output-neighbor' else ''), path=str(path),
            execution_id=capture['execution_id'], independent_run=entry['independent_run'], median_us=statistics.median(samples), setup_us=capture['setup_us']))
    result = []
    for identity, observations in cohorts.items():
        if len({v['execution_id'] for v in observations}) != len(observations):
            raise ValueError('reused process identity')
        modes = {}
        for observation in observations:
            modes.setdefault(observation['mode'], []).append(observation['median_us'])
        medians = {mode: statistics.median(values) for mode, values in modes.items() if len(values) >= 2}
        baseline = 'int8' if 'int8' in medians else 'ordinary'
        result.append(dict(identity=json.loads(identity), observations=observations, independent_process_medians_us=medians,
            baseline_over_candidate={mode: medians[baseline]/value for mode, value in medians.items() if mode != baseline and baseline in medians},
            promotion_eligible=False, qualification='not_inferred'))
    with (args.output/'summary.json').open('x') as stream:
        json.dump(dict(schema='research_kernels_summary_v1', cohorts=result, promotion_eligible=False), stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
