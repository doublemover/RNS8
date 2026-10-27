"""Immutable binding of hook-disabled host latency to a matched event diagnostic.

No capture is edited or annotated: variance identities and artifact receipts remain
unchanged. This binding proves matching captured contracts, NOT executed ISA,
measured bandwidth, repeatability, target qualification or a performance win.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from benchmark_schema import load_capture, validate_capture
from build_manifest import manifest_provenance_complete
from timing_eligibility import capture_identity

SCHEMA = 'rns8_production_diagnostic_binding_v1'
GPU_BACKENDS = {'hip-direct', 'hipblaslt', 'ck', 'rocwmma', 'amdgpu-builtins', 'hip-vector-alu-int64'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_arguments(arguments):
    if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
        raise ValueError('argument vector required')
    result = []; i = 0; seen = False
    while i < len(arguments):
        arg = arguments[i]
        if arg == '--host-timing':
            if seen or i+1 == len(arguments) or arguments[i+1] not in ('production', 'instrumented'):
                raise ValueError('invalid/duplicate host timing option')
            seen = True; i += 2; continue
        if arg.startswith('--host-timing='):
            raise ValueError('unsupported host timing option spelling')
        result.append(arg); i += 1
    return result


def common_identity(capture):
    # Preserve unknown top-level fields: a future workload/control change fails
    # closed until its observation-only nature is explicitly reviewed here.
    identity = copy.deepcopy({k: v for k, v in capture.items() if not k.startswith('_')})
    for key in list(identity):
        if key.endswith('_us') or key in {'execution_id', 'request_identity', 'artifact_capture', 'command_line',
                'derived_tops_equivalent', 'comparison_baseline', 'timing_note', 'capture_timestamp_utc'}:
            identity.pop(key)
    for section,fields in (('native_reference_verification',('elapsed_us',)),('exact_wide_magnitude',('admission_us','oracle_us'))):
        for field in fields:
            if isinstance(identity.get(section),dict):identity[section].pop(field,None)
    timing = identity.get('timing_metadata') or {}
    for key in list(timing):
        if key.startswith('gpu_event_') or key in {'host_instrumentation_state', 'phase_states', 'timer_pair_median_us',
                'timer_pair_p95_us', 'censor_threshold_us', 'physical_accuracy_ns'}:
            timing.pop(key)
    artifact = capture.get('artifact_capture') or {}
    identity['compiled_artifact'] = {k: artifact.get(k) for k in ('executable_sha256', 'rns8_linkage', 'shared_library_sha256')}
    request = capture.get('request_identity')
    if not isinstance(request, dict) or request.get('schema') != 'rns8_benchmark_request_v1':
        raise ValueError('versioned execution request required for companion binding')
    identity['execution_request'] = {k: copy.deepcopy(v) for k, v in request.items() if k not in ('identity_sha256', 'arguments')}
    identity['execution_request']['arguments'] = normalize_arguments(request.get('arguments'))
    command = capture.get('command_line')
    if isinstance(command, str):
        # Request argv is authoritative (preserves Windows quoting). Keep the
        # printed command for inspection but do not reparsed-string-bind it.
        pass
    elif command is not None:
        raise ValueError('printed command line must be a string')
    return identity


def _pair_contract(production, diagnostic):
    p = production.get('timing_metadata') or {}; d = diagnostic.get('timing_metadata') or {}
    if p.get('host_instrumentation_state') != 'hooks_disabled' or p.get('gpu_event_timing') is not False or p.get('gpu_event_timing_status') != 'not_requested_production_timing' or p.get('gpu_event_timing_reason') != 'explicit_production_hooks_disabled':
        raise ValueError('explicit hook-disabled production capture required')
    if d.get('host_instrumentation_state') != 'diagnostic_gpu_events' or d.get('gpu_event_timing') is not True or d.get('gpu_event_timing_status') != 'available' or d.get('gpu_event_timing_source') != 'hipEventElapsedTime' or not d.get('gpu_event_phase_order'):
        raise ValueError('complete explicit HIP-event diagnostic required')
    if production.get('backend_selected') not in GPU_BACKENDS:
        raise ValueError('companion is only needed for a supported GPU backend')
    for capture in (production, diagnostic):
        if not manifest_provenance_complete(capture.get('build_manifest')):
            raise ValueError('complete compiled build identity required')
        artifact = capture.get('artifact_capture') or {}
        if artifact.get('unchanged_during_run') is not True or re.fullmatch(r'[0-9a-f]{64}',str(artifact.get('executable_sha256',''))) is None:
            raise ValueError('stable executable receipt required')
        request = capture.get('request_identity') or {}
        if request.get('executable_sha256') != artifact['executable_sha256']:
            raise ValueError('request/artifact mismatch')
        payload={k:v for k,v in request.items() if k!='identity_sha256'}
        if request.get('identity_sha256')!=digest(payload) or not isinstance(request.get('environment'),dict):
            raise ValueError('execution request digest/environment mismatch')
        args=request.get('arguments')
        normalize_arguments(args)
        requested=args[args.index('--host-timing')+1] if '--host-timing' in args else 'instrumented'
        if requested!=('production' if capture is production else 'instrumented'):
            raise ValueError('requested instrumentation disagrees with capture')
        if not isinstance(capture.get('execution_id'), str) or not capture['execution_id']:
            raise ValueError('separate child-process execution receipt required')
    if production['execution_id'] == diagnostic['execution_id']:
        raise ValueError('diagnostic and production cannot be one execution')
    left, right = common_identity(production), common_identity(diagnostic)
    if left != right:
        differing = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
        raise ValueError('companion contract differs: '+','.join(differing))
    return digest(left)


def build_binding(production_path, diagnostic_path):
    production_path, diagnostic_path = Path(production_path).resolve(), Path(diagnostic_path).resolve()
    if production_path == diagnostic_path:
        raise ValueError('distinct immutable captures required')
    production, diagnostic = load_capture(production_path), load_capture(diagnostic_path)
    validate_capture(production, production_path); validate_capture(diagnostic, diagnostic_path)
    common = _pair_contract(production, diagnostic)
    return dict(schema=SCHEMA, production_capture=str(production_path), diagnostic_capture=str(diagnostic_path),
                production_file_sha256=file_digest(production_path), diagnostic_file_sha256=file_digest(diagnostic_path),
                production_capture_sha256=capture_identity(production), diagnostic_capture_sha256=capture_identity(diagnostic),
                common_identity_sha256=common, evidence_scope='matched_contract_only_not_performance_or_ISA_qualification')


def validate_binding(binding, production=None):
    if not isinstance(binding, dict) or binding.get('schema') != SCHEMA:
        raise ValueError('diagnostic binding ABI')
    try:
        actual = build_binding(binding['production_capture'], binding['diagnostic_capture'])
    except (KeyError, OSError, TypeError) as error:
        raise ValueError('diagnostic binding source unavailable') from error
    if actual != binding:
        raise ValueError('diagnostic binding changed or forged receipt')
    if production is not None and capture_identity(production) != actual['production_capture_sha256']:
        raise ValueError('reviewed production differs from bound capture')
    return actual


def load_bindings(paths):
    result = {}
    for path in paths:
        binding = json.loads(Path(path).read_text(encoding='utf-8'))
        validate_binding(binding)
        key = str(Path(binding['production_capture']).resolve())
        if key in result:
            raise ValueError('duplicate production companion binding')
        result[key] = binding
    return result


def binding_for_capture(capture, bindings):
    path = capture.get('_path')
    return (bindings or {}).get(str(Path(path).resolve())) if path else None


def production_evidence(capture, binding=None):
    blockers = []
    timing = capture.get('timing_metadata') or {}
    if timing.get('host_instrumentation_state') != 'hooks_disabled' or timing.get('gpu_event_timing') is True:
        blockers.append('diagnostic_instrumented_host_time_not_production_evidence')
    if capture.get('backend_selected') in GPU_BACKENDS:
        if binding is None:
            blockers.append('matched_diagnostic_companion_missing')
        else:
            try:
                validate_binding(binding, capture)
            except (ValueError, OSError, TypeError) as error:
                blockers.append('matched_diagnostic_companion_invalid:'+str(error))
    return dict(eligible=not blockers, blockers=blockers, binding=binding if not blockers else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('production', type=Path); parser.add_argument('diagnostic', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); binding = build_binding(args.production, args.diagnostic)
    # Never edit the captures or replace an existing immutable binding receipt.
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(binding, indent=2, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
