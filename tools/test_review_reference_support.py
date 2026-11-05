"""Synthetic protocol fixtures, never hardware or numerical qualification evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from benchmark_schema import load_capture, validate_capture
from benchmark_sweep_lib.review import review_captures
from reference_verdict import numerical_contract
from benchmark_sweep_lib.capture_metadata import capture_contract_key
from timing_eligibility import capture_identity
from review_contract import REVIEW_SCHEMA_VERSION
from test_benchmark_sweep_support import bounded_capture, exact_wide_capture, finite_capture


def add_synthetic_full_reference(capture: dict) -> dict:
    raw = {key: value for key, value in capture.items() if key != 'native_reference_verification'}
    key = hashlib.sha256(('synthetic-reference-fixture-v1\n' + numerical_contract(raw)).encode()).hexdigest()
    tasks = max((capture.get('host_api_batch') or {}).get('tasks_per_measured_repeat', 1),
                (capture.get('grouped_dispatch') or {}).get('task_count', 1))
    capture['native_reference_verification'] = dict(method='boost_cpp_int_full_integer_reference',
        cache_key=key, payload_sha256='', checked_cells=capture['m'] * capture['n'] * tasks,
        elapsed_us=1.0, inside_production_interval=False)
    for name in ('verification_amortization', 'error_detection_policy', 'incremental_result_cache'):
        policy = capture.get(name)
        if isinstance(policy, dict) and policy.get('enabled') is True:
            policy['final_exact_comparison_status'] = 'exact_cpu_reference_compared'
    return capture


def synthetic_reference_controls(subject: dict) -> list[dict]:
    semantics = subject['semantics']
    factory = (finite_capture if semantics in ('finite_ring_u8', 'finite_field_u8') else
               exact_wide_capture if semantics == 'exact_wide_signed' else
               bounded_capture if semantics == 'bounded_i64' else None)
    if factory is None:
        raise ValueError('no authored reference model for this fixture semantic')
    expected = numerical_contract({key: value for key, value in subject.items() if key != 'native_reference_verification'})
    controls = []
    for backend in ('cpu-reference', 'hip-direct'):
        capture = factory(backend, 500 if backend == 'cpu-reference' else 300)
        if 'scenario_metadata' in subject:
            capture['scenario_metadata'] = copy.deepcopy(subject['scenario_metadata'])
        for name in ('verification_amortization', 'error_detection_policy'):
            capture.pop(name, None) # controls are references, not additional policy rows
        actual = numerical_contract({key: value for key, value in capture.items() if key != 'native_reference_verification'})
        if actual != expected:
            raise ValueError('synthetic control and subject have different mathematical work; repair the fixture')
        controls.append(add_synthetic_full_reference(capture))
    return controls


def write_policy_review(path: Path, capture_path: Path, *, include_cpu: bool = True,
                        candidate_promotable: bool | None = None, cache_status: str | None = None) -> None:
    subject = load_capture(capture_path)
    captures = []
    for control in synthetic_reference_controls(subject):
        backend = control['backend_selected']
        if backend == 'cpu-reference' and not include_cpu:
            continue
        target = path.parent / (backend + '.json')
        target.write_text(json.dumps(control), encoding='utf-8')
        validate_capture(control, target)
        control['_path'] = str(target)
        captures.append(control)
    validate_capture(subject, capture_path)
    subject['_path'] = str(capture_path)
    captures.append(subject)
    report = review_captures(captures, review_mode='release')
    for group in report['groups']:
        for candidate in group['candidates']:
            if candidate['capture'] == str(capture_path):
                if candidate_promotable is not None:
                    candidate['promotable'] = candidate_promotable
                if cache_status is not None:
                    candidate['cache_write_status'] = cache_status
    path.write_text(json.dumps(report), encoding='utf-8')


def write_variant_review(path: Path, capture_paths: list[Path]) -> None:
    """A variant study keeps CPU mathematical anchors outside its GPU timing rows."""
    subject = load_capture(capture_paths[0])
    anchors = []
    for control in synthetic_reference_controls(subject):
        control.pop('export_variant', None)
        control.pop('reconstruction_variant', None)
        target = path.parent / (control['backend_selected'] + '-reference.json')
        validate_capture(control, target)
        target.write_text(json.dumps(control), encoding='utf-8')
        control['_path'] = str(target)
        anchors.append(control)
    cpu = next(control for control in anchors if control['backend_selected'] == 'cpu-reference')
    groups = []
    for capture_path in capture_paths:
        subject = load_capture(capture_path);validate_capture(subject,capture_path);subject['_path'] = str(capture_path)
        candidates = []
        for capture in [subject,*anchors]:
            match = capture['checksum_u64'] == cpu['checksum_u64']
            candidates.append(dict(backend=capture['backend_selected'],capture=capture['_path'],
                capture_sha256=capture_identity(capture),checksum=capture['checksum_u64'],
                checksum_reference=cpu['checksum_u64'],checksum_reference_backend='cpu-reference',
                checksum_matches_reference=match,release_review_capture=True,promotable=False,
                promotion_blockers=['research_variant_not_promoted'] if match else ['checksum_mismatch_vs_reference'],
                promotion_reason='synthetic_reference_bound_variant_study',cache_write_status='not_eligible'))
        groups.append(dict(contract_key=capture_contract_key(subject),review_mode='release',release_review_satisfied=True,
            required_baselines=['cpu-reference','hip-direct'],missing_required_baselines=[],duplicate_backends=[],
            checksum_consistent=all(candidate['checksum_matches_reference'] for candidate in candidates),
            checksum_reference=cpu['checksum_u64'],checksum_reference_backend='cpu-reference',candidates=candidates))
    path.write_text(json.dumps(dict(schema_version=REVIEW_SCHEMA_VERSION,review_mode='release',groups=groups)),encoding='utf-8')
