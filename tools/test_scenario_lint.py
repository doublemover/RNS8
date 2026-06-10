#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from benchmark_sweep_lib.scenario_lint import validate_scenario_catalog


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + chr(10), encoding='utf-8')


_INLINE_CASE = {
    'name': 'test-64',
    'm': 64,
    'n': 64,
    'k': 64,
    'tile_m': 128,
    'tile_n': 128,
    'bound_mode': 'global',
    'input_profile': 'uniform-small',
    'promotable': True,
}


def _valid_item(name: str, **overrides: object) -> dict:
    base = {
        'name': name,
        'semantics': 'bounded-i64',
        'case': dict(_INLINE_CASE, name=f'case-{name}'),
        'evidence_scope': 'release_review_candidate',
        'output_domain': 'host_export',
        'rationale': 'test item',
        'review_mode_expectation': 'release',
        'promotion_eligibility': 'release_review_candidate',
        'backends': ['cpu', 'hip-direct'],
    }
    for k, v in overrides.items():
        if v is not None:
            base[k] = v
    return base


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        valid_dir = tmp / 'valid'
        _write(valid_dir / 'test.json', {
            'schema_version': 1,
            'family': 'valid-family',
            'items': [
                _valid_item('release-row'),
                _valid_item('smoke-row', promotion_eligibility='proxy_evidence_only', review_mode_expectation='smoke'),
            ],
        })
        errors = validate_scenario_catalog(valid_dir)
        assert errors == [], f'Expected no errors, got: {errors}'

        smoke_release_dir = tmp / 'smoke-release'
        _write(smoke_release_dir / 'bad.json', {
            'schema_version': 1,
            'family': 'bad-family',
            'items': [
                _valid_item('bad-row', review_mode_expectation='smoke', promotion_eligibility='release_review_candidate'),
            ],
        })
        errors = validate_scenario_catalog(smoke_release_dir)
        assert len(errors) == 1
        assert 'review_mode_expectation=release' in errors[0]

        bad_domain_dir = tmp / 'bad-domain'
        _write(bad_domain_dir / 'bad.json', {
            'schema_version': 1,
            'family': 'bad-domain-family',
            'items': [_valid_item('bad-row', output_domain='nonexistent')],
        })
        errors = validate_scenario_catalog(bad_domain_dir)
        assert any('unregistered output_domain=nonexistent' in e for e in errors)

        residue_dir = tmp / 'residue-no-next'
        _write(residue_dir / 'bad.json', {
            'schema_version': 1,
            'family': 'residue-family',
            'items': [_valid_item('bad-row', output_domain='residue_current_rns', review_mode_expectation='release')],
        })
        errors = validate_scenario_catalog(residue_dir)
        assert any('residue_current_rns output requires next_op_hint' in e for e in errors)

        graph_dir = tmp / 'graph-no-baseline'
        _write(graph_dir / 'bad.json', {
            'schema_version': 1,
            'family': 'graph-family',
            'items': [_valid_item('graph-row', hip_graph_replay=True)],
        })
        errors = validate_scenario_catalog(graph_dir)
        assert any('hip_graph_replay requires a non-graph baseline' in e for e in errors)

        graph_valid_dir = tmp / 'graph-valid'
        _write(graph_valid_dir / 'ok.json', {
            'schema_version': 1,
            'family': 'graph-valid-family',
            'items': [_valid_item('baseline-row', case=dict(_INLINE_CASE, name='shared-case')), _valid_item('graph-row', hip_graph_replay=True, case=dict(_INLINE_CASE, name='shared-case'))],
        })
        errors = validate_scenario_catalog(graph_valid_dir)
        assert not any('hip_graph_replay requires' in e for e in errors)

        bad_backend_dir = tmp / 'bad-backend'
        _write(bad_backend_dir / 'bad.json', {
            'schema_version': 1,
            'family': 'backend-family',
            'items': [_valid_item('bad-row', semantics='bounded-i64', backends=['cpu', 'nvidia-cuda'])],
        })
        errors = validate_scenario_catalog(bad_backend_dir)
        assert any('backend=nvidia-cuda not supported for semantics=bounded-i64' in e for e in errors)

        bad_prefix_dir = tmp / 'bad-prefix'
        _write(bad_prefix_dir / 'bad.json', {
            'schema_version': 1,
            'family': 'prefix-family',
            'items': [_valid_item('bad-row', prefix_policy='fixed-requested', max_prefix=99)],
        })
        errors = validate_scenario_catalog(bad_prefix_dir)
        assert any('max_prefix=99 out of valid range' in e for e in errors)

    print('scenario lint self-test: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
