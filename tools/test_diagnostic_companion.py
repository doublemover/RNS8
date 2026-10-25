"""Authored synthetic companion receipts; no benchmark executable is invoked."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from benchmark_schema import load_capture, validate_capture, validation_errors
from benchmark_sweep_lib.review import review_captures, write_promoted_cache_entries
from promotion_ledger import capture_entry
from collect_diagnostic_pairs import paired_plan
from diagnostic_companion import build_binding, common_identity, digest, load_bindings, production_evidence, validate_binding
from test_capture_identity_support import synthetic_identity, synthetic_host_measurements

FIXTURES = Path(__file__).parents[1]/'tests/fixtures/benchmark_schema'


def authored_pair(root, name='v4_bounded_i64_rocwmma.json'):
    diagnostic = load_capture(FIXTURES/name)
    synthetic_identity(diagnostic); synthetic_host_measurements(diagnostic)
    diagnostic['timing_metadata']['host_instrumentation_state'] = 'diagnostic_gpu_events'
    production = copy.deepcopy(diagnostic); production['execution_id'] = str(uuid.uuid4())
    p = production['timing_metadata']; p.update(host_instrumentation_state='hooks_disabled', gpu_event_timing=False,
        gpu_event_timing_status='not_requested_production_timing', gpu_event_timing_reason='explicit_production_hooks_disabled')
    for key in ('gpu_event_phase_order', 'gpu_event_timing_source', 'gpu_event_timing_source_scope'):
        p[key] = None
    production['gpu_event_timings_us'] = production['gpu_event_timing_summary_us'] = None
    paths = []
    for capture, mode in ((production, 'production'), (diagnostic, 'instrumented')):
        request = dict(schema='rns8_benchmark_request_v1', arguments=['--backend', capture['backend_selected'], '--host-timing', mode],
                       executable_sha256='b'*64, environment={'HIP_VISIBLE_DEVICES': '0'}, overrides_sha256='e'*64)
        request['identity_sha256'] = digest(request); capture['request_identity'] = request
        path = root/(mode+'.json'); path.write_text(json.dumps(capture)); paths.append(path)
    return production, diagnostic, paths


class DiagnosticCompanion(unittest.TestCase):
    def test_eventless_accelerator_requires_explicit_production_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            production, diagnostic, paths = authored_pair(Path(temp))
            validate_capture(production); validate_capture(diagnostic)
            bad = copy.deepcopy(production); bad['timing_metadata'].pop('host_instrumentation_state')
            self.assertTrue(any('HIP event' in error for error in validation_errors(bad)))
            bad = copy.deepcopy(production); bad['gpu_event_timings_us'] = diagnostic['gpu_event_timings_us']
            self.assertTrue(any('must be null' in error for error in validation_errors(bad)))

    def test_raw_files_and_all_bound_contracts_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); production, diagnostic, paths = authored_pair(root)
            before = [p.read_bytes() for p in paths]
            binding = build_binding(*paths); self.assertEqual(validate_binding(binding, production), binding)
            self.assertTrue(production_evidence(production, binding)['eligible'])
            self.assertEqual(before, [p.read_bytes() for p in paths])
            self.assertEqual(common_identity(production), common_identity(diagnostic))
            changed = copy.deepcopy(production); changed['selected_kernel'] += '_other'
            self.assertFalse(production_evidence(changed, binding)['eligible'])
            for field, value in [('selected_kernel', 'other'), ('seed', 99), ('warmups', 0), ('checksum_u64', 0), ('packed_layout_version', 99)]:
                bad = copy.deepcopy(diagnostic); bad[field] = value
                self.assertNotEqual(common_identity(production), common_identity(bad), field)
            paths[1].write_text(paths[1].read_text()+'\n')  # even byte-only post-binding replacement is detectable
            with self.assertRaises(ValueError):
                validate_binding(binding)

    def test_independent_process_artifact_environment_and_request_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); production, diagnostic, paths = authored_pair(root)
            mutations = [lambda c: c.update(execution_id=production['execution_id']),
                lambda c: c['artifact_capture'].update(executable_sha256='0'*64),
                lambda c: c['request_identity']['environment'].update(HIP_VISIBLE_DEVICES='1'),
                lambda c: c['request_identity'].update(identity_sha256='0'*64),
                lambda c: c['build_manifest'].update(source_digest='f'*64)]
            for mutate in mutations:
                bad = copy.deepcopy(diagnostic); mutate(bad); paths[1].write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    build_binding(*paths)
            paths[1].write_text(json.dumps(diagnostic)); binding = build_binding(*paths)
            sidecar = root/'binding.json'; sidecar.write_text(json.dumps(binding))
            index = load_bindings([sidecar]); self.assertIn(str(paths[0].resolve()), index)
            with self.assertRaises(ValueError):
                load_bindings([sidecar, sidecar])

    def test_binding_satisfies_only_diagnostic_gate_not_other_release_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            production, diagnostic, paths = authored_pair(Path(temp))
            binding = build_binding(*paths); production['_path'] = str(paths[0])
            missing = review_captures([production], review_mode='release')
            candidate = missing['groups'][0]['candidates'][0]
            self.assertIn('missing_required_gpu_events', candidate['promotion_blockers'])
            report = review_captures([production], review_mode='release', diagnostic_bindings={str(paths[0].resolve()): binding})
            candidate = report['groups'][0]['candidates'][0]
            self.assertNotIn('missing_required_gpu_events', candidate['promotion_blockers'])
            self.assertTrue(candidate['production_timing_evidence']['eligible'])
            self.assertIn('missing_required_baselines', candidate['promotion_blockers'])
            self.assertFalse(candidate['promotable'])
            ledger=capture_entry(paths[0],diagnostic_binding=binding)
            self.assertTrue(ledger['production_timing_evidence']['eligible'])
            self.assertNotIn('matched_diagnostic_companion_missing',ledger['promotion_blockers'])
            cache = Path(temp)/'not-written.json'
            self.assertEqual(write_promoted_cache_entries(report, [production], cache), 0)
            self.assertFalse(cache.exists())
            # A mutated/stale report cannot turn instrumented timings into a cache write.
            forged = dict(promotable_autotune_entries=[dict(source_capture=str(paths[0]), source_capture_sha256='0'*64, diagnostic_companion=binding)])
            self.assertEqual(write_promoted_cache_entries(forged, [production], cache), 0)
            self.assertFalse(cache.exists())

    def test_collection_counterbalances_modes_without_running_binary(self):
        plan = paired_plan(Path('not-executed'), ['--m', '17', '--host-timing', 'production'], Path('out'), 3, 'fixture')
        self.assertEqual([r['captures'][0]['mode'] for r in plan], ['production', 'instrumented', 'production'])
        self.assertEqual(len({c['output'] for r in plan for c in r['captures']}), 6)
        for row in plan:
            self.assertEqual(len(row['captures']), 2)
        with self.assertRaises(ValueError):
            paired_plan(Path('not-executed'), ['--sustained-batches', '3'], Path('out'), 2, 'fixture')


if __name__ == '__main__':
    unittest.main()
