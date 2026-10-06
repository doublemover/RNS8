#!/usr/bin/env python3
"""Test qualification plans and fail-closed evidence checks without using a GPU."""
import json
import sys
import tempfile
from pathlib import Path

import gpu_qualification as q
from process_progress import run_capture


def main():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        for target in q.TARGETS:
            plan = q.command_plan(target, list(q.BACKENDS), out, 2)
            assert len({item['name'] for item in plan}) == len(plan)
            assert all('--write-autotune-cache' not in item['argv'] for item in plan)
            assert sum('inspect_target' in item for item in plan) == len(q.BACKENDS)
            assert sum('junit' in item for item in plan) == len(q.BACKENDS)
            assert any('hip-direct-graph' == item['name'] for item in plan)
            assert all(sys.executable not in item['argv'] for item in plan)
            for item in plan:
                assert all(isinstance(arg, str) for arg in item['argv'])
        good = {'returncode': 0, 'timed_out': False, 'stdout': json.dumps({'gcn_arch': 'gfx942:sramecc+:xnack-', 'backend': 'hip-direct'})}
        check = {'inspect_target': 'gfx942', 'inspect_backend': 'hip-direct'}
        assert not q.validate_result(check, good)
        assert q.validate_result({**check, 'inspect_target': 'gfx1100'}, good)
        assert q.validate_result(check, {**good, 'stdout': '{}'})
        assert q.validate_result({}, {**good, 'timed_out': True})
        xml = out / 'junit.xml'
        item = {'junit': str(xml), 'require_test': 'required'}
        assert q.validate_result(item, good)
        xml.write_text('<testsuites><testsuite><testcase name="required"/></testsuite></testsuites>')
        assert not q.validate_result(item, good)
        xml.write_text('<testsuites><testsuite><testcase name="required"><skipped/></testcase></testsuite></testsuites>')
        assert q.validate_result(item, good)
        xml.write_text('<testsuites><testsuite><testcase name="wrong"/></testsuite></testsuites>')
        assert q.validate_result(item, good)
        result = run_capture([sys.executable, '-c', 'print("ok")'], cwd=q.ROOT, label='test', timeout_seconds=10)
        assert result['returncode'] == 0 and result['stdout'] == 'ok\n'
        result = run_capture([sys.executable, '-c', 'import time; time.sleep(5)'], cwd=q.ROOT, label='timeout-test', timeout_seconds=0.1, heartbeat_seconds=0.02)
        assert result['timed_out'] and result['returncode'] != 0
        result = run_capture([str(out / 'missing-program')], cwd=q.ROOT, label='missing-test')
        assert result['returncode'] is None
    kernels = q.kernel_inventory()
    assert kernels and any(item['name'] == 'cdna3_dense_mfma_kernel' for item in kernels)
    assert not any('skinny_kernel' in item['name'] and 'amdgpu_builtin' in item['name'] for item in kernels)
    print('GPU qualification self-test: PASS (no GPU used)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
