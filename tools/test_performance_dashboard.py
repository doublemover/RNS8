#!/usr/bin/env python3
"""Capture explorer tests: real schema checks and escaped, non-promotional output."""
import json
import tempfile
from pathlib import Path
from benchmark_schema import BenchmarkSchemaError
from generate_performance_dashboard import build

ROOT = Path(__file__).resolve().parents[1]


def main():
    fixture = ROOT / 'tests/fixtures/benchmark_schema/v4_bounded_i64_ck.json'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "capture & 'sample'.json"
        capture = json.loads(fixture.read_text(encoding='utf-8'))
        capture['device']['name'] = '<script>alert("capture")</script>'
        source.write_text(json.dumps(capture), encoding='utf-8')
        output = root / 'new' / 'report.html'
        assert build(root, output) == 1
        text = output.read_text()
        assert 'capture &amp; &#x27;sample&#x27;.json' in text and source.name not in text
        assert '&lt;script&gt;' in text and '&lt;/script&gt;' in text and '<script>' not in text
        assert 'not a release review' in text
        capture = json.loads(source.read_text())
        capture['selected_kernel'] = 'invalid_kernel'
        source.write_text(json.dumps(capture))
        try:
            build(root, output)
        except BenchmarkSchemaError:
            pass
        else:
            raise AssertionError('malformed capture must not be reported as schema-valid')
    print('performance dashboard self-test: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
