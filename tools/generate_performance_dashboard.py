#!/usr/bin/env python3
"""Build a schema-validated capture explorer; never infer cross-contract winners."""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

from report_capture_inputs import expand_report_inputs, load_report_capture
from result_compare import CONTRACT_KEYS, contract_value


def build(sweep_dir: Path, out_path: Path) -> int:
    paths = expand_report_inputs([sweep_dir])
    rows = []
    started = heartbeat = time.monotonic()
    print(f"capture explorer: reading {len(paths)} JSON inputs", flush=True)
    for index, (path, from_directory) in enumerate(paths, 1):
        capture = load_report_capture(path, from_directory=from_directory)
        if capture is not None:
            contract = {key: contract_value(capture, key) for key in CONTRACT_KEYS}
            contract.update({"device": capture.get("device"), "seed": capture.get("seed"),
                             "correctness": capture.get("correctness")})
            escape = lambda value: html.escape(str(value), quote=True)
            shape = ' × '.join(str(capture.get(key)) for key in ('m', 'n', 'k'))
            cells = [path.name, capture.get('semantics'), shape, capture.get('backend_selected'),
                     capture.get('target_id') or capture.get('device', {}).get('gcn_arch'),
                     capture.get('avg_end_to_end_us'), capture.get('checksum_u64')]
            row = '<tr>' + ''.join(f'<td>{escape(value)}</td>' for value in cells)
            row += '<td><details><summary>Contract</summary><pre>'
            row += escape(json.dumps(contract, indent=2, sort_keys=True))
            row += '</pre></details></td></tr>'
            rows.append(row)
        now = time.monotonic()
        if now - heartbeat >= 10:
            print(f"capture explorer: {index}/{len(paths)} inputs elapsed={now-started:.1f}s", flush=True)
            heartbeat = now
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>RNS8 capture explorer</title>
<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #bbb;padding:.5rem;text-align:left;vertical-align:top}
pre{white-space:pre-wrap;max-width:40rem}th{background:#eee}</style>
<h1>RNS8 capture explorer</h1>
<p>Schema-valid captures only. This is not a release review or a winner ranking.
Different bounds, prefixes, targets, seeds, lifetimes, output policies, or build
revisions must not be compared as if they were the same workload. Use the
benchmark review and variance tools before making performance claims.</p>
'''
    document += f'<p>{len(rows)} captures from {len(paths)} JSON inputs.</p>'
    document += '<table><thead><tr>' + ''.join(f'<th>{title}</th>' for title in
        ('Capture', 'Semantics', 'M × N × K', 'Backend', 'Target', 'Mean E2E (µs)', 'Checksum', 'Details'))
    document += '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></html>\n'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding='utf-8')
    print(f"capture explorer: {len(rows)} validated captures, elapsed={time.monotonic()-started:.1f}s: {out_path}", flush=True)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('temp/performance-dashboard.html'))
    args = parser.parse_args()
    build(args.capture_root, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
