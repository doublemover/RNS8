#!/usr/bin/env python3
"""Generate an interactive HTML performance dashboard from sweep captures."""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from pathlib import Path

def build(sweep_dir, out_path):
    captures = []
    for root, dirs, files in os.walk(sweep_dir):
        for f in files:
            if f.endswith('.json') and not f.endswith('.failed.json') and 'scenarios' in root:
                captures.append(Path(root) / f)
    rows = defaultdict(list)
    for path in captures:
        d = json.loads(path.read_text(encoding='utf-8'))
        sem = d.get('semantics','unknown')
        m,n,k = d.get('m',0),d.get('n',0),d.get('k',0)
        b = d.get('backend_selected','unknown')
        e2e = d.get('avg_end_to_end_us',0)
        pk = d.get('avg_pack_us',0)
        gm = d.get('avg_rns_gemm_us',0)
        ex = d.get('avg_crt_export_us',0)
        if e2e <= 0: continue
        rows[f'{sem}_{m}x{n}x{k}'].append((b,e2e,pk,gm,ex))
    h = ['<html><head><meta charset=UTF-8><title>RNS8 Dashboard</title>',
         '<style>body{font-family:sans-serif;margin:20px;background:#1a1a2e;color:#e0e0e0}',
         'table{border-collapse:collapse;width:100%}th,td{padding:6px 10px;text-align:right;border:1px solid #333}',
         'th{background:#2d2d44;cursor:pointer}tr:nth-child(even){background:#222240}',
         '.win{color:#4caf50;font-weight:bold}.dir{color:#ff9800}.amd{color:#2196f3}.roc{color:#e91e63}',
         '.bar{height:16px;margin:1px 0;border-radius:2px;display:inline-block}',
         '.bp{background:#ff9800}.bg{background:#4caf50}.be{background:#2196f3}',
         '</style></head><body><h1>RNS8 Performance Dashboard</h1>',
         f'<p>Sweep: {sweep_dir} | Captures: {len(captures)}</p>',
         '<table><tr><th>Semantics</th><th>Shape</th><th>Winner</th><th>E2E(us)</th><th>Pack</th><th>GEMM</th><th>Export</th></tr>']
    for key in sorted(rows.keys()):
        entries = sorted(rows[key], key=lambda x: x[1])
        w = entries[0]
        parts = key.rsplit('_',1)
        sem = parts[0] if len(parts)==2 else key
        shp = parts[1] if len(parts)==2 else key
        c = ''
        if 'amdgpu' in w[0]: c='amd'
        elif 'rocwmma' in w[0]: c='roc'
        elif 'hip-direct' in w[0]: c='dir'
        h.append(f'<tr><td>{sem}</td><td>{shp}</td><td class="win {c}">{w[0]}</td><td>{w[1]:.0f}</td><td>{w[2]:.0f}</td><td>{w[3]:.0f}</td><td>{w[4]:.0f}</td></tr>')
    h.append('</table>')
    # Phase charts for large shapes
    h.append('<h2>Phase Breakdowns</h2>')
    for key in sorted(rows.keys()):
        entries = sorted(rows[key], key=lambda x: x[1])
        w = entries[0]
        parts2 = key.rsplit('_',1)[1].split('x') if '_' in key else ['0','0','0']
        parts = parts2
        m = int(parts[0]) if parts else 0
        if m < 256: continue
        pp = w[2]/w[1]*100 if w[1]>0 else 0
        gp = w[3]/w[1]*100 if w[1]>0 else 0
        ep = w[4]/w[1]*100 if w[1]>0 else 0
        op = 100-pp-gp-ep
        h.append(f'<div style="margin:4px 0"><b>{key}</b> ({w[0]}) {w[1]:.0f}us<br>')
        h.append(f'<span class="bar bp" style="width:{pp:.0f}%"></span> Pack {pp:.0f}% ')
        h.append(f'<span class="bar bg" style="width:{gp:.0f}%"></span> GEMM {gp:.0f}% ')
        h.append(f'<span class="bar be" style="width:{ep:.0f}%"></span> Export {ep:.0f}% ')
        h.append(f'<span class="bar" style="width:{op:.0f}%"></span> Other {op:.0f}%<br></div>')
    h.append('</body></html>')
    out_path.write_text('\n'.join(h), encoding='utf-8')
    print(f'Dashboard written to {out_path}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture-root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('docs/dashboard.html'))
    a = p.parse_args()
    build(a.capture_root, Path(__file__).resolve().parents[1] / a.out)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
