#!/usr/bin/env python3
"""Generate an interactive HTML performance dashboard from sweep captures.

Reads all JSON captures from a sweep output directory and produces a
single self-contained dashboard.html with sortable tables, phase breakdown
charts, backend comparison bar charts, and shape x backend heatmap.

Output: docs/dashboard.html (static, no server required)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RNS8 Performance Dashboard</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }
  h1 { color: #7b68ee; }
  table { border-collapse: collapse; width: 100%%; margin: 10px 0; }
  th, td { padding: 8px 12px; text-align: right; border: 1px solid #333; }
  th { background: #2d2d44; cursor: pointer; position: sticky; top: 0; }
  tr:nth-child(even) { background: #222240; }
  tr:hover { background: #333355; }
  .winner { color: #4caf50; font-weight: bold; }
  .backend-direct { color: #ff9800; }
  .backend-builtin { color: #2196f3; }
  .backend-rocwmma { color: #e91e63; }
  .chart-container { display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }
  .chart-box { background: #2d2d44; padding: 15px; border-radius: 8px; min-width: 300px; }
  .bar { height: 20px; margin: 2px 0; border-radius: 3px; display: inline-block; }
  .bar-pack { background: #ff9800; }
  .bar-gemm { background: #4caf50; }
  .bar-export { background: #2196f3; }
  .heatmap-cell { display: inline-block; width: 40px; height: 20px; margin: 1px; border-radius: 2px; }
  #search { padding: 8px; width: 300px; margin: 10px 0; background: #2d2d44; color: #e0e0e0; border: 1px solid #444; border-radius: 4px; }
</style>
</head>
<body>
<h1>RNS8 Performance Dashboard</h1>
<p>Sweep: {sweep_dir} | Captures: {capture_count} | Generated: {generated_date}</p>
<input id="search" placeholder="Filter by semantics, shape, or backend..." onkeyup="filterTable()">

<h2>Backend Winners by Shape</h2>
<table id="winners">
<thead><tr><th onclick="sortTable(0)">Semantics</th><th onclick="sortTable(1)">Shape</th><th onclick="sortTable(2)">Winner</th><th onclick="sortTable(3)">E2E (us)</th><th onclick="sortTable(4)">Pack</th><th onclick="sortTable(5)">GEMM</th><th onclick="sortTable(6)">Export</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>

<h2>Phase Breakdowns</h2>
<div class="chart-container">
{phase_charts}
</div>

<h2>Backend Comparison: {comparison_shape}</h2>
<div class="chart-container">
{backend_bars}
</div>

<script>
function sortTable(n) {{
  const table = document.getElementById("winners");
  const rows = Array.from(table.rows).slice(1);
  const asc = table.getAttribute("data-sort") !== n.toString();
  table.setAttribute("data-sort", asc ? n.toString() : "");
  rows.sort((a, b) => {{
    const va = a.cells[n].textContent, vb = b.cells[n].textContent;
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => table.appendChild(r));
}}
function filterTable() {{
  const q = document.getElementById("search").value.toLowerCase();
  const rows = document.querySelectorAll("#winners tbody tr");
  rows.forEach(r => r.style.display = r.textContent.toLowerCase().includes(q) ? "" : "none");
}}
</script>
</body>
</html>
"""


def build_dashboard(sweep_dir: Path, out_path: Path) -> None:
    captures = []
    for root, dirs, files in os.walk(sweep_dir):
        for f in files:
            if f.endswith('.json') and not f.endswith('.failed.json') and 'scenarios' in root:
                captures.append(Path(root) / f)

    rows = defaultdict(list)
    for path in captures:
        d = json.loads(path.read_text(encoding='utf-8'))
        sem = d.get('semantics', 'unknown')
        m, n, k = d.get('m', 0), d.get('n', 0), d.get('k', 0)
        backend = d.get('backend_selected', 'unknown')
        e2e = d.get('avg_end_to_end_us', 0)
        pack = d.get('avg_pack_us', 0)
        gemm = d.get('avg_rns_gemm_us', 0)
        export = d.get('avg_crt_export_us', 0)
        if e2e <= 0:
            continue
        key = f'{sem}_{m}x{n}x{k}'
        rows[key].append((backend, e2e, pack, gemm, export))

    # Build winner table
    table_rows = []
    phase_charts = []
    for key in sorted(rows.keys()):
        entries = sorted(rows[key], key=lambda x: x[1])
        winner = entries[0]
        cls = ''
        if 'amdgpu' in winner[0]: cls = 'backend-builtin'
        elif 'rocwmma' in winner[0]: cls = 'backend-rocwmma'
        elif 'hip-direct' in winner[0]: cls = 'backend-direct'
        sem, shape = key.split('_', 1)
        pack_pct = winner[2] / winner[1] * 100 if winner[1] > 0 else 0
        gemm_pct = winner[3] / winner[1] * 100 if winner[1] > 0 else 0
        exp_pct = winner[4] / winner[1] * 100 if winner[1] > 0 else 0
        table_rows.append(
            f'<tr><td>{sem}</td><td>{shape}</td>'
            f'<td class="winner {cls}">{winner[0]}</td>'
            f'<td>{winner[1]:.0f}</td><td>{winner[2]:.0f}</td>'
            f'<td>{winner[3]:.0f}</td><td>{winner[4]:.0f}</td></tr>'
        )
        # Phase chart for shapes >= 512
        if m >= 256:
            chart = f'<div class="chart-box"><b>{key}</b> ({winner[0]})<br>'
            chart += f'<span class="bar bar-pack" style="width:{pack_pct:.0f}%"></span> Pack {pack_pct:.0f}%<br>'
            chart += f'<span class="bar bar-gemm" style="width:{gemm_pct:.0f}%"></span> GEMM {gemm_pct:.0f}%<br>'
            chart += f'<span class="bar bar-export" style="width:{exp_pct:.0f}%"></span> Export {exp_pct:.0f}%<br>'
            chart += f'{winner[1]:.0f} us total</div>'
            phase_charts.append(chart)

    # Best comparison shape
    comp_shape = max(rows.keys(), key=lambda k: len(rows[k]), default='none')
    backend_bars = ''
    if comp_shape in rows:
        for backend, e2e, pack, gemm, export in sorted(rows[comp_shape], key=lambda x: x[1]):
            backend_bars += f'<div class="chart-box"><b>{backend}</b>: {e2e:.0f} us<br>'
            backend_bars += f'Pack: {pack:.0f} | GEMM: {gemm:.0f} | Export: {export:.0f}</div>
'

    html = HTML_TEMPLATE.format(
        sweep_dir=str(sweep_dir),
        capture_count=len(captures),
        generated_date='2026-06-10',
        table_rows='
'.join(table_rows),
        phase_charts='
'.join(phase_charts[:20]),
        comparison_shape=comp_shape,
        backend_bars=backend_bars,
    )

    out_path.write_text(html, encoding='utf-8')
    print(f'Dashboard written to {out_path}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True,
                        help='sweep output root directory')
    parser.add_argument('--out', type=Path,
                        default=Path('docs/dashboard.html'),
                        help='output HTML file path')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / args.out
    build_dashboard(args.capture_root, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
