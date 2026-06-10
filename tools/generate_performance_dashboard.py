#!/usr/bin/env python3
"""Generate an interactive performance dashboard from sweep captures."""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from pathlib import Path

CSS = """body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:#0d1117;color:#c9d1d9}
h1{color:#58a6ff;border-bottom:1px solid#30363d;padding-bottom:10px}
h2{color:#f0883e;margin-top:30px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}
th,td{padding:6px 10px;text-align:right;border:1px solid#30363d}
th{background:#161b22;cursor:pointer;position:sticky;top:0;z-index:1}
th:hover{background:#1f2937}
tr:nth-child(even){background:#0d1117}
tr:nth-child(odd){background:#161b22}
tr:hover{background:#1f2937}
.win{color:#3fb950;font-weight:bold}.dir{color:#d2991d}.amd{color:#58a6ff}.roc{color:#f778ba}.hip{color:#a371f7}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.card{background:#161b22;border:1px solid#30363d;border-radius:8px;padding:16px}
.card h3{color:#8b949e;font-size:11px;text-transform:uppercase;margin:0 0 8px 0}
.card .val{font-size:28px;font-weight:700;color:#c9d1d9}
.card .sub{font-size:12px;color:#8b949e}
.bar-row{display:flex;align-items:center;margin:2px 0;font-size:11px}
.bar-row span{min-width:50px}
.bar-bg{flex:1;height:14px;background:#21262d;border-radius:7px;margin:0 8px;overflow:hidden}
.bar-fill{height:100%;border-radius:7px;transition:width.3s}
.bp{background:#d2991d}.bg{background:#3fb950}.be{background:#58a6ff}.bo{background:#484f58}
.tabs{display:flex;gap:4px;margin:12px 0}
.tab{padding:6px 16px;background:#21262d;border:1px solid#30363d;border-radius:6px 6px 0 0;cursor:pointer;color:#8b949e;font-size:13px}
.tab.active{background:#0d1117;color:#c9d1d9;border-bottom-color:#0d1117}
#search{padding:6px 12px;width:280px;background:#21262d;color:#c9d1d9;border:1px solid#30363d;border-radius:6px;font-size:13px;margin-right:8px}
#search:focus{outline:none;border-color:#58a6ff}
.heatmap{display:flex;flex-wrap:wrap;gap:1px;margin:8px 0}
.hcell{width:18px;height:18px;border-radius:2px;font-size:8px}
.speedup-bar{display:inline-block;height:8px;border-radius:4px;margin:0 4px;vertical-align:middle}
"""

JS = """function sortTable(n,t){const r=document.getElementById("t-"+t);if(!r)return;const o=Array.from(r.rows).slice(1);const a=r.getAttribute("data-sort")!==n.toString();r.setAttribute("data-sort",a?n.toString():"");o.sort((e,d)=>{const s=e.cells[n].textContent,l=d.cells[n].textContent,v=parseFloat(s),u=parseFloat(l);return!isNaN(v)&&!isNaN(u)?a?v-u:u-v:a?s.localeCompare(l):l.localeCompare(s)});o.forEach(e=>r.appendChild(e))}
function filterTable(){const q=document.getElementById("search").value.toLowerCase();["winners","shapes","phases"].forEach(t=>{const r=document.getElementById("t-"+t);if(!r)return;Array.from(r.rows).slice(1).forEach(e=>e.style.display=e.textContent.toLowerCase().includes(q)?"":"none")})}
function showTab(t){document.querySelectorAll(".tab").forEach(e=>e.classList.remove("active"));document.querySelectorAll(".tab-content").forEach(e=>e.style.display="none");const el=document.getElementById("tab-"+t);const btn=document.getElementById("btn-"+t);if(el)el.style.display="block";if(btn)btn.classList.add("active")}
"""

def build(sweep_dir, out_path):
    captures = []
    for root, dirs, files in os.walk(sweep_dir):
        for f in files:
            if f.endswith('.json') and not f.endswith('.failed.json') and 'scenarios' in root:
                captures.append(Path(root) / f)
    rows = defaultdict(list)
    backends = set()
    for path in captures:
        d = json.loads(path.read_text(encoding='utf-8'))
        sem = d.get('semantics','?')
        m,n,k = d.get('m',0),d.get('n',0),d.get('k',0)
        b = d.get('backend_selected','?')
        e2e = d.get('avg_end_to_end_us',0)
        pk = d.get('avg_pack_us',0)
        gm = d.get('avg_rns_gemm_us',0)
        ex = d.get('avg_crt_export_us',0)
        if e2e <= 0: continue
        backends.add(b)
        rows[f'{sem}|{m}x{n}x{k}'].append((b,e2e,pk,gm,ex,d.get('checksum_u64',0)))

    # Summary cards
    fastest = {}
    for k, vs in rows.items():
        vs_sorted = sorted(vs, key=lambda x: x[1])
        fastest[k] = vs_sorted[0]
    direct_wins = sum(1 for v in fastest.values() if 'hip-direct' in v[0])
    builtin_wins = sum(1 for v in fastest.values() if 'amdgpu' in v[0])
    rocwmma_wins = sum(1 for v in fastest.values() if 'rocwmma' in v[0])
    hipblaslt_wins = sum(1 for v in fastest.values() if 'hipblaslt' in v[0])
    cpu_wins = sum(1 for v in fastest.values() if 'cpu' in v[0])

    html = ['<!DOCTYPE html><html lang=en><head><meta charset=UTF-8><meta name=viewport content="width=device-width,initial-scale=1">',
            '<title>RNS8 Performance Dashboard</title><style>', CSS, '</style></head><body>',
            '<h1>RNS8 Performance Dashboard</h1>',
            '<div class=summary>',
            f'<div class=card><h3>Captures</h3><div class=val>{len(captures)}</div><div class=sub>schema-valid</div></div>',
            f'<div class=card><h3>Direct HIP Wins</h3><div class=val style=color:#d2991d>{direct_wins}</div><div class=sub>{direct_wins*100//max(len(fastest),1)}% of groups</div></div>',
            f'<div class=card><h3>AMDGPU Builtin Wins</h3><div class=val style=color:#58a6ff>{builtin_wins}</div><div class=sub>{builtin_wins*100//max(len(fastest),1)}% of groups</div></div>',
            f'<div class=card><h3>rocWMMA Wins</h3><div class=val style=color:#f778ba>{rocwmma_wins}</div><div class=sub>{rocwmma_wins*100//max(len(fastest),1)}% of groups</div></div>',
            f'<div class=card><h3>hipBLASLt Wins</h3><div class=val style=color:#a371f7>{hipblaslt_wins}</div><div class=sub>{hipblaslt_wins*100//max(len(fastest),1)}% of groups</div></div>',
            f'<div class=card><h3>CPU Wins</h3><div class=val>{cpu_wins}</div><div class=sub>tiny shapes</div></div>',
            '</div>',
            '<div style="margin:8px 0"><input id=search placeholder="Filter..." onkeyup=filterTable()></div>',
            '<div class=tabs>',
            '<div class="tab active" id=btn-winners onclick=showTab("winners")>Winners</div>',
            '<div class=tab id=btn-shapes onclick=showTab("shapes")>Shapes</div>',
            '<div class=tab id=btn-phases onclick=showTab("phases")>Phases</div>',
            '</div>']

    # Winners tab
    html.append('<div class=tab-content id=tab-winners>')
    html.append('<table id=t-winners><thead><tr><th onclick=sortTable(0,"winners")>Semantics</th><th onclick=sortTable(1,"winners")>Shape</th><th onclick=sortTable(2,"winners")>Winner</th><th onclick=sortTable(3,"winners")>E2E(us)</th><th onclick=sortTable(4,"winners")>Pack</th><th onclick=sortTable(5,"winners")>GEMM</th><th onclick=sortTable(6,"winners")>Export</th></tr></thead><tbody>')
    for key in sorted(rows.keys()):
        entries = sorted(rows[key], key=lambda x: x[1])
        w = entries[0]
        sem,shp = key.split('|',1)
        c = 'dir'; 
        if 'amdgpu' in w[0]: c='amd'
        elif 'rocwmma' in w[0]: c='roc'
        elif 'hipblaslt' in w[0]: c='hip'
        elif 'cpu' in w[0]: c=''
        html.append(f'<tr><td>{sem}</td><td>{shp}</td><td class="win {c}">{w[0]}</td><td>{w[1]:.0f}</td><td>{w[2]:.0f}</td><td>{w[3]:.0f}</td><td>{w[4]:.0f}</td></tr>')
    html.append('</tbody></table></div>')

    # Shapes tab
    html.append('<div class=tab-content id=tab-shapes style=display:none>')
    html.append('<table id=t-shapes><thead><tr><th onclick=sortTable(0,"shapes")>Semantics</th><th onclick=sortTable(1,"shapes")>M</th><th onclick=sortTable(2,"shapes")>N</th><th onclick=sortTable(3,"shapes")>K</th><th onclick=sortTable(4,"shapes")>Best Backend</th><th onclick=sortTable(5,"shapes")>E2E</th><th>vs Direct HIP</th></tr></thead><tbody>')
    for key in sorted(rows.keys()):
        entries = sorted(rows[key], key=lambda x: x[1])
        w = entries[0]
        sem,shp = key.split('|',1)
        parts = shp.split('x')
        direct = next((e for e in entries if e[0]=='hip-direct'), None)
        speedup = direct[1]/w[1] if direct and w[1]>0 else 0
        c = 'dir'
        if 'amdgpu' in w[0]: c='amd'
        elif 'rocwmma' in w[0]: c='roc'
        elif 'hipblaslt' in w[0]: c='hip'
        bar = f'<span class=speedup-bar style="width:{min(speedup*20,100)}px;background:{"#d2991d"if c=="dir"else"#58a6ff"if c=="amd"else"#f778ba"if c=="roc"else"#a371f7"}"></span>' if speedup>0 else ''
        html.append(f'<tr><td>{sem}</td><td>{parts[0]}</td><td>{parts[1] if len(parts)>1 else "?"}</td><td>{parts[2] if len(parts)>2 else "?"}</td><td class="win {c}">{w[0]}</td><td>{w[1]:.0f}</td><td>{bar}{speedup:.2f}x</td></tr>')
    html.append('</tbody></table></div>')

    # Phases tab
    html.append('<div class=tab-content id=tab-phases style=display:none>')
    for key in sorted(rows.keys()):
        vs = sorted(rows[key], key=lambda x: x[1])[:6]
        sem,shp = key.split('|',1)
        parts = shp.split('x')
        m = int(parts[0]) if parts and parts[0].isdigit() else 0
        if m < 128: continue
        html.append(f'<div style="margin:12px 0"><b>{key}</b>')
        for b,e2e,pk,gm,ex,_ in vs:
            pp = pk/e2e*100 if e2e>0 else 0
            gp = gm/e2e*100 if e2e>0 else 0
            ep = ex/e2e*100 if e2e>0 else 0
            op = max(0,100-pp-gp-ep)
            html.append(f'<div class=bar-row><span>{b}</span><span style=color:#8b949e;font-size:10px;min-width:55px>{e2e:.0f}us</span>')
            html.append(f'<div class=bar-bg>')
            if pp>0: html.append(f'<span class="bar-fill bp" style="width:{pp:.1f}%" title="Pack {pk:.0f}us"></span>')
            if gp>0: html.append(f'<span class="bar-fill bg" style="width:{gp:.1f}%" title="GEMM {gm:.0f}us"></span>')
            if ep>0: html.append(f'<span class="bar-fill be" style="width:{ep:.1f}%" title="Export {ex:.0f}us"></span>')
            if op>0: html.append(f'<span class="bar-fill bo" style="width:{op:.1f}%" title="Other {op:.0f}%"></span>')
            html.append('</div></div>')
        html.append('</div>')
    html.append('</div>')

    html.append(f'<script>{JS}</script></body></html>')
    out_path.write_text('\n'.join(html), encoding='utf-8')
    print(f'Dashboard: {out_path} ({out_path.stat().st_size} bytes, {len(captures)} captures)')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture-root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('docs/dashboard.html'))
    a = p.parse_args()
    build(a.capture_root, Path(__file__).resolve().parents[1] / a.out)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
