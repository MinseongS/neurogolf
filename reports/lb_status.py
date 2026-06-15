"""LB status tracker — surfaces the stored-vs-LB gap every time.

stored  = sum of manifest points (local, OPTIMISTIC: counts non-generalizing/overcounted base nets).
LB      = last CONFIRMED Kaggle publicScore (reports/lb_anchor.json).
gap     = stored_at_last_submit - LB  (the structural base-net overcount; ~stable since our floor-break
          wins translate ~1:1, confirmed 2026-06-15).
proj_LB = current_stored - gap  (what the LB should read now, assuming new wins translate 1:1).

Attribution (rough, from genverify.json n=40 batch — false-negative prone per memory, isolated n=200 is
truth): per task overcount ≈ stored_pts * (1 - ok/run). Sum ≈ where the gap lives. Replacing a flagged
task with a generalizing net closes that slice of the gap (and then proj_LB UNDER-estimates until re-anchored).

Usage: `PYTHONPATH=. .venv/bin/python reports/lb_status.py`  -> prints summary, writes reports/lb_status.md
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def R(p): return os.path.join(ROOT, p)

man = json.load(open(R('reports/manifest.json')))['tasks']
anchor = json.load(open(R('reports/lb_anchor.json')))
stored = sum(v['points'] for v in man.values())
gap = anchor['stored_at_submit'] - anchor['lb']
proj_lb = stored - gap
new_since_anchor = stored - anchor['stored_at_submit']

# attribution from genverify batch (if present)
attrib = []
gv_path = R('reports/genverify.json')
gv_sum = 0.0
if os.path.exists(gv_path):
    gv = json.load(open(gv_path))
    for k, v in man.items():
        rec = gv.get(str(k))
        if not rec or not rec.get('run'):
            continue
        rate = rec['ok'] / rec['run']
        if rate < 0.999:
            over = v['points'] * (1 - rate)
            gv_sum += over
            attrib.append((int(k), v['points'], rate, over, v.get('method', '?')))
    attrib.sort(key=lambda t: -t[3])

lines = []
lines.append('# LB status (stored ↔ real LB gap tracker)\n')
lines.append(f"- **stored (local, optimistic):** {stored:.2f}")
lines.append(f"- **last confirmed LB:** {anchor['lb']:.2f}  (stored {anchor['stored_at_submit']:.2f} @ {anchor['time_utc']} UTC)")
lines.append(f"- **structural gap (stored−LB at anchor):** {gap:.2f}  ← base-net overcount, ~stable")
lines.append(f"- **PROJECTED current LB:** {proj_lb:.2f}  (= stored − gap; +{new_since_anchor:.2f} of un-submitted wins since anchor)")
lines.append(f"- next submit at +5 adopted wins re-anchors this.\n")
lines.append(f"## Gap attribution (genverify n=40 batch — rough, false-neg prone; isolated n=200 is truth)")
lines.append(f"Estimated overcount across {len(attrib)} sub-100% tasks ≈ **{gv_sum:.1f}** pts (cf. gap {gap:.2f}).")
lines.append(f"Top offenders (replacing these with generalizing nets closes the gap directly):\n")
lines.append('| task | stored | fresh rate | est. overcount | method |')
lines.append('|---|---|---|---|---|')
for t, pts, rate, over, meth in attrib[:20]:
    lines.append(f"| {t} | {pts:.2f} | {rate:.2f} | {over:.2f} | {meth} |")
open(R('reports/lb_status.md'), 'w').write('\n'.join(lines) + '\n')

print(f"stored {stored:.2f} | LB(confirmed) {anchor['lb']:.2f} | gap {gap:.2f} | PROJ LB {proj_lb:.2f} "
      f"(+{new_since_anchor:.2f} unsubmitted)")
print(f"gap attribution: ~{gv_sum:.1f} pts across {len(attrib)} sub-100% tasks (top: "
      f"{', '.join(str(t[0]) for t in attrib[:8])})")
print('-> reports/lb_status.md')
