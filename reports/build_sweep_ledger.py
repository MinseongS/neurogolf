"""Build a 1->400 sweep ledger: per-task current score, generator signature,
feasibility heuristic, and merged known verdicts. Output reports/sweep_ledger.{json,md}."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def R(p): return os.path.join(ROOT, p)

manifest = json.load(open(R('reports/manifest.json')))['tasks']
amap = json.load(open(R('reports/arc_mapping.json')))

# Known-INFEASIBLE / BAIL (from memory, authoritative — never retry)
BAIL = set(map(str, [
    187,251,286,338, 96,319, 358, 255, 219,  # structural infeasible classes
    66,77,25,23,157,243,162,173,54,112,71,364,5,90,280,145,193,192,69,76,379,9,89,4,
    44,118,2,387,17,383,359,368,154,148,390,  # triaged BAIL
    110,101,133,216,392,232,                  # earlier-confirmed infeasible
]))
# Known FEASIBLE-unbuilt queues (from memory/reports)
FEAS_QUEUE = set(map(str, [70,175,198,131,228,324,333,86,51,397,20,34,161,224,306,55,
                           92,110,222,165,265]))

# Generator keyword heuristics
BAIL_KW = ['continuous_creature','num_sprites','overlaps','flood','connect','mega',
           'random_pixels','scatter','for rotate','correspond']
CLEAN_KW = ['hmirror','vmirror','rot90','rot180','transpose','tile','periodic',
            'recolor','mirror','dmirror','cmirror']

def gen_src(task):
    arc_id = amap[str(task)]['arc_id']
    p = f'/tmp/arc-gen/tasks/task_{arc_id}.py'
    try: return open(p).read()
    except Exception: return ''

def classify(task):
    if str(task) in BAIL: return 'BAIL', 'memory'
    src = gen_src(task).lower()
    if not src: return 'CHECK', 'no-src'
    for kw in BAIL_KW:
        if kw in src: return 'BAIL?', kw
    hits = [kw for kw in CLEAN_KW if kw in src]
    if hits: return 'FEASIBLE?', ','.join(hits[:2])
    return 'CHECK', '-'

rows = []
for n in range(1, 401):
    info = manifest[str(n)]
    cls, why = classify(n)
    if str(n) in FEAS_QUEUE and cls in ('CHECK','FEASIBLE?'):
        cls = 'FEASIBLE'  # promote known-queue
    rows.append({
        'task': n, 'arc_id': amap[str(n)]['arc_id'],
        'points': round(info['points'], 2), 'memory': info['memory'],
        'params': info['params'], 'method': info.get('method', '?'),
        'class': cls, 'sig': why,
        'status': 'pending',  # pending|done|skip
        'verdict': '',        # filled as we sweep
    })

json.dump(rows, open(R('reports/sweep_ledger.json'), 'w'), indent=1)

# readable md
with open(R('reports/sweep_ledger.md'), 'w') as f:
    f.write('# Sweep ledger 1->400 (targeted one-by-one)\n\n')
    f.write('class: FEASIBLE(build) / FEASIBLE?(verify) / CHECK(triage) / BAIL?(likely skip) / BAIL(skip)\n\n')
    f.write('| # | arc_id | pts | mem | method | class | sig | status | verdict |\n')
    f.write('|---|---|---|---|---|---|---|---|---|\n')
    for r in rows:
        f.write(f"| {r['task']} | {r['arc_id']} | {r['points']} | {r['memory']} | "
                f"{r['method']} | {r['class']} | {r['sig']} | {r['status']} | {r['verdict']} |\n")

# summary
import collections
c = collections.Counter(r['class'] for r in rows)
print('total', len(rows), 'stored', round(sum(r['points'] for r in rows), 2))
print('class counts:', dict(c))
# headroom targets: pts<16 and not BAIL
tgt = [r for r in rows if r['points'] < 16 and not r['class'].startswith('BAIL')]
print('pts<16 & not-bail:', len(tgt))
