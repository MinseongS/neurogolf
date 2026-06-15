"""Refine triage for sub-16 non-bail tasks by grepping generator structure."""
import json, os
rows = json.load(open('reports/sweep_ledger.json'))
amap = json.load(open('reports/arc_mapping.json'))

# structural markers -> hint
HARD = {
    'connect': 'connect/line', 'flood': 'flood', 'while ': 'iterate',
    'num_sprites': 'sprites', 'continuous': 'creature', 'sample(': 'random-pick',
    'random_pixels': 'noise', 'overlaps': 'overlap', 'count': 'counting',
    'sorted': 'sort', 'matches_up': 'match', 'bitmap': 'bitmap-grow',
}
EASY = {
    'hmirror': 'mirror', 'vmirror': 'mirror', 'rot90': 'rot', 'transpose': 'transp',
    'tile': 'tile', 'recolor': 'recolor', 'mirror': 'mirror',
}

def src(t):
    try: return open(f"/tmp/arc-gen/tasks/task_{amap[str(t)]['arc_id']}.py").read().lower()
    except Exception: return ''

cand = [r for r in rows if r['points'] < 16 and not r['class'].startswith('BAIL')]
cand.sort(key=lambda r: r['task'])
for r in cand:
    s = src(r['task'])
    hard = [v for k, v in HARD.items() if k in s]
    easy = [v for k, v in EASY.items() if k in s]
    tag = 'EASY:' + ','.join(sorted(set(easy))) if easy else ''
    if hard: tag += ' HARD:' + ','.join(sorted(set(hard)))
    print(f"{r['task']:3d} {r['points']:5.2f} {r['method'][:18]:18s} | {tag}")
print('total', len(cand))
