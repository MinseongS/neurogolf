"""task219 v4: exact generator-parse enumeration (a,bw,col in {1,2} space)."""
import sys, importlib
import numpy as np
sys.path.insert(0, 'arc-gen')


def runs_of(rowocc, H):
    runs = []
    r = 0
    while r < H:
        if rowocc[r]:
            r2 = r
            while r2 + 1 < H and rowocc[r2 + 1]:
                r2 += 1
            runs.append((r, r2))
            r = r2 + 1
        else:
            r += 1
    return runs


def seg_options(runs, tall_est):
    opts = [runs]
    if tall_est >= 3:
        merged = [list(runs[0])]
        for a, b in runs[1:]:
            if a - merged[-1][1] == 2 and (b - merged[-1][0] + 1) <= tall_est:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        m = [tuple(x) for x in merged]
        if m != runs:
            opts.append(m)
    return opts


def sig_consistent(bands, occ):
    if len(bands) < 3:
        return True
    masks = []
    for a, b in bands[1:]:
        masks.append(tuple(occ[a:b + 1, :].any(1).tolist()))
    return len(set(masks)) == 1


def periodic_ok(band0, s, cw, W):
    if s + cw > W:
        return False
    pat = band0[:, s:s + cw]
    return all(np.array_equal(band0[:, c], pat[:, (c - s) % cw]) for c in range(s, W))


def a_tiling_ok(band0, col0, a):
    # A occupies [0, col0) tiled with period a: column c ≡ c+a for c+a < col0
    for c in range(0, col0 - a):
        if not np.array_equal(band0[:, c], band0[:, c + a]):
            return False
    return True


def solve_bands(g, occ, bands, H, W):
    b0 = bands[0]
    tall = b0[1] - b0[0] + 1
    lowers = bands[1:]
    if not lowers:
        return None
    band0 = occ[b0[0]:b0[0] + tall, :]
    infos = []  # (a_k(row), s_k, lower_full builder per d)
    for a, b in lowers:
        sub = occ[a:b + 1, :]
        cols = np.where(sub.any(0))[0]
        if not len(cols):
            continue
        infos.append((a, b, sub, cols.max() + 1))
    if not infos:
        return None
    # global parse: d (row offset of lower visible top within band), sigma, cw
    votes = {}
    for d in range(0, tall):
        ok_d = True
        parse_sets = []
        for (ra, rb, sub, s_k) in infos:
            h = sub.shape[0]
            if d + h > tall:
                ok_d = False
                break
            full = np.zeros((tall, W), bool)
            full[d:d + h, :] = sub
            local = set()
            for bw in (1, 2):
                col_k = s_k - bw
                if col_k < 1:
                    continue
                for aw in (1, 2):
                    if col_k not in (aw, 2 * aw):
                        continue
                    for col_0 in (aw, 2 * aw):
                        sigma = col_0 + bw
                        if sigma >= W:
                            continue
                        m = min(col_0, col_k)
                        if not np.array_equal(band0[:, :m], full[:, :m]):
                            continue
                        if not np.array_equal(band0[:, sigma - bw:sigma],
                                              full[:, s_k - bw:s_k]):
                            continue
                        if not a_tiling_ok(band0, col_0, aw):
                            continue
                        for cw in (1, 2):
                            if periodic_ok(band0, sigma, cw, W):
                                local.add((sigma, cw))
                                break
            if not local:
                ok_d = False
                break
            parse_sets.append(local)
        if not ok_d:
            continue
        common = set.intersection(*parse_sets) if parse_sets else set()
        for key in common:
            votes[(d,) + key] = votes.get((d,) + key, 0) + 1
    if not votes:
        return None
    # render all winning parses; if fills identical → deterministic
    outs = []
    for (d, sigma, cw) in sorted(votes):
        pat = band0[:, sigma:sigma + cw]
        out = g.copy()
        for (ra, rb, sub, s_k) in infos:
            top = ra - d
            for rr in range(tall):
                r = top + rr
                if r < 0 or r >= H:
                    continue
                for c in range(s_k, W):
                    if pat[rr, (c - s_k) % cw] and out[r, c] == 0:
                        out[r, c] = 1
        outs.append(out)
    # prefer parse with smallest sigma... but if fills differ it's ambiguity; majority impossible → first
    uniq = {o.tobytes(): o for o in outs}
    return outs[0] if outs else None


def predict(gi):
    g = np.array(gi)
    H, W = g.shape
    occ = g != 0
    runs = runs_of(occ.any(1), H)
    if not runs:
        return g.copy()
    tall_est = max(b - a + 1 for a, b in runs)
    for bands in seg_options(runs, tall_est):
        if not sig_consistent(bands, occ):
            continue
        o = solve_bands(g, occ, bands, H, W)
        if o is not None:
            return o
    return g.copy()


def main(n):
    gen = importlib.import_module('tasks.task_90f3ed37')
    runs = fails = 0
    ex_fails = []
    for _ in range(n):
        try:
            ex = gen.generate()
        except Exception:
            continue
        gi = np.array(ex['input']); go = np.array(ex['output'])
        if max(gi.shape) > 30 or max(go.shape) > 30:
            continue
        runs += 1
        pred = predict(gi)
        if not np.array_equal(pred, go):
            fails += 1
            if len(ex_fails) < 2:
                ex_fails.append((gi, go, pred))
    print(f'v4: runs={runs} fails={fails} rate={fails/runs:.4%}')
    for gi, go, pr in ex_fails:
        print('IN:'); print(gi); print('EXP:'); print(go); print('GOT:'); print(pr); print('---')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)

# ============================================================================
# FINAL VALIDATED VERSION (v6b) — 2026-07-11 diagnosis session
# The predict() above is v4; the validated v6b adds, inside the parse loop:
#   (1) A max-col occupancy:  full[:, aw-1].any()          [awide = max(acols)+1]
#   (2) B max-col occupancy:  band0[:, sigma-1].any()      [bwide = max(bcols)+1]
#   (3) C max-col occupancy:  pat[:, cw-1].any()           [cwide = max(ccols)+1]
#   (4) seg_options: always offer gap-1 pair-merge (combined span <= 3), pick
#       segmentation by max parse votes (all lower bands share one row-mask)
#   (5) tie-break: (-votes, -sigma, d)  [prefer larger sigma]
# Measured: bundled 265/265; fresh 20k fail 62 (0.31%) vs deployed net 43.9%.
# See state/tasks/task219.md 2026-07-11 entry for the full derivation.
# ============================================================================
