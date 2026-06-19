"""Numpy mirror of the planned ONNX graph for task158 — uses only ops
expressible in opset-10 ONNX. Validate this 500/500, then translate to ONNX.

Input: one-hot [10,30,30] (we drop the batch dim here for clarity).
Output: one-hot [10,30,30] bool.
"""
import numpy as np

N = 30
CH = np.arange(10)


def mirror(inp):
    # inp: [10,30,30] float {0,1}
    # ---- occupancy / in-grid ------------------------------------------------
    ingrid = inp.sum(0) > 0  # [30,30] bool : exactly-one-hot in grid, zero off

    # colf = sum_k k*inp_k  -> colour index where in-grid; 0 where off-grid OR colour0
    colf = (CH[:, None, None] * inp).sum(0)  # [30,30] float

    # ---- bg = mode colour ---------------------------------------------------
    cnt = inp.sum((1, 2))  # [10] per-channel pixel count
    bg = int(np.argmax(cnt))  # scalar bg channel index

    # bg presence plane (in-grid bg cells): inp[bg]==1
    isbg = inp[bg] > 0  # [30,30] bool
    nonbg = ingrid & ~isbg  # in-grid, non-background

    # ---- c2 = non-bg colour with min bbox span -----------------------------
    # per-channel row/col occupancy
    rowocc = inp.max(2) > 0  # [10,30] bool : channel k present in row r
    colocc = inp.max(1) > 0  # [10,30] bool
    ramp = np.arange(N)
    BIG = 1e6
    rmin = np.where(rowocc, ramp[None, :], BIG).min(1)  # [10]
    rmax = np.where(rowocc, ramp[None, :], -BIG).max(1)
    cmin = np.where(colocc, ramp[None, :], BIG).min(1)
    cmax = np.where(colocc, ramp[None, :], -BIG).max(1)
    rspan = rmax - rmin
    cspan = cmax - cmin
    span = np.maximum(rspan, cspan) + 1  # [10]
    present = cnt > 0  # [10]
    # exclude bg channel and absent channels: set span huge
    spanmask = span.copy().astype(float)
    spanmask[bg] = BIG
    spanmask[~present] = BIG
    c2 = int(np.argmin(spanmask))  # min-span non-bg colour

    # ref bbox from c2 occupancy
    ry0 = int(rmin[c2]); rx0 = int(cmin[c2])
    # The reference 3x3 top-left: c2 cells span <=2; the 3x3 window top-left is
    # somewhere in [ry1-2, ry0] x [rx1-2, rx0]. In the canonical sprite c2 never
    # touches row0/col... actually body occupies rows1-2,cols0-1 region (mirrored).
    # We will locate ref via the corner colours instead in ONNX. For the mirror,
    # replicate solve(): try candidate windows.
    return colf, bg, c2, ry0, rx0, ingrid, isbg


# quick self-test against solve
if __name__ == "__main__":
    import sys
    from src.custom.task158 import solve, _bg
    sys.path.insert(0, "/tmp/arc-gen")
    import importlib.util
    spec = importlib.util.spec_from_file_location("g", "/tmp/arc-gen/tasks/task_6aa20dc0.py")
    gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
    okbg = okc2 = run = 0
    for i in range(500):
        ex = gen.generate(); g = np.array(ex["input"])
        H, W = g.shape
        inp = np.zeros((10, 30, 30), np.float32)
        for r in range(H):
            for c in range(W):
                inp[g[r, c], r, c] = 1
        colf, bg, c2, ry0, rx0, ingrid, isbg = mirror(inp)
        run += 1
        # compare bg
        rbg = _bg(g)
        okbg += int(bg == rbg)
        # compare c2 via solve internals
        colsu = [c for c in np.unique(g) if c != rbg]
        cand = []
        for c in colsu:
            ys, xs = np.where(g == c)
            sp = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
            cand.append((sp, (g == c).sum(), c))
        cand.sort()
        rc2 = cand[0][2]
        okc2 += int(c2 == rc2)
    print(f"bg {okbg}/{run}  c2 {okc2}/{run}")
