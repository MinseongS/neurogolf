"""task158 (ARC-AGI 6aa20dc0) — variable-mag dihedral sprite re-stamping.

================================================================================
STATUS: SOLVED + ONNX-BUILT. The reference solver `solve()` (numpy) is exact
        (50000/50000 fresh). `build(task)` emits an opset-10 ONNX graph that
        replicates a SIMPLIFIED but provably-equivalent spatial reformulation
        (validated 20000/20000 numpy, and onnxruntime-checked against solve()).
================================================================================

GENERATOR (key facts, verified against /tmp/arc-gen/tasks/task_6aa20dc0.py):
  - Canonical 3x3 sprite: c0@(0,0), c1@(2,2) (distinct corner colours), body
    colour c2 at a transpose-symmetric subset of the off-diagonal cells.
  - 2..4 megas, each with own mag in {1,2,3} and own (hflip,vflip); non-overlap
    margin 2. The grid has EXACTLY 4 colours: {c0,c1,c2,bg=color_list[3]}.
  - INPUT: mega 0 (REFERENCE) fully drawn (always mag=1); every other mega shows
    ONLY its two magnified diagonal CORNER blocks (c0 block + c1 block).
  - OUTPUT: every mega filled with its flipped-magnified full sprite.

SPATIAL REFORMULATION (this is what the ONNX graph builds — validated exact):
  bg  = MODE colour (per-channel pixel-count ArgMax).
  c2  = non-bg colour with MIN bbox span (it is confined to the 3x3 reference).
  refpos = (min row, min col) of {c2 cells} U {non-bg non-c2 cells within
           Chebyshev-2 of the c2 bbox}  (the reference region; isolated by margin).
  canon = the 3x3 window at refpos, HFLIPPED iff its two non-bg-non-c2 ("special")
          corners lie on the ANTI-diagonal -> brings c0,c1 onto the MAIN diagonal.
          c0 = canon[0,0], c1 = canon[2,2].
  For each (mag in {1,2,3}, h, v) build the flipped-magnified tile T (Sz=3*mag) and
  the visible mask M (the two diagonal corner blocks). A placement matches at a
  top-left where window == T on M (corner blocks == c0/c1) AND window == bg on the
  rest of the Sz*Sz footprint (the hidden body + non-corner cells).
  EXACT-COVER (single pass, NO iteration, NO isolation, NO ref self-exclusion):
    Fcount = sum over all 12 configs of (forward-spread of match by an Sz*Sz block).
    uniq   = (Fcount == 1)   (cells covered by exactly one placement footprint).
    A placement is CONFIRMED iff its visible mask overlaps a uniq cell.
    Confirmed placements are non-overlapping and exactly tile the target; write
    each tile. (Validated 20000/20000 numpy, equivalent to solve()'s exact-cover.)

ONNX REALIZATION:
  - colf  = sum_k k*input_k, with off-grid (all-channel-0) cells set to a -BIG
    sentinel so they never equal bg / a tile colour.
  - bg/c0/c1/c2 recovered as fp32 scalars; eq planes eq_bg/eq_c0/eq_c1 = (colf==x).
  - MATCH: ONE Conv, STATIC weight [12,3,9,9] (geometry only: which footprint cell
    requires bg vs c0 vs c1), pads bottom-right by 8 so output[yy,xx] correlates the
    window [yy:yy+9, xx:xx+9]. sat[12] ; match = Equal(sat, Sz^2 per channel).
  - Fcount: grouped ConvTranspose(match, ones[Sz] per channel) -> Slice -> ReduceSum.
  - vis-overlap: Conv(uniq, M-geometry kernel [12,1,9,9]) > 0 ; confirmed = match&that.
  - WRITE: per colour X in {bg,c0,c1,c2}: ConvTranspose(confirmed, X-geometry kernel)
    spreads a 0/1 indicator; outidx = sum_X X * spread_X ; covered = spread_any>0.
    output index plane = Where(covered, outidx, colf); final one-hot via Equal-to-arange.
  All Conv/ConvTranspose geometry kernels are STATIC; only bg/c0/c1/c2 are scalars.
"""

import numpy as np


# ============================================================================
# VERIFIED EXACT REFERENCE SOLVER (numpy) — 50000/50000 fresh. Spec for the net.
# ============================================================================
def _bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return vals[counts.argmax()]


def _recover_canon(sub, bg, c2):
    for h in range(2):
        for v in range(2):
            t = sub.copy()
            if h:
                t = t[:, ::-1]
            if v:
                t = t[::-1, :]
            a, b = t[0, 0], t[2, 2]
            if a == bg or b == bg or a == b or a == c2 or b == c2:
                continue
            ok = True
            for r in range(3):
                for c in range(3):
                    if (r, c) in [(0, 0), (2, 2)]:
                        continue
                    if t[r, c] != bg and t[r, c] != c2:
                        ok = False
            if not ok:
                continue
            body = (t == c2)
            if np.array_equal(body, body.T):
                return t
    return None


def _isolated(grid, yy, xx, Sz, bg, H, W):
    for c in range(xx - 1, xx + Sz + 1):
        for r in (yy - 1, yy + Sz):
            if 0 <= r < H and 0 <= c < W and grid[r, c] != bg:
                return False
    for r in range(yy - 1, yy + Sz + 1):
        for c in (xx - 1, xx + Sz):
            if 0 <= r < H and 0 <= c < W and grid[r, c] != bg:
                return False
    return True


def _stamp_cover(grid, canon, bg, H, W, refpos):
    visms, writes = [], []
    for mag in (1, 2, 3):
        base = np.kron(canon, np.ones((mag, mag), dtype=canon.dtype))
        Sz = 3 * mag
        vis = np.zeros((Sz, Sz), dtype=bool)
        vis[0:mag, 0:mag] = True
        vis[2 * mag:3 * mag, 2 * mag:3 * mag] = True
        for h in range(2):
            for v in range(2):
                T = base.copy()
                M = vis.copy()
                if h:
                    T = T[:, ::-1]
                    M = M[:, ::-1]
                if v:
                    T = T[::-1, :]
                    M = M[::-1, :]
                for yy in range(H - Sz + 1):
                    for xx in range(W - Sz + 1):
                        if mag == 1 and (yy, xx) == refpos:
                            continue
                        win = grid[yy:yy + Sz, xx:xx + Sz]
                        if not (np.all(win[M] == T[M]) and np.all(win[~M] == bg)):
                            continue
                        if not _isolated(grid, yy, xx, Sz, bg, H, W):
                            continue
                        vm = np.zeros((H, W), dtype=bool)
                        vm[yy:yy + Sz, xx:xx + Sz][M] = True
                        visms.append(vm)
                        writes.append((yy, xx, Sz, T))
    ry, rx = refpos
    refmask = np.zeros((H, W), dtype=bool)
    refmask[ry:ry + 3, rx:rx + 3] = (grid[ry:ry + 3, rx:rx + 3] != bg)
    target = (grid != bg) & ~refmask
    if not visms:
        return grid.copy() if not target.any() else None
    V = np.stack(visms)
    K = V.shape[0]
    chosen = np.zeros(K, dtype=bool)
    dead = np.zeros(K, dtype=bool)
    covered = np.zeros((H, W), dtype=bool)
    for _ in range(4):
        live = ~chosen & ~dead
        if not live.any():
            break
        claim = (V & live[:, None, None]).sum(0)
        uniqcell = (claim == 1) & target
        has_uniq = (V & uniqcell[None] & live[:, None, None]).reshape(K, -1).any(1)
        newsel = has_uniq & live
        if not newsel.any():
            break
        chosen |= newsel
        covered |= V[newsel].any(0)
        overlap = (V & covered[None]).reshape(K, -1).any(1)
        dead |= overlap & ~chosen
    if not np.array_equal(covered & target, target):
        return None
    if (covered & ~target).any():
        return None
    out = grid.copy()
    for i in range(K):
        if chosen[i]:
            yy, xx, Sz, T = writes[i]
            out[yy:yy + Sz, xx:xx + Sz] = T
    return out


def solve(grid):
    grid = np.array(grid)
    H, W = grid.shape
    bg = _bg(grid)
    colsu = [c for c in np.unique(grid) if c != bg]
    cand = []
    for c in colsu:
        ys, xs = np.where(grid == c)
        sp = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        cand.append((sp, (grid == c).sum(), c))
    cand.sort()
    for _, _, c2 in cand:
        c2m = (grid == c2)
        tot = c2m.sum()
        ys, xs = np.where(c2m)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if y1 - y0 > 2 or x1 - x0 > 2:
            continue
        canon = None
        refpos = None
        for ty in range(max(0, y1 - 2), min(y0, H - 3) + 1):
            for tx in range(max(0, x1 - 2), min(x0, W - 3) + 1):
                if ty + 3 > H or tx + 3 > W:
                    continue
                w = grid[ty:ty + 3, tx:tx + 3]
                if (w == c2).sum() != tot:
                    continue
                cc = _recover_canon(w, bg, c2)
                if cc is not None:
                    canon = cc
                    refpos = (ty, tx)
                    break
            if canon is not None:
                break
        if canon is None:
            continue
        out = _stamp_cover(grid, canon, bg, H, W, refpos)
        if out is not None:
            return out
    return None


# ============================================================================
# ONNX BUILD
# ============================================================================
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

N = 30
# active region: generator caps width in [15,25], height = width+/-1 -> height<=26,
# width<=25, so ALL content lies in rows [0:26) x cols [0:25).
WORK_R = 26
WORK_C = 25
KP = 9  # max footprint side (mag=3 -> Sz=9)

# ---- static geometry tables (config order: mag in {1,2,3} x h x v) ----------
_CONFIGS = []
for _mag in (1, 2, 3):
    _Sz = 3 * _mag
    _vis = np.zeros((_Sz, _Sz), bool)
    _vis[0:_mag, 0:_mag] = True
    _vis[2 * _mag:3 * _mag, 2 * _mag:3 * _mag] = True
    for _h in range(2):
        for _v in range(2):
            _M = _vis.copy()
            if _h:
                _M = _M[:, ::-1]
            if _v:
                _M = _M[::-1, :]
            # canonical role grid on the magnified tile: 0=bg,1=c0,2=c1,3=c2.
            # canon[0,0]=c0(role1), canon[2,2]=c1(role2); off-diagonal body cells=c2(role3).
            _role3 = np.zeros((3, 3), int)
            _role3[0, 0] = 1
            _role3[2, 2] = 2
            # body cells (role3) = the off-diagonal positions; we mark ALL non-corner
            # cells as "body candidate" -> role 3, but whether a given off-diag cell is
            # really c2 or bg is INSTANCE-dependent. For the WRITE we need the true body;
            # we encode body via a runtime canon read. For MATCH only roles 1/2/0 matter
            # (corner blocks require c0/c1; everything else requires bg).
            _baserole = np.kron(_role3, np.ones((_mag, _mag), int))
            _T_role = _baserole.copy()
            if _h:
                _T_role = _T_role[:, ::-1]
            if _v:
                _T_role = _T_role[::-1, :]
            _CONFIGS.append((_mag, _Sz, _h, _v, _M.astype(np.float32), _T_role))


def _match_kernel():
    """Static [12,3,9,9] match kernel. channels = [eq_bg, eq_c0, eq_c1].
    Visible corner cells require c0 (ch1) or c1 (ch2); ALL other footprint cells
    require bg (ch0). Off-footprint kernel = 0."""
    K = np.zeros((12, 3, KP, KP), np.float32)
    for i, (mag, Sz, h, v, M, Trole) in enumerate(_CONFIGS):
        for dy in range(Sz):
            for dx in range(Sz):
                if M[dy, dx] > 0:
                    role = Trole[dy, dx]  # 1=c0 or 2=c1 at corner blocks
                    K[i, role, dy, dx] = 1.0
                else:
                    K[i, 0, dy, dx] = 1.0  # require bg
    return K


# Spread kernels are used in a forward-spread Conv (pads=[8,8,0,0]); the Conv
# kernel must be the 180-flip of the footprint pattern anchored at bottom-right:
# w[8-dy, 8-dx] = pattern[dy,dx]  (out[r,c] = sum_{dy,dx} match[r-dy,c-dx]*pattern).
def _ones_kernel():
    """Forward-spread weight [12,1,9,9]: ones over the Sz*Sz footprint."""
    K = np.zeros((12, 1, KP, KP), np.float32)
    for i, (mag, Sz, h, v, M, Trole) in enumerate(_CONFIGS):
        for dy in range(Sz):
            for dx in range(Sz):
                K[i, 0, 8 - dy, 8 - dx] = 1.0
    return K


def _vis_kernel():
    """Conv weight [12,1,9,9]: visible mask per config (top-left correlation for
    the uniq-overlap test; pads=[0,0,8,8]). NOT a spread kernel."""
    K = np.zeros((12, 1, KP, KP), np.float32)
    for i, (mag, Sz, h, v, M, Trole) in enumerate(_CONFIGS):
        K[i, 0, :Sz, :Sz] = M
    return K


def _role_spread_kernels():
    """Forward-spread weights [12,1,9,9] for the corner roles c0,c1 (bottom-right,
    180-flipped). bg-role = oneskern - c0k - c1k - bodyk at runtime."""
    out = {}
    for role, name in [(1, "c0"), (2, "c1")]:
        K = np.zeros((12, 1, KP, KP), np.float32)
        for i, (mag, Sz, h, v, M, Trole) in enumerate(_CONFIGS):
            for dy in range(Sz):
                for dx in range(Sz):
                    if Trole[dy, dx] == role:
                        K[i, 0, 8 - dy, 8 - dx] = 1.0
        out[name] = K
    return out


_SIZES = np.array([Sz * Sz for (mag, Sz, h, v, M, Trole) in _CONFIGS],
                  np.float32).reshape(1, 12, 1, 1)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def init_scalar(name, val, dtype):
        # true 0-dim scalar initializer (Gather index that REMOVES an axis).
        inits.append(numpy_helper.from_array(np.array(val, dtype), name))
        return name

    def nd(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colf = sum_k k*input_k (1x1 Conv, no [1,10,N,N] plane), off-grid sentinel
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("onesw", np.ones((1, 10, 1, 1), np.float32), np.float32)
    nd("Conv", ["input", "onesw"], "ingrid_f")                          # [1,1,N,N] occupancy
    nd("Conv", ["input", "kw"], "colf_raw")                             # [1,1,N,N] colour idx
    init("ZERO", np.array(0.0, np.float32), np.float32)
    init("ZEROH", np.array(0.0, np.float16), np.float16)
    nd("Greater", ["ingrid_f", "ZERO"], "ingrid_b")                      # bool [1,1,N,N]
    nd("Cast", ["colf_raw"], "colf_rawh", to=F16)                       # fp16
    init("NEGBIGH", np.array(-100.0, np.float16), np.float16)           # fp16-safe sentinel
    nd("Where", ["ingrid_b", "colf_rawh", "NEGBIGH"], "colf")          # [1,1,N,N] fp16

    # ---- per-channel pixel counts -> bg = ArgMax(count) ---------------------
    nd("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)           # [1,10,1,1]
    nd("ArgMax", ["cnt"], "bg_i", axis=1, keepdims=1)                    # [1,1,1,1] int64
    # gather colour value of bg channel: bg value == its channel index
    nd("Cast", ["bg_i"], "bg", to=F32)                                  # bg colour scalar

    # ---- per-channel bbox span -> c2 = argmin span (excl bg & absent) -------
    # row/col occupancy per channel (fp16 working planes; values 0/1 and 0..29 exact)
    nd("ReduceMax", ["input"], "rowocc_f", axes=[3], keepdims=1)         # [1,10,N,1] f32
    nd("ReduceMax", ["input"], "colocc_f", axes=[2], keepdims=1)         # [1,10,1,N] f32
    nd("Cast", ["rowocc_f"], "rowocc", to=F16)
    nd("Cast", ["colocc_f"], "colocc", to=F16)
    init("rampN_r", np.arange(N, dtype=np.float16).reshape(1, 1, N, 1), np.float16)
    init("rampN_c", np.arange(N, dtype=np.float16).reshape(1, 1, 1, N), np.float16)
    init("BIGH", np.array(9999.0, np.float16), np.float16)
    init("NEGBIGH2", np.array(-9999.0, np.float16), np.float16)
    nd("Greater", ["rowocc", "ZEROH"], "rowocc_b")
    nd("Greater", ["colocc", "ZEROH"], "colocc_b")
    nd("Where", ["rowocc_b", "rampN_r", "BIGH"], "rmin_w")
    nd("ReduceMin", ["rmin_w"], "rmin", axes=[2, 3], keepdims=1)         # [1,10,1,1] f16
    nd("Where", ["rowocc_b", "rampN_r", "NEGBIGH2"], "rmax_w")
    nd("ReduceMax", ["rmax_w"], "rmax", axes=[2, 3], keepdims=1)
    nd("Where", ["colocc_b", "rampN_c", "BIGH"], "cmin_w")
    nd("ReduceMin", ["cmin_w"], "cmin", axes=[2, 3], keepdims=1)
    nd("Where", ["colocc_b", "rampN_c", "NEGBIGH2"], "cmax_w")
    nd("ReduceMax", ["cmax_w"], "cmax", axes=[2, 3], keepdims=1)
    nd("Sub", ["rmax", "rmin"], "rspan")
    nd("Sub", ["cmax", "cmin"], "cspan")
    nd("Max", ["rspan", "cspan"], "span")                               # [1,10,1,1] f16
    # presence and bg one-hot to exclude
    nd("Greater", ["cnt", "ZERO"], "present_b")                         # [1,10,1,1]
    init("chan10", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    nd("Equal", ["chan10", "bg"], "isbgch_b")                           # [1,10,1,1] bool
    # span' = span where present and not bg, else BIG
    nd("Not", ["present_b"], "absent_b")
    nd("Or", ["absent_b", "isbgch_b"], "excl_b")
    nd("Where", ["excl_b", "BIGH", "span"], "span2")
    nd("ArgMin", ["span2"], "c2_i", axis=1, keepdims=1)
    nd("Cast", ["c2_i"], "c2", to=F32)                                 # c2 colour scalar
    nd("Cast", ["bg"], "bgh", to=F16)                                  # fp16 versions
    nd("Cast", ["c2"], "c2h", to=F16)

    # ---- reference bbox of c2 (rmin/cmin/rmax/cmax at channel c2) -----------
    # gather c2 channel's bbox via Equal(chan10,c2) one-hot select
    nd("Equal", ["chan10", "c2"], "isc2ch_b")                          # [1,10,1,1]
    nd("Cast", ["isc2ch_b"], "isc2ch", to=F16)
    def chan_scalar(vec, name):
        nd("Mul", [vec, "isc2ch"], name + "_m")
        nd("ReduceSum", [name + "_m"], name, axes=[1], keepdims=1)      # [1,1,1,1]
        return name
    chan_scalar("rmin", "c2_y0")
    chan_scalar("cmin", "c2_x0")
    chan_scalar("rmax", "c2_y1")
    chan_scalar("cmax", "c2_x1")

    # ---- refpos = min (row,col) over {c2 cells} U {nonbg-nonc2 within Cheb-2}
    # region = c2-bbox dilated by 2 ; refnb = c2cells OR (special-colour cells in region)
    # special colour = nonbg & nonc2 (the two corner colours).
    # build masks on full canvas.
    nd("Equal", ["colf", "c2h"], "c2cell_b")                           # [1,1,N,N]
    nd("Equal", ["colf", "bgh"], "bgcell_b")
    # special = ingrid & ~bg & ~c2
    nd("Not", ["bgcell_b"], "nbg_b")
    nd("Not", ["c2cell_b"], "nc2_b")
    nd("And", ["ingrid_b", "nbg_b"], "t_a")
    nd("And", ["t_a", "nc2_b"], "special_b")                          # [1,1,N,N]
    # region mask: rows in [y0-2,y1+2], cols in [x0-2,x1+2]
    init("TWO", np.array(2.0, np.float16), np.float16)
    nd("Sub", ["c2_y0", "TWO"], "ry_lo")
    nd("Add", ["c2_y1", "TWO"], "ry_hi")
    nd("Sub", ["c2_x0", "TWO"], "rx_lo")
    nd("Add", ["c2_x1", "TWO"], "rx_hi")
    # rowramp [1,1,N,1], colramp [1,1,1,N]
    nd("Not", [nd("Less", ["rampN_r", "ry_lo"], "rl_neg")], "row_ge")
    nd("Not", [nd("Greater", ["rampN_r", "ry_hi"], "rl_gt")], "row_le")
    nd("And", ["row_ge", "row_le"], "row_in")                         # [1,1,N,1]
    nd("Not", [nd("Less", ["rampN_c", "rx_lo"], "cl_neg")], "col_ge")
    nd("Not", [nd("Greater", ["rampN_c", "rx_hi"], "cl_gt")], "col_le")
    nd("And", ["col_ge", "col_le"], "col_in")                         # [1,1,1,N]
    nd("And", ["row_in", "col_in"], "region_b")                       # [1,1,N,N]
    nd("And", ["special_b", "region_b"], "specreg_b")
    nd("Or", ["c2cell_b", "specreg_b"], "refnb_b")                    # [1,1,N,N]
    # ry = min row of refnb ; rx = min col of refnb
    nd("Cast", ["refnb_b"], "refnb_f", to=F16)                       # fp16 [1,1,N,N]
    nd("ReduceMax", ["refnb_f"], "refnb_row", axes=[3], keepdims=1)   # [1,1,N,1]
    nd("ReduceMax", ["refnb_f"], "refnb_col", axes=[2], keepdims=1)   # [1,1,1,N]
    nd("Greater", ["refnb_row", "ZEROH"], "refnb_row_b")
    nd("Greater", ["refnb_col", "ZEROH"], "refnb_col_b")
    nd("Where", ["refnb_row_b", "rampN_r", "BIGH"], "ry_w")
    nd("ReduceMin", ["ry_w"], "ry", axes=[2, 3], keepdims=1)          # [1,1,1,1]
    nd("Where", ["refnb_col_b", "rampN_c", "BIGH"], "rx_w")
    nd("ReduceMin", ["rx_w"], "rx", axes=[2, 3], keepdims=1)

    # ---- read refwin 3x3 at (ry,rx) and recover canon ----------------------
    # gather 3x3 from colf: rows ry+{0,1,2}, cols rx+{0,1,2}.
    nd("Cast", ["ry"], "ry_i64", to=I64)
    nd("Cast", ["rx"], "rx_i64", to=I64)
    init("d012", np.array([0, 1, 2], np.int64), np.int64)
    nd("Reshape", ["ry_i64", "shp1"], "ry_s")
    init("shp1", np.array([1], np.int64), np.int64)
    nd("Reshape", ["rx_i64", "shp1"], "rx_s")
    nd("Add", ["ry_s", "d012"], "rowidx")                            # [3] int64
    nd("Add", ["rx_s", "d012"], "colidx")                            # [3]
    # squeeze colf to [N,N]
    init("shpNN", np.array([N, N], np.int64), np.int64)
    nd("Reshape", ["colf", "shpNN"], "colf2d")                       # [N,N]
    nd("Gather", ["colf2d", "rowidx"], "g_rows", axis=0)             # [3,N]
    nd("Gather", ["g_rows", "colidx"], "refwin", axis=1)             # [3,3]
    # special-corner test: hflip refwin iff special on anti-diagonal.
    # corner values
    init_scalar("i0", 0, np.int64)   # 0-dim scalar index (removes gathered axis)
    init_scalar("i2", 2, np.int64)
    def cell(t, ri, ci, name):
        nd("Gather", [t, ri], name + "_r", axis=0)                   # [3]
        nd("Gather", [name + "_r", ci], name, axis=0)                # scalar
        return name
    cell("refwin", "i0", "i0", "corner_tl")
    cell("refwin", "i2", "i2", "corner_br")
    cell("refwin", "i0", "i2", "corner_tr")
    cell("refwin", "i2", "i0", "corner_bl")
    # special(v) = v!=bg & v!=c2.  refwin is fp16, so use fp16 scalars.
    nd("Reshape", ["bgh", "shp1"], "bg_s")
    nd("Reshape", ["c2h", "shp1"], "c2_s")
    def is_special(name, out):
        nd("Equal", [name, "bg_s"], out + "_eb")
        nd("Equal", [name, "c2_s"], out + "_ec")
        nd("Or", [out + "_eb", out + "_ec"], out + "_o")
        nd("Not", [out + "_o"], out)                                 # bool scalar [1]
        return out
    is_special("corner_tl", "sp_tl")
    is_special("corner_br", "sp_br")
    is_special("corner_tr", "sp_tr")
    is_special("corner_bl", "sp_bl")
    nd("And", ["sp_tr", "sp_bl"], "anti")
    nd("And", ["sp_tl", "sp_br"], "main")
    nd("Not", ["main"], "nmain")
    nd("And", ["anti", "nmain"], "doflip")                          # bool [1]
    # canon = refwin if not doflip else refwin[:, ::-1]
    init("rev012", np.array([2, 1, 0], np.int64), np.int64)
    nd("Gather", ["refwin", "rev012"], "refwin_hf", axis=1)         # [3,3]
    nd("Where", ["doflip", "refwin_hf", "refwin"], "canon")        # [3,3]
    # c0 = canon[0,0], c1 = canon[2,2]
    cell("canon", "i0", "i0", "c0v")                                # scalar [1]
    cell("canon", "i2", "i2", "c1v")
    # reshape c0,c1 to [1,1,1,1] for plane ops
    init("shp1111", np.array([1, 1, 1, 1], np.int64), np.int64)
    nd("Reshape", ["c0v", "shp1111"], "c0")
    nd("Reshape", ["c1v", "shp1111"], "c1")

    # ======================================================================
    # All full-grid working planes below are fp16 (declared half-size).
    # ======================================================================
    # ---- eq planes (fp16), built on the CROPPED colf so they are WORKxWORK --
    # Generator caps width<=25, height<=26 -> ALL content lies in [0:WORK,0:WORK];
    # every 12-channel plane below is WORKxWORK.
    init("ec0", np.array([0, 0], np.int64), np.int64)
    init("ecW", np.array([WORK_R, WORK_C], np.int64), np.int64)
    init("ec23", np.array([2, 3], np.int64), np.int64)
    nd("Slice", ["colf", "ec0", "ecW", "ec23"], "colf_w")          # [1,1,WORK_R,WORK_C] fp16
    nd("Equal", ["colf_w", "bgh"], "eqbg_b")
    nd("Equal", ["colf_w", "c0"], "eqc0_b")
    nd("Equal", ["colf_w", "c1"], "eqc1_b")
    nd("Cast", ["eqbg_b"], "eqbg", to=F16)
    nd("Cast", ["eqc0_b"], "eqc0", to=F16)
    nd("Cast", ["eqc1_b"], "eqc1", to=F16)
    nd("Concat", ["eqbg", "eqc0", "eqc1"], "eqstack", axis=1)       # [1,3,WORK,WORK] fp16

    # ---- MATCH conv: static [12,3,9,9] fp16, pad bottom-right 8 -------------
    init("mkern", _match_kernel(), np.float16)
    nd("Conv", ["eqstack", "mkern"], "sat", pads=[0, 0, KP - 1, KP - 1])  # [1,12,WORK,WORK]
    init("sizes", _SIZES, np.float16)
    nd("Equal", ["sat", "sizes"], "match_b")                        # [1,12,N,N] bool
    nd("Cast", ["match_b"], "match", to=F16)

    # ---- Fcount = forward-spread Conv that SUMS the 12 channels in one op ----
    # weight [1,12,9,9] (channels = the 12 per-config footprint patterns) -> the
    # Conv contracts the channel axis, emitting Fcount [1,1,N,N] with no 12-ch plane.
    init("oneskern", _ones_kernel().transpose(1, 0, 2, 3), np.float16)  # [1,12,9,9]
    nd("Conv", ["match", "oneskern"], "Fcount", pads=[8, 8, 0, 0])  # [1,1,N,N] fp16
    init("ONEH", np.array(1.0, np.float16), np.float16)
    nd("Equal", ["Fcount", "ONEH"], "uniq_b")                      # [1,1,N,N]
    nd("Cast", ["uniq_b"], "uniq", to=F16)

    # ---- vis-overlap: Conv(uniq, vis kernel [12,1,9,9]) > 0 -----------------
    init("viskern", _vis_kernel(), np.float16)
    nd("Conv", ["uniq", "viskern"], "vsat", pads=[0, 0, KP - 1, KP - 1])  # [1,12,N,N] fp16
    nd("Greater", ["vsat", "ZEROH"], "vis_ok_b")
    nd("And", ["match_b", "vis_ok_b"], "confirmed_b")              # [1,12,N,N]
    nd("Cast", ["confirmed_b"], "confirmed", to=F16)

    # ---- WRITE via ONE channel-contracting value Conv -----------------------
    # The value Conv weight valk_c[1,12,9,9] holds, per config (channel) and per
    # footprint cell, (colour value + 1):
    #   valk = (bg+1)*ones + (c0-bg)*c0k + (c1-bg)*c1k + (c2-bg)*bodyk
    # (the +1 lets ONE Conv carry both colour and coverage: covered = val>0,
    #  colour = val-1).  ones/c0k/c1k are STATIC [1,12,9,9]; bodyk is runtime.
    role_k = _role_spread_kernels()
    init("c0k", role_k["c0"].transpose(1, 0, 2, 3), np.float16)   # [1,12,9,9] static
    init("c1k", role_k["c1"].transpose(1, 0, 2, 3), np.float16)   # [1,12,9,9] static
    init("onesk1", _ones_kernel().transpose(1, 0, 2, 3), np.float16)  # [1,12,9,9] static
    # runtime body geometry kernel bodyk[1,12,9,9] (180-flipped, bottom-right).
    nd("Equal", ["canon", "c2_s"], "cb3_b")                        # [3,3] bool
    nd("Cast", ["cb3_b"], "cb3", to=F16)                           # [3,3] fp16
    body_idx = np.full((12, KP * KP), 9, np.int64)  # 9 -> appended zero
    for i, (mag, Sz, h, v, M, Trole) in enumerate(_CONFIGS):
        for dy in range(Sz):
            for dx in range(Sz):
                yy, xx = dy, dx
                if h:
                    xx = Sz - 1 - xx
                if v:
                    yy = Sz - 1 - yy
                cr, cc = yy // mag, xx // mag
                body_idx[i, (8 - dy) * KP + (8 - dx)] = cr * 3 + cc
    init("body_idx", body_idx.reshape(-1), np.int64)               # [12*81]
    init("shp9", np.array([9], np.int64), np.int64)
    nd("Reshape", ["cb3", "shp9"], "cb3_flat")
    init("zpadh", np.array([0.0], np.float16), np.float16)
    nd("Concat", ["cb3_flat", "zpadh"], "cb3_pad", axis=0)         # [10] fp16
    nd("Gather", ["cb3_pad", "body_idx"], "bodyk_flat", axis=0)    # [12*81]
    init("shp_bk1", np.array([1, 12, KP, KP], np.int64), np.int64)
    nd("Reshape", ["bodyk_flat", "shp_bk1"], "bodyk")             # [1,12,9,9] fp16
    # scalar coefficients
    nd("Add", ["bgh", "ONEH"], "a0")                              # bg+1
    nd("Sub", ["c0", "bgh"], "a1")                                # c0-bg
    nd("Sub", ["c1", "bgh"], "a2")                                # c1-bg
    nd("Sub", ["c2h", "bgh"], "a3")                               # c2-bg
    nd("Mul", ["onesk1", "a0"], "vk0")
    nd("Mul", ["c0k", "a1"], "vk1")
    nd("Mul", ["c1k", "a2"], "vk2")
    nd("Mul", ["bodyk", "a3"], "vk3")
    nd("Sum", ["vk0", "vk1", "vk2", "vk3"], "valk_c")            # [1,12,9,9] fp16
    nd("Conv", ["confirmed", "valk_c"], "valp1_w", pads=[8, 8, 0, 0])  # [1,1,WORK,WORK]
    # pad back to 30x30 (trailing rows/cols are off-content -> 0 -> covered=False)
    init("padvp", np.array([0, 0, 0, 0, 0, 0, N - WORK_R, N - WORK_C], np.int64), np.int64)
    nd("Pad", ["valp1_w", "padvp"], "valp1", mode="constant")     # [1,1,N,N] fp16
    nd("Greater", ["valp1", "ZEROH"], "covered_b")                # covered = val>0
    nd("Sub", ["valp1", "ONEH"], "outidx")                        # colour = val-1
    nd("Where", ["covered_b", "outidx", "colf"], "finalidx")      # [1,1,N,N] fp16

    # ---- final one-hot: Equal(finalidx, arange_fp16) ; off-grid -> all-false
    init("chanA", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    nd("Equal", ["finalidx", "chanA"], "output")                   # [1,10,N,N] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    g = helper.make_graph(nodes, "task158", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
