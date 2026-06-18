"""task017 (ARC-AGI 0dfd9992) — fill black cutouts in a doubly-periodic pattern.

Rule (from the ARC-GEN generator, verified fresh):
  A `size`=21 grid is filled with a DOUBLY-PERIODIC pattern of period `length`
  (4..9) in BOTH axes:
      cell(r, c) = ((rr*rr + cc*cc) % mod) + 1,
      rr = (offset+r) % length - length//2,  cc = (offset+c) % length - length//2
  rr depends only on r, cc only on c, so the value depends ONLY on
  (r % length, c % length): the pattern repeats with period `length` in r and c
  (values 1..mod, mod 4..9 -> 1..9).  The INPUT has 5 black rectangles (colour 0)
  stamped over the pattern; the OUTPUT is the SAME pattern with the cutouts
  removed.  Off the 21x21 grid everything is background 0.

Reconstruction (periodic MAX, robust to cutouts — no scalar mod/offset needed):
  Every cell of a periodic class (r%p, c%p) shares the same value v>=1, and a cut
  copy is 0 <= v, so out(r,c) = MAX over all (r',c') with r'==r, c'==c (mod p) of
  val(r',c').  At least one copy of each class is uncut, so the max recovers v.

  val = sum_k k*input_k  (one 3600B fp32 entry plane), sliced to 21x21 and cast to
  fp16 for all downstream full-region work.

  PERIOD p (= a valid period, the smallest one) is the smallest p in 2..9 such
  that val shifted by p matches val on every jointly-occupied cell in BOTH axes
  (forward shift via Gather, out-of-grid routed to all-zero index 20).  Detected
  on a 14x14 crop (sufficient overlap, verified).  p = min over candidates of
  (p if valid else 99).  (A smaller-than-length period is only ever returned when
  the pattern genuinely has that period, so folding by it is exact.)

  Periodic max = forward sparse-table doubling along each axis (shifts
  p,2p,4p,8p via Gather, oob->zero) -> F[r] = max over forward class members
  {r, r+p, ...}.  The smallest member of r's class is exactly (r mod p) (p<=9<=20),
  so the full class max is F[r mod p]: one Gather by (arange mod p) per axis
  broadcasts the class max to every cell.

  Output: pad the 21x21 class-max plane to 30x30 with sentinel -1 (off-grid never
  equals any colour 0..9), then Equal(M, arange) -> the FREE bool one-hot output.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
G = 21              # active grid is always 21x21
CD = 14             # crop size used for period detection


def build(task):
    inits, nodes = [], []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    NI = TensorProto.INT32
    B = TensorProto.BOOL

    # ---- colour-index value plane: val = sum_k k*input_k --------------------
    chW = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("chW", chW, np.float32)
    n("Conv", ["input", "chW"], "val30")              # [1,1,30,30] fp32 (entry)
    # slice to 21x21 then cast fp16
    init("s0", np.array([0, 0], np.int64), np.int64)
    init("sG", np.array([G, G], np.int64), np.int64)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["val30", "s0", "sG", "ax23"], "valf")  # [1,1,21,21] fp32
    n("Cast", ["valf"], "val", to=H)                   # [1,1,21,21] fp16

    # ---- period detection on a CDxCD crop (fp16) ----------------------------
    init("sCD", np.array([CD, CD], np.int64), np.int64)
    n("Slice", ["val", "s0", "sCD", "ax23"], "vc")     # [1,1,CD,CD] fp16
    init("zh", np.array(0.0, np.float16), np.float16)
    n("Greater", ["vc", "zh"], "vocc")                 # [1,1,CD,CD] bool (reused)

    base_cd = np.arange(CD, dtype=np.int64)
    init("zeroS", np.array(0.0, np.float16), np.float16)
    init("big99", np.array(99.0, np.float16), np.float16)
    cand_pts = []
    for p in range(2, 10):
        # forward shift index on the crop: i+p, oob (> CD-1) -> CD-1 ... but
        # CD-1 is in-grid (nonzero) -> instead route oob to a guaranteed-zero
        # lane.  The crop has no zero lane, so clamp the shifted Gather to valid
        # range and rely on the occupancy AND-mask to ignore wrap cells: oob
        # cells map to index CD-1; we mark them not-occupied by also masking the
        # shifted occupancy with an in-range indicator.
        raw = base_cd + p
        inr = (raw <= CD - 1)
        idx = np.where(inr, raw, 0).astype(np.int32)
        iname = init(f"sidx{p}", idx, np.int32)
        # in-range mask as bool vector broadcast: [1,1,1,CD] for cols, [1,1,CD,1] rows
        inr_row = inr.reshape(1, 1, CD, 1)
        inr_col = inr.reshape(1, 1, 1, CD)
        init(f"inrR{p}", inr_row, np.bool_)
        init(f"inrC{p}", inr_col, np.bool_)

        for nm, ax, inrm in (("r", 2, f"inrR{p}"), ("c", 3, f"inrC{p}")):
            sh = n("Gather", ["vc", iname], f"sh_{nm}{p}", axis=ax)        # shifted vals
            so = n("Gather", ["vocc", iname], f"so_{nm}{p}", axis=ax)      # shifted occ
            n("And", [so, inrm], f"so2_{nm}{p}")                          # occ & in-range
            n("And", ["vocc", f"so2_{nm}{p}"], f"both_{nm}{p}")           # jointly occ
            n("Equal", ["vc", sh], f"eq_{nm}{p}")
            n("Not", [f"eq_{nm}{p}"], f"ne_{nm}{p}")
            n("And", [f"both_{nm}{p}", f"ne_{nm}{p}"], f"bad_{nm}{p}")
            n("Cast", [f"bad_{nm}{p}"], f"badf_{nm}{p}", to=H)
            n("ReduceSum", [f"badf_{nm}{p}"], f"mm_{nm}{p}", keepdims=0)
        n("Cast", [f"both_r{p}"], f"ovf{p}", to=H)
        n("ReduceSum", [f"ovf{p}"], f"ov{p}", keepdims=0)
        n("Add", [f"mm_r{p}", f"mm_c{p}"], f"mmsum{p}")
        n("Equal", [f"mmsum{p}", "zeroS"], f"nomis{p}")
        n("Greater", [f"ov{p}", "zeroS"], f"hasov{p}")
        n("And", [f"nomis{p}", f"hasov{p}"], f"valid{p}")
        init(f"pval{p}", np.array(float(p), np.float16), np.float16)
        n("Where", [f"valid{p}", f"pval{p}", "big99"], f"cp{p}")
        cand_pts.append(f"cp{p}")

    # p = min over candidates (fp16 scalar)
    init("one1", np.array([1], np.int64), np.int64)
    cps1 = [n("Reshape", [nm, "one1"], nm + "_1") for nm in cand_pts]
    n("Concat", cps1, "cpvec", axis=0)
    n("ReduceMin", ["cpvec"], "pF", keepdims=0)        # scalar fp16 period

    # ---- forward periodic-max fold (fp16) on the 21x21 plane ----------------
    base = np.arange(G, dtype=np.int64)
    init("arangeF", base.astype(np.float16), np.float16)     # [21] fp16 0..20
    init("g20", np.array(float(G - 1), np.float16), np.float16)
    # we need oob -> a zero lane.  Append a zero row/col?  Instead clamp oob index
    # to G-1 and rely on... but G-1 is in-grid.  So: build the fold on a 22-wide
    # padded plane whose last lane (index 21) is zero, via Pad.
    init("pad1", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), np.int64)  # +1 on H&W ends
    init("padv", np.array(0.0, np.float16), np.float16)
    n("Pad", ["val", "pad1", "padv"], "valp")          # [1,1,22,22] fp16, lane 21 = 0

    base22 = np.arange(G + 1, dtype=np.float16)
    init("ar22", base22, np.float16)                   # [22] fp16 0..21
    init("z21", np.array(float(G), np.float16), np.float16)   # 21.0 (the zero lane)

    def fold(src, axis, tag):
        cur = src
        for k in (1, 2, 4, 8):
            init(f"k{k}_{tag}", np.array(float(k), np.float16), np.float16)
            n("Mul", ["pF", f"k{k}_{tag}"], f"kp{k}_{tag}")
            n("Add", ["ar22", f"kp{k}_{tag}"], f"raw{k}_{tag}")       # arange + k*p
            n("Greater", [f"raw{k}_{tag}", "g20"], f"oob{k}_{tag}")   # > 20 -> oob
            n("Where", [f"oob{k}_{tag}", "z21vec", f"raw{k}_{tag}"], f"sf{k}_{tag}")
            n("Cast", [f"sf{k}_{tag}"], f"si{k}_{tag}", to=NI)
            sh = n("Gather", [cur, f"si{k}_{tag}"], f"sh{k}_{tag}", axis=axis)
            cur = n("Max", [cur, sh], f"mx{k}_{tag}")
        return cur

    init("z21vec", np.full((G + 1,), float(G), np.float16), np.float16)
    fr = fold("valp", 2, "R")
    fc = fold(fr, 3, "C")                              # [1,1,22,22] fp16

    # ---- class gather: M[r][c] = fc[r mod p][c mod p] -----------------------
    n("Mod", ["arangeF", "pF"], "clsF", fmod=1)        # [21] fp16 arange mod p
    n("Cast", ["clsF"], "clsI", to=NI)                 # [21] int32
    n("Gather", [fc, "clsI"], "mr", axis=2)            # [1,1,21,22]
    n("Gather", ["mr", "clsI"], "M21", axis=3)         # [1,1,21,21] fp16

    # ---- pad to 30x30 with sentinel -1, then one-hot into FREE bool output ---
    init("pad30", np.array([0, 0, 0, 0, 0, 0, S - G, S - G], np.int64), np.int64)
    init("neg1", np.array(-1.0, np.float16), np.float16)
    n("Pad", ["M21", "pad30", "neg1"], "M", )          # [1,1,30,30] fp16, off-grid=-1
    kvals = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("kvals", kvals, np.float16)
    n("Equal", ["M", "kvals"], "output")               # [1,10,30,30] bool (FREE)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task017", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
