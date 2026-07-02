"""Task 286 candidate v4: maze two-colour parity flood as chained walk-counting
Einsums over a single materialized 25x25 traversability plane.

Rule (b782dc8a): cyan(8) walls; seeds = non-{0,8} cells; flood the 4-connected
non-cyan component containing the seeds; every reached cell -> pair[(r+c)%2]
where pair[p] = colour of the seed(s) with checkerboard parity p.

v4 vs v1/v2 (same walk-counting mechanism):
- The v1/v2 free-input trick (shared black colour letter) repeats the 9000-element
  input 46x per einsum -> ~330 ms/run in ORT. v4 materializes ONE t25 plane
  (non-cyan traversability, 2500B) as the repeated mask operand (625 elements,
  task187-style) -> ~ms latency. With a real mask plane, walks may traverse any
  non-cyan cell (the true rule), so no colour letters at all: 45/48/48/48 slots
  (189 total) vs measured max 171 slots over 80000 fresh (proto2, and the
  black-only dynamics upper-bound these t-dynamics).
- t25/label25 via single-tap 6x6 Convs (spatial crop for free: 30-6+1=25).
- link1 seeds = free input * Qs, cropped to 25 by identity P[30,25] operands
  (identity = no movement, so no wall-tunnelling; params are element count ->
  P costs 750).
- S = tridiagonal+self 0.5 at [25,25]; Sign after link2 re-binarizes (min path
  weight >= 0.5^93 ~ 1e-28, max <= ~3e19: fp32-safe, nonneg so >0 never flips).
- Epilogue at 25x25: parity seed-colour extraction (rank-1 stacked parity
  einsums + Greater binarize + arange dot), pv = Where(PARb, c1, c0),
  lbl_out = Where(W4>0, pv, label25), Pad(value=10) -> Equal(levels) = output.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION

LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8

    # step matrix at 25: tridiagonal + self, 0.5
    S = np.zeros((25, 25), np.float32)
    for i in range(25):
        S[i, i] = 0.5
        if i > 0:
            S[i, i - 1] = 0.5
        if i < 24:
            S[i, i + 1] = 0.5
    init("S", S, np.float32)
    # identity projection 30 -> 25 (crop); subscript order (30-letter, 25-letter)
    P = np.zeros((30, 25), np.float32)
    for i in range(25):
        P[i, i] = 1.0
    init("P", P, np.float32)

    qs = np.ones(10, np.float32); qs[0] = 0.0; qs[8] = 0.0  # seeds: non-{black,cyan}
    init("Qs", qs, np.float32)

    # t25: traversable = non-cyan in-grid, via single-tap 6x6 Conv (crops 30->25)
    wt = np.zeros((1, 10, 6, 6), np.float32)
    wt[0, :, 0, 0] = 1.0
    wt[0, 8, 0, 0] = 0.0
    init("Wt", wt, np.float32)
    n("Conv", ["input", "Wt"], "t25")  # [1,1,25,25]

    def walk_link(seed_ops, seed_subs, fixed, K, start_axis, project_seed, out_name):
        pool = list(LETTERS)
        n_l = pool.pop(0)
        z_l = pool.pop(0)
        L = {}
        for name in fixed:
            L[name] = pool.pop(0)
        r0, c0 = L['r0'], L['c0']
        subs, ops = [], []
        for o, s in zip(seed_ops, seed_subs):
            ops.append(o)
            subs.append(s.format(n=n_l, qs=L.get('qs', ''), r0=r0, c0=c0))
        r_cur, c_cur = r0, c0
        if project_seed:  # seed letters are 30-dim; identity-crop both to 25
            nr, nc = pool.pop(0), pool.pop(0)
            ops.append("P"); subs.append(r_cur + nr)
            ops.append("P"); subs.append(c_cur + nc)
            r_cur, c_cur = nr, nc
        axis = start_axis
        for _ in range(K):
            new = pool.pop(0)
            if axis == 0:
                subs.append(r_cur + new); ops.append("S"); r_cur = new
            else:
                subs.append(c_cur + new); ops.append("S"); c_cur = new
            axis ^= 1
            ops.append("t25"); subs.append(n_l + z_l + r_cur + c_cur)
        equation = ",".join(subs) + "->" + n_l + r_cur + c_cur
        nodes.append(helper.make_node("Einsum", ops, [out_name], equation=equation))
        return axis

    # link1: seeds = coloured cells of the free input, cropped by P; 45 slots
    axis = walk_link(["input", "Qs"], ["{n}{qs}{r0}{c0}", "{qs}"],
                     ['qs', 'r0', 'c0'], 45, 0, True, "W1")
    axis = walk_link(["W1"], ["{n}{r0}{c0}"], ['r0', 'c0'], 48, axis, False, "W2")
    n("Sign", ["W2"], "B2")
    axis = walk_link(["B2"], ["{n}{r0}{c0}"], ['r0', 'c0'], 48, axis, False, "W3")
    axis = walk_link(["W3"], ["{n}{r0}{c0}"], ['r0', 'c0'], 48, axis, False, "W4")

    # ---- epilogue (25x25) ----
    init("zero_f", np.array(0.0, np.float32), np.float32)
    n("Greater", ["W4", "zero_f"], "g")  # [1,25,25] bool: reached non-cyan cells

    # label at 25x25 with sentinel 10, via single-tap 6x6 Conv
    wl = np.zeros((1, 10, 6, 6), np.float32)
    wl[0, :, 0, 0] = np.arange(10, dtype=np.float32) - 10.0
    init("Wl", wl, np.float32)
    init("Bl", np.array([10.0], np.float32), np.float32)
    n("Conv", ["input", "Wl", "Bl"], "label_f")  # [1,1,25,25]
    n("Cast", ["label_f"], "label_u8", to=U8)

    # parity seed-colour extraction (over the full free input)
    pe = (np.arange(30) % 2 == 0).astype(np.float32)
    po = 1.0 - pe
    init("A", np.stack([pe, po]), np.float32)   # (r+c) even
    init("Bm", np.stack([po, pe]), np.float32)  # (r+c) odd
    init("AR10", np.arange(10, dtype=np.float32), np.float32)
    n("Einsum", ["input", "A", "A", "Qs"], "e0", equation="nqrc,sr,sc,q->nq")
    n("Einsum", ["input", "A", "Bm", "Qs"], "e1", equation="nqrc,sr,sc,q->nq")
    for p in (0, 1):
        n("Greater", [f"e{p}", "zero_f"], f"gb{p}")
        n("Cast", [f"gb{p}"], f"f{p}", to=F)
        n("Einsum", [f"f{p}", "AR10"], f"c{p}f", equation="nq,q->n")
        n("Cast", [f"c{p}f"], f"c{p}u", to=U8)

    par = ((np.add.outer(np.arange(25), np.arange(25)) % 2) == 1).reshape(1, 1, 25, 25)
    init("PARb", par, np.bool_)
    init("levels", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Where", ["PARb", "c1u", "c0u"], "pv")         # [1,1,25,25] u8
    n("Where", ["g", "pv", "label_u8"], "lbl_out")   # [1,1,25,25] u8
    init("pads", np.array([0, 0, 0, 0, 0, 0, 5, 5], np.int64), np.int64)
    init("ten_u8", np.array(10, np.uint8), np.uint8)
    n("Pad", ["lbl_out", "pads", "ten_u8"], "lbl30")  # [1,1,30,30] u8
    n("Equal", ["lbl30", "levels"], "output")         # free: [1,10,30,30] bool

    graph = helper.make_graph(
        nodes,
        "task286_walk_einsum_chain_v4",
        [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = IR_VERSION
    return model
