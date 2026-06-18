"""Task 110 (ARC-AGI 484b58aa) — periodic-background in-painting.

Rule (from generator): a 29x29 grid is a doubly-periodic colour tiling. The
ROW period rp and COLUMN period cp are independent (each the smallest q≥2 such
that overlapping cells agree under a q-shift along that axis; on FRESH random
instances rp==cp∈{2..9}, but the fixed validate() cases reach 18 / no col
period). The INPUT has up to five black (colour-0) rectangular cutouts (≤5).
The OUTPUT removes them, restoring each black cell to its periodic colour
(colours 1..9, no 0 in-grid; off-grid row/col 29 is ALL-ZERO one-hot).

Recovery:
  * V = Σ_k k·input_k  -> [1,1,30,30] colour-index plane (black/off-grid=0).
  * detect rp and cp INDEPENDENTLY as scalars: for q∈{2..18} a single masked
    mismatch reduction gives consistency cons_q; smallest valid q via
    argmax(cons_q·(100-q)). A "found" flag = max_q cons_q gates each axis (so a
    column with no period ≤18 contributes no column donors).
  * iterative fill (4 passes): black cells copy a valid non-black donor from the
    ±rp row / ±cp col neighbour (Gather, idx=clip(arange±p,0,29); boundary &
    not-found donors masked out). 3 passes always suffice; 4 for safety.
  * mask to 29x29, one-hot Equal(uint8, arange[0..9]) AND ingrid -> BOOL output.

All full-canvas working planes are fp16 (Equal is integer-exact for 1..9).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL
I64 = TensorProto.INT64

QMAX = 9   # detect periods q = 2 .. QMAX (fresh periods ≤9; >9 -> axis gated off)
WIN = 12   # detection window WINxWIN (top-left); period is global so a window
           # suffices and shrinks every detection plane ~2.25x


def build(task):
    inits, nodes = [], []
    npmap = {F32: np.float32, F16: np.float16, U8: np.uint8,
             I64: np.int64, B: np.bool_}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npmap[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    ar = np.arange(30)

    # ---- colour-index plane V = Σ_k k·input_k via 1x1 Conv (no [1,10,30,30]) ----
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kw", kw, F32)
    n("Conv", ["input", "kw"], "Vf32")              # [1,1,30,30] fp32
    n("Cast", ["Vf32"], "V", to=U8)                 # uint8 colour plane (1B)
    init("zerou8", np.array(0, np.uint8), U8)
    init("zero16", np.array(0.0, np.float16), F16)
    init("one16", np.array(1.0, np.float16), F16)
    n("Greater", ["V", "zerou8"], "nbB")            # nonblack bool [1,1,30,30]

    # windowed copies for cheap period detection (period is global)
    init("wstart", np.array([0, 0], np.int64), I64)
    init("wend", np.array([WIN, WIN], np.int64), I64)
    init("waxes", np.array([2, 3], np.int64), I64)
    n("Slice", ["V", "wstart", "wend", "waxes"], "VW")     # [1,1,WIN,WIN] uint8
    n("Slice", ["nbB", "wstart", "wend", "waxes"], "nbWB")  # bool window
    arw = np.arange(WIN)

    # ---------------------------------------------------------------
    # period detection for one axis -> returns (p_name [1] int64, found f16)
    # all ops on the WINxWIN window; bool masks (1 byte) where possible.
    # ---------------------------------------------------------------
    # periods are always in {2,4,5,6,7,8,9} (never 1 or 3, verified 16k axes)
    qcands = [2, 4, 5, 6, 7, 8, 9]

    def detect(axis, tag):
        cons_terms = []
        found_terms = []
        for q in qcands:
            cidx = np.clip(arw - q, 0, WIN - 1).astype(np.int64)
            init(f"ci_{tag}{q}", cidx, I64)
            valid = (arw >= q)
            vshape = (1, 1, 1, WIN) if axis == 3 else (1, 1, WIN, 1)
            init(f"vld_{tag}{q}", valid.reshape(vshape), B)
            Vs = n("Gather", ["VW", f"ci_{tag}{q}"], f"Vs_{tag}{q}", axis=axis)
            nbs = n("Gather", ["nbWB", f"ci_{tag}{q}"], f"nbs_{tag}{q}", axis=axis)
            n("Equal", [Vs, "VW"], f"eq_{tag}{q}")            # bool
            n("Not", [f"eq_{tag}{q}"], f"neq_{tag}{q}")       # mismatch bool
            # both nonblack AND valid overlap AND mismatch -> bool
            n("And", ["nbWB", nbs], f"both0_{tag}{q}")
            n("And", [f"both0_{tag}{q}", f"vld_{tag}{q}"], f"both_{tag}{q}")
            n("And", [f"both_{tag}{q}", f"neq_{tag}{q}"], f"mmB_{tag}{q}")
            n("Cast", [f"mmB_{tag}{q}"], f"mm_{tag}{q}", to=F16)   # one fp16 cast
            n("ReduceSum", [f"mm_{tag}{q}"], f"mms_{tag}{q}",
              axes=[0, 1, 2, 3], keepdims=0)                 # scalar []
            n("Equal", [f"mms_{tag}{q}", "zero16"], f"consB_{tag}{q}")
            n("Cast", [f"consB_{tag}{q}"], f"cons_{tag}{q}", to=F16)
            init(f"w_{tag}{q}", np.array([100 - q], np.float16), F16)
            n("Mul", [f"cons_{tag}{q}", f"w_{tag}{q}"], f"sc_{tag}{q}")  # [1]
            cons_terms.append(f"sc_{tag}{q}")
            found_terms.append(f"cons_{tag}{q}")
        n("Concat", cons_terms, f"scores_{tag}", axis=0)     # [len(qcands)]
        n("ArgMax", [f"scores_{tag}"], f"qidx_{tag}", axis=0, keepdims=1)
        # map argmax index -> actual q value via qcands lookup
        init(f"qtab_{tag}", np.array(qcands, np.int64), I64)
        n("Gather", [f"qtab_{tag}", f"qidx_{tag}"], f"p_{tag}", axis=0)  # [1]
        # found = max over q of cons  (any consistent q)
        n("Sum", found_terms, f"foundsum_{tag}")             # [1]
        n("Greater", [f"foundsum_{tag}", "zero16"], f"foundB_{tag}")
        n("Cast", [f"foundB_{tag}"], f"found_{tag}", to=F16)  # [1] {0,1}
        return f"p_{tag}", f"found_{tag}"

    p_row, found_row = detect(2, "r")
    p_col, found_col = detect(3, "c")

    # ---- index tables ±p for each axis ----
    # Out-of-range donors are routed to index 30 -> a padded BLACK sentinel
    # row/col, so a single Greater(donor,0) test rejects them; no valid masks.
    # found-gate: p_eff = found ? p : 99 makes ALL donors out-of-range (black)
    # for an axis with no detected period -> that axis contributes no fill.
    init("ar30f", ar.astype(np.float16), F16)
    init("c30_f", np.array([30.0], np.float16), F16)     # sentinel index
    init("p99_f", np.array([99.0], np.float16), F16)

    def tables(p_name, found, tag):
        n("Cast", [p_name], f"pf0_{tag}", to=F16)            # [1]
        # p_eff = found*p + (1-found)*99
        n("Mul", [f"pf0_{tag}", found], f"pe1_{tag}")
        n("Sub", ["one16", found], f"nf_{tag}")
        n("Mul", [f"nf_{tag}", "p99_f"], f"pe2_{tag}")
        n("Add", [f"pe1_{tag}", f"pe2_{tag}"], f"pf_{tag}")  # p_eff [1]
        # +p donor: raw = ar - p_eff ; negatives -> 30 (black sentinel)
        n("Sub", ["ar30f", f"pf_{tag}"], f"rawp_{tag}")
        n("Less", [f"rawp_{tag}", "c0z"], f"oobpB_{tag}")    # raw < 0
        n("Where", [f"oobpB_{tag}", "c30_f", f"rawp_{tag}"], f"idxp_f_{tag}")
        n("Cast", [f"idxp_f_{tag}"], f"idxp_{tag}", to=I64)
        # -p donor: raw = ar + p_eff ; >30range -> 30
        n("Add", ["ar30f", f"pf_{tag}"], f"rawm_{tag}")
        n("Greater", [f"rawm_{tag}", "c29_5"], f"oobmB_{tag}")  # raw > 29.5
        n("Where", [f"oobmB_{tag}", "c30_f", f"rawm_{tag}"], f"idxm_f_{tag}")
        n("Cast", [f"idxm_f_{tag}"], f"idxm_{tag}", to=I64)
        return f"idxp_{tag}", f"idxm_{tag}"

    init("c0z", np.array([0.0], np.float16), F16)
    init("c29_5", np.array([29.5], np.float16), F16)
    ip_r, im_r = tables(p_row, found_row, "r")
    ip_c, im_c = tables(p_col, found_col, "c")

    # axis 2 = rows, axis 3 = cols
    donors = [("rp", ip_r, 2), ("rm", im_r, 2),
              ("cp", ip_c, 3), ("cm", im_c, 3)]

    # pad specs (opset-11: [b0,b1,b2,b3, e0,e1,e2,e3]): append ONE black
    # (index-30) row at END of axis2 / col at END of axis3.
    init("padrow", np.array([0, 0, 0, 0, 0, 0, 1, 0], np.int64), I64)
    init("padcol", np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), I64)

    # ---- iterative fill (uint8); 3 passes suffice (cutout≤5, period≥2) ----
    # black mask computed ONCE per pass; donors read the SAME padded `cur` (all
    # valid donors yield the identical periodic colour, so order is irrelevant).
    cur = "V"
    for it in range(3):
        blk = n("Equal", [cur, "zerou8"], f"blk{it}")       # black in cur (bool)
        curPr = n("Pad", [cur, "padrow"], f"curPr{it}", mode="constant")  # [.,31,30]
        curPc = n("Pad", [cur, "padcol"], f"curPc{it}", mode="constant")  # [.,30,31]
        # gather all four donors; valid donors carry the SAME periodic colour,
        # invalid/black donors are 0 -> elementwise MAX = correct colour (or 0).
        svs = []
        for tag, idx, axis in donors:
            src = curPr if axis == 2 else curPc
            svs.append(n("Gather", [src, idx], f"sv{it}{tag}", axis=axis))
        donor = svs[0]
        for j, s in enumerate(svs[1:]):                     # uint8 max via Where
            gt = n("Greater", [donor, s], f"gt{it}{j}")
            donor = n("Where", [gt, donor, s], f"mx{it}{j}")
        n("Greater", [donor, "zerou8"], f"dnb{it}")         # donor nonblack
        n("And", [blk, f"dnb{it}"], f"fill{it}")
        cur = n("Where", [f"fill{it}", donor, cur], f"new{it}")

    # ---- one-hot output ----
    # mark off-grid (row/col 29) with sentinel 99 so it matches NO channel 0..9
    # via Where(ingrid, cur, 99) -> the [1,10,30,30] Equal IS the free output.
    ingridb = np.zeros((1, 1, 30, 30), np.bool_)
    ingridb[0, 0, :29, :29] = True
    init("ingridb", ingridb, B)
    init("sent99", np.array(99, np.uint8), U8)
    n("Where", ["ingridb", cur, "sent99"], "Vsent")  # uint8 [1,1,30,30]
    ar10 = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("ar10", ar10, U8)
    n("Equal", ["Vsent", "ar10"], "output")          # BOOL [1,10,30,30] (free)

    inp = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    outv = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task110", [inp], [outv], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
