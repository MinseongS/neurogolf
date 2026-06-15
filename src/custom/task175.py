"""task175 (ARC-AGI 73251a56) — restore a deterministic position-pattern grid.

Rule (from the generator): on a 21x21 grid every cell holds
    color = ((base[r,c] + modset) % mod) + 1
where base[r,c] = (r+2)//(c+2) if r>c else (c+2)//(r+2), with base=2 on the
diagonal (r==c).  `mod` in 5..9 and `modset` in 1..4 are per-puzzle constants.
The INPUT then has up to 5 black (colour 0) rectangular cutouts overwriting the
pattern; the OUTPUT is the clean pattern (cutouts restored).

So the output is a PURE function of position, parameterised only by (mod,modset)
— 20 possibilities.  Each (mod,modset) defines a lookup table
    LUT_k[b] = ((b + modset) % mod) + 1   for base value b in 1..11.
The output of every cell is LUT_winner[base[r,c]].

Recovery (generalising): base[r,c] is a fixed known constant (values 1..11 on
the 21x21 region, 0 elsewhere).  For each base value b we read the observed
colour obs[b] from any non-cut cell with that base value (segmented max of the
observed colour index over the base==b cells; cut cells read 0, so max yields
the true colour, or 0 if every such cell is cut).  We then score each of the 20
candidate LUTs by how many present base values it matches, take the argmax, and
emit LUT_winner[base[r,c]] as a uint8 label map -> BOOL output via Equal.

Memory: only small planes are materialised.  cidx is one [1,1,30,30] uint8 plane
(900B); the segmented-max product over the 11 base values is the one larger
intermediate.  Everything else is <=300B.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

SIZE = 21


def _base_grid():
    B = np.zeros((30, 30), dtype=np.int64)
    for r in range(SIZE):
        for c in range(SIZE):
            if r > c:
                col = (r + 2) // (c + 2)
            elif c > r:
                col = (c + 2) // (r + 2)
            else:
                col = 2
            B[r][c] = col  # values 1..11
    return B  # 0 off the 21x21 region


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    base = _base_grid()                       # [30,30] int values 0..11
    nbase = 11                                # base values 1..11

    # candidate LUTs ------------------------------------------------------
    luts = []
    for mod in range(5, 10):
        for modset in range(1, 5):
            luts.append([((b + modset) % mod) + 1 for b in range(1, nbase + 1)])
    luts = np.array(luts, dtype=np.int64)     # [20,11], values 1..9
    ncand = luts.shape[0]

    # --- observed colour index per cell: cidx = Conv(input, arange) ------
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)   # ch0 weight 0
    init("cidxW", w, np.float32)
    n("Conv", ["input", "cidxW"], "cidxf")     # [1,1,30,30] float (colour 0..9)
    n("Cast", ["cidxf"], "cidxu", to=U8)       # [1,1,30,30] uint8 (900B)
    init("flat900", np.array([900], np.int64), np.int64)
    n("Reshape", ["cidxu", "flat900"], "cflat")  # [900] uint8

    # --- per base value, gather a fixed set of K representative cells and
    #     take the max (survives black cutouts unless EVERY cell is cut) ---
    K = 8
    idx = np.zeros((nbase, K), dtype=np.int64)
    for bi in range(nbase):
        cells = [r * 30 + c for r in range(SIZE) for c in range(SIZE)
                 if base[r][c] == (bi + 1)]
        if len(cells) >= K:
            cells = cells[:K]
        else:
            cells = cells + [cells[0]] * (K - len(cells))   # pad by repeat
        idx[bi] = cells
    init("gidx", idx.reshape(-1), np.int64)    # [11*K]
    n("Gather", ["cflat", "gidx"], "gvals")    # [11*K] uint8 (88B)
    init("gshape", np.array([nbase, K], np.int64), np.int64)
    n("Reshape", ["gvals", "gshape"], "gvals2")  # [11,K] uint8
    n("Cast", ["gvals2"], "gvalsf", to=F)        # [11,K] float
    n("ReduceMax", ["gvalsf"], "obs", axes=[1], keepdims=0)  # [11] observed

    # --- score 20 candidates: score_k = sum_b (obs[b]==LUT_k[b]).
    # Absent base values have obs[b]==0, and 0 is never a LUT value (LUT in
    # 1..9), so absent values contribute no false match -> no present mask. ---
    lutf = luts.astype(np.float32)                            # [20,11]
    init("lutf", lutf, np.float32)
    # broadcast: obs[1,11] vs lutf[20,11] -> match[20,11]
    init("obsshape", np.array([1, nbase], np.int64), np.int64)
    n("Reshape", ["obs", "obsshape"], "obs2")                # [1,11]
    n("Equal", ["lutf", "obs2"], "match_b")                  # [20,11] bool
    n("Cast", ["match_b"], "matchf", to=F)                   # [20,11]
    n("ReduceSum", ["matchf"], "scores", axes=[1], keepdims=0)  # [20]

    # --- argmax candidate -> winner index --------------------------------
    init("sshape", np.array([1, ncand], np.int64), np.int64)
    n("Reshape", ["scores", "sshape"], "scores2")            # [1,20]
    n("ArgMax", ["scores2"], "winner", axis=1, keepdims=0)   # [1] int64

    # --- gather winning LUT row -> lut_w[11], build lut_full[12] ----------
    # lut_data[12]: index 0 = sentinel(10) for off-grid base==0, index b = LUT[b]
    # We gather per-candidate full lut tables [20,12].
    lut_full = np.zeros((ncand, nbase + 1), dtype=np.uint8)
    lut_full[:, 0] = 10                                       # sentinel off-grid
    lut_full[:, 1:] = luts.astype(np.uint8)                  # base 1..11 -> value
    init("lutfull", lut_full, np.uint8)                      # [20,12] params
    n("Gather", ["lutfull", "winner"], "lutw")               # [1,12] uint8

    # --- map base grid -> label: L[r,c] = lutw[base[r,c]] ----------------
    init("basegrid", base.reshape(1, 1, 30, 30).astype(np.int64),
         np.int64)                                            # [1,1,30,30] params
    # squeeze lutw to [12]
    init("lwshape", np.array([nbase + 1], np.int64), np.int64)
    n("Reshape", ["lutw", "lwshape"], "lutw1")               # [12] uint8
    n("Gather", ["lutw1", "basegrid"], "L")                  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                      # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task175", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
