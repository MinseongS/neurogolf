"""task102 (ARC-AGI 44d8ac46) — fill the interior of every empty square gray
ring with red.

Rule (verified fresh, 0 mismatch / 2000+):
  The 12x12 input contains ONLY black (0) and gray (5) cells: axis-aligned gray
  1px-outline rectangles on a black background.  A box's black interior is
  recoloured RED (2) IFF the gray outline is a SQUARE with an all-black (gray-
  free) interior, side s in {3,4,5,6}; every other cell copies the input.

Encoding (gray-only square-ring detection -> colour-index plane -> FREE Equal):
  Detection needs ONLY the gray plane (input is 2-colour, black = not-gray):
  * Slice ch5 (gray) to the 12x12 active grid, cast fp16.
  * For each side s in {3,4,5,6}: ONE SAME-pad Conv whose kernel is +1 on the
    perimeter and -10 on the interior; Greater(4*(s-1)-0.5) fires only where the
    full gray perimeter is present AND no gray sits inside (rejects nested boxes
    / non-empty interiors).  A ConvTranspose with an (s-2)x(s-2) kernel of VALUE
    2.0 stamps "red=2" into the interior block.
  * Every red cell is interior of exactly one detected square, so the summed
    stamps are exactly {0,2}; add 5*gray to get a colour-index plane lab in
    {0(black bg), 2(red), 5(gray ring)}.  Cast uint8, Pad to 30x30 with sentinel
    99 (off-grid -> all channels false), and expand to the 10-channel BOOL
    output with ONE Equal(lab, arange) -- the whole expansion lands in the FREE
    output, only a single 30x30 plane (the uint8 colour-index) materialises.

Wins over the 15.67 public import: gray-only detection (no black channel), the
stamp value carries the red label directly (no separate fill mask / threshold),
sides 7/8 dropped, and one 30x30 colour-index plane replaces the 10-channel
output concat.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 30
G = 12  # active grid side
SIDES = [3, 4, 5, 6]


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    U = TensorProto.UINT8

    # ---- slice ch5 (gray) to the 12x12 active grid, cast fp16 ----
    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    init("s5", np.array([0, 5, 0, 0], np.int64), np.int64)
    init("e5", np.array([1, 6, G, G], np.int64), np.int64)
    n("Slice", ["input", "s5", "e5", "ax"], "gray_f")   # [1,1,12,12] fp32
    n("Cast", ["gray_f"], "gray", to=H)                 # [1,1,12,12] fp16

    stamps = []
    for s in SIDES:
        inner = s - 2
        # +1 perimeter, -10 interior  (interior gray => score drops below thr)
        k = np.zeros((1, 1, s, s), np.float16)
        k[0, 0, 0, :] = 1.0; k[0, 0, -1, :] = 1.0
        k[0, 0, :, 0] = 1.0; k[0, 0, :, -1] = 1.0
        k[0, 0, 1:s - 1, 1:s - 1] = -10.0
        init(f"kh{s}", k, np.float16)
        init(f"thr{s}", np.array([4 * (s - 1) - 0.5], np.float16), np.float16)
        n("Conv", ["gray", f"kh{s}"], f"score{s}", kernel_shape=[s, s], pads=[1, 1, 1, 1])
        n("Greater", [f"score{s}", f"thr{s}"], f"hitb{s}")
        n("Cast", [f"hitb{s}"], f"hith{s}", to=H)
        # stamp value 2.0 into the interior block (red label)
        ks = np.full((1, 1, inner, inner), 2.0, np.float16)
        init(f"ks{s}", ks, np.float16)
        n("ConvTranspose", [f"hith{s}", f"ks{s}"], f"stamp{s}", kernel_shape=[inner, inner])
        stamps.append(f"stamp{s}")

    # sum stamps (each interior cell hit by exactly one square) -> {0,2}
    cur = stamps[0]
    for i, st in enumerate(stamps[1:], 1):
        out = f"adds{i}" if i < len(stamps) - 1 else "stamp_sum"
        n("Add", [cur, st], out)
        cur = out

    # colour-index lab = 5*gray + stamp_sum  in {0 bg, 2 red, 5 gray}
    init("five", np.array([5.0], np.float16), np.float16)
    n("Mul", ["gray", "five"], "graylab")
    n("Add", ["graylab", "stamp_sum"], "lab12h")        # [1,1,12,12] fp16
    n("Cast", ["lab12h"], "lab12u", to=U)               # [1,1,12,12] uint8

    # pad to 30x30 with sentinel 99 (off-grid -> all channels false)
    init("pad", np.array([0, 0, 0, 0, 0, 0, N - G, N - G], np.int64), np.int64)
    init("sent", np.array([99], np.uint8), np.uint8)
    n("Pad", ["lab12u", "pad", "sent"], "lab30u")       # [1,1,30,30] uint8

    # expand to 10-channel BOOL output via Equal (FREE)
    arange = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arange", arange, np.uint8)
    n("Equal", ["lab30u", "arange"], "output")          # [1,10,30,30] bool FREE

    x = helper.make_tensor_value_info("input", F, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task102", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
