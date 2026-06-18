"""task208 (ARC-AGI 890034e9): draw the missing box frame.

Rule: a 21x21 random 2-colour field holds two identical h x w all-black holes
(h,w in [2,5]). One hole (box0) already has a `boxcolor` frame drawn one cell
outside it; the other has none. Output = input with the frame stamped around
BOTH holes (re-stamping box0 is idempotent). boxcolor (color_list[0]) is NEVER
used in the random field, so its channel is exactly box0's frame ring.

Pipeline (all integer-valued, exact in fp32/fp16):
  bc_chan : boxcolor one-hot [1,10,1,1] = min-positive-count channel (channel>0)
  h,w     : boxcolor occupies exactly h+2 distinct rows / w+2 cols (frame is a
            rectangle ring + field never reuses boxcolor) -> h = (#occupied
            rows)-2 via two collapse-Convs + ReduceSum (NO bbox min/max plane,
            NO 3600B boxcolor 30x30 plane).
  cnt     : Conv(input, Tw=h x w top-left block) NO-pad -> 26x26 black-count; a
            cell is a hole top-left corner iff cnt == h*w.
  ring    : ConvTranspose(corner map, F=(h+2)x(w+2) ring kernel) cropped directly
            to 30x30 (no 36x36 intermediate / no Slice).
  output  : Where(ring, boxcolor_onehot, input)  (10-ch expansion stays free).

Floor-break levers vs the original public net (15.13 -> 15.47, +0.34):
  * h,w from occupied-row/col COUNT (ReduceSum) instead of a min/max bbox scan
    -> deletes ~14 small planes + the ridx/cidx ramps.
  * collapse-Convs for boxcolor row/col occupancy -> never materialise the
    3600B boxcolor 30x30 plane.
  * NO-pad cnt Conv -> 26x26 (2704B) instead of padded 30x30 (3600B); holes'
    top-left corners (rows/cols <= 17) all fall inside 26x26.
  * ConvTranspose cropped straight to 30x30 (no 36x36 ringbig / Slice).
  * every downstream canvas plane is fp16 / bool, never fp32.
"""

import numpy as np
import onnx

from ..harness import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

BIG = 1.0e6
F16 = onnx.TensorProto.FLOAT16


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=np.float32):
        a = np.ascontiguousarray(arr, dtype=dtype)
        inits.append(onnx.numpy_helper.from_array(a, name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(onnx.helper.make_node(op, ins, [out], **attrs))
        return out

    INT = onnx.TensorProto.INT32

    # ---- per-channel counts -> boxcolor one-hot [1,10,1,1] ----
    n("ReduceSum", ["input"], "s", axes=[2, 3], keepdims=0)
    n("Greater", ["s", init("zero", np.array(0.0))], "posb")
    n("Cast", ["posb"], "pos", to=DATA_TYPE)
    notch0 = np.ones((1, 10), np.float32); notch0[0, 0] = 0.0
    n("Mul", ["pos", init("notch0", notch0)], "keep")
    n("Mul", ["s", "keep"], "skeep")
    n("Sub", [init("one", np.array(1.0)), "keep"], "nkeep")
    n("Mul", ["nkeep", init("big", np.array(BIG))], "bigm")
    n("Add", ["skeep", "bigm"], "sm")
    n("ReduceMin", ["sm"], "mn", axes=[1], keepdims=1)
    n("Cast", ["sm"], "sm_i", to=INT)
    n("Cast", ["mn"], "mn_i", to=INT)
    n("Equal", ["sm_i", "mn_i"], "bcb")
    n("Cast", ["bcb"], "bc2d", to=DATA_TYPE)
    n("Reshape", ["bc2d", init("sh_chan", np.array([1, 10, 1, 1]), np.int64)],
      "bc_chan")

    # ---- per-row / per-col boxcolor occupancy via collapsing convs ----
    # (avoids materialising the 3600B boxcolor 30x30 plane B)
    n("Expand", ["bc_chan", init("ew_row", np.array([1, 10, 1, 30]), np.int64)],
      "Wrow")                                                         # [1,10,1,30] f32
    n("Conv", ["input", "Wrow"], "rowsum", kernel_shape=[1, 30])      # [1,1,30,1]
    n("Expand", ["bc_chan", init("ew_col", np.array([1, 10, 30, 1]), np.int64)],
      "Wcol")                                                         # [1,10,30,1] f32
    n("Conv", ["input", "Wcol"], "colsum", kernel_shape=[30, 1])      # [1,1,1,30]
    # boxcolor occurs ONLY in box0's frame (field never uses it), so the frame
    # spans exactly h+2 distinct rows and w+2 distinct cols. Count occupied
    # rows/cols and subtract 2 -> h, w (no min/max bbox machinery).
    n("Greater", ["rowsum", "zero"], "rowoccb")                      # [1,1,30,1] bool
    n("Greater", ["colsum", "zero"], "coloccb")                      # [1,1,1,30] bool
    n("Cast", ["rowoccb"], "rowocc", to=DATA_TYPE)
    n("Cast", ["coloccb"], "colocc", to=DATA_TYPE)
    init("one16", np.array(1.0), np.float32)
    init("two16", np.array(2.0), np.float32)
    n("ReduceSum", ["rowocc"], "hframe", axes=[2], keepdims=1)       # [1,1,1,1] = h+2
    n("ReduceSum", ["colocc"], "wframe", axes=[3], keepdims=1)       # [1,1,1,1] = w+2
    n("Sub", ["hframe", "two16"], "h")
    n("Sub", ["wframe", "two16"], "w")
    n("Mul", ["h", "w"], "hw")

    # ---- Tpad: 5x5 top-left h x w block, fp32 runtime Conv weight ----
    ii5 = np.tile(np.arange(5, dtype=np.float32).reshape(5, 1), (1, 5)).reshape(1, 1, 5, 5)
    jj5 = np.tile(np.arange(5, dtype=np.float32).reshape(1, 5), (5, 1)).reshape(1, 1, 5, 5)
    init("ii5", ii5); init("jj5", jj5)
    n("Less", ["ii5", "h"], "ilt")
    n("Less", ["jj5", "w"], "jlt")
    n("Cast", ["ilt"], "iltf", to=DATA_TYPE)
    n("Cast", ["jlt"], "jltf", to=DATA_TYPE)
    n("Mul", ["iltf", "jltf"], "Tpad")
    n("Pad", ["Tpad"], "Tw", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 9, 0, 0])
    # no-pad cnt: output 26x26, cnt[y,x]=window top-left (y,x). Hole corners are
    # at rows/cols <= 17 so all fall inside the 26x26 region.
    n("Conv", ["input", "Tw"], "cnt", kernel_shape=[5, 5],
      strides=[1, 1])                                                # f32 26x26
    n("Sub", ["hw", "half"], "hwm")
    init("half", np.array(0.5))
    n("Greater", ["cnt", "hwm"], "holeb")                            # bool 26x26
    n("Cast", ["holeb"], "holef", to=F16)                            # f16 26x26

    # ---- F: 7x7 frame ring of an (h+2)x(w+2) box ----
    ii7 = np.tile(np.arange(7, dtype=np.float32).reshape(7, 1), (1, 7)).reshape(1, 1, 7, 7)
    jj7 = np.tile(np.arange(7, dtype=np.float32).reshape(1, 7), (7, 1)).reshape(1, 1, 7, 7)
    init("ii7", ii7); init("jj7", jj7)
    n("Add", ["h", "one16"], "hp1")
    n("Add", ["w", "one16"], "wp1")
    n("Add", ["hp1", "one16"], "hp2")
    n("Add", ["wp1", "one16"], "wp2")
    n("Less", ["ii7", "hp2"], "in_i")
    n("Less", ["jj7", "wp2"], "in_j")
    n("Cast", ["in_i"], "in_if", to=F16)
    n("Cast", ["in_j"], "in_jf", to=F16)
    n("Mul", ["in_if", "in_jf"], "within")
    n("Cast", ["ii7"], "ii7i", to=INT)
    n("Cast", ["jj7"], "jj7i", to=INT)
    n("Cast", ["hp1"], "hp1i", to=INT)
    n("Cast", ["wp1"], "wp1i", to=INT)
    init("zeroi", np.array(0, np.int32), np.int32)
    n("Equal", ["ii7i", "zeroi"], "b_i0")
    n("Equal", ["ii7i", "hp1i"], "b_ih")
    n("Equal", ["jj7i", "zeroi"], "b_j0")
    n("Equal", ["jj7i", "wp1i"], "b_jw")
    n("Or", ["b_i0", "b_ih"], "or1b")
    n("Or", ["b_j0", "b_jw"], "or2b")
    n("Or", ["or1b", "or2b"], "borb")
    n("Cast", ["borb"], "borf", to=F16)
    n("Mul", ["within", "borf"], "F")

    # ---- ring: ConvTranspose directly to 30x30 via cropping pads ----
    # input 30x30, kernel 7 -> 36; pads crop 6 total. Offset tuned to align frame.
    # holef 26x26, kernel 7 -> 32x32; crop 1 each side -> 30x30 (same stamp offset)
    n("ConvTranspose", ["holef", "F"], "ringbig", kernel_shape=[7, 7],
      strides=[1, 1], pads=[1, 1, 1, 1])                              # f16 30x30
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Greater", ["ringbig", "zero16"], "ringb")                     # bool 30x30

    # ---- compose: output = ring ? boxcolor_onehot : input ----
    n("Where", ["ringb", "bc_chan", "input"], "output")

    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = onnx.helper.make_graph(nodes, "task208", [x], [y], inits)
    return onnx.helper.make_model(graph, ir_version=IR_VERSION,
                                  opset_imports=OPSET_IMPORTS)
