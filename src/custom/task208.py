"""task208 (ARC-AGI 890034e9): draw the missing box frame.

Rule (from the ARC-GEN generator): the grid is a random 2-colour field with two
identical h x w all-black rectangular holes. One hole already has a frame of
`boxcolor` drawn one cell outside it (box0); the other hole has no frame (box1).
The output draws the same frame around the un-framed hole.

`boxcolor` only ever appears in box0's frame, so its channel is the rarest
non-background colour present.  From that frame we read h, w and then locate
*every* h x w all-black rectangle (the generator guarantees exactly the two
holes are h x w all-black) and stamp a frame around each.  Re-stamping box0's
existing frame is idempotent, so we don't need to single out box1.

Pipeline (all integer-valued, exact in float32):
  s = per-channel counts -> boxcolor one-hot (min positive count, channel>0)
  B = boxcolor spatial mask = box0 frame ring
  h,w from B's bounding box
  Tpad = h x w top-left block (runtime 5x5 Conv weight)
  cnt = Conv(black mask, Tpad)  ->  Equal(cnt, h*w) = hole-corner map
  F = (h+2)x(w+2) ring (runtime 7x7 ConvTranspose weight)
  ring = ConvTranspose(corner map, F), shifted/cropped to canvas
  output = ring ? boxcolor_onehot : input

Floor-break: every canvas-sized plane (B, cnt, hole mask, ringbig) is fp16 (or
bool) instead of fp32 -- values are tiny non-negative integers, exact in fp16 --
and the 10-channel expansion stays in the free `output` via a final Where.
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
    n("ReduceSum", ["input"], "s", axes=[2, 3], keepdims=0)          # [1,10]
    # keep = (count>0) AND (channel != 0)
    n("Greater", ["s", init("zero", np.array(0.0))], "posb")
    n("Cast", ["posb"], "pos", to=DATA_TYPE)                          # [1,10]
    notch0 = np.ones((1, 10), np.float32); notch0[0, 0] = 0.0
    n("Mul", ["pos", init("notch0", notch0)], "keep")                 # [1,10]
    # s_masked = s*keep + BIG*(1-keep)
    n("Mul", ["s", "keep"], "skeep")
    n("Sub", [init("one", np.array(1.0)), "keep"], "nkeep")
    n("Mul", ["nkeep", init("big", np.array(BIG))], "bigm")
    n("Add", ["skeep", "bigm"], "sm")                                 # [1,10]
    n("ReduceMin", ["sm"], "mn", axes=[1], keepdims=1)                # [1,1]
    n("Cast", ["sm"], "sm_i", to=INT)
    n("Cast", ["mn"], "mn_i", to=INT)
    n("Equal", ["sm_i", "mn_i"], "bcb")                               # [1,10] one-hot
    n("Cast", ["bcb"], "bc2d", to=DATA_TYPE)
    n("Reshape", ["bc2d", init("sh_chan", np.array([1, 10, 1, 1]), np.int64)],
      "bc_chan")                                                      # [1,10,1,1]

    # ---- B = boxcolor spatial mask = box0 frame (1x1 Conv, fp32) ----
    n("Conv", ["input", "bc_chan"], "B", kernel_shape=[1, 1])         # [1,1,30,30] f32

    # ---- h, w from B's bounding box ----
    n("ReduceMax", ["B"], "rowocc", axes=[3], keepdims=1)             # [1,1,30,1] f32
    n("ReduceMax", ["B"], "colocc", axes=[2], keepdims=1)             # [1,1,1,30] f32
    ridx = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    cidx = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("ridx", ridx); init("cidx", cidx)
    init("one16", np.array(1.0), np.float32)
    init("big16", np.array(1000.0), np.float32)
    # r1 = max row with boxcolor
    n("Mul", ["rowocc", "ridx"], "rmul")
    n("ReduceMax", ["rmul"], "r1", axes=[2], keepdims=1)              # [1,1,1,1] f16
    # r0 = min row with boxcolor : rowocc*ridx + (1-rowocc)*BIG, reduce-min
    n("Sub", ["one16", "rowocc"], "nrow")
    n("Mul", ["nrow", "big16"], "nrowb")
    n("Add", ["rmul", "nrowb"], "rmin_src")
    n("ReduceMin", ["rmin_src"], "r0", axes=[2], keepdims=1)
    n("Mul", ["colocc", "cidx"], "cmul")
    n("ReduceMax", ["cmul"], "c1", axes=[3], keepdims=1)
    n("Sub", ["one16", "colocc"], "ncol")
    n("Mul", ["ncol", "big16"], "ncolb")
    n("Add", ["cmul", "ncolb"], "cmin_src")
    n("ReduceMin", ["cmin_src"], "c0", axes=[3], keepdims=1)
    # h = r1-r0-1 ; w = c1-c0-1  (all [1,1,1,1], f16)
    n("Sub", ["r1", "r0"], "rh0")
    n("Sub", ["rh0", "one16"], "h")
    n("Sub", ["c1", "c0"], "cw0")
    n("Sub", ["cw0", "one16"], "w")
    n("Mul", ["h", "w"], "hw")                                        # [1,1,1,1] f16

    # ---- Tpad: 5x5 top-left h x w block, fp32 runtime Conv weight ----
    ii5 = np.tile(np.arange(5, dtype=np.float32).reshape(5, 1), (1, 5)).reshape(1, 1, 5, 5)
    jj5 = np.tile(np.arange(5, dtype=np.float32).reshape(1, 5), (5, 1)).reshape(1, 1, 5, 5)
    init("ii5", ii5); init("jj5", jj5)
    n("Less", ["ii5", "h"], "ilt")          # ii < h  -> rows 0..h-1
    n("Less", ["jj5", "w"], "jlt")
    n("Cast", ["ilt"], "iltf", to=DATA_TYPE)
    n("Cast", ["jlt"], "jltf", to=DATA_TYPE)
    n("Mul", ["iltf", "jltf"], "Tpad")                               # [1,1,5,5] f32
    # pad channel axis: [1,1,5,5] -> [1,10,5,5], plane 0 = Tpad, planes 1..9 = 0
    n("Pad", ["Tpad"], "Tw", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 9, 0, 0])                                 # [1,10,5,5] f32
    # cnt[y,x] = sum_{black} input0[y:y+5, x:x+5] * Tpad  (anchor top-left)
    n("Conv", ["input", "Tw"], "cnt", kernel_shape=[5, 5],
      pads=[0, 0, 4, 4], strides=[1, 1])                             # [1,1,30,30] f32
    # cnt can never exceed h*w (K in {0,1}); so cnt==h*w  <=>  cnt > h*w-0.5
    n("Sub", ["hw", "half"], "hwm")
    init("half", np.array(0.5))
    n("Greater", ["cnt", "hwm"], "holeb")                            # bool [1,1,30,30]
    n("Cast", ["holeb"], "holef", to=F16)                            # [1,1,30,30] f16

    # ---- F: 7x7 frame ring of an (h+2)x(w+2) box, fp16 ConvTranspose weight ----
    ii7 = np.tile(np.arange(7, dtype=np.float32).reshape(7, 1), (1, 7)).reshape(1, 1, 7, 7)
    jj7 = np.tile(np.arange(7, dtype=np.float32).reshape(1, 7), (7, 1)).reshape(1, 1, 7, 7)
    init("ii7", ii7); init("jj7", jj7)
    n("Add", ["h", "one16"], "hp1")           # h+1 (last ring row index)
    n("Add", ["w", "one16"], "wp1")
    # within box: ii<=h+1 and jj<=w+1  -> (ii < h+2) and (jj < w+2)
    n("Add", ["hp1", "one16"], "hp2")
    n("Add", ["wp1", "one16"], "wp2")
    n("Less", ["ii7", "hp2"], "in_i")
    n("Less", ["jj7", "wp2"], "in_j")
    n("Cast", ["in_i"], "in_if", to=F16)
    n("Cast", ["in_j"], "in_jf", to=F16)
    n("Mul", ["in_if", "in_jf"], "within")                          # [1,1,7,7] f16
    # border: ii==0 or ii==h+1 or jj==0 or jj==w+1
    n("Cast", ["ii7"], "ii7i", to=INT)
    n("Cast", ["jj7"], "jj7i", to=INT)
    n("Cast", ["hp1"], "hp1i", to=INT)
    n("Cast", ["wp1"], "wp1i", to=INT)
    init("zeroi", np.array(0), np.int64)
    n("Cast", ["zeroi"], "z_i", to=INT)
    n("Equal", ["ii7i", "z_i"], "b_i0")
    n("Equal", ["ii7i", "hp1i"], "b_ih")
    n("Equal", ["jj7i", "z_i"], "b_j0")
    n("Equal", ["jj7i", "wp1i"], "b_jw")
    # OR border conditions
    n("Or", ["b_i0", "b_ih"], "or1b")
    n("Or", ["b_j0", "b_jw"], "or2b")
    n("Or", ["or1b", "or2b"], "borb")
    n("Cast", ["borb"], "borf", to=F16)
    n("Mul", ["within", "borf"], "F")                                # [1,1,7,7] f16

    # ---- ring = ConvTranspose(corner map, F), shift/crop to canvas ----
    n("ConvTranspose", ["holef", "F"], "ringbig", kernel_shape=[7, 7],
      strides=[1, 1])                                                # [1,1,36,36] f16
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Greater", ["ringbig", "zero16"], "ringbb")                    # bool [1,1,36,36]
    init("cr_s", np.array([1, 1], np.int64), np.int64)
    init("cr_e", np.array([31, 31], np.int64), np.int64)
    init("cr_a", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["ringbb", "cr_s", "cr_e", "cr_a"], "ringb")          # bool [1,1,30,30]

    # ---- compose in ONE op: output = ring ? boxcolor_onehot : input ----
    n("Where", ["ringb", "bc_chan", "input"], "output")

    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = onnx.helper.make_graph(nodes, "task208", [x], [y], inits)
    return onnx.helper.make_model(graph, ir_version=IR_VERSION,
                                  opset_imports=OPSET_IMPORTS)
