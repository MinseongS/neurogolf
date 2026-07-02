"""Task 145 candidate — guillotine rooms min/max area via TWO phase-walk Einsums (fp16).

Rule (6455b5f5): red(2) guillotine cut lines partition a <=20x20 top-left-anchored
grid into rectangular rooms; output keeps red lines, paints max-area room(s) blue(1),
min-area room(s) cyan(8), other bg 0, off-grid no channel.

Mechanism (walk-einsum lens: grader counts only NODE OUTPUTS; op internals free):
- Rooms are rectangles => per-cell area = (horizontal run length) * (vertical run
  length). Run length is computed EXACTLY (zero multiplicity) by a 3-phase monotone
  walk over phases {S(tay-at-start), R(ight/down), L(eft/up)}: every walk is
  S^s -> dir^d, so each cell of the run is reached by exactly ONE walk =>
  Einsum value = run length on bg cells, 0 on walls/off-grid (seed mask kills them).
  19 steps cover runs up to 20 (measured max run = 20 over 20000 fresh).
- bg16 = Cast(Slice(input ch0 20x20)) : the bg plane, fp16, doubles as walk mask
  (all run cells are bg; mask is idempotent under S-stays).
- width16 = Einsum(bg16, eS, (PHI,V,bg16)*19) [1,1,20,20] fp16 800B (43 letters)
- area16  = Einsum(width16, eS, (PHI,V,bg16)*19 over ROW letters) = w*h, 800B
  V is shared between axes (R slice = +1 shift = right/down, L = -1 = left/up).
  Exact small integers (area <= 360 measured; fp16 exact <= 2048), operands nonneg.
- selection: maxa = ReduceMax(area); mina = ReduceMin(Where(bg, area, 1024));
  Equal masks (walls/off-grid have area 0, never equal to maxa/mina >= 1).
- label u8 Where-chain; in-grid mask from input row/col profiles (ReduceMax);
  Pad(10) to 30x30; Equal(levels) -> free bool output.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION

LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
STEPS = 19
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- entry: bg plane (channel 0) over the 20x20 window, fp16
    init('z_st', np.zeros(4, np.int64), np.int64)
    init('z_en', np.array([1, 1, 20, 20], np.int64), np.int64)
    n('Slice', ['input', 'z_st', 'z_en'], 'bg20')
    n('Cast', ['bg20'], 'bg16', to=F16)

    # ---- walk machine params (shared by both einsums)
    # phases: 0=S(tay at start), 1=R(+1 shift), 2=L(-1 shift)
    PHI = np.zeros((3, 3), np.float16)
    PHI[0, 0] = PHI[0, 1] = PHI[0, 2] = PHI[1, 1] = PHI[2, 2] = 1.0
    init('PHI', PHI, np.float16)
    V = np.zeros((3, 20, 20), np.float16)
    V[0] = np.eye(20)
    V[1, np.arange(19), np.arange(1, 20)] = 1.0   # +1 shift (right / down)
    V[2, np.arange(1, 20), np.arange(19)] = 1.0   # -1 shift (left / up)
    init('V', V, np.float16)
    init('eS', np.array([1, 0, 0], np.float16), np.float16)

    def walk_einsum(seed_name, out_name, axis):
        """axis='col': walk over the last letter (horizontal runs);
        axis='row': walk over the row letter (vertical runs)."""
        pool = list(LETTERS)
        nl, zl, rl, cl = pool.pop(0), pool.pop(0), pool.pop(0), pool.pop(0)
        p_cur = pool.pop(0)
        subs = [nl + zl + rl + cl, p_cur]
        ops = [seed_name, 'eS']
        cur = cl if axis == 'col' else rl
        for _ in range(STEPS):
            p_new = pool.pop(0)
            x_new = pool.pop(0)
            subs.append(p_cur + p_new)
            ops.append('PHI')
            subs.append(p_new + cur + x_new)
            ops.append('V')
            p_cur, cur = p_new, x_new
            mask_sub = (nl + zl + rl + cur) if axis == 'col' else (nl + zl + cur + cl)
            subs.append(mask_sub)
            ops.append('bg16')
        equation = ','.join(subs) + '->' + nl + zl + rl + cl
        nodes.append(helper.make_node('Einsum', ops, [out_name], equation=equation))
        return out_name

    walk_einsum('bg16', 'width16', 'col')     # horizontal run length per cell
    walk_einsum('width16', 'area16', 'row')   # * vertical run length = room area

    # ---- min/max selection (walls & off-grid have area 0; rooms >= 1)
    n('Cast', ['bg20'], 'fmask', to=BOOL)
    init('big16', np.float16(1024.0), np.float16)
    n('Where', ['fmask', 'area16', 'big16'], 'afm')
    n('ReduceMax', ['area16'], 'maxa', keepdims=1)
    n('ReduceMin', ['afm'], 'mina', keepdims=1)
    n('Equal', ['area16', 'maxa'], 'maxm')
    n('Equal', ['area16', 'mina'], 'minm')

    # ---- in-grid mask from full-input row/col occupancy profiles
    n('ReduceMax', ['input'], 'rowp', axes=[1, 3], keepdims=1)   # [1,1,30,1]
    n('ReduceMax', ['input'], 'colp', axes=[1, 2], keepdims=1)   # [1,1,1,30]
    n('Cast', ['rowp'], 'rowb', to=BOOL)
    n('Cast', ['colp'], 'colb', to=BOOL)
    init('s0', np.array([0], np.int64), np.int64)
    init('e20', np.array([20], np.int64), np.int64)
    init('ax2', np.array([2], np.int64), np.int64)
    init('ax3', np.array([3], np.int64), np.int64)
    n('Slice', ['rowb', 's0', 'e20', 'ax2'], 'rows20')           # [1,1,20,1]
    n('Slice', ['colb', 's0', 'e20', 'ax3'], 'cols20')           # [1,1,1,20]
    n('And', ['rows20', 'cols20'], 'insideb')                    # [1,1,20,20]

    # ---- label chain (u8) and free output
    for name, v in [('zero_u8', 0), ('one_u8', 1), ('two_u8', 2),
                    ('eight_u8', 8), ('ten_u8', 10)]:
        init(name, np.uint8(v), np.uint8)
    n('Where', ['insideb', 'two_u8', 'ten_u8'], 'm10')   # in-grid default 2, off-grid 10
    n('Where', ['fmask', 'zero_u8', 'm10'], 'l1')        # bg -> 0 (walls stay 2, off 10)
    n('Where', ['maxm', 'one_u8', 'l1'], 'l2')           # max room(s) -> blue
    n('Where', ['minm', 'eight_u8', 'l2'], 'l3')         # min room(s) -> cyan
    init('pads', np.array([0, 0, 0, 0, 0, 0, 10, 10], np.int64), np.int64)
    n('Pad', ['l3', 'pads', 'ten_u8'], 'l30')            # [1,1,30,30], off-crop -> 10
    init('levels', np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n('Equal', ['l30', 'levels'], 'output')              # [1,10,30,30] bool, FREE

    graph = helper.make_graph(
        nodes,
        'task145_room_walk_einsum',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info('output', BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = IR_VERSION
    return model
