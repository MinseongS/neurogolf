"""Task 234 recast candidate — fp16 final routing + uint8 presence profiles.

The final `Einsum(['weight','rsel','csel'], 'tnk,tr,tc->nkrc')` has NO fp32 `input`
operand, so all three operands + their fp16 upstream (onehot_f, minus_bg, weight310,
weight) can be jointly recast to fp16 and the einsum runs / outputs fp16. Values are
small integers ({-1,0,1} for weight, {0,1} for rsel/csel), fp16-exact; grader
thresholds output>0.0 so bit-identical.

Recast tensors: submask/onehot0row (inits->fp16), onehot_f, minus_bg, weight310,
weight, rsel, csel -> fp16. Save 600B (rsel 180 + csel 180 + weight/weight310/
onehot_f/minus_bg 60 each).

S12 follow-up: the `Sign(profile)` planes feeding `ArgMax` only encode nonzero
presence.  `Greater(profile,0) -> Cast(uint8)` is exact and accepted by ArgMax,
so each 30-vector costs 60B instead of a 120B fp32 Sign plane.  Applied to
present9/rp0/cp0/rp1/cp1.

NOT recast (dtype-bound): g0/g1/rp*/cp*/present* -> einsum co-operand is the fp32
free input (rp0=Einsum(input,g0)); scalar Where/Concat chain stays fp32.
"""
import numpy as np

from onnx import TensorProto, helper, numpy_helper
from ._exact import arr_b64, model, tensor

F16 = TensorProto.FLOAT16


def build(task):
    inits = [
        tensor('slice1_start', np.array([1], dtype=np.int64)),
        tensor('slice10_end', np.array([10], dtype=np.int64)),
        tensor('one_i64', np.array([1], dtype=np.int64)),
        tensor('one_f', np.array([1.0], dtype=np.float32)),
        tensor('arange30', np.arange(30, dtype=np.float32).reshape(1, 30)),
        tensor('arange10', np.arange(10, dtype=np.float32).reshape(1, 10)),
        tensor('zerof1', np.array([0.0], dtype=np.float32)),
        tensor('submask', np.array([[0.0], [1.0], [1.0]], dtype=np.float16)),
        tensor('shape1', np.array([1], dtype=np.int64)),
        tensor('shape31', np.array([3, 1], dtype=np.int64)),
        tensor('onehot0row', np.eye(10, dtype=np.float16)[0].reshape(1, 10)),
        tensor('shape3_1_10', np.array([3, 1, 10], dtype=np.int64)),
    ]
    nodes = [
        # ---- presence / object colours from pixel counts ----
        helper.make_node('ReduceSum', ['input'], ['pixel_all'], axes=[0, 2, 3], keepdims=0),
        helper.make_node('Slice', ['pixel_all', 'slice1_start', 'slice10_end'], ['present9']),
        helper.make_node('Greater', ['present9', 'zerof1'], ['present9s_b']),
        helper.make_node('Cast', ['present9s_b'], ['present9s'], to=2),
        helper.make_node('ArgMax', ['present9s'], ['color0_idx'], axis=0, keepdims=0),
        helper.make_node('ArgMax', ['present9s'], ['color1_idx'], axis=0, keepdims=0, select_last_index=1),
        helper.make_node('Add', ['color0_idx', 'one_i64'], ['color0']),
        helper.make_node('Add', ['color1_idx', 'one_i64'], ['color1']),
        # ---- one-hot colour gates ----
        helper.make_node('Cast', ['color0'], ['color0_g'], to=1),
        helper.make_node('Cast', ['color1'], ['color1_g'], to=1),
        helper.make_node('Equal', ['arange10', 'color0_g'], ['g0_b']),
        helper.make_node('Equal', ['arange10', 'color1_g'], ['g1_b']),
        helper.make_node('Cast', ['g0_b'], ['g0'], to=1),
        helper.make_node('Cast', ['g1_b'], ['g1'], to=1),
        # ---- per-object row/col count profiles (free input) -> sign -> argmax bbox ----
        helper.make_node('Einsum', ['input', 'g0'], ['rp0'], equation='bchw,xc->h'),
        helper.make_node('Einsum', ['input', 'g0'], ['cp0'], equation='bchw,xc->w'),
        helper.make_node('Einsum', ['input', 'g1'], ['rp1'], equation='bchw,xc->h'),
        helper.make_node('Einsum', ['input', 'g1'], ['cp1'], equation='bchw,xc->w'),
        helper.make_node('Greater', ['rp0', 'zerof1'], ['rp0s_b']),
        helper.make_node('Cast', ['rp0s_b'], ['rp0s'], to=2),
        helper.make_node('Greater', ['cp0', 'zerof1'], ['cp0s_b']),
        helper.make_node('Cast', ['cp0s_b'], ['cp0s'], to=2),
        helper.make_node('Greater', ['rp1', 'zerof1'], ['rp1s_b']),
        helper.make_node('Cast', ['rp1s_b'], ['rp1s'], to=2),
        helper.make_node('Greater', ['cp1', 'zerof1'], ['cp1s_b']),
        helper.make_node('Cast', ['cp1s_b'], ['cp1s'], to=2),
        helper.make_node('ArgMax', ['rp0s'], ['obj0_r0'], axis=0, keepdims=0),
        helper.make_node('ArgMax', ['rp0s'], ['obj0_r1'], axis=0, keepdims=0, select_last_index=1),
        helper.make_node('ArgMax', ['cp0s'], ['obj0_c0'], axis=0, keepdims=0),
        helper.make_node('ArgMax', ['cp0s'], ['obj0_c1'], axis=0, keepdims=0, select_last_index=1),
        helper.make_node('ArgMax', ['rp1s'], ['obj1_r0'], axis=0, keepdims=0),
        helper.make_node('ArgMax', ['rp1s'], ['obj1_r1'], axis=0, keepdims=0, select_last_index=1),
        helper.make_node('ArgMax', ['cp1s'], ['obj1_c0'], axis=0, keepdims=0),
        helper.make_node('ArgMax', ['cp1s'], ['obj1_c1'], axis=0, keepdims=0, select_last_index=1),
        # ---- pixel counts per object ----
        helper.make_node('Gather', ['pixel_all', 'color0'], ['obj0_pixel'], axis=0),
        helper.make_node('Gather', ['pixel_all', 'color1'], ['obj1_pixel'], axis=0),
        # ---- grid extent (any colour) ----
        helper.make_node('ReduceSum', ['input'], ['anyrow'], axes=[0, 1, 3], keepdims=0),
        helper.make_node('ReduceSum', ['input'], ['anycol'], axes=[0, 1, 2], keepdims=0),
        helper.make_node('ArgMax', ['anyrow'], ['bg_r1'], axis=0, keepdims=0, select_last_index=1),
        helper.make_node('ArgMax', ['anycol'], ['bg_c1'], axis=0, keepdims=0, select_last_index=1),
        # ---- unchanged incumbent logic from here on ----
        helper.make_node('Cast', ['obj0_r0'], ['obj0_r0_f'], to=1),
        helper.make_node('Cast', ['obj0_r1'], ['obj0_r1_f'], to=1),
        helper.make_node('Cast', ['obj0_c0'], ['obj0_c0_f'], to=1),
        helper.make_node('Cast', ['obj0_c1'], ['obj0_c1_f'], to=1),
        helper.make_node('Sub', ['obj0_r1_f', 'obj0_r0_f'], ['obj0_bbox_h0']),
        helper.make_node('Add', ['obj0_bbox_h0', 'one_f'], ['obj0_bbox_h']),
        helper.make_node('Sub', ['obj0_c1_f', 'obj0_c0_f'], ['obj0_bbox_w0']),
        helper.make_node('Add', ['obj0_bbox_w0', 'one_f'], ['obj0_bbox_w']),
        helper.make_node('Mul', ['obj0_bbox_h', 'obj0_bbox_w'], ['obj0_bbox_area']),
        helper.make_node('Less', ['obj0_pixel', 'obj0_bbox_area'], ['obj0_armed']),
        helper.make_node('Where', ['obj0_armed', 'color0', 'color1'], ['arm_color']),
        helper.make_node('Where', ['obj0_armed', 'color1', 'color0'], ['pure_color']),
        helper.make_node('Where', ['obj0_armed', 'obj0_pixel', 'obj1_pixel'], ['arm_pixel']),
        helper.make_node('Where', ['obj0_armed', 'obj0_r0', 'obj1_r0'], ['r0']),
        helper.make_node('Where', ['obj0_armed', 'obj0_r1', 'obj1_r1'], ['r1']),
        helper.make_node('Where', ['obj0_armed', 'obj0_c0', 'obj1_c0'], ['c0']),
        helper.make_node('Where', ['obj0_armed', 'obj0_c1', 'obj1_c1'], ['c1']),
        helper.make_node('Where', ['obj0_armed', 'obj1_r0', 'obj0_r0'], ['pure_r0']),
        helper.make_node('Where', ['obj0_armed', 'obj1_r1', 'obj0_r1'], ['pure_r1']),
        helper.make_node('Where', ['obj0_armed', 'obj1_c0', 'obj0_c0'], ['pure_c0']),
        helper.make_node('Where', ['obj0_armed', 'obj1_c1', 'obj0_c1'], ['pure_c1']),
        helper.make_node('Cast', ['r0'], ['r0_f'], to=1),
        helper.make_node('Cast', ['r1'], ['r1_f'], to=1),
        helper.make_node('Cast', ['c0'], ['c0_f'], to=1),
        helper.make_node('Cast', ['c1'], ['c1_f'], to=1),
        helper.make_node('Cast', ['pure_r0'], ['pure_r0_f'], to=1),
        helper.make_node('Cast', ['pure_r1'], ['pure_r1_f'], to=1),
        helper.make_node('Cast', ['pure_c0'], ['pure_c0_f'], to=1),
        helper.make_node('Cast', ['pure_c1'], ['pure_c1_f'], to=1),
        helper.make_node('Sub', ['r1_f', 'r0_f'], ['bbox_h0']),
        helper.make_node('Add', ['bbox_h0', 'one_f'], ['bbox_h']),
        helper.make_node('Sub', ['c1_f', 'c0_f'], ['bbox_w0']),
        helper.make_node('Add', ['bbox_w0', 'one_f'], ['bbox_w']),
        helper.make_node('Less', ['pure_r1', 'r0'], ['top_arm']),
        helper.make_node('Greater', ['pure_r0', 'r1'], ['bottom_arm']),
        helper.make_node('Less', ['pure_c1', 'c0'], ['left_arm']),
        helper.make_node('Greater', ['pure_c0', 'c1'], ['right_arm']),
        helper.make_node('Or', ['top_arm', 'bottom_arm'], ['vertical']),
        helper.make_node('Sub', ['bbox_h', 'one_f'], ['bbox_h_m1']),
        helper.make_node('Sub', ['bbox_w', 'one_f'], ['bbox_w_m1']),
        helper.make_node('Sub', ['arm_pixel', 'bbox_h'], ['vertical_num']),
        helper.make_node('Sub', ['arm_pixel', 'bbox_w'], ['horizontal_num']),
        helper.make_node('Where', ['vertical', 'vertical_num', 'horizontal_num'], ['rect_num']),
        helper.make_node('Where', ['vertical', 'bbox_w_m1', 'bbox_h_m1'], ['rect_den']),
        helper.make_node('Div', ['rect_num', 'rect_den'], ['rect_extent']),
        helper.make_node('Where', ['vertical', 'rect_extent', 'bbox_h'], ['rect_h']),
        helper.make_node('Where', ['vertical', 'bbox_w', 'rect_extent'], ['rect_w']),
        helper.make_node('Sub', ['rect_h', 'one_f'], ['rect_h_m1']),
        helper.make_node('Sub', ['rect_w', 'one_f'], ['rect_w_m1']),
        helper.make_node('Sub', ['r1_f', 'rect_h_m1'], ['row_bottom_start']),
        helper.make_node('Add', ['r0_f', 'rect_h_m1'], ['row_top_end']),
        helper.make_node('Sub', ['c1_f', 'rect_w_m1'], ['col_right_start']),
        helper.make_node('Add', ['c0_f', 'rect_w_m1'], ['col_left_end']),
        helper.make_node('Where', ['bottom_arm', 'row_bottom_start', 'r0_f'], ['row_out_start']),
        helper.make_node('Where', ['top_arm', 'row_top_end', 'r1_f'], ['row_out_end']),
        helper.make_node('Where', ['right_arm', 'col_right_start', 'c0_f'], ['col_out_start']),
        helper.make_node('Where', ['left_arm', 'col_left_end', 'c1_f'], ['col_out_end']),
        helper.make_node('Cast', ['bg_r1'], ['bg_r1_f'], to=1),
        helper.make_node('Cast', ['bg_c1'], ['bg_c1_f'], to=1),
        helper.make_node('Cast', ['pure_color'], ['pure_color_f'], to=1),
        helper.make_node('Cast', ['arm_color'], ['arm_color_f'], to=1),
        helper.make_node('Reshape', ['pure_color_f', 'shape1'], ['pure_color_f_1']),
        helper.make_node('Reshape', ['arm_color_f', 'shape1'], ['arm_color_f_1']),
        helper.make_node('Reshape', ['bg_r1_f', 'shape1'], ['bg_r1_f_1']),
        helper.make_node('Reshape', ['bg_c1_f', 'shape1'], ['bg_c1_f_1']),
        helper.make_node('Reshape', ['pure_r0_f', 'shape1'], ['pure_r0_f_1']),
        helper.make_node('Reshape', ['pure_r1_f', 'shape1'], ['pure_r1_f_1']),
        helper.make_node('Reshape', ['pure_c0_f', 'shape1'], ['pure_c0_f_1']),
        helper.make_node('Reshape', ['pure_c1_f', 'shape1'], ['pure_c1_f_1']),
        helper.make_node('Reshape', ['row_out_start', 'shape1'], ['row_out_start_1']),
        helper.make_node('Reshape', ['row_out_end', 'shape1'], ['row_out_end_1']),
        helper.make_node('Reshape', ['col_out_start', 'shape1'], ['col_out_start_1']),
        helper.make_node('Reshape', ['col_out_end', 'shape1'], ['col_out_end_1']),
        helper.make_node('Concat', ['zerof1', 'pure_color_f_1', 'arm_color_f_1'], ['colors3'], axis=0),
        helper.make_node('Reshape', ['colors3', 'shape31'], ['colors31']),
        helper.make_node('Equal', ['colors31', 'arange10'], ['onehot_eq']),
        # ---- fp16 weight path (feeds final free-output einsum; no fp32 input operand) ----
        helper.make_node('Cast', ['onehot_eq'], ['onehot_f'], to=10),
        helper.make_node('Mul', ['submask', 'onehot0row'], ['minus_bg']),
        helper.make_node('Sub', ['onehot_f', 'minus_bg'], ['weight310']),
        helper.make_node('Reshape', ['weight310', 'shape3_1_10'], ['weight']),
        helper.make_node('Concat', ['zerof1', 'pure_r0_f_1', 'row_out_start_1'], ['rstart3'], axis=0),
        helper.make_node('Concat', ['bg_r1_f_1', 'pure_r1_f_1', 'row_out_end_1'], ['rend3'], axis=0),
        helper.make_node('Concat', ['zerof1', 'pure_c0_f_1', 'col_out_start_1'], ['cstart3'], axis=0),
        helper.make_node('Concat', ['bg_c1_f_1', 'pure_c1_f_1', 'col_out_end_1'], ['cend3'], axis=0),
        helper.make_node('Reshape', ['rstart3', 'shape31'], ['rstart3b']),
        helper.make_node('Reshape', ['rend3', 'shape31'], ['rend3b']),
        helper.make_node('Reshape', ['cstart3', 'shape31'], ['cstart3b']),
        helper.make_node('Reshape', ['cend3', 'shape31'], ['cend3b']),
        helper.make_node('GreaterOrEqual', ['arange30', 'rstart3b'], ['rge']),
        helper.make_node('LessOrEqual', ['arange30', 'rend3b'], ['rle']),
        helper.make_node('And', ['rge', 'rle'], ['rsel_b']),
        helper.make_node('Cast', ['rsel_b'], ['rsel'], to=10),
        helper.make_node('GreaterOrEqual', ['arange30', 'cstart3b'], ['cge']),
        helper.make_node('LessOrEqual', ['arange30', 'cend3b'], ['cle']),
        helper.make_node('And', ['cge', 'cle'], ['csel_b']),
        helper.make_node('Cast', ['csel_b'], ['csel'], to=10),
        helper.make_node('Einsum', ['weight', 'rsel', 'csel'], ['output'], equation='tnk,tr,tc->nkrc'),
    ]
    return model('task234_recast', nodes, inits, output_dtype=F16, opset=12, value_infos=[])
