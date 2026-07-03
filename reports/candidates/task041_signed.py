"""Task 041 candidate — FLOOR CONTROL (bit-identical reconstruction).

Investigation result: task041 is already at its mechanism floor (1730 mem / 59
params = 1789B, 17.5106 pts).  BOTH proposed levers were refuted:

  A) fp16 recast of the fp32 Conv index plane `color_f` [1,1,10,10] (400B):
     REFUTED.  ORT Conv binds output dtype to input dtype, and the only fp32
     source is the FREE graph input.  Emitting fp16 requires either
       - an extra fp16 Cast AFTER the fp32 Conv  -> plane is ADDED (+200B -> 1930), or
       - a fp16 Cast of the [1,10,30,30] input   -> 18000B copy (-> 19530).
     Conv(fp16 w, fp32 in) errors (type mismatch).  No cheaper float
     channel-collapse exists; the 400B fp32 index plane is structural.

  B) task259 sub-900 carrier trick: REAL but NOT portable.  task259's output
     grids are 3x3, so it Pads the full 10-channel one-hot at content res
     ([1,10,3,3]=90B) straight to the free output.  For task041's 10x10 content
     that same structure costs [1,10,10,10]=1000B (Equal-then-Pad) vs the current
     900B (Pad single-channel index then free Equal).  Crossover: 10*h*w < 900
     <=> h*w < 90.  task041 (h*w=100) sits just past it -> 259 route is +100B worse.

This file reproduces the live graph EXACTLY (verified 0 divergence, 1730/59) so
the anatomy above is grounded in a runnable control, not just prose.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import model, tensor


def build(task):
    conv_w = np.zeros((1, 10, 2, 2), np.float32)
    for k in range(10):
        conv_w[0, k, 0, 0] = k
    inits = [
        tensor('conv_w', conv_w),
        tensor('pad_pads', np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64)),
        tensor('outside_u8', np.uint8(10)),
        tensor('channel_ids', np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
    ]
    nodes = [
        helper.make_node('Conv', ['input', 'conv_w'], ['color_f'], dilations=[20, 20]),
        helper.make_node('Cast', ['color_f'], ['colors'], to=2),
        helper.make_node('Split', ['colors'], [f'col{i}' for i in range(10)], axis=3, num_outputs=10),
        helper.make_node('BitwiseXor', ['col0', 'col1'], ['px1']),
        helper.make_node('BitwiseXor', ['px1', 'col2'], ['px2']),
        helper.make_node('BitwiseXor', ['px2', 'col3'], ['px3']),
        helper.make_node('BitwiseXor', ['px3', 'col4'], ['px4']),
        helper.make_node('BitwiseXor', ['px4', 'col5'], ['px5']),
        helper.make_node('BitwiseXor', ['px5', 'col6'], ['px6']),
        helper.make_node('BitwiseXor', ['px6', 'col7'], ['px7']),
        helper.make_node('BitwiseOr', ['px1', 'col2'], ['y2']),
        helper.make_node('BitwiseOr', ['px2', 'col3'], ['y3']),
        helper.make_node('BitwiseOr', ['px3', 'col4'], ['y4']),
        helper.make_node('BitwiseOr', ['px4', 'col5'], ['y5']),
        helper.make_node('BitwiseOr', ['px5', 'col6'], ['y6']),
        helper.make_node('BitwiseOr', ['px6', 'col7'], ['y7']),
        helper.make_node('Concat', ['col0', 'px1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'px7', 'col9'], ['y_u8'], axis=3),
        helper.make_node('Pad', ['y_u8', 'pad_pads', 'outside_u8'], ['y_full_u8'], mode='constant'),
        helper.make_node('Equal', ['y_full_u8', 'channel_ids'], ['output']),
    ]
    return model('task041_floor_control', nodes, inits, output_dtype=9, opset=18)
