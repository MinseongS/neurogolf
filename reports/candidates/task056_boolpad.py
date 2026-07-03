"""Task 056 boolpad candidate — opset-13 bool Pad (drop the fp16 Cast).

Attempt: the incumbent casts active_bool(BOOL[1,6,1,1]) -> fp16 (12B plane) only so
the opset-9 Pad accepts it. bool Pad is legal at opset 13, so Pad the bool directly and
let the graph output be BOOL[1,10,30,30] (free final output; grader decodes out>0.0).

CATCH (this is why it is built — to MEASURE the loss): opset 9 -> 13 also converts the two
Slice nodes (starts/ends/axes) and the Pad (pads) from FREE node-attributes into COUNTED
int64 initializers (calculate_params counts init.dims, not node attrs). That adds ~26 params.
"""
import numpy as np
from onnx import TensorProto, helper
from src.custom._exact import model


def build(task):
    def i64(name, vals):
        return helper.make_tensor(name, TensorProto.INT64, [len(vals)], vals)

    inits = [
        # Slice-13 input-form starts/ends/axes (were free attrs at opset 9)
        i64('s_start', [0, 0, 0]), i64('s_end', [1, 1, 1]), i64('s_axes', [1, 2, 3]),
        i64('b_start', [0, 0, 2]), i64('b_end', [1, 1, 3]),  # shares s_axes
        # Pad-13 input-form pads (was a free attr at opset 9)
        i64('pad_amt', [0, 1, 0, 0, 0, 3, 29, 29]),
    ]
    nodes = [
        helper.make_node('Slice', ['input', 's_start', 's_end', 's_axes'], ['b0_f']),
        helper.make_node('Slice', ['input', 'b_start', 'b_end', 's_axes'], ['b2_f']),
        helper.make_node('Greater', ['b2_f', 'b0_f'], ['h1']),
        helper.make_node('Greater', ['b0_f', 'b2_f'], ['h3']),
        helper.make_node('Equal', ['h1', 'h3'], ['equal_bits']),
        helper.make_node('Cast', ['b0_f'], ['b0_bool'], to=9),
        helper.make_node('Not', ['b0_bool'], ['not_b0']),
        helper.make_node('And', ['equal_bits', 'not_b0'], ['h2']),
        helper.make_node('And', ['equal_bits', 'b0_bool'], ['h6']),
        helper.make_node('And', ['h1', 'h3'], ['false_bool']),
        helper.make_node('Concat', ['h1', 'h2', 'h3', 'false_bool', 'false_bool', 'h6'], ['active_bool'], axis=1),
        # bool Pad directly (opset 13) -> free BOOL output
        helper.make_node('Pad', ['active_bool', 'pad_amt'], ['output'], mode='constant'),
    ]
    return model('task056_boolpad', nodes, inits, output_dtype=TensorProto.BOOL, opset=13, value_infos=[])
