"""Task 042 — mechanism-16 (runtime-parameterized stamp) redesign attempt.

REPORT-ONLY candidate.  This is the best-case mech-16 STRUCTURE for task042:
detect the single global magnification m in {1,2,3} from the green-cell count,
then assemble ONE matched-filter kernel at runtime (ScatterND base_idx*m into a
fixed centered zeros tensor) and apply ONE Conv, replacing the incumbent's
m1/m2/m3 template bank + Max.

VERDICT: KILL.  Two independent walls:

(1) PARAMETRIC-FAMILY IS FALSE.  The three matched filters are NOT a scale
    family.  A fitted m=1 kernel Kronecker-upscaled by m fails 1500/1500 fresh
    grids at m=2 and m=3 (cell-err 3.5-5%).  A single fitted conv even at native
    m=1 fails ~106/1500 (that is exactly why the incumbent m=1 path is a
    two-stage pair-conv + ConvTranspose, not a single conv).  There is no single
    K(m) that reproduces detect+block-fill across scales; you would still need
    3 independently-fitted kernels = the incumbent bank.  So this candidate
    FAILS the fresh gate by construction.

(2) COST FLOOR.  The incumbent already folds detect+stamp into compact fitted
    single convs (m2: 81 params/200B plane, m3: 143 params/200B plane).  A clean
    parametric kernel cannot fill the m x m cyan block from a single conv (that
    block-fill came from fitting), so a correct clean build needs TWO stages
    (anchor-detect + m x m stamp).  The runtime kernel must be a FIXED [1,1,K,K]
    (Resize-runtime-scales makes the kernel shape dynamic -> grader shape
    inference returns None), K>=13 for m=3 => ~338B counted plane, PLUS the
    int64 ScatterND index buffer (counted) PLUS the stamp stage.  These new
    intermediates ~= the ~992B of bank-plane savings.  The two dominant costs
    (green_f 400B fp32 entry crop + cyan30 900B routing mask) are structural and
    identical in every approach; the incumbent (mem 2792 / params 318, cost 3110)
    already sits at that floor.

The kernel taps below are the fitted m=1 significant offsets scaled by m; they
are placeholders for cost measurement only (correctness already refuted above).
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import tensor, model


def build(task):
    K = 13          # fixed centered kernel, covers m=3 offsets (+-6)
    C = K // 2      # center = 6

    # 8 "knight" green-pair offsets of the m=1 matched filter (unit scale).
    offs = np.array([[-2, 1], [-1, 2], [1, -2], [2, -1],
                     [-2, -1], [-1, -2], [1, 2], [2, 1]], dtype=np.int64)
    T = offs.shape[0]

    inits = [
        # green channel (index 3) slice -> [1,1,10,10]
        tensor('starts_0', np.array([3, 0, 0], dtype=np.int64)),
        tensor('ends_1', np.array([4, 10, 10], dtype=np.int64)),
        tensor('axes_2', np.array([1, 2, 3], dtype=np.int64)),
        # m-probe thresholds (green count 2|4 -> m1, 8|16 -> m2, 18|36 -> m3)
        tensor('thr4', np.array(4.0, dtype=np.float16)),
        tensor('thr16', np.array(16.0, dtype=np.float16)),
        # runtime-kernel assembly
        tensor('kzeros', np.zeros((1, 1, K, K), dtype=np.float16)),
        tensor('kones', np.ones((T,), dtype=np.float16)),
        tensor('offs', offs),                                  # [T,2]
        tensor('center', np.array([C, C], dtype=np.int64)),    # [2]
        tensor('nc_zeros', np.zeros((T, 2), dtype=np.int64)),  # [T,2] leading n,c index
        tensor('conv_bias', np.array([-1.5], dtype=np.float16)),
        # epilogue
        tensor('pads_11', np.array([0, 0, 20, 20], dtype=np.int64)),
        tensor('axes_12', np.array([2, 3], dtype=np.int64)),
        tensor('cyan_rep', np.zeros((1, 10, 1, 1), dtype=np.float32)),
    ]
    # set cyan (colour 8) replacement channel
    rep = inits[-1]
    # rebuild cyan_rep with channel 8 = 1.0
    rep_arr = np.zeros((1, 10, 1, 1), dtype=np.float32)
    rep_arr[0, 8, 0, 0] = 1.0
    inits[-1] = tensor('cyan_rep', rep_arr)

    nodes = [
        helper.make_node('Slice', ['input', 'starts_0', 'ends_1', 'axes_2'], ['green_f']),
        helper.make_node('Cast', ['green_f'], ['green_f16'], to=10),
        # --- detect global m from green count (scalars) ---
        helper.make_node('ReduceSum', ['green_f16'], ['gcount'], keepdims=0),
        helper.make_node('Greater', ['gcount', 'thr4'], ['g4']),
        helper.make_node('Greater', ['gcount', 'thr16'], ['g16']),
        helper.make_node('Cast', ['g4'], ['g4i'], to=TensorProto.INT64),
        helper.make_node('Cast', ['g16'], ['g16i'], to=TensorProto.INT64),
        helper.make_node('Add', ['g4i', 'g16i'], ['m_sum']),      # 0..2
        # m = 1 + g4 + g16  (scalar int64)
        helper.make_node('Add', ['m_sum', 'one_i'], ['m']),
        # --- assemble kernel: idx = center + offs*m ; ScatterND ones ---
        helper.make_node('Mul', ['offs', 'm'], ['offs_scaled']),  # [T,2]
        helper.make_node('Add', ['offs_scaled', 'center'], ['rc']),  # [T,2]
        helper.make_node('Concat', ['nc_zeros', 'rc'], ['idx'], axis=1),  # [T,4]
        helper.make_node('ScatterND', ['kzeros', 'idx', 'kones'], ['kernel']),
        # --- single matched conv ---
        helper.make_node('Conv', ['green_f16', 'kernel', 'conv_bias'], ['score'],
                         pads=[C, C, C, C]),
        helper.make_node('Greater', ['score', 'zero_h'], ['cyan']),
        helper.make_node('Pad', ['cyan', 'pads_11', '', 'axes_12'], ['cyan30'], mode='constant'),
        helper.make_node('Where', ['cyan30', 'cyan_rep', 'input'], ['output']),
    ]
    inits += [
        tensor('one_i', np.array(1, dtype=np.int64)),
        tensor('zero_h', np.array(0.0, dtype=np.float16)),
    ]
    return model('task042_rts', nodes, inits, output_dtype=TensorProto.FLOAT,
                 opset=18, value_infos=[])
