"""Build hardened candidates by in-place editing incumbent initializers.
Structure (nodes/shapes/opset/dtypes) is byte-preserved except initializer
raw_data, so mem/params are guaranteed identical.

Hardening = shift the decision boundary away from 0 for IN-GRID cells:
  bias-free nets (220,230): add -delta to W[out_ch, :, 1, 1] (all 10 input
    center taps). For an in-grid pixel exactly one input channel is 1 at the
    center, so sum_in input[in,center] = 1 -> adds -delta to in-grid logits,
    0 to padded (all-zero) cells. No new params/ops.
  biased net (294): subtract delta from every bias entry (uniform -delta).
delta < min_on and > 0 so decode is provably preserved AND on/off populations
(on>=min_on grid, off<=0 grid) both move >= delta away from the 0 threshold.
"""
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
sys.path.insert(0, ".")

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

def get_init(model, name):
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            return i, init
    raise KeyError(name)

def set_init(model, idx, arr):
    new = numpy_helper.from_array(arr.astype(np.float32), model.graph.initializer[idx].name)
    model.graph.initializer[idx].CopyFrom(new)

def harden_weight_centertap(inpath, outpath, wname, delta):
    m = onnx.load(inpath)
    idx, init = get_init(m, wname)
    W = numpy_helper.to_array(init).copy()  # (10,10,3,3)
    assert W.shape == (10, 10, 3, 3), W.shape
    W[:, :, 1, 1] -= delta  # all out ch, all in ch, center tap
    set_init(m, idx, W)
    onnx.save(m, outpath)
    print(f"saved {outpath} (center-tap -{delta})")

def harden_bias(inpath, outpath, bname, delta):
    m = onnx.load(inpath)
    idx, init = get_init(m, bname)
    b = numpy_helper.to_array(init).copy()
    b = b - delta
    set_init(m, idx, b)
    onnx.save(m, outpath)
    print(f"saved {outpath} (bias -{delta}), new bias={b.tolist()}")

if __name__ == "__main__":
    harden_weight_centertap("networks/task220.onnx", OUT / "task220_cand.onnx", "conv_weight", 0.5)
    harden_weight_centertap("networks/task230.onnx", OUT / "task230_cand.onnx", "conv_weight", 0.25)
    harden_bias("networks/task294.onnx", OUT / "task294_cand.onnx", "score_bias", 0.5)
