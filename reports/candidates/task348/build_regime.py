import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Transformation (verified): vertical 7-line at column c0, rows 0..R (apex at (R,c0)).
# Output: checkerboard cone.  triangle = {r+c<=A} & {r-c<=B}, A=R+c0, B=R-c0.
# value 7 if |c-c0| even else 8.  Within r<=R region sign(s1i*s2i) gives membership.
# s1i=2A+1-2r-2c, s2i=2B+1-2r+2c ; poly = s1i*s2i = P_i[r]+Q_i[c]
#   P_i[r]=C0 + C1*r + 4 r^2 ; Q_i[c]=C2*c - 4 c^2 ; C0=(2A+1)(2B+1), C1=-4(2R+1), C2=8 c0
# Final free-output einsum: out[b,k,h,w] = sum_{c,m} input[b,c,h,w]*COEF[k,m]*RFAC[m,h]*CFAC[m,w]
#   monomials m: 0:RMP*ones 1:RM*Q 2:RMP*sig 3:RM*sigQ 4:RMc*ones
#   COEF k=7:[.5,.5,.5p,.5p,0] k=8:[.5,.5,-.5p,-.5p,0] k=0:[-1,-1,0,0,1]

N = 30
rr = np.arange(N, dtype=np.float32)
sig = ((-1.0) ** np.arange(N)).astype(np.float32)  # (-1)^c

def c(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)

inits = [
    c("e7", np.eye(10, dtype=np.float32)[7]),      # channel-7 onehot [10]
    c("rramp", rr),                                # [30]
    c("rsq", rr * rr),
    c("cramp", rr),
    c("csq", rr * rr),
    c("sig", sig),                                 # [30]
    c("ones30", np.ones(N, dtype=np.float32)),
    c("one", np.array(1.0, np.float32)),
    c("four", np.array(4.0, np.float32)),
    c("negfour", np.array(-4.0, np.float32)),
    c("eight", np.array(8.0, np.float32)),
    c("two", np.array(2.0, np.float32)),
    # COEF base and delta [10,5]
    c("coef_base", np.array([
        [-1,-1,0,0,1],   # 0
        [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],  #1-6
        [0.5,0.5,0,0,0], # 7
        [0.5,0.5,0,0,0], # 8
        [0,0,0,0,0],     # 9
    ], np.float32)),
    c("coef_delta", np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],
        [0,0,0.5,0.5,0],   # 7 : +0.5 p
        [0,0,-0.5,-0.5,0], # 8 : -0.5 p
        [0,0,0,0,0],
    ], np.float32)),
]

nodes = []
def nd(op, ins, outs, **kw):
    nodes.append(helper.make_node(op, ins, outs, **kw))

# row/col line profiles via einsum (no 30x30 intermediate)
nd("Einsum", ["input", "e7"], ["rowMask"], equation="bchw,c->bh")   # [1,30] = 1 for r<=R
nd("Einsum", ["input", "e7"], ["colLine"], equation="bchw,c->bw")   # [1,30] = L at c0
nd("ReduceSum", ["colLine"], ["L"], keepdims=0)                     # scalar L=R+1
nd("Sub", ["L", "one"], ["R"])
# c0 = sum_c c*colLine / L ; p = sum_c sig*colLine / L
nd("Einsum", ["colLine", "cramp"], ["sumc"], equation="bw,w->b")
nd("Div", ["sumc", "L"], ["c0"])
nd("Einsum", ["colLine", "sig"], ["sump"], equation="bw,w->b")
nd("Div", ["sump", "L"], ["p"])                                    # scalar (-1)^c0
# A=R+c0 B=R-c0
nd("Add", ["R", "c0"], ["A"])
nd("Sub", ["R", "c0"], ["B"])
# C0=(2A+1)(2B+1)
nd("Mul", ["A", "two"], ["A2"]); nd("Add", ["A2", "one"], ["A21"])
nd("Mul", ["B", "two"], ["B2"]); nd("Add", ["B2", "one"], ["B21"])
nd("Mul", ["A21", "B21"], ["C0"])
# C1=-4(2R+1)
nd("Mul", ["R", "two"], ["R2"]); nd("Add", ["R2", "one"], ["R21"])
nd("Mul", ["R21", "negfour"], ["C1"])
# C2=8 c0
nd("Mul", ["c0", "eight"], ["C2"])
# P_i[r]=C0 + C1*rramp + 4*rsq  -> broadcast scalars over [30]
nd("Mul", ["rramp", "C1"], ["c1r"])
nd("Mul", ["rsq", "four"], ["fr2"])
nd("Add", ["c1r", "fr2"], ["pr_a"])
nd("Add", ["pr_a", "C0"], ["P_i"])                # [1,30]? C0 scalar broadcast -> shape [1,30]
# Q_i[c]=C2*cramp -4*csq
nd("Mul", ["cramp", "C2"], ["c2c"])
nd("Mul", ["csq", "negfour"], ["nf c2"]) if False else nd("Mul", ["csq", "negfour"], ["nfc2"])
nd("Add", ["c2c", "nfc2"], ["Q_i"])               # [1,30]
# RM, RMc, RMP, sigQ
nd("Identity", ["rowMask"], ["RM"])
nd("Sub", ["ones30", "RM"], ["RMc"])
nd("Mul", ["RM", "P_i"], ["RMP"])
nd("Mul", ["sig", "Q_i"], ["sigQ"])
# Build RFAC[5,30], CFAC[5,30] via Concat of reshaped [1,30] rows
def row(name, src):
    nd("Reshape", [src, "shape1_30"], [name])
inits.append(numpy_helper.from_array(np.array([1,30], np.int64), "shape1_30"))
row("r0", "RMP"); row("r1", "RM"); row("r2", "RMP"); row("r3", "RM"); row("r4", "RMc")
nd("Concat", ["r0","r1","r2","r3","r4"], ["RFAC"], axis=0)
row("k0", "ones30"); row("k1", "Q_i"); row("k2", "sig"); row("k3", "sigQ"); row("k4", "ones30")
nd("Concat", ["k0","k1","k2","k3","k4"], ["CFAC"], axis=0)
# COEF = coef_base + p*coef_delta
nd("Mul", ["coef_delta", "p"], ["coef_pd"])
nd("Add", ["coef_base", "coef_pd"], ["COEF"])
# final free-output einsum
nd("Einsum", ["input", "COEF", "RFAC", "CFAC"], ["output"], equation="bchw,km,mh,mw->bkhw")

graph = helper.make_graph(
    nodes, "task348_regime",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,10,30,30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,10,30,30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.save(model, "reports/candidates/task348/regime.onnx")
print("saved")
