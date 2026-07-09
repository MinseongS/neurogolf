import onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto as TP

src = onnx.load('submission/overfit_nets/task168.onnx')
g = src.graph

# Keep every node up to and including 'ray_scores' (the [1,1,10,10] u8 ray mask).
# Drop: 'ray_b10' Cast, 'ray_b30' Pad, 'output' Where, and the color-onehot Mul? keep color.
keep = []
for n in g.node:
    keep.append(n)
    if n.output[0] == 'ray_scores':
        break
# color_presence / color_onehot_f nodes are needed -> ensure included
have = {o for n in keep for o in n.output}
extra = []
for n in g.node:
    if n.output[0] in ('color_presence','color_onehot_f') and n.output[0] not in have:
        extra.append(n)
# Reorder: color nodes depend only on input, safe to append after
nodes = list(keep) + [n for n in extra if n.output[0] not in have]

inits = {i.name: i for i in g.initializer}
new_inits = []
used = set()
def addinit(arr, name):
    if name in used: return name
    used.add(name)
    new_inits.append(numpy_helper.from_array(arr, name)); return name

# carry over initializers referenced by kept nodes
ref = set()
for n in nodes:
    ref.update(n.input)
for name in ref:
    if name in inits and name not in used:
        new_inits.append(inits[name]); used.add(name)

# --- Build the fold operands ---
# ray10 float [1,10,10]: Cast ray_scores(u8 [1,1,10,10]) -> float, Reshape [1,10,10]
nodes.append(helper.make_node('Cast', ['ray_scores'], ['ray10f_4d'], to=TP.FLOAT))
addinit(np.array([1,10,10], dtype=np.int64), 'shp_1_10_10')
nodes.append(helper.make_node('Reshape', ['ray10f_4d','shp_1_10_10'], ['ray10f']))
# R[2,10,10]: layer0=ones, layer1=ray10f
addinit(np.ones((1,10,10), dtype=np.float32), 'ones_1_10_10')
nodes.append(helper.make_node('Concat', ['ones_1_10_10','ray10f'], ['R'], axis=0))

# dcolor[10] = color_onehot_f_vec - e0 ; color_onehot_f is [1,10,1,1]
addinit(np.array([10], dtype=np.int64), 'shp_10')
nodes.append(helper.make_node('Reshape', ['color_onehot_f','shp_10'], ['color_vec']))
e0 = np.zeros(10, dtype=np.float32); e0[0]=1.0
addinit(e0, 'e0_vec')
nodes.append(helper.make_node('Sub', ['color_vec','e0_vec'], ['dcolor']))
# A1[10,10] = e0col(10,1) x dcolor(1,10)
addinit(e0.reshape(10,1).astype(np.float32), 'e0_col')  # [10,1]
addinit(np.array([1,10], dtype=np.int64), 'shp_1_10')
nodes.append(helper.make_node('Reshape', ['dcolor','shp_1_10'], ['dcolor_row']))
nodes.append(helper.make_node('MatMul', ['e0_col','dcolor_row'], ['A1_2d']))  # [10,10]
addinit(np.array([1,10,10], dtype=np.int64), 'shp_1_10_10b')
nodes.append(helper.make_node('Reshape', ['A1_2d','shp_1_10_10b'], ['A1']))
# A0 = eye [1,10,10]
addinit(np.eye(10, dtype=np.float32).reshape(1,10,10), 'eye_1_10_10')
nodes.append(helper.make_node('Concat', ['eye_1_10_10','A1'], ['A'], axis=0))

# Sel[30,10]: identity in top-left
sel = np.zeros((30,10), dtype=np.float32)
for k in range(10): sel[k,k]=1.0
addinit(sel, 'Sel_30_10')

# Final Einsum -> output
nodes.append(helper.make_node('Einsum',
    ['input','A','Sel_30_10','Sel_30_10','R'],
    ['output'], equation='ekHW,skc,Hh,Ww,shw->ecHW'))

ng = helper.make_graph(nodes, 'task168_regime', g.input, g.output, new_inits)
m = helper.make_model(ng, opset_imports=list(src.opset_import), ir_version=src.ir_version)
onnx.save(m, 'reports/candidates/task168/regime.onnx')
print('saved, nodes=', len(nodes), 'inits=', len(new_inits))
