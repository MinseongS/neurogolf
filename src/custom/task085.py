"""task085 v6: bool entry cast (900) instead of fp16 (1800) before slice.
occ32 {0,1} -> Cast bool (900) -> Slice (480 bool) -> Cast fp16 (960).
"""
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
from ..harness import IR_VERSION
F=TensorProto.FLOAT; H16=TensorProto.FLOAT16; BOOL=TensorProto.BOOL
HROWS=16
def build(task):
    inits,nodes=[],[]
    def init(nm,a,d): inits.append(numpy_helper.from_array(np.ascontiguousarray(a,dtype=d),nm)); return nm
    def n(op,i,o,**k): nodes.append(helper.make_node(op,i,[o],**k)); return o
    w_occ=np.zeros((1,10,1,1),np.float32); w_occ[0,1:,0,0]=1.0; init("w_occ",w_occ,np.float32)
    n("Conv",["input","w_occ"],"occ32",kernel_shape=[1,1])
    n("Cast",["occ32"],"occ_bool",to=BOOL)                # [1,1,30,30] bool = 900
    init("sl_st",np.array([0,0,0,0],np.int64),np.int64)
    init("sl_en",np.array([1,1,HROWS,30],np.int64),np.int64)
    init("sl_ax",np.array([0,1,2,3],np.int64),np.int64)
    n("Slice",["occ_bool","sl_st","sl_en","sl_ax"],"occb") # [1,1,16,30] bool = 480
    n("Cast",["occb"],"occ",to=H16)                        # [1,1,16,30] f16 = 960

    ramp=(30-np.arange(30)).astype(np.float16).reshape(1,1,1,30); init("ramp",ramp,np.float16)
    n("Mul",["occ","ramp"],"occr")
    n("ReduceMax",["occr"],"startw",axes=[3],keepdims=1)
    init("two16",np.array(2.0,np.float16),np.float16)
    n("Mod",["startw","two16"],"sparf",fmod=1)
    init("half16",np.array(0.5,np.float16),np.float16)
    n("Greater",["sparf","half16"],"spar_b")
    cpar=(np.arange(30)%2).astype(bool).reshape(1,1,1,30); init("cpar",cpar,np.bool_)
    n("Xor",["cpar","spar_b"],"odd_b")

    U=np.tril(np.ones((HROWS,HROWS),np.float16)); init("Umat",U,np.float16)
    n("MatMul",["Umat","occ"],"vpre")
    init("three16",np.array(3.0,np.float16),np.float16)
    n("Mod",["vpre","three16"],"vmod",fmod=1)
    init("onehalf16",np.array(1.5,np.float16),np.float16)
    n("Greater",["vmod","onehalf16"],"mid_b")

    n("And",["mid_b","odd_b"],"rem16b")
    init("zpad", np.zeros((1,1,30-HROWS,30),bool), np.bool_)
    n("Concat",["rem16b","zpad"],"removed",axis=2)
    e0=np.zeros((1,10,1,1),np.float32); e0[0,0,0,0]=1.0; init("e0",e0,np.float32)
    n("Where",["removed","e0","input"],"output")
    x=helper.make_tensor_value_info("input",F,[1,10,30,30])
    y=helper.make_tensor_value_info("output",F,[1,10,30,30])
    g=helper.make_graph(nodes,"task085",[x],[y],inits)
    return helper.make_model(g,ir_version=IR_VERSION,opset_imports=[helper.make_opsetid("",11)])
