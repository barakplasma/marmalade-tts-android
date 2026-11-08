# /// script
# dependencies = ["numpy", "huggingface_hub", "onnx", "onnxruntime"]
# ///
import os,sys,itertools,numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from huggingface_hub import snapshot_download
p=snapshot_download("ConwayResearch/Underdog-Woof",revision="be702960539777fc3f698e54335276a8391b00f9",allow_patterns=["woof_int8.npz","underdog_woof/*.py"])
sys.path.insert(0,p)
from underdog_woof.runtime import unflatten,Speech
with np.load(os.path.join(p,"woof_int8.npz"),allow_pickle=False) as z: st=unflatten(z)
speech=Speech(st["speech"])
class Graph:
 def __init__(self):
  self.nodes=[];self.initializers=[];self.counter=itertools.count()
 def name(self,tag):return f"{tag}_{next(self.counter)}"
 def constant(self,arr,tag="c"):
  name=self.name(tag);self.initializers.append(numpy_helper.from_array(np.asarray(arr),name=name));return name
 def op(self,kind,*args,**attrs):
  result=self.name(kind.lower());self.nodes.append(helper.make_node(kind,list(args),[result],**attrs));return result
 def mat(self,w,b,x):
  v=self.op("MatMul",self.constant(np.asarray(w,np.float32),"W"),x)
  if b is not None:v=self.op("Add",v,self.constant(np.asarray(b,np.float32),"bias"))
  return v
 def silu(self,x):return self.op("Mul",x,self.op("Sigmoid",x))
 def norm(self,x):
  mean=self.op("ReduceMean",x,axes=[0],keepdims=1)
  c=self.op("Sub",x,mean)
  var=self.op("ReduceMean",self.op("Mul",c,c),axes=[0],keepdims=1)
  return self.op("Div",c,self.op("Sqrt",self.op("Add",var,self.constant(np.array(1e-5,np.float32)))))
 def cut(self,x,start,end):
  return self.op("Slice",x,self.constant([start],"start"),self.constant([end],"end"),self.constant([0],"axis"))
g=Graph();hw=speech.hw
A=g.mat(*speech.ada,"s");h=g.mat(*speech.h_inp,"z")
for j,((w1,b1),(w2,b2)) in enumerate(speech.hblocks):
 k=3*hw*j
 y=g.op("Add",g.cut(A,k,k+hw),g.op("Mul",g.norm(h),g.cut(A,k+hw,k+2*hw)))
 ff=g.mat(w1,b1,y)
 ff=g.mat(w2,b2,g.silu(ff))
 h=g.op("Add",h,g.op("Mul",g.cut(A,k+2*hw,k+3*hw),ff))
k=3*hw*speech.depth
y=g.op("Add",g.cut(A,k,k+hw),g.op("Mul",g.norm(h),g.cut(A,k+hw,k+2*hw)))
out=g.mat(*speech.h_out,y);out=g.op("Identity",out)
inputs=[helper.make_tensor_value_info("s",TensorProto.FLOAT,[hw]),helper.make_tensor_value_info("z",TensorProto.FLOAT,[speech.latent])]
outputs=[helper.make_tensor_value_info(out,TensorProto.FLOAT,[speech.latent])]
graph=helper.make_graph(g.nodes,"woof-flow-head",inputs,outputs,g.initializers)
model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)],producer_name="woof-onnx-feasibility-probe")
model.ir_version=9;onnx.checker.check_model(model);onnx.save_model(model,"woof_head.onnx")
print("EXPORTED bytes",os.path.getsize("woof_head.onnx"),"nodes",len(g.nodes),"initializers",len(g.initializers),flush=True)
session=ort.InferenceSession("woof_head.onnx",providers=["CPUExecutionProvider"])
maxabs=0
for i in range(10):
 rng=np.random.default_rng(i)
 s=rng.normal(0,0.5,hw).astype(np.float32)
 z=rng.normal(0,0.5,speech.latent).astype(np.float32)
 want=speech._head(s,z)
 got=session.run([out],{"s":s,"z":z})[0]
 err=np.max(np.abs(want-got));maxabs=max(maxabs,float(err))
 print(f"probe {i:02d}: max_abs_err={err:.8g}, shape={got.shape}",flush=True)
print(f"RESULT: max_absolute_error={maxabs:.8g}, PASS={maxabs < 0.0001}",flush=True)
assert maxabs<0.0001, "ONNX head failed numerical equivalence threshold"
