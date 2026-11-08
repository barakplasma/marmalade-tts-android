# /// script
# dependencies = ["numpy", "huggingface_hub", "onnx", "onnxruntime"]
# ///
import os,sys,itertools,numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from huggingface_hub import snapshot_download
p=snapshot_download("ConwayResearch/Underdog-Woof",allow_patterns=["woof_int8.npz","underdog_woof/*.py"])
sys.path.insert(0,p)
from underdog_woof.runtime import unflatten,Speech, EPS
with np.load(os.path.join(p,"woof_int8.npz"),allow_pickle=False) as z:st=unflatten(z)
speech=Speech(st['speech'])
class G:
 def __init__(self):self.nodes=[];self.inits=[];self.count=itertools.count()
 def new(self,n):return f"{n}_{next(self.count)}"
 def c(self,a):
  n=self.new("c");self.inits.append(numpy_helper.from_array(np.asarray(a),name=n));return n
 def op(self,k,*args,**opts):
  n=self.new(k.lower());self.nodes.append(helper.make_node(k,list(args),[n],**opts));return n
 def mat(self,x,w):return self.op('MatMul',self.c(np.asarray(w,np.float32)),x)
 def addc(self,x,c):return self.op('Add',x,self.c(np.asarray(c,np.float32)))
 def gather(self,x,i,axis):return self.op('Gather',x,self.c(np.array(i,np.int64)),axis=axis)
 def unsq(self,x,axes):return self.op('Unsqueeze',x,self.c(np.array(axes,np.int64)))
 def reshape(self,x,shape):return self.op('Reshape',x,self.c(np.array(shape,np.int64)))
 def rms(self,x):
  r=self.op('ReduceMean',self.op('Mul',x,x),axes=[0],keepdims=0)
  return self.op('Div',x,self.op('Sqrt',self.addc(r,np.array(EPS,np.float32))))
 def silu(self,x):return self.op('Mul',x,self.op('Sigmoid',x))
g=G();pos='pos';x='x';kv='kv'
co=g.op('Gather',g.c(speech.cos),pos,axis=0)
si=g.op('Gather',g.c(speech.sin),pos,axis=0)
ks=[]
for i,((wqkv,_),(wo,_),(w1,_),(w2,_)) in enumerate(speech.layers):
 qkv=g.reshape(g.mat(g.rms(x),wqkv),[3,speech.H,speech.Dh])
 qk=g.op('Slice',qkv,g.c(np.array([0],np.int64)),g.c(np.array([2],np.int64)),g.c(np.array([0],np.int64)))
 qk_swap=g.reshape(g.op('Gather',g.reshape(qk,[2,speech.H,2,speech.Dh//2]),g.c(np.array([1,0],np.int64)),axis=2),[2,speech.H,speech.Dh])
 qk=g.op('Add',g.op('Mul',qk,co),g.op('Mul',qk_swap,si))
 old=g.gather(kv,i,0)
 k_prev=g.gather(old,0,0);v_prev=g.gather(old,1,0)
 k=g.op('Concat',k_prev,g.unsq(g.gather(qk,1,0),[1]),axis=1)
 v=g.op('Concat',v_prev,g.unsq(g.gather(qkv,2,0),[1]),axis=1)
 ks.append(g.unsq(g.op('Concat',g.unsq(k,[0]),g.unsq(v,[0]),axis=0),[0]))
 q=g.gather(qk,0,0)
 scores=g.op('Mul',g.op('ReduceSum',g.op('Mul',k,g.unsq(q,[1])),g.c(np.array([2],np.int64)),keepdims=0),g.c(np.array(speech.scale,np.float32)))
 att=g.op('ReduceSum',g.op('Mul',g.unsq(g.op('Softmax',scores,axis=1),[2]),v),g.c(np.array([1],np.int64)),keepdims=0)
 x=g.op('Add',x,g.mat(g.reshape(att,[speech.d]),wo))
 gu=g.mat(g.rms(x),w1)
 fst=g.op('Slice',gu,g.c(np.array([0],np.int64)),g.c(np.array([speech.ff],np.int64)),g.c(np.array([0],np.int64)))
 snd=g.op('Slice',gu,g.c(np.array([speech.ff],np.int64)),g.c(np.array([speech.ff*2],np.int64)),g.c(np.array([0],np.int64)))
 x=g.op('Add',x,g.mat(g.op('Mul',g.silu(fst),snd),w2))
kvout=g.op('Concat',*ks,axis=0)
cee=g.mat(g.rms(x),speech.ce[0]);cee=g.addc(cee,speech.ce[1])
s=g.silu(g.op('Slice',cee,g.c(np.array([0],np.int64)),g.c(np.array([speech.hw],np.int64)),g.c(np.array([0],np.int64))))
e=g.gather(cee,speech.hw,0)
model=helper.make_model(helper.make_graph(g.nodes,'woof-speech-step',[
 helper.make_tensor_value_info('x',TensorProto.FLOAT,[speech.d]),
 helper.make_tensor_value_info('kv',TensorProto.FLOAT,[speech.L,2,speech.H,'P',speech.Dh]),
 helper.make_tensor_value_info('pos',TensorProto.INT64,[]),
 ],[
 helper.make_tensor_value_info(x,TensorProto.FLOAT,[speech.d]),
 helper.make_tensor_value_info(kvout,TensorProto.FLOAT,[speech.L,2,speech.H,'P_plus_1',speech.Dh]),
 helper.make_tensor_value_info(s,TensorProto.FLOAT,[speech.hw]),
 helper.make_tensor_value_info(e,TensorProto.FLOAT,[]),
 ],g.inits),opset_imports=[helper.make_opsetid('',17)]);model.ir_version=9
onnx.checker.check_model(model);onnx.save_model(model,'woof_speech_step.onnx')
print('EXPORTED',os.path.getsize('woof_speech_step.onnx'),'bytes',len(g.nodes),'nodes',flush=True)
sess=ort.InferenceSession('woof_speech_step.onnx',providers=['CPUExecutionProvider'])
for seed in range(5):
 rng=np.random.default_rng(seed);p0=seed+1
 old=rng.normal(0,.02,(speech.L,2,speech.H,p0,speech.Dh)).astype(np.float32)
 speech.kv[:,:,:,:p0,:]=old
 xin=rng.normal(0,.25,speech.d).astype(np.float32)
 ref=xin.copy()
 for i in range(speech.L):ref=speech._block1(i,ref,p0)
 sr,er=speech._cond(ref)
 arr=sess.run([x,kvout,s,e],{'x':xin,'kv':old,'pos':np.array(p0,np.int64)})
 delta=np.max(np.abs(ref-arr[0]));kv_delta=np.max(np.abs(speech.kv[:,:,:,:p0+1,:]-arr[1]));sdelta=np.max(np.abs(sr-arr[2]));edelta=np.abs(er-arr[3])
 print('test',seed,'x_err',delta,'kv_err',kv_delta,'s_err',sdelta,'e_err',edelta,flush=True)
 assert delta<.0001 and kv_delta<.0001 and sdelta<.0001 and edelta<.0001
print('SPEECH STEP RESULT: PASS',flush=True)
