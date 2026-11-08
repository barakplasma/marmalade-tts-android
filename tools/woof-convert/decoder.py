# /// script
# dependencies = ["numpy", "huggingface_hub", "onnx", "onnxruntime"]
# ///
import os,sys,itertools,numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from huggingface_hub import snapshot_download
p=snapshot_download("ConwayResearch/Underdog-Woof",revision="be702960539777fc3f698e54335276a8391b00f9",allow_patterns=["woof_int8.npz","underdog_woof/*.py"])
sys.path.insert(0,p)
from underdog_woof.runtime import unflatten,Decoder,deq
with np.load(os.path.join(p,"woof_int8.npz"),allow_pickle=False) as z:st=unflatten(z)
dec=Decoder(st["decoder"],mode="int8")
T=8 # now test dynamic sequence lengths
class G:
 def __init__(self):self.nodes=[];self.inits=[];self.count=itertools.count()
 def new(self,n):return f"{n}_{next(self.count)}"
 def c(self,a):
  n=self.new("c");self.inits.append(numpy_helper.from_array(np.asarray(a),name=n));return n
 def op(self,k,*args,**opts):
  n=self.new(k.lower());self.nodes.append(helper.make_node(k,list(args),[n],**opts));return n
 def mat(self,x,w):return self.op("MatMul",x,self.c(np.asarray(w,np.float32)))
 def plus(self,x,c):return self.op("Add",x,self.c(np.asarray(c,np.float32)))
 def sl(self,x,start,end,axis):
  return self.op("Slice",x,self.c(np.array([start],np.int64)),self.c(np.array([end],np.int64)),self.c(np.array([axis],np.int64)))
 def ln(self,x,w,b):
  mu=self.op("ReduceMean",x,axes=[-1],keepdims=1);v=self.op("Sub",x,mu)
  var=self.op("ReduceMean",self.op("Mul",v,v),axes=[-1],keepdims=1)
  n=self.op("Div",v,self.op("Sqrt",self.plus(var,np.array(float(dec.eps),np.float32))))
  return self.plus(self.op("Mul",n,self.c(np.asarray(w,np.float32))),np.asarray(b,np.float32))
 def rope(self,x,cos,sin):
  hd=dec.hd;rot=self.op("Concat",self.op("Neg",self.sl(x,hd//2,hd,2)),self.sl(x,0,hd//2,2),axis=2)
  return self.op("Add",self.op("Mul",x,cos),self.op("Mul",rot,sin))
 def conv(self,x,op):
  w=deq(op.st["w"]).astype(np.float32);k=w.shape[-1]
  x3=self.op("Unsqueeze",self.op("Transpose",x,perm=[1,0]),self.c(np.array([0],np.int64)))
  y3=self.op("Conv",x3,self.c(w),self.c(op.b),pads=[k-1,0],strides=[1])
  return self.op("Transpose",self.op("Squeeze",y3,self.c(np.array([0],np.int64))),perm=[1,0])
 def convt(self,x,op,Tin):
  w=deq(op.st["w"]).astype(np.float32);cin,cout,k=w.shape;s=op.s
  assert k==2*s,(k,s)
  # Exact upstream _ConvT: Z = x @ w.transpose(0,2,1).reshape(cin,k*cout);
  # output[t] = Z[t,:s*cout] + (t>0 ? Z[t-1,s*cout:] : 0) + tile(b,s)
  z=self.mat(x,w.transpose(0,2,1).reshape(cin,k*cout))
  first=self.sl(z,0,s*cout,1)
  second=self.sl(z,s*cout,k*cout,1)
  shifted=self.op("Concat",self.c(np.zeros((1,s*cout),np.float32)),self.op("Slice",second,self.c(np.array([0],np.int64)),self.op("Unsqueeze",self.op("Sub",Tin,self.c(np.array(1,np.int64))),self.c(np.array([0],np.int64))),self.c(np.array([0],np.int64))),axis=0)
  flat=self.plus(self.op("Add",first,shifted),np.tile(op.st["b"].astype(np.float32),s))
  return self.op("Reshape",flat,self.c(np.array([-1,cout],np.int64)))
 def gelu(self,x):
  q=self.op("Div",x,self.c(np.array(np.sqrt(2.),np.float32)))
  return self.op("Mul",self.op("Mul",x,self.c(np.array(.5,np.float32))),self.plus(self.op("Erf",q),np.array(1.,np.float32)))
g=G();h=g.plus(g.mat("lat_in",dec.inp_w),dec.inp_b);D=dec.D;nh=dec.nh;hd=dec.hd
fr=np.arange(1024,dtype=np.float32)[:,None]*dec.inv_freq
emb=np.concatenate([fr,fr],axis=-1)
pos=g.op("Range",g.c(np.array(0,np.int64)),g.op("Gather",g.op("Shape","lat_in"),g.c(np.array(0,np.int64)),axis=0),g.c(np.array(1,np.int64)))
co=g.op("Gather",g.c(np.cos(emb).astype(np.float32)[None,:,:]),pos,axis=1)
si=g.op("Gather",g.c(np.sin(emb).astype(np.float32)[None,:,:]),pos,axis=1)
row=g.op("Unsqueeze",pos,g.c(np.array([1],np.int64)))
col=g.op("Unsqueeze",pos,g.c(np.array([0],np.int64)))
bad=g.op("Or",g.op("Greater",col,row),g.op("LessOrEqual",col,g.op("Sub",row,g.c(np.array(dec.window,np.int64)))))
mask=g.op("Unsqueeze",g.op("Where",bad,g.c(np.array(-1e9,np.float32)),g.c(np.array(0.,np.float32))),g.c(np.array([0],np.int64)))
for layer_st in dec.tf_st:
 l=dec._tf_layer(layer_st);y=g.ln(h,*l["ln1"])
 qkv=g.mat(y,l["qkv"]);qkv=g.op("Reshape",qkv,g.c(np.array([-1,3,nh,hd],np.int64)))
 qkv=g.op("Transpose",qkv,perm=[1,2,0,3])
 q=g.op("Gather",qkv,g.c(np.array(0,np.int64)),axis=0)
 k=g.op("Gather",qkv,g.c(np.array(1,np.int64)),axis=0)
 v=g.op("Gather",qkv,g.c(np.array(2,np.int64)),axis=0)
 q=g.rope(q,co,si);k=g.rope(k,co,si)
 score=g.op("Mul",g.op("MatMul",q,g.op("Transpose",k,perm=[0,2,1])),g.c(np.array(1/np.sqrt(hd),np.float32)))
 score=g.op("Softmax",g.op("Add",score,mask),axis=-1)
 att=g.op("MatMul",score,v)
 att=g.op("Reshape",g.op("Transpose",att,perm=[1,0,2]),g.c(np.array([-1,D],np.int64)))
 h=g.op("Add",h,g.mat(att,l["o"]))
 h=g.op("Add",h,g.mat(g.gelu(g.mat(g.ln(h,*l["ln2"]),l["fc1"])),l["fc2"]))
length=g.op("Gather",g.op("Shape","lat_in"),g.c(np.array(0,np.int64)),axis=0)
for kind,op in dec.ops:
 if kind=="elu":h=g.op("Elu",h,alpha=1.0)
 elif kind=="conv":h=g.conv(h,op)
 elif kind=="convt":h=g.convt(h,op,length);length=g.op("Mul",length,g.c(np.array(op.s,np.int64)))
 elif kind=="res":
  r=g.op("Elu",h,alpha=1.0);r=g.conv(r,op[0]);r=g.op("Elu",r,alpha=1.0);r=g.conv(r,op[1]);h=g.op("Add",h,r)
 else:raise ValueError(kind)
h=g.op("Gather",h,g.c(np.array(0,np.int64)),axis=1)
graph=helper.make_graph(g.nodes,"woof-full-decoder-convt-exact",[helper.make_tensor_value_info("lat_in",TensorProto.FLOAT,["T",32])],[helper.make_tensor_value_info(h,TensorProto.FLOAT,["wave"])],g.inits)
model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)]);model.ir_version=9
onnx.checker.check_model(model);onnx.save_model(model,"woof_full_decoder_fixed_T8.onnx")
print("EXPORTED bytes",os.path.getsize("woof_full_decoder_fixed_T8.onnx"),"nodes",len(g.nodes),flush=True)
sess=ort.InferenceSession("woof_full_decoder_fixed_T8.onnx",providers=["CPUExecutionProvider"])
maxerr=0.
for seed,T in enumerate([4,8,12]):
 inp=np.random.default_rng(seed).normal(0,.5,(T,32)).astype(np.float32)
 ref=dec(inp.copy());actual=sess.run([h],{"lat_in":inp})[0]
 err=float(np.max(np.abs(ref-actual)));mae=float(np.mean(np.abs(ref-actual)));maxerr=max(maxerr,err)
 print(f"full decoder probe {seed}: shape {actual.shape}, max_abs_error={err:.8g}, MAE={mae:.8g}, rms_ref={np.std(ref):.5g}",flush=True)
print("RESULT: dynamic full decoder PASS",maxerr<0.01,"max_abs_error",maxerr,flush=True)
assert maxerr<0.01
