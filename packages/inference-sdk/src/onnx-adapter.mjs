/**
 * Optional ONNX inference adapter. Runtime and artifact injected by the host.
 * Compatible with @microsoft/onnxruntime-web sessions after the host imports that package.
 * This module downloads nothing and will not classify without verified weights and labels.
 */
export class OnnxMaterialClassifier {
 constructor({session,Tensor,labels,inputSize=160,version,normalization={mean:[.485,.456,.406],std:[.229,.224,.225]}}){
  if(!session?.run||!Tensor||!Array.isArray(labels)||!labels.length||!version)throw new Error('A verified session, Tensor, labels and version are required');
  this.session=session;this.Tensor=Tensor;this.labels=labels;this.inputSize=inputSize;this.version=version;this.normalization=normalization;this.name='onnx-materialnet';
 }
 async classify(rgba,{width,height}={}){
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||rgba.length!==width*height*4)throw new Error('Input image dimensions must be provided');
  const n=this.inputSize,pixels=new Float32Array(3*n*n);
  // Bilinear center-aligned resize (research preview). Downsampling differs from
  // TorchVision's anti-aliased resize; verify pixel parity before production release.
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
   const sourceX=(x+.5)*width/n-.5,sourceY=(y+.5)*height/n-.5;
   const x0=Math.max(0,Math.min(width-1,Math.floor(sourceX)));
   const x1=Math.max(0,Math.min(width-1,Math.ceil(sourceX)));
   const y0=Math.max(0,Math.min(height-1,Math.floor(sourceY)));
   const y1=Math.max(0,Math.min(height-1,Math.ceil(sourceY)));
   const wx=Math.max(0,Math.min(1,sourceX-Math.floor(sourceX)));
   const wy=Math.max(0,Math.min(1,sourceY-Math.floor(sourceY)));
   for(let c=0;c<3;c++){
    const at=(xx,yy)=>rgba[(yy*width+xx)*4+c];
    const upper=(1-wx)*at(x0,y0)+wx*at(x1,y0);
    const lower=(1-wx)*at(x0,y1)+wx*at(x1,y1);
    const v=((1-wy)*upper+wy*lower)/255;
    pixels[c*n*n+y*n+x]=(v-this.normalization.mean[c])/this.normalization.std[c];
   }
  }
  const input=this.session.inputNames?.[0]||'image',output=this.session.outputNames?.[0]||'logits';
  const result=await this.session.run({[input]:new this.Tensor('float32',pixels,[1,3,n,n])});
  const logits=Array.from(result[output]?.data||[]);
  if(logits.length!==this.labels.length)throw new Error('Classifier outputs do not match manifest labels');
  const max=Math.max(...logits),exps=logits.map(v=>Math.exp(v-max)),sum=exps.reduce((a,b)=>a+b,0);
  if(!Number.isFinite(sum)||sum<=0)throw new Error('Invalid logits');
  const probs=exps.map(v=>v/sum),index=probs.indexOf(Math.max(...probs));
  const alternatives=probs.map((rawSoftmax,i)=>({family:this.labels[i],rawSoftmax}))
   .sort((a,b)=>b.rawSoftmax-a.rawSoftmax).slice(0,3);
  return {family:this.labels[index],confidence:probs[index],finish:null,alternatives};
 }
}
