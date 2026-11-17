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
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
   const sx=Math.min(width-1,Math.floor((x+.5)*width/n)),sy=Math.min(height-1,Math.floor((y+.5)*height/n)),i=(sy*width+sx)*4;
   for(let c=0;c<3;c++)pixels[c*n*n+y*n+x]=(rgba[i+c]/255-this.normalization.mean[c])/this.normalization.std[c];
  }
  const input=this.session.inputNames?.[0]||'image',output=this.session.outputNames?.[0]||'logits';
  const result=await this.session.run({[input]:new this.Tensor('float32',pixels,[1,3,n,n])});
  const logits=Array.from(result[output]?.data||[]);
  if(logits.length!==this.labels.length)throw new Error('Classifier outputs do not match manifest labels');
  const max=Math.max(...logits),exps=logits.map(v=>Math.exp(v-max)),sum=exps.reduce((a,b)=>a+b,0);
  if(!Number.isFinite(sum)||sum<=0)throw new Error('Invalid logits');
  const probs=exps.map(v=>v/sum),index=probs.indexOf(Math.max(...probs));
  return {family:this.labels[index],confidence:probs[index],finish:null};
 }
}
