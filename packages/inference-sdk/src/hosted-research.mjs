import {OnnxMaterialClassifier} from './onnx-adapter.mjs';
import {summarizeCropPredictions} from './research-consensus.mjs';
const RUNTIME='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
let runtimePromise;
export function verifyResearchManifest(manifest){
 if(!manifest||manifest.kind!=='experimental-catalog-proxy'||manifest.approved!==false||
    manifest.releaseEligible!==false||manifest.expertReviewed!==false)throw Error('Unapproved research manifest');
 if(!Array.isArray(manifest.labels)||manifest.labels.length<2||
    !manifest.labels.every(x=>typeof x==='string'&&/^[a-z-]+$/.test(x)))throw Error('Invalid model labels');
 if(!Number.isInteger(manifest.inputSize)||manifest.inputSize<96||manifest.inputSize>512)throw Error('Invalid model input');
 if(!Number.isInteger(manifest.bytes)||manifest.bytes<1024||manifest.bytes>45000000)throw Error('Invalid model size');
 if(!/^[a-f0-9]{64}$/.test(manifest.sha256)||manifest.weights!=='vision-probe-research.onnx')throw Error('Invalid model artifact specification');
 return manifest;
}
export async function verifyModelBytes(buffer,manifest){
 if(buffer.byteLength!==manifest.bytes)throw Error('Model file size mismatch');
 if(!globalThis.crypto?.subtle)throw Error('Secure cryptographic hashing unavailable');
 const hash=await crypto.subtle.digest('SHA-256',buffer);
 const hex=Array.from(new Uint8Array(hash),x=>x.toString(16).padStart(2,'0')).join('');
 if(hex!==manifest.sha256)throw Error('Model checksum mismatch');
 return true;
}
async function getRuntime(){
 if(globalThis.ort?.InferenceSession)return globalThis.ort;
 if(!runtimePromise)runtimePromise=new Promise((resolve,reject)=>{
  const tag=document.createElement('script');
  tag.src=RUNTIME;tag.crossOrigin='anonymous';
  tag.onload=()=>globalThis.ort?.InferenceSession?resolve(globalThis.ort):reject(Error('ONNX runtime unavailable'));
  tag.onerror=()=>reject(Error('Unable to load WebAssembly runtime'));
  document.head.append(tag);
 }).catch(error=>{runtimePromise=null;throw error});
 return runtimePromise;
}
export async function loadHostedResearchModel(root,{onProgress=()=>{}}={}){
 const url=new URL(root,globalThis.location?.href||'https://example.invalid/');
 const response=await fetch(new URL('manifest.json',url),{cache:'no-store'});
 if(!response.ok)throw Error('Research model manifest unavailable');
 const manifest=verifyResearchManifest(await response.json());
 onProgress('Downloading research weights…');
 const weights=await fetch(new URL(manifest.weights,url));
 if(!weights.ok)throw Error('Research model file unavailable');
 const bytes=await weights.arrayBuffer();
 onProgress('Verifying model checksum…');
 await verifyModelBytes(bytes,manifest);
 onProgress('Starting on-device inference runtime…');
 const ort=await getRuntime();
 ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
 ort.env.wasm.numThreads=1;
 const session=await ort.InferenceSession.create(new Uint8Array(bytes),{executionProviders:['wasm']});
 return {
   manifest,session,
   classifier:new OnnxMaterialClassifier({session,Tensor:ort.Tensor,labels:manifest.labels,inputSize:manifest.inputSize,version:manifest.version}),
   async dispose(){await session.release?.();}
 };
}
export async function identifyResearchMaterial(classifier,rgba,width,height,{minimumScore=.85,minimumAgreement=.75}={}){
 if(!classifier?.classify)throw TypeError('Classifier unavailable');
 if(!rgba||rgba.length!==width*height*4||!Number.isInteger(width)||!Number.isInteger(height))throw TypeError('Invalid pixels');
 const crops=[
  {x:0,y:0,width,height},
  {x:0,y:0,width:Math.max(1,Math.floor(width*.8)),height},
  {x:width-Math.max(1,Math.floor(width*.8)),y:0,width:Math.max(1,Math.floor(width*.8)),height}
 ];
 const predictions=[];
 for(const crop of crops){
  const subset=new Uint8ClampedArray(crop.width*crop.height*4);
  for(let y=0;y<crop.height;y++){
   const from=((crop.y+y)*width+crop.x)*4;
   subset.set(rgba.subarray(from,from+crop.width*4),y*crop.width*4);
  }
  predictions.push(await classifier.classify(subset,{width:crop.width,height:crop.height}));
 }
 return summarizeCropPredictions(predictions,{minimumScore,minimumAgreement});
}
