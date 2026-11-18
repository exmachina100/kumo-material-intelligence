import test from 'node:test';
import assert from 'node:assert/strict';
import {verifyResearchManifest,verifyModelBytes,identifyResearchMaterial} from '../packages/inference-sdk/src/hosted-research.mjs';
const manifest={kind:'experimental-catalog-proxy',approved:false,releaseEligible:false,expertReviewed:false,
labels:['wood','metal'],inputSize:224,bytes:1200,sha256:'a'.repeat(64),weights:'vision-probe-research.onnx'};
test('only explicitly unapproved research manifests can load',()=>{
 assert.equal(verifyResearchManifest(manifest),manifest);
 assert.throws(()=>verifyResearchManifest({...manifest,approved:true}),/Unapproved/);
 assert.throws(()=>verifyResearchManifest({...manifest,weights:'../other.onnx'}),/Invalid/);
});
test('model checksum and size are enforced',async()=>{
 await assert.rejects(verifyModelBytes(new Uint8Array(100).buffer,manifest),/size mismatch/);
});
test('three-crop inference identifies consistent guesses without confirming them',async()=>{
 let calls=0;
 const classifier={classify:async(rgba,opts)=>{
  calls++;assert.equal(rgba.length,opts.width*opts.height*4);
  return {family:'metal',confidence:.92};
 }};
 const data=new Uint8ClampedArray(10*8*4);
 const result=await identifyResearchMaterial(classifier,data,10,8);
 assert.equal(calls,3);assert.equal(result.topGuess,'metal');
 assert.equal(result.withheld,false);assert.equal(result.disclaimer.includes('Research'),true);
 assert.equal(result.hypotheses[0].family,'metal');assert.ok(result.hypotheses[0].rawMeanSoftmax>.9);
});
test('crop disagreement yields unknown',async()=>{
 let n=0;const classifier={classify:async()=>({family:++n===1?'metal':'wood',confidence:.97})};
 const out=await identifyResearchMaterial(classifier,new Uint8ClampedArray(10*8*4),10,8);
 assert.equal(out.family,'unknown');
});
