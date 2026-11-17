import test from 'node:test';import assert from 'node:assert/strict';
import {OnnxMaterialClassifier} from '../packages/inference-sdk/src/onnx-adapter.mjs';
class MockTensor{constructor(type,data,dims){this.type=type;this.data=data;this.dims=dims}}
test('rejects missing session and weights',()=>assert.throws(()=>new OnnxMaterialClassifier({labels:['metal']})));
test('valid session returns calibrated softmax estimate',async()=>{
 const session={inputNames:['image'],outputNames:['logits'],run:async({image})=>{assert.deepEqual(image.dims,[1,3,2,2]);return {logits:{data:Float32Array.from([0,3])}}}};
 const model=new OnnxMaterialClassifier({session,Tensor:MockTensor,labels:['polymer','metal'],inputSize:2,version:'test-only'});
 const r=await model.classify(new Uint8ClampedArray([255,0,0,255]),{width:1,height:1});
 assert.equal(r.family,'metal');assert.ok(r.confidence>.9);
});
test('dimension mismatch fails safely',async()=>{
 const model=new OnnxMaterialClassifier({session:{run:async()=>({})},Tensor:MockTensor,labels:['metal'],version:'test'});
 await assert.rejects(model.classify(new Uint8ClampedArray(4),{width:2,height:2}),/dimensions/);
});

test('bilinear RGB preprocessing interpolates the center of a 2x2 gradient',async()=>{
 let input;
 const session={inputNames:['image'],outputNames:['logits'],run:async(inputs)=>{
   input=inputs.image;
   return {logits:{data:Float32Array.from([0,1])}};
 }};
 const model=new OnnxMaterialClassifier({session,Tensor:MockTensor,labels:['wood','metal'],inputSize:3,version:'test',normalization:{mean:[0,0,0],std:[1,1,1]}});
 const rgba=new Uint8ClampedArray([
   0,0,0,255,   255,0,0,255,
   0,0,0,255,   255,0,0,255
 ]);
 await model.classify(rgba,{width:2,height:2});
 assert.ok(Math.abs(input.data[4]-0.5)<0.01,'center pixel should contain roughly 50% red, not nearest neighbor');
 assert.ok(input.data[0]<0.01,'left border remains black');
 assert.ok(input.data[2]>.99,'right border remains red');
});
