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
