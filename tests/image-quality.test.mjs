import test from 'node:test';
import assert from 'node:assert/strict';
import {inspectImageQuality} from '../packages/cmf-core/src/image-quality.mjs';
const solid=(w,h,v)=>{const rgba=new Uint8ClampedArray(w*h*4);for(let i=0;i<rgba.length;i+=4){rgba[i]=v;rgba[i+1]=v;rgba[i+2]=v;rgba[i+3]=255}return rgba};
test('reject invalid data or dimensions',()=>{
 assert.throws(()=>inspectImageQuality(new Uint8ClampedArray(4),20,20));
});
test('dark frame generates underexposure advisory',()=>{
 const quality=inspectImageQuality(solid(128,128,12),128,128);
 assert.ok(quality.warnings.some(w=>w.code==='underexposed'));
});
test('bright white frame generates highlight advisory',()=>{
 const quality=inspectImageQuality(solid(128,128,255),128,128);
 assert.ok(quality.warnings.some(w=>w.code==='clipped_highlights'));
});
test('uniform non-clipped material is not called underexposed',()=>{
 const quality=inspectImageQuality(solid(128,128,135),128,128);
 assert.equal(quality.warnings.some(w=>w.code==='underexposed'),false);
 assert.equal(quality.warnings.some(w=>w.code==='clipped_highlights'),false);
 assert.ok(quality.warnings.some(w=>w.code==='low_visual_detail'));
});
test('transparent frame does not fake image quality',()=>{
 const q=inspectImageQuality(new Uint8ClampedArray(128*128*4),128,128);
 assert.ok(q.warnings.some(w=>w.code==='no_visible_pixels'));
});
