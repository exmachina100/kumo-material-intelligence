import test from 'node:test';import assert from 'node:assert/strict';
import {rgbToHex,rgbToLab,deltaE76,sampleDominantColor} from '../packages/cmf-core/src/color.mjs';
import {analyzePixels} from '../packages/inference-sdk/src/index.mjs';
test('RGB hexadecimal representation',()=>assert.equal(rgbToHex([13,125,255]),'#0D7DFF'));
test('CIELAB reference white',()=>{const [l,a,b]=rgbToLab([255,255,255]);assert.ok(Math.abs(l-100)<.01);assert.ok(Math.abs(a)<.02);assert.ok(Math.abs(b)<.02)});
test('delta E zero for identical colors',()=>assert.equal(deltaE76([40,10,-12],[40,10,-12]),0));
test('dominant color ignores transparent pixels',()=>assert.deepEqual(sampleDominantColor(new Uint8ClampedArray([200,0,0,255,0,255,0,0]),{stride:1}),[200,0,0]));
test('unknown classification is explicit and not invented',()=>{let r=analyzePixels(new Uint8ClampedArray([12,34,56,255]),{stride:1});assert.equal(r.classification.family,'unknown');assert.equal(r.classification.confidence,null);assert.equal(r.color.hex,'#0C2238')});