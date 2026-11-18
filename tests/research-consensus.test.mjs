import test from 'node:test';
import assert from 'node:assert/strict';
import {summarizeCropPredictions} from '../packages/inference-sdk/src/research-consensus.mjs';

test('inconsistent crop votes produce unknown',()=>{
 const result=summarizeCropPredictions([{family:'metal',confidence:.98},{family:'wood',confidence:.94},{family:'metal',confidence:.98}]);
 assert.equal(result.family,'unknown');assert.equal(result.reason,'crop_disagreement');
});
test('weak scores produce unknown even with consistent votes',()=>{
 const result=summarizeCropPredictions([{family:'wood',confidence:.65},{family:'wood',confidence:.73},{family:'wood',confidence:.80}]);
 assert.equal(result.family,'unknown');assert.equal(result.reason,'low_raw_score');
});
test('agreement only permits an unverified research guess',()=>{
 const r=summarizeCropPredictions([{family:'textile',confidence:.93},{family:'textile',confidence:.9},{family:'textile',confidence:.95}]);
 assert.equal(r.family,'textile');assert.equal(r.withheld,false);
 assert.equal(r.reason,'unverified_guess');
});
test('invalid scores and empty input are rejected',()=>{
 assert.throws(()=>summarizeCropPredictions([{family:'metal',confidence:NaN},{family:'wood',confidence:.9}]));
 assert.throws(()=>summarizeCropPredictions([]));
 assert.throws(()=>summarizeCropPredictions([{family:'metal',confidence:.9},{family:'metal',confidence:.9}],{minimumScore:3}));
});
