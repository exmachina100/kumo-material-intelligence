import test from 'node:test';import assert from 'node:assert/strict';
import {makeReview,validateReview,reviewStats} from '../packages/material-dna/src/review.mjs';
test('material review includes only manually confirmed valid family',()=>{
 const good=makeReview({asset_id:'a1',family:'wood',decision:'include',reviewer:'Designer'});
 assert.equal(validateReview(good,new Set(['a1'])).valid,true);
 assert.equal(good.evidence,'visual-only');
});
test('unconfirmed family cannot slip into training',()=>{
 const item=makeReview({asset_id:'a1',family:'unknown',decision:'include',reviewer:'Designer'});
 assert.equal(validateReview(item,new Set(['a1'])).valid,false);
});
test('rejects fake ids and missing reviewer',()=>{
 const item=makeReview({asset_id:'a2',family:'wood',decision:'include',reviewer:'X'});
 const out=validateReview(item,new Set(['a1']));
 assert.equal(out.valid,false);assert.ok(out.errors.length>=2);
});
test('review stats distinguish pending from confirmed',()=>{
 const sample=[{asset_id:'a1'},{asset_id:'a2'}];
 const r=makeReview({asset_id:'a1',family:'metal',decision:'include',reviewer:'Expert'});
 const stats=reviewStats(sample,[r]);
 assert.deepEqual(stats,{total:2,reviewed:1,excluded:0,uncertain:0,pending:1});
});
