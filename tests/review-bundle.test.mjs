import test from 'node:test';
import assert from 'node:assert/strict';
import {makeReview} from '../packages/material-dna/src/review.mjs';
import {importReviewBundle,mergeReviews,exportReviewBundle,summarizeReviewQuality} from '../packages/material-dna/src/review-bundle.mjs';

const sources=new Set(['wood_1','metal_2']);
const approved=()=>makeReview({asset_id:'wood_1',decision:'include',family:'wood',reviewer:'CMF Reviewer',evidence:'visual-only'});

test('review bundles export, validate, and round-trip',()=>{
 const bundle=exportReviewBundle([approved()]);
 const imported=importReviewBundle(bundle,sources);
 assert.equal(imported.size,1);
 assert.equal(imported.get('wood_1').family,'wood');
});
test('no catalog or external unknown source may become ground truth',()=>{
 assert.throws(()=>importReviewBundle(exportReviewBundle([{...approved(),asset_id:'fabricated'}]),sources),/absent/);
});
test('duplicate decisions are rejected rather than silently overwritten',()=>{
 assert.throws(()=>importReviewBundle(exportReviewBundle([approved(),approved()]),sources),/Duplicate/);
});
test('reject malformed positive reviews without confirmed family or reviewer',()=>{
 assert.throws(()=>importReviewBundle(exportReviewBundle([{...approved(),family:'unknown',reviewer:''}]),sources),/Invalid review/);
});
test('merging local reviews retains other samples unless explicitly replaced',()=>{
 const a=new Map([['metal_2',makeReview({asset_id:'metal_2',decision:'uncertain',reviewer:'CMF Reviewer'})]]);
 const b=importReviewBundle(exportReviewBundle([approved()]),sources);
 assert.equal(mergeReviews(a,b).size,2);
 assert.equal(mergeReviews(a,b,{replace:true}).size,1);
});
test('audit separates visual-only from source-verified evidence',()=>{
 const inputs=[{asset_id:'wood_1'},{asset_id:'metal_2'}];
 const map=new Map([['wood_1',approved()]]);
 assert.deepEqual(summarizeReviewQuality(inputs,map),{included:1,excluded:0,uncertain:0,pending:1,visualOnly:1,sourceVerified:0,invalid:0});
});
