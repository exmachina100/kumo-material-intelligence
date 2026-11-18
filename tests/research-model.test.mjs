import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,statSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {verifyResearchManifest} from '../packages/inference-sdk/src/hosted-research.mjs';

const root='models/research/';
test('committed research weights match the license-gated manifest and SHA-256',()=>{
 const manifest=verifyResearchManifest(JSON.parse(readFileSync(root+'manifest.json','utf8')));
 const model=readFileSync(root+manifest.weights);
 assert.equal(model.byteLength,manifest.bytes);
 assert.equal(createHash('sha256').update(model).digest('hex'),manifest.sha256);
 const labels=JSON.parse(readFileSync(root+'labels.json','utf8'));
 assert.deepEqual(labels,manifest.labels);
 assert.equal(manifest.approved,false);
 assert.equal(manifest.releaseEligible,false);
 assert.equal(manifest.expertReviewed,false);
});
test('research evaluation remains weak-label and explicitly disallowed for production',()=>{
 const report=JSON.parse(readFileSync(root+'evaluation.json','utf8'));
 assert.equal(report.release_eligible,false);
 assert.equal(report.expert_reviewed,false);
 assert.equal(report.status,'RESEARCH_ONLY');
 assert.equal(report.onnx_export.success,true);
});
test('main workstation exposes opt-in research identification and manual confirmation',()=>{
 const html=readFileSync('apps/demo/index.html','utf8');
 assert.match(html,/id="identify"/);
 assert.match(html,/id="acceptAI"/);
 assert.match(html,/identifyResearchMaterial/);
 assert.match(html,/confirmedByDesigner:true/);
 assert.match(html,/experimental-catalog-proxy/);
});
test('GitHub Pages copies the pinned model without depending on expiring training artifacts',()=>{
 const workflow=readFileSync('.github/workflows/pages.yml','utf8');
 assert.match(workflow,/cp models\/research\/vision-probe-research.onnx/);
 assert.match(workflow,/sha256/);
 assert.doesNotMatch(workflow,/gh run download/);
});
