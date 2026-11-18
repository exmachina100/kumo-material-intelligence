import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';

const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{
 const page=await browser.newPage();
 const errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 await page.goto('http://127.0.0.1:8080/apps/demo/index.html',{waitUntil:'networkidle',timeout:45000});
 await page.locator('#file').setInputFiles(resolve('training/external/polyhaven_cc0_staging/leather_white.jpg'));
 await page.waitForFunction(()=>!document.querySelector('#identify').disabled,{timeout:15000});
 assert.equal(await page.locator('#acceptAI').isDisabled(),true);
 await page.locator('#identify').click();
 await page.waitForFunction(()=>{
  const text=document.querySelector('#visionStatus')?.textContent||'';
  return text.includes('Suggestion available')||text.includes('AI suggestion withheld')||text.includes('Vision unavailable');
 },{timeout:120000});
 const status=await page.locator('#visionStatus').textContent();
 const suggested=await page.locator('#aiFamily').textContent();
 const detail=await page.locator('#aiEvidence').textContent();
 console.log(JSON.stringify({status,suggested,detail,errors}));
 assert.ok(!status.includes('Vision unavailable'),'Actual ONNX inference failed: '+status);
 assert.match(detail,/crop agreement/);
 assert.match(await page.locator('#aiAlternatives').textContent(),/Other visual possibilities:/);
 assert.equal(await page.locator('#recognition').textContent(),'Unknown / unconfirmed');
 if(await page.locator('#acceptAI').isEnabled()){
  await page.locator('#acceptAI').click();
  assert.equal(await page.locator('#recognition').textContent(),'Designer-confirmed');
 }
 // The expanded model is opt-in and must never silently assign a verified family.
 await page.locator('#visionVariant').selectOption('research-expanded');
 assert.equal(await page.locator('#acceptAI').isDisabled(),true);
 await page.locator('#identify').click();
 await page.waitForFunction(()=>{
  const text=document.querySelector('#visionStatus')?.textContent||'';
  return text.includes('Suggestion available')||text.includes('AI suggestion withheld')||text.includes('Vision unavailable');
 },{timeout:120000});
 const expandedStatus=await page.locator('#visionStatus').textContent();
 assert.ok(!expandedStatus.includes('Vision unavailable'),'Expanded ONNX model failed: '+expandedStatus);
 assert.match(await page.locator('#aiAlternatives').textContent(),/Other visual possibilities:/);
 const dna=JSON.parse(await page.locator('#json').textContent());
 assert.equal(dna.visionResearch.modelVariant,'research-expanded');
 assert.equal(dna.visionResearch.notGroundTruth,true);
 assert.equal(dna.visionResearch.confirmedByDesigner,false);
 console.log('Expanded 10-family model local inference PASSED');
 assert.deepEqual(errors,[],'Uncaught page JS errors');
 const reviewPage=await browser.newPage();
 const reviewErrors=[];
 reviewPage.on('pageerror',e=>reviewErrors.push(e.message));
 await reviewPage.goto('http://127.0.0.1:8080/apps/review/index.html',{waitUntil:'networkidle',timeout:45000});
 await reviewPage.waitForFunction(()=>document.querySelectorAll('#gallery .tile').length>=300,{timeout:20000});
 const count=await reviewPage.locator('#gallery .tile').count();
 assert.ok(count>=300,'Combined Poly Haven and ambientCG review images missing: '+count);
 assert.match(await reviewPage.locator('#source').textContent(),/ambientCG/);
 await reviewPage.locator('#search').fill('leather');
 assert.ok(await reviewPage.locator('#gallery .tile').count()>0);
 assert.deepEqual(reviewErrors,[]);
 console.log('Combined-source material review browser smoke test PASSED with '+count+' references');
 await reviewPage.close();
 console.log('Browser vision end-to-end inference PASSED');
}finally{await browser.close()}
