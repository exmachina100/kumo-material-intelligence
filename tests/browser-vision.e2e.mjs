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
 assert.deepEqual(errors,[],'Uncaught page JS errors');
 console.log('Browser vision end-to-end inference PASSED');
}finally{await browser.close()}
