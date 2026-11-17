import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync, readFileSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
import {spawnSync} from 'node:child_process';

const entrypoints=[
  'apps/demo/index.html',
  'apps/review/index.html',
  'apps/research/index.html'
];
for(const page of entrypoints){
  test(page+' inline JavaScript parses and local imports resolve',()=>{
    const html=readFileSync(page,'utf8');
    const scripts=[...html.matchAll(/<script\s+type="module"\s*>([\s\S]*?)<\/script>/g)];
    assert.ok(scripts.length>0,'Expected a module script in '+page);
    for(const match of scripts){
      const code=match[1];
      const result=spawnSync(process.execPath,['--check','--input-type=module'],{input:code,encoding:'utf8'});
      assert.equal(result.status,0,result.stderr||'Failed to parse '+page);
      for(const imp of code.matchAll(/from\s+['"](\.\.?\/[^'"]+)['"]/g)){
        const file=resolve(dirname(page),imp[1]);
        assert.ok(existsSync(file),'Unresolved import '+imp[1]+' from '+page);
      }
    }
  });
}
test('GitHub Pages workflow deploys all three CMF applications',()=>{
  const workflow=readFileSync('.github/workflows/pages.yml','utf8');
  for(const app of ['demo','review','research']){
    assert.ok(workflow.includes('cp -R apps/'+app+' site/apps/'+app),
      'GitHub Pages must deploy '+app);
  }
});
