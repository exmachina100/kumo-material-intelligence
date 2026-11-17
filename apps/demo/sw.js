const CACHE='kumo-cmf-v2';
const FILES=['/apps/demo/','/apps/demo/index.html','/apps/demo/manifest.webmanifest','/packages/cmf-core/src/color.mjs','/packages/cmf-core/src/palette.mjs','/packages/inference-sdk/src/index.mjs','/packages/material-dna/src/index.mjs'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(FILES)).then(()=>self.skipWaiting()))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(names=>Promise.all(names.filter(x=>x!==CACHE).map(x=>caches.delete(x)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{if(e.request.method!=='GET'||new URL(e.request.url).origin!==self.location.origin)return;e.respondWith(caches.match(e.request).then(hit=>hit||fetch(e.request).then(resp=>{if(resp.ok){const copy=resp.clone();caches.open(CACHE).then(c=>c.put(e.request,copy))}return resp})))});
