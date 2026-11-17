const CACHE='kumo-cmf-v3';
const ROOT=new URL('./',self.registration.scope);
const urls=['./','./index.html','./manifest.webmanifest','../../packages/cmf-core/src/color.mjs','../../packages/cmf-core/src/palette.mjs','../../packages/inference-sdk/src/index.mjs','../../packages/material-dna/src/index.mjs'].map(path=>new URL(path,ROOT).href);
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(c=>c.addAll(urls)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key!==CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',event=>{
 const request=event.request;
 if(request.method!=='GET'||new URL(request.url).origin!==location.origin)return;
 event.respondWith(caches.match(request).then(cached=>cached||fetch(request).then(response=>{
  if(response.ok&&response.type==='basic'){
   const copy=response.clone();
   event.waitUntil(caches.open(CACHE).then(c=>c.put(request,copy)));
  }
  return response;
 })));
});
