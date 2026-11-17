import {rgbToHex,rgbToLab,deltaE76} from './color.mjs';
/** Deterministic quantized histogram palette extraction; no external network or model. */
export function extractPalette(rgba,{maxColors=6,stride=4,alphaThreshold=128}={}) {
 if(!rgba || rgba.length%4!==0)throw new TypeError('Expected RGBA pixel array');
 const bins=new Map();let total=0;
 for(let i=0;i<rgba.length;i+=4*Math.max(1,Math.floor(stride))){
  if(rgba[i+3]<alphaThreshold)continue;
  let r=rgba[i],g=rgba[i+1],b=rgba[i+2],key=[r>>4,g>>4,b>>4].join(':');
  let e=bins.get(key);if(!e){e={n:0,r:0,g:0,b:0};bins.set(key,e)}
  e.n++;e.r+=r;e.g+=g;e.b+=b;total++;
 }
 if(!total)return [];
 let ranked=[...bins.values()].sort((a,b)=>b.n-a.n).slice(0,Math.max(1,maxColors)*8);
 let merged=[];
 for(const e of ranked){
  const rgb=[e.r/e.n,e.g/e.n,e.b/e.n].map(Math.round),lab=rgbToLab(rgb);
  const nearby=merged.find(m=>deltaE76(m.lab,lab)<11);
  if(nearby){nearby.r+=e.r;nearby.g+=e.g;nearby.b+=e.b;nearby.n+=e.n;nearby.rgb=[nearby.r,nearby.g,nearby.b].map(v=>Math.round(v/nearby.n));nearby.lab=rgbToLab(nearby.rgb)}
  else merged.push({...e,rgb,lab});
 }
 return merged.sort((a,b)=>b.n-a.n).slice(0,maxColors).map(e=>({
  hex:rgbToHex(e.rgb),rgb:e.rgb,lab:e.lab,share:Math.round(1000*e.n/total)/10
 }));
}
/** CIEDE2000 color difference, standardized reference implementation in JS. */
export function deltaE2000(lab1,lab2){
 const [L1,a1,b1]=lab1,[L2,a2,b2]=lab2;
 const rad=Math.PI/180,deg=180/Math.PI;const C1=Math.hypot(a1,b1),C2=Math.hypot(a2,b2);
 const Cbar=(C1+C2)/2,G=.5*(1-Math.sqrt(Cbar**7/(Cbar**7+25**7)));
 const ap1=(1+G)*a1,ap2=(1+G)*a2,Cp1=Math.hypot(ap1,b1),Cp2=Math.hypot(ap2,b2);
 const angle=(x,y)=>((Math.atan2(y,x)*deg)%360+360)%360;
 const h1=angle(ap1,b1),h2=angle(ap2,b2);
 const dL=L2-L1,dC=Cp2-Cp1;
 let dh=h2-h1;if(Cp1*Cp2===0)dh=0;else if(dh>180)dh-=360;else if(dh< -180)dh+=360;
 const dH=2*Math.sqrt(Cp1*Cp2)*Math.sin(dh*rad/2);
 const Lbar=(L1+L2)/2,Cbarp=(Cp1+Cp2)/2;
 let hbar;if(Cp1*Cp2===0)hbar=h1+h2;else if(Math.abs(h1-h2)<=180)hbar=(h1+h2)/2;else hbar=(h1+h2+ (h1+h2<360?360:-360))/2;
 const T=1-.17*Math.cos((hbar-30)*rad)+.24*Math.cos(2*hbar*rad)+.32*Math.cos((3*hbar+6)*rad)-.20*Math.cos((4*hbar-63)*rad);
 const dTheta=30*Math.exp(-((hbar-275)/25)**2),Rc=2*Math.sqrt(Cbarp**7/(Cbarp**7+25**7));
 const Sl=1+.015*(Lbar-50)**2/Math.sqrt(20+(Lbar-50)**2),Sc=1+.045*Cbarp,Sh=1+.015*Cbarp*T;
 const Rt=-Math.sin(2*dTheta*rad)*Rc;
 const x=dL/Sl,y=dC/Sc,z=dH/(Sh*Cbarp||1);
 return Math.sqrt(Math.max(0,x*x+y*y+z*z+Rt*y*z));
}
export function rankColorMatches(target,records,limit=8){
 return records.filter(r=>Array.isArray(r.color?.lab)).map(r=>({...r,deltaE:deltaE2000(target,r.color.lab)})).sort((a,b)=>a.deltaE-b.deltaE).slice(0,limit);
}