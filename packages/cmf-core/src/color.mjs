/** Lightweight deterministic color operations, no ML inference or server required. */
export function rgbToHex([r,g,b]) {
  return '#' + [r,g,b].map(x => Math.max(0,Math.min(255,Math.round(x))).toString(16).padStart(2,'0')).join('').toUpperCase();
}
export function rgbToLab(rgb) {
  const linear=rgb.map(v=>{let c=v/255;return c<=0.04045?c/12.92:((c+0.055)/1.055)**2.4;});
  const [r,g,b]=linear;
  const x=(r*.4124564+g*.3575761+b*.1804375)/.95047;
  const y=(r*.2126729+g*.7151522+b*.0721750);
  const z=(r*.0193339+g*.1191920+b*.9503041)/1.08883;
  const f=t=>t>0.008856451679?t**(1/3):t/0.128418549346+4/29;
  const [fx,fy,fz]=[x,y,z].map(f);
  return [116*fy-16,500*(fx-fy),200*(fy-fz)];
}
export function deltaE76(a,b) {return Math.hypot(...a.map((v,i)=>v-b[i]));}
export function sampleDominantColor(data,{stride=16,alphaThreshold=128}={}) {
  const bins=new Map();
  for(let i=0;i<data.length;i+=4*Math.max(1,stride)){
    if(data[i+3]<alphaThreshold)continue;
    const key=[data[i]>>4,data[i+1]>>4,data[i+2]>>4].join(',');
    const prev=bins.get(key)||{count:0,sums:[0,0,0]}; prev.count++;
    for(let c=0;c<3;c++)prev.sums[c]+=data[i+c]; bins.set(key,prev);
  }
  let best=null;for(const b of bins.values())if(!best||b.count>best.count)best=b;
  return best?best.sums.map(v=>Math.round(v/best.count)):[0,0,0];
}