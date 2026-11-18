/** Local capture-condition diagnostics. No classifier and no material claims. */
export function inspectImageQuality(rgba,width,height,{stride=3}={}){
 if(!Number.isInteger(width)||width<1||!Number.isInteger(height)||height<1
    ||!rgba||rgba.length!==width*height*4)throw new TypeError('Valid RGBA data and dimensions required');
 const step=Math.max(1,Math.floor(stride));
 let samples=0,lumSum=0,lumSquared=0,nearBlack=0,clipped=0,valid=0;
 const brightness=new Float32Array(width*height);
 const opacity=new Uint8Array(width*height);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
   const i=(y*width+x)*4;
   if(rgba[i+3]<128)continue;
   opacity[y*width+x]=1;valid++;
   const l=.2126*rgba[i]+.7152*rgba[i+1]+.0722*rgba[i+2];
   brightness[y*width+x]=l;
   if(x%step||y%step)continue;
   samples++;lumSum+=l;lumSquared+=l*l;
   if(l<25)nearBlack++;
   if(Math.max(rgba[i],rgba[i+1],rgba[i+2])>=250)clipped++;
 }
 const mean=samples?lumSum/samples:0;
 const sigma=samples?Math.sqrt(Math.max(0,lumSquared/samples-mean*mean)):0;
 let edgeSum=0,edgeCount=0;
 for(let y=1;y<height-1;y+=step)for(let x=1;x<width-1;x+=step){
   const k=y*width+x;
   if(!(opacity[k]&&opacity[k-1]&&opacity[k+1]&&opacity[k-width]&&opacity[k+width]))continue;
   const value=4*brightness[k]-brightness[k-1]-brightness[k+1]-brightness[k-width]-brightness[k+width];
   edgeSum+=value*value;edgeCount++;
 }
 const edgeEnergy=edgeCount?Math.sqrt(edgeSum/edgeCount):0;
 const nearBlackFraction=samples?nearBlack/samples:0;
 const highlightFraction=samples?clipped/samples:0;
 const warnings=[];
 if(valid===0)warnings.push({code:'no_visible_pixels',text:'No opaque image pixels were detected'});
 if(mean<45||nearBlackFraction>.45)warnings.push({code:'underexposed',text:'Very dark image; consider more diffuse lighting'});
 if(highlightFraction>.15)warnings.push({code:'clipped_highlights',text:'Many bright clipped pixels; reduce glare or exposure'});
 if(width<96||height<96)warnings.push({code:'low_resolution',text:'Low-resolution capture; fine texture may be missed'});
 if(edgeEnergy<2.5 && sigma<20 && valid>0)warnings.push({code:'low_visual_detail',text:'Very little visible texture or contrast; check focus and framing'});
 return {
  schemaVersion:'0.1.0',kind:'capture-condition-hints',width,height,
  meanLuma:Math.round(mean*10)/10,lumaStd:Math.round(sigma*10)/10,
  clippedHighlightFraction:Math.round(highlightFraction*1000)/1000,
  nearBlackFraction:Math.round(nearBlackFraction*1000)/1000,
  edgeEnergy:Math.round(edgeEnergy*10)/10,
  warnings,
  note:'Advisory visual diagnostics; uniform materials can legitimately have low detail. No physical material properties inferred.'
 };
}
