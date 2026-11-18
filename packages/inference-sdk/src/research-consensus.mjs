/**
 * Research-only inference consistency summary over distinct image crops.
 * This is NOT a confidence-calibration procedure, nor a production permission.
 */
export function summarizeCropPredictions(predictions,{minimumScore=0.85,minimumAgreement=0.75}={}){
 if(!Array.isArray(predictions)||predictions.length<2)throw new Error('At least two crop predictions required');
 if(!Number.isFinite(minimumScore)||minimumScore<0||minimumScore>1)throw new RangeError('Invalid score threshold');
 if(!Number.isFinite(minimumAgreement)||minimumAgreement<0||minimumAgreement>1)throw new RangeError('Invalid agreement threshold');
 const votes=new Map();
 for(const p of predictions){
  if(!p||typeof p.family!=='string'||!p.family||!Number.isFinite(p.confidence)||p.confidence<0||p.confidence>1)
   throw new TypeError('Invalid crop prediction');
  const v=votes.get(p.family)||{count:0,sum:0};
  v.count++;v.sum+=p.confidence;votes.set(p.family,v);
 }
 const ranking=[...votes.entries()].sort((a,b)=>b[1].count-a[1].count||b[1].sum-a[1].sum);
 const [family,top]=ranking[0],agreement=top.count/predictions.length,meanRawScore=top.sum/top.count;
 const withheld=agreement<minimumAgreement||meanRawScore<minimumScore;
 return {family:withheld?'unknown':family,topGuess:family,
   agreement,meanRawScore,
   withheld,reason:agreement<minimumAgreement?'crop_disagreement':meanRawScore<minimumScore?'low_raw_score':'unverified_guess',
   predictions:predictions.map(p=>({family:p.family,rawSoftmax:p.confidence})),
   disclaimer:'Research crop-consistency diagnostic, not calibrated probability or physical-material verification'};
}
