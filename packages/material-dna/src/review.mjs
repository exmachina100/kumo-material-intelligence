import {FAMILIES} from './index.mjs';
export const REVIEW_SCHEMA_VERSION='0.1.0';
export const DECISIONS=['include','exclude','uncertain'];
export function validateReview(review,sourceIds){
  const errors=[];
  if(review?.schemaVersion!==REVIEW_SCHEMA_VERSION)errors.push('Unsupported review schema');
  if(typeof review?.asset_id!=='string'||!review.asset_id)errors.push('Missing asset_id');
  else if(sourceIds && !sourceIds.has(review.asset_id))errors.push('Asset absent from source manifest');
  if(!DECISIONS.includes(review?.decision))errors.push('Invalid decision');
  if(typeof review?.reviewer!=='string'||review.reviewer.trim().length<2)errors.push('Reviewer name required');
  if(review?.decision==='include'){
    if(!FAMILIES.includes(review.family)||review.family==='unknown')errors.push('An explicit CMF family is required');
    if(!['visual-only','source-verified'].includes(review.evidence))errors.push('Select an evidence level');
  }
  if(typeof review?.reviewed_at!=='string'||!Number.isFinite(Date.parse(review.reviewed_at)))errors.push('Invalid review timestamp');
  return {valid:errors.length===0,errors};
}
export function makeReview({asset_id,decision,family=null,evidence='visual-only',notes='',reviewer}){
  return {schemaVersion:REVIEW_SCHEMA_VERSION,asset_id,decision,
    family:decision==='include'?family:null,
    evidence:decision==='include'?evidence:null,
    notes:String(notes).slice(0,1200),reviewer:reviewer.trim(),
    reviewed_at:new Date().toISOString()};
}
export function reviewStats(items,reviews){
 const ids=new Set(items.map(x=>x.asset_id)),valid=reviews.filter(x=>validateReview(x,ids).valid);
 return {total:ids.size,reviewed:valid.filter(x=>x.decision==='include').length,
  excluded:valid.filter(x=>x.decision==='exclude').length,
  uncertain:valid.filter(x=>x.decision==='uncertain').length,
  pending:Math.max(0,ids.size-valid.length)};
}
