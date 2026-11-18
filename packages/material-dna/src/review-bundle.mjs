import {validateReview} from './review.mjs';

/**
 * Review files are untrusted user-supplied JSON. Validate identifiers, decisions,
 * evidence and timestamps before adding to local review state.
 * Never infer material-family labels from the upstream catalog.
 */
export function importReviewBundle(bundle,sourceAssetIds,{maxReviews=10000}={}){
 if(!bundle||typeof bundle!=='object'||bundle.schemaVersion!=='0.1.0'
    ||!Array.isArray(bundle.reviews))throw new Error('Unsupported Kumo review bundle');
 if(bundle.reviews.length>maxReviews)throw new Error('Review bundle exceeds size limit');
 if(!(sourceAssetIds instanceof Set))throw new TypeError('Expected set of source asset IDs');
 const results=new Map();
 for(const [i,record] of bundle.reviews.entries()){
   const validation=validateReview(record,sourceAssetIds);
   if(!validation.valid)throw new Error('Invalid review '+i+': '+validation.errors.join(', '));
   if(results.has(record.asset_id))throw new Error('Duplicate review for '+record.asset_id);
   results.set(record.asset_id,{...record,notes:String(record.notes||'').slice(0,1200)});
 }
 return results;
}

export function mergeReviews(current,incoming,{replace=false}={}){
 if(!(current instanceof Map)||!(incoming instanceof Map))throw new TypeError('Expected review maps');
 const result=replace?new Map():new Map(current);
 for(const [id,review] of incoming)result.set(id,review);
 return result;
}

export function exportReviewBundle(reviews){
 const values=reviews instanceof Map?[...reviews.values()]:reviews;
 if(!Array.isArray(values))throw new TypeError('Expected review map or array');
 return {schemaVersion:'0.1.0',source:'Poly Haven',exported_at:new Date().toISOString(),reviews:values};
}

export function summarizeReviewQuality(items,reviews){
 const ids=new Set(items.map(x=>x.asset_id));
 const counts={included:0,excluded:0,uncertain:0,pending:0,visualOnly:0,sourceVerified:0,invalid:0};
 for(const id of ids){
  const review=reviews.get(id);
  if(!review){counts.pending++;continue;}
  if(!validateReview(review,ids).valid){counts.invalid++;continue;}
  if(review.decision==='include'){
    counts.included++;
    if(review.evidence==='visual-only')counts.visualOnly++;
    if(review.evidence==='source-verified')counts.sourceVerified++;
  } else if(review.decision==='exclude')counts.excluded++;
  else counts.uncertain++;
 }
 return counts;
}
