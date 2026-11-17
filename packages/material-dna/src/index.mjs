export const FAMILIES=['unknown','metal','polymer','leather','synthetic-leather','textile','wood','stone','ceramic','glass','rubber','composite','paper','coating'];
export const FINISHES=['matte','satin','gloss','high-gloss','brushed','polished','textured','embossed','anodized','painted','unfinished'];
export function validateMaterial(record){
 const errors=[];
 if(!record||typeof record!=='object')return {valid:false,errors:['Record is not an object']};
 if(record.schemaVersion!=='0.1.0')errors.push('Unsupported schemaVersion');
 if(typeof record.id!=='string'||!record.id)errors.push('Invalid id');
 if(!record.color||!/^#[0-9A-Fa-f]{6}$/.test(record.color.hex||''))errors.push('Invalid color.hex');
 if(!Array.isArray(record.color?.rgb)||record.color.rgb.length!==3||record.color.rgb.some(x=>!Number.isInteger(x)||x<0||x>255))errors.push('Invalid color.rgb');
 if(!FAMILIES.includes(record.classification?.family))errors.push('Unknown material family');
 if(!['algorithmic','model-estimated','designer-authored','supplier-verified'].includes(record.provenance?.level))errors.push('Missing provenance level');
 if(record.classification?.confidence!=null&&(!Number.isFinite(record.classification.confidence)||record.classification.confidence<0||record.classification.confidence>1))errors.push('Invalid confidence');
 return {valid:errors.length===0,errors};
}
export function normalizeMaterial(input){
 const out=structuredClone(input);
 if(!FAMILIES.includes(out.classification?.family)){out.classification={...out.classification,family:'unknown',confidence:null,verified:false}}
 return out;
}
