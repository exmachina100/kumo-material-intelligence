import {rgbToHex,rgbToLab,sampleDominantColor} from '../../cmf-core/src/color.mjs';
/** MVP adapter; classification intentionally unknown until a validated model is installed. */
export function analyzePixels(rgba, {sourceName='uploaded image', stride=12}={}) {
 const rgb=sampleDominantColor(rgba,{stride});
 return {
  schemaVersion:'0.1.0',id:globalThis.crypto?.randomUUID?.()||'local-material',
  source:{type:'image',name:sourceName},color:{hex:rgbToHex(rgb),rgb,lab:rgbToLab(rgb)},
  classification:{family:'unknown',finish:null,confidence:null,verified:false},
  appearance:{roughnessEstimate:null,metallicEstimate:null},
  provenance:{level:'algorithmic',modelVersion:null},createdAt:new Date().toISOString()
 };
}