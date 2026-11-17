import {analyzePixels} from './index.mjs';
/** Injection-based engine: caller provides an ONNX adapter only when validated weights are installed. */
export class MaterialEngine {
 constructor({classifier=null}={}){this.classifier=classifier}
 async analyze(rgba,options={}){
  const base=analyzePixels(rgba,options);
  if(!this.classifier)return {...base,execution:{mode:'local',engine:'deterministic',classifierAvailable:false}};
  try {
   const estimate=await this.classifier.classify(rgba,options);
   if(!estimate||typeof estimate.family!=='string'||!Number.isFinite(estimate.confidence)||estimate.confidence<0||estimate.confidence>1)throw Error('Invalid classifier output');
   const uncertain=estimate.confidence<(options.minConfidence??0.65);
   base.classification={family:uncertain?'unknown':estimate.family,finish:estimate.finish??null,confidence:estimate.confidence,verified:false};
   base.provenance={level:'model-estimated',modelVersion:this.classifier.version??'unversioned'};
   return {...base,execution:{mode:'local',engine:this.classifier.name??'classifier',classifierAvailable:true}};
  }catch(error){return {...base,execution:{mode:'local',engine:'deterministic',classifierAvailable:false,error:String(error.message||error)}}}
 }
}
