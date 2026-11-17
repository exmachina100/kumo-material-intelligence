"""Create a verified browser model manifest from trained local artifacts.

Only packages artifacts that actually exist. Does not push weights to Git.
Usage: python training/package_model.py --model training/out/materialnet.onnx
"""
import argparse, hashlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--labels',type=Path,default=Path('training/out/labels.json'))
    p.add_argument('--output',type=Path,default=Path('training/out/browser-manifest.json'))
    p.add_argument('--version',default='materialnet-s-0.1.0')
    p.add_argument('--image-size',type=int,default=160)
    a=p.parse_args()
    if not a.model.is_file() or not a.labels.is_file():p.error('Trained model and labels.json are required')
    labels=json.loads(a.labels.read_text())
    if not isinstance(labels,list) or len(labels)<2 or not all(isinstance(v,str) and v for v in labels):p.error('Invalid labels')
    digest=hashlib.file_digest(a.model.open('rb'),'sha256').hexdigest()
    manifest={
      'schemaVersion':'0.1.0','version':a.version,'format':'onnx',
      'artifact':a.model.name,'sha256':digest,'bytes':a.model.stat().st_size,
      'input':{'name':'image','shape':[1,3,a.image_size,a.image_size],
               'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},
               'resize':'bilinear','colorspace':'RGB'},
      'labels':labels,
      'review':{'approved':False,'datasetLicenseReviewed':False,
                'accuracyBenchmarked':False,'browserParityVerified':False},
      'note':'Not deployable until review flags are supported by documented evidence'
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(a.output)
if __name__=='__main__':main()
