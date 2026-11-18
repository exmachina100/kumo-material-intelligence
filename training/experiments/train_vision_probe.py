"""Research-only compact pretrained vision encoder + linear material-category probe.

This uses genuine pretrained MobileNetV3-Small visual features, but only weak
Poly Haven catalog labels. NEVER deploy or represent the result as a verified
CMF material classifier. The held-out set measures agreement with noisy
catalog hints only.

Usage:
  python training/experiments/train_vision_probe.py --source training/external/polyhaven_cc0_staging

All generated results go to training/out/vision-probe (ignored by Git).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from vision_dataset import select_multisource_samples
from rejection_metrics import rejection_summary

SEED = 42
IMAGE_SIZE = 224
MEAN = [0.485,0.456,0.406]
STD = [0.229,0.224,0.225]

class PreviewImages(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE),interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        with Image.open(self.rows[index]["path"]) as source:
            image = source.convert("RGB")
        return self.transform(image)

class Encoder(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self,x):
        return torch.flatten(self.avgpool(self.features(x)),1)

class FullProbe(nn.Module):
    def __init__(self,encoder,scaler_mean,scaler_scale,coef,intercept):
        super().__init__()
        self.encoder=encoder
        self.register_buffer("center",torch.tensor(scaler_mean,dtype=torch.float32))
        self.register_buffer("scale",torch.tensor(scaler_scale,dtype=torch.float32))
        self.register_buffer("coef",torch.tensor(coef,dtype=torch.float32))
        self.register_buffer("intercept",torch.tensor(intercept,dtype=torch.float32))

    def forward(self,x):
        features=(self.encoder(x)-self.center)/self.scale
        return features @ self.coef.T + self.intercept

def extract_embeddings(encoder,rows,batch_size):
    loader=DataLoader(PreviewImages(rows),batch_size=batch_size,shuffle=False,num_workers=0)
    chunks=[]
    encoder.eval()
    with torch.inference_mode():
        for images in loader:
            chunks.append(encoder(images).cpu().numpy())
    return np.concatenate(chunks,axis=0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=Path("training/external/polyhaven_cc0_staging"))
    ap.add_argument("--output",type=Path,default=Path("training/out/vision-probe"))
    ap.add_argument("--min-class-samples",type=int,default=8)
    ap.add_argument("--batch-size",type=int,default=12)
    ap.add_argument("--ambient-source",type=Path,default=Path("training/external/ambientcg_cc0_staging"))
    ap.add_argument("--skip-onnx",action="store_true",help="Only run research evaluation")
    args=ap.parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    source_dirs=[args.source]
    if (args.ambient_source/"provenance.json").is_file():
        source_dirs.append(args.ambient_source)
    rows=select_multisource_samples(source_dirs,args.min_class_samples)
    labels=np.array([x["weak_label"] for x in rows])
    classes=sorted(set(labels.tolist()))
    idx=np.arange(len(rows))
    train,test=train_test_split(idx,test_size=.25,stratify=labels,random_state=SEED)
    # A frozen pretrained ImageNet visual encoder rather than a randomly trained
    # classifier on a few dozen CMF reference thumbnails.
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    backbone=models.mobilenet_v3_small(weights=weights)
    encoder=Encoder(backbone).eval()
    start=time.perf_counter()
    embeddings=extract_embeddings(encoder,rows,args.batch_size)
    scaler=StandardScaler().fit(embeddings[train])
    head=LogisticRegression(max_iter=1500,class_weight="balanced",random_state=SEED)
    head.fit(scaler.transform(embeddings[train]),labels[train])
    predicted=head.predict(scaler.transform(embeddings[test]))
    scores=head.predict_proba(scaler.transform(embeddings[test]))
    # Report performance separately by upstream material library. This is
    # still a weak-label benchmark, but it reveals source/domain imbalance.
    by_source={}
    for upstream in sorted({row["source"] for row in rows}):
        positions=[j for j,i in enumerate(test) if rows[int(i)]["source"]==upstream]
        if not positions:continue
        actual_source=np.asarray(labels[test])[positions]
        predicted_source=np.asarray(predicted)[positions]
        by_source[upstream]={
            "test_count":len(positions),
            "proxy_label_accuracy":float(accuracy_score(actual_source,predicted_source)),
            "macro_f1":float(f1_score(actual_source,predicted_source,average="macro",zero_division=0))
        }
    report={
        "experiment":"vision-probe-0.3-multisource","status":"RESEARCH_ONLY",
        "release_eligible":False,"expert_reviewed":False,
        "warning":"Predictions are against weak catalog categories and do NOT measure real-world material recognition.",
        "image_encoder":"torchvision mobilenet_v3_small IMAGENET1K_V1",
        "frozen_backbone":True,
        "features_dimension":int(embeddings.shape[1]),
        "preprocessing":{"size":[IMAGE_SIZE,IMAGE_SIZE],"resize":"bilinear","normalization":{"mean":MEAN,"std":STD}},
        "dataset":{"source":"Poly Haven + ambientCG CC0 preview textures","samples":len(rows),
                   "source_counts":dict(Counter(item["source"] for item in rows)),
                   "class_counts":dict(Counter(labels)),"train":len(train),"test":len(test),
                   "split_seed":SEED,"split_unit":"unique source asset id + matching checksum",
                   "labels_are_weak_catalog_hints":True},
        "classes":classes,
        "accuracy_vs_proxy":float(accuracy_score(labels[test],predicted)),
        "macro_f1_vs_proxy":float(f1_score(labels[test],predicted,average="macro",zero_division=0)),
        "by_source_weak_label_metrics":by_source,
        "classification_report":classification_report(labels[test],predicted,output_dict=True,zero_division=0),
        "confusion_matrix":confusion_matrix(labels[test],predicted,labels=classes).tolist(),
        "heldout_predictions":[{"asset_id":rows[int(i)]["asset_id"],"weak_label":str(labels[i]),
                                "prediction":str(predicted[j]),"max_softmax":float(max(scores[j]))}
                               for j,i in enumerate(test)],
        "runtime_seconds":round(time.perf_counter()-start,1),
    }
    # Probe explicit out-of-label catalog categories as a tiny OOD sanity check.
    # These remain weak source hints, not a real automotive unknown-detection set.
    source_records=[]
    for folder in source_dirs:
        source_records.append((folder,json.loads((folder/"provenance.json").read_text())))
    trained_ids={item["asset_id"] for item in rows}
    seen_ood_hash=set()
    ood=[]
    for folder,records in source_records:
        for item in records:
            if item.get("source") not in ("Poly Haven","ambientCG") or item.get("license")!="CC0":
                continue
            if item.get("asset_id") in trained_ids or item.get("catalog_suggestion") in classes:
                continue
            name=item.get("path")
            if not isinstance(name,str) or not name.endswith(".jpg") or Path(name).name!=name:
                continue
            image_path=folder/name
            if not image_path.is_file():
                continue
            hash_value=hashlib.sha256(image_path.read_bytes()).hexdigest()
            if hash_value!=item.get("sha256") or hash_value in seen_ood_hash:
                continue
            seen_ood_hash.add(hash_value)
            ood.append({"asset_id":item["asset_id"],"path":str(image_path),
                        "catalog_category":item.get("catalog_suggestion") or "other"})
    if ood:
        out_embedding=extract_embeddings(encoder,ood,args.batch_size)
        out_scores=head.predict_proba(scaler.transform(out_embedding)).max(axis=1)
        diag=rejection_summary(scores.max(axis=1),out_scores)
        diag["excluded_catalog_categories"]=dict(Counter(x["catalog_category"] for x in ood))
    else:
        diag=rejection_summary(scores.max(axis=1),[])
    report["unknown_rejection_catalog_proxy"]=diag
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output/"labels.json").write_text(json.dumps(list(head.classes_),indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("classification_report","confusion_matrix","heldout_predictions")},indent=2),flush=True)
    if not args.skip_onnx:
        if len(head.classes_)<3:
            raise ValueError("Multiclass ONNX export requires >=3 classes for this experiment")
        probe=FullProbe(encoder,scaler.mean_,scaler.scale_,head.coef_,head.intercept_).eval()
        example=torch.rand(1,3,IMAGE_SIZE,IMAGE_SIZE)
        onnx_path=args.output/"vision-probe-research.onnx"
        torch.onnx.export(probe,example,str(onnx_path),
                          input_names=["image"],output_names=["logits"],
                          opset_version=17,dynamo=False,do_constant_folding=True)
        import onnxruntime as ort
        inference=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected=probe(example).cpu().numpy()
        actual=inference.run(["logits"],{"image":example.numpy()})[0]
        max_diff=float(np.max(np.abs(actual-expected)))
        report["onnx_export"]={"success":True,"max_abs_logit_diff":max_diff,
                                "bytes":onnx_path.stat().st_size}
        if max_diff>0.025:
            raise RuntimeError(f"ONNX parity failed: {max_diff}")
        (args.output/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n")
        print("ONNX parity OK, max absolute logits difference:",max_diff,flush=True)
    print("WARNING: model is RESEARCH ONLY; never auto-deploy into production.",flush=True)

if __name__=="__main__":
    main()
