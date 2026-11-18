"""MaterialNet-v2 research fine-tuning: transfer-learning, not a frozen linear probe.

Uses source-grouped train/val/test, class-aware sampling, augmentation,
two-stage MobileNetV3 fine-tuning, calibration diagnostics, and a same-split
frozen-encoder baseline. NO GPT synthetic data enters this benchmark.
Unverified CC0 source catalog labels remain *weak*, NOT physical truth.

Example:
 python training/experiments/finetune_materialnet.py --head-epochs 2 --finetune-epochs 4
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from vision_dataset import select_multisource_samples
from material_splits import group_stratified_split

SIZE=224
MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]


def transforms_for(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(SIZE,scale=(.74,1.0),ratio=(.87,1.15),antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=.25),
            transforms.ColorJitter(brightness=.14,contrast=.13,saturation=.12,hue=.025),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3,sigma=(.1,.5))],p=.10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD)
        ])
    return transforms.Compose([
        transforms.Resize((SIZE,SIZE),antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])


class MaterialRows(Dataset):
    def __init__(self,rows,indices,labels,train=False):
        self.rows=rows
        self.indices=indices
        self.label_indices=[labels.index(rows[i]["weak_label"]) for i in indices]
        self.tf=transforms_for(train)

    def __len__(self):return len(self.indices)

    def __getitem__(self,k):
        with Image.open(self.rows[self.indices[k]]["path"]) as img:
            tensor=self.tf(img.convert("RGB"))
        return tensor,self.label_indices[k]


def balanced_loader(ds,batch=16):
    counts=Counter(ds.label_indices)
    weights=[1.0/(counts[x]**.65) for x in ds.label_indices]
    sampler=WeightedRandomSampler(weights,num_samples=len(weights),replacement=True)
    return DataLoader(ds,batch_size=batch,sampler=sampler,num_workers=0)


def predict_logits(model,loader,device):
    model.eval()
    scores=[];targets=[]
    with torch.inference_mode():
        for x,y in loader:
            result=model(x.to(device))
            scores.append(result.cpu().numpy())
            targets.extend(y.numpy().tolist())
    return np.concatenate(scores),np.asarray(targets)


def measure(y_true,y_pred,labels):
    return {
        "count":len(y_true),
        "accuracy":float(accuracy_score(y_true,y_pred)),
        "macro_f1":float(f1_score(y_true,y_pred,labels=list(range(len(labels))),average="macro",zero_division=0)),
        "classification_report":classification_report(y_true,y_pred,labels=list(range(len(labels))),
                              target_names=labels,output_dict=True,zero_division=0),
        "confusion_matrix":confusion_matrix(y_true,y_pred,labels=list(range(len(labels)))).tolist()
    }


def set_trainable_stage(model,phase):
    for p in model.features.parameters():p.requires_grad=False
    for p in model.classifier.parameters():p.requires_grad=True
    if phase=="finetune":
        for block in list(model.features.children())[-3:]:
            for p in block.parameters():p.requires_grad=True


def train_one_epoch(model,loader,optim,device,phase):
    model.train()
    # Locked feature extractor and early normalization layers do not drift.
    if phase=="head":
        model.features.eval()
    else:
        for block in list(model.features.children())[:-3]:
            block.eval()
    losses=[]
    for x,y in loader:
        x=x.to(device);y=y.to(device)
        optim.zero_grad(set_to_none=True)
        logits=model(x)
        loss=F.cross_entropy(logits,y,label_smoothing=.08)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.5)
        optim.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def calibrate_temperature(logits,targets):
    tlog=torch.tensor(logits,dtype=torch.float32)
    ys=torch.tensor(targets,dtype=torch.long)
    values=[.75,1.0,1.25,1.5,1.75,2.,2.5,3.,4.,5.]
    losses=[float(F.cross_entropy(tlog/t,ys).item()) for t in values]
    best=int(np.argmin(losses))
    return {"temperature":values[best],"validation_nll":losses[best],
            "grid":dict(zip(map(str,values),[round(x,5) for x in losses]))}


def baseline_features(backbone,ds,batch,device):
    loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0)
    backbone.eval()
    feats=[]
    with torch.inference_mode():
        for x,_ in loader:
            f=backbone.avgpool(backbone.features(x.to(device))).flatten(1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats,axis=0)


def run(args):
    t0=time.time()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    folders=[Path(args.polyhaven),Path(args.ambientcg)]
    rows=select_multisource_samples(folders,args.min_class_samples)
    labels=sorted({r["weak_label"] for r in rows})
    splits=group_stratified_split(rows,seed=args.seed)
    train_ds=MaterialRows(rows,splits["train"],labels,train=True)
    train_eval_ds=MaterialRows(rows,splits["train"],labels)
    val_ds=MaterialRows(rows,splits["val"],labels)
    test_ds=MaterialRows(rows,splits["test"],labels)
    device="cuda" if torch.cuda.is_available() else "cpu"
    train_loader=balanced_loader(train_ds,args.batch)
    val_loader=DataLoader(val_ds,batch_size=args.batch)
    test_loader=DataLoader(test_ds,batch_size=args.batch)
    backbone=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    baseline_backbone=copy.deepcopy(backbone).to(device).eval()
    for p in baseline_backbone.parameters():p.requires_grad=False

    # Equal test split: compare to a frozen visual representation baseline.
    xb=baseline_features(baseline_backbone,train_eval_ds,args.batch,device)
    xt=baseline_features(baseline_backbone,test_ds,args.batch,device)
    sc=StandardScaler().fit(xb)
    base=LogisticRegression(max_iter=1200,class_weight="balanced",random_state=args.seed)
    base.fit(sc.transform(xb),np.array(train_eval_ds.label_indices))
    baseline_pred=base.predict(sc.transform(xt))
    baseline=measure(np.array(test_ds.label_indices),baseline_pred,labels)
    del baseline_backbone
    print(f"Same-split frozen-baseline macro F1: {baseline['macro_f1']:.4f}",flush=True)

    backbone.classifier[-1]=nn.Linear(backbone.classifier[-1].in_features,len(labels))
    model=backbone.to(device)
    best_state=None;best_f1=-1;history=[];stale=0
    phases=[("head",args.head_epochs),("finetune",args.finetune_epochs)]
    for phase,epochs in phases:
        if epochs<=0:continue
        set_trainable_stage(model,phase)
        params=[{"params":model.classifier.parameters(),"lr":args.head_lr if phase=="head" else args.finetune_head_lr}]
        if phase=="finetune":
            tail=list(model.features.children())[-3:]
            params.append({"params":[p for block in tail for p in block.parameters() if p.requires_grad],"lr":args.backbone_lr})
        optimizer=torch.optim.AdamW(params,weight_decay=.018)
        for epoch in range(epochs):
            loss=train_one_epoch(model,train_loader,optimizer,device,phase)
            val_logits,val_y=predict_logits(model,val_loader,device)
            val_pred=val_logits.argmax(axis=1)
            val_f1=float(f1_score(val_y,val_pred,labels=list(range(len(labels))),average="macro",zero_division=0))
            val_acc=float(accuracy_score(val_y,val_pred))
            history.append({"phase":phase,"epoch":epoch+1,"training_loss":loss,
                            "val_macro_f1":val_f1,"val_accuracy":val_acc})
            print(f"{phase} epoch={epoch+1}, loss={loss:.4f}, val_macro_f1={val_f1:.4f}, val_acc={val_acc:.4f}",flush=True)
            if val_f1>best_f1+0.0001:
                best_f1=val_f1
                best_state=copy.deepcopy({k:v.cpu().clone() for k,v in model.state_dict().items()})
                stale=0
            else:
                stale+=1
            if stale>=args.patience and phase=="finetune":
                print("Early stopping based only on validation macro F1",flush=True)
                break
        if stale>=args.patience and phase=="finetune":break
    if best_state is None:raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state);model.eval()
    val_logits,val_y=predict_logits(model,val_loader,device)
    calibration=calibrate_temperature(val_logits,val_y)
    test_logits,test_y=predict_logits(model,test_loader,device)
    test_pred=test_logits.argmax(axis=1)
    fine=measure(test_y,test_pred,labels)
    by_source={}
    for source in sorted({rows[i]["source"] for i in splits["test"]}):
        positions=[j for j,i in enumerate(splits["test"]) if rows[i]["source"]==source]
        if not positions:continue
        by_source[source]=measure(test_y[positions],test_pred[positions],labels)
    all_probs=torch.softmax(torch.tensor(test_logits)/calibration["temperature"],dim=1).numpy()
    confidence=all_probs.max(axis=1)
    uncertain={str(t):{
        "accepted":int((confidence>=t).sum()),
        "coverage":float((confidence>=t).mean()),
        "accepted_correct_fraction":float((test_pred[confidence>=t]==test_y[confidence>=t]).mean())
          if (confidence>=t).sum() else None
    } for t in (.5,.65,.8,.9)}
    # All outcomes are against catalog-derived weak labels and have no ground-truth status.
    report={
       "experiment":"materialnet-v2-finetune-research","status":"RESEARCH_ONLY",
       "release_eligible":False,"expert_reviewed":False,"synthetic_training_used":False,
       "warning":"All labels are weak catalog hints; physical material recognition is NOT verified.",
       "architecture":"torchvision ImageNet-pretrained MobileNetV3-Small, head warmup + last-three-feature-block fine-tune",
       "augmentation":"moderate random crop/flip/color jitter/minor blur; class-aware weighted sampling",
       "split_method":"deterministic source-group stratified train/validation/test",
       "dataset_count":len(rows),"train":len(splits["train"]),"val":len(splits["val"]),"test":len(splits["test"]),
       "class_counts":dict(Counter(x["weak_label"] for x in rows)),
       "classes":labels,"seed":args.seed,
       "baseline_frozen_same_split":baseline,
       "fine_tuned_same_split":fine,
       "macro_f1_delta_vs_frozen":float(fine["macro_f1"]-baseline["macro_f1"]),
       "validation_best_macro_f1":best_f1,
       "temperature_diagnostic_only":calibration,
       "raw_softmax_not_probabilities":True,
       "threshold_diagnostics_not_ood_validation":uncertain,
       "source_heldout_metrics":by_source,
       "history":history,
       "split_assets":{split:[rows[i]["asset_id"] for i in indices] for split,indices in splits.items()},
       "duration_seconds":round(time.time()-t0,2),
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"labels.json").write_text(json.dumps(labels,indent=2)+"\n")
    torch.save(model.cpu().state_dict(),out/"materialnet-v2-research.pt")
    # Research-only ONNX exports only after train/test metrics already locked.
    model.eval()
    x=torch.rand(1,3,SIZE,SIZE)
    onnx_path=out/"materialnet-v2-research.onnx"
    torch.onnx.export(model,x,str(onnx_path),
        input_names=["image"],output_names=["logits"],
        opset_version=17,do_constant_folding=True,dynamo=False)
    import onnxruntime as ort
    session=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"])
    torch_out=model(x).detach().numpy()
    ort_out=session.run(["logits"],{"image":x.numpy()})[0]
    delta=float(np.max(np.abs(torch_out-ort_out)))
    report["onnx_export"]={"verified":delta<.03,"max_abs_logit_diff":delta,
                          "bytes":onnx_path.stat().st_size}
    if delta>=.03:raise RuntimeError(f"ONNX parity failed: {delta}")
    (out/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
       "classes":labels,"samples":len(rows),"train":len(train_ds),"val":len(val_ds),"test":len(test_ds),
       "baseline_macro_f1":baseline["macro_f1"],"finetuned_macro_f1":fine["macro_f1"],
       "macro_f1_delta":report["macro_f1_delta_vs_frozen"],
       "onnx_verified":True,"runtime_seconds":report["duration_seconds"],
       "release_eligible":False
    },indent=2),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--polyhaven",default="training/external/polyhaven_cc0_staging")
    p.add_argument("--ambientcg",default="training/external/ambientcg_cc0_staging")
    p.add_argument("--out",default="training/out/materialnet-v2")
    p.add_argument("--head-epochs",type=int,default=2)
    p.add_argument("--finetune-epochs",type=int,default=4)
    p.add_argument("--batch",type=int,default=16)
    p.add_argument("--patience",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--min-class-samples",type=int,default=8)
    p.add_argument("--head-lr",type=float,default=.001)
    p.add_argument("--finetune-head-lr",type=float,default=.00035)
    p.add_argument("--backbone-lr",type=float,default=.000065)
    run(p.parse_args())
