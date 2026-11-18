"""Research ablation: exactly the same REAL holdout, with/without approved GPT synthetic TRAIN data.

This fits two new material-family classifiers over a pretrained MobileNetV3
visual encoder. No synthetic images, source fingerprints, or GPT prompt labels
may appear in real validation/test. Each arm independently selects regularization
with the same real validation set and then reports real test metrics only.
Not an expert-verified physical material accuracy benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import models

from vision_dataset import select_multisource_samples
from material_splits import group_stratified_split
from finetune_materialnet import MaterialRows,transforms_for,SIZE

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"gpt_synthetic"))
from import_approved_seed import validate as validate_synthetic_approvals


def load_approved_synthetic(archive:Path,real_rows:list,labels:list):
    real_shas={r["sha256"] for r in real_rows}
    included=[];skipped=[];seen=set()
    with zipfile.ZipFile(archive) as pkg:
        names=set(pkg.namelist())
        if any(name.startswith(("val/","test/")) or ".." in Path(name).parts
               or name.startswith("/") for name in names):
            raise ValueError("Unexpected validation/test or unsafe path in synthetic archive")
        manifest=json.loads(pkg.read("manifest.json"))
        records=validate_synthetic_approvals(manifest)
        for r in records:
            name=f'train/{r["family"]}/{r["asset_id"]}.png'
            if name not in names:raise ValueError("Missing synthetic training file: "+name)
            blob=pkg.read(name)
            digest=hashlib.sha256(blob).hexdigest()
            if digest!=r["sha256"] or len(blob)!=r["bytes"]:
                raise ValueError("Synthetic image checksum/size mismatch: "+r["asset_id"])
            if digest in seen or digest in real_shas:
                raise ValueError("Duplicate/repeated real-source or synthetic image")
            seen.add(digest)
            img=Image.open(io.BytesIO(blob));img.verify()
            if r["family"] in labels:
                included.append({"asset_id":r["asset_id"],"weak_label":r["family"],
                                 "sha256":digest,"bytes":blob,
                                 "source":"GPT synthetic train-only"})
            else:
                skipped.append({"asset_id":r["asset_id"],"family":r["family"],
                                "reason":"not an existing model class"})
        expected={f'train/{r["family"]}/{r["asset_id"]}.png' for r in records}|{"manifest.json"}
        if names!=expected:
            raise ValueError("Unexpected files in synthetic training archive")
    return included,skipped


def extract_features(backbone,images,transform,batch,device):
    """Input images: PIL images. Return frozen MobileNetV3 embeddings."""
    feats=[]
    for start in range(0,len(images),batch):
        chunk=[transform(x.convert("RGB")) for x in images[start:start+batch]]
        x=torch.stack(chunk).to(device)
        with torch.inference_mode():
            representation=backbone.avgpool(backbone.features(x)).flatten(1)
        feats.append(representation.cpu().numpy())
    return np.concatenate(feats)


def real_embeddings(backbone,rows,indices,batch,device):
    data=MaterialRows(rows,indices,sorted({r["weak_label"] for r in rows}))
    loader=DataLoader(data,batch_size=batch,shuffle=False,num_workers=0)
    encoded=[]
    with torch.inference_mode():
        backbone.eval()
        for x,_ in loader:
            encoded.append(backbone.avgpool(backbone.features(x.to(device))).flatten(1).cpu().numpy())
    return np.concatenate(encoded)


def select_regularization(X,y,X_val,y_val,class_ids,weights=None):
    leaderboard=[]
    for c in (.03,.10,.3,1.0,3.0,10.0):
        model=LogisticRegression(C=c,max_iter=1700,class_weight="balanced",random_state=42)
        model.fit(X,y,sample_weight=weights)
        predicted=model.predict(X_val)
        score=f1_score(y_val,predicted,labels=list(range(len(class_ids))),
                       average="macro",zero_division=0)
        leaderboard.append((float(score),-c,model))
    score,neg_c,best=max(leaderboard,key=lambda x:(x[0],x[1]))
    return best,score,-neg_c


def metrics(y_true,y_pred,classes):
    return {
        "n":len(y_true),
        "accuracy_weak_catalog":float(accuracy_score(y_true,y_pred)),
        "macro_f1_weak_catalog":float(f1_score(y_true,y_pred,average="macro",
                    labels=list(range(len(classes))),zero_division=0)),
        "by_family":classification_report(y_true,y_pred,labels=list(range(len(classes))),
                    target_names=classes,output_dict=True,zero_division=0),
        "confusion":confusion_matrix(y_true,y_pred,labels=list(range(len(classes)))).tolist(),
    }


def train_and_compare(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    real=select_multisource_samples([Path(args.polyhaven),Path(args.ambientcg)],min_per_class=8)
    labels=sorted(set(r["weak_label"] for r in real))
    splits=group_stratified_split(real,seed=args.seed)
    synthetic,skipped=load_approved_synthetic(Path(args.synthetic_zip),real,labels)

    device="cuda" if torch.cuda.is_available() else "cpu"
    backbone=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    backbone=backbone.to(device).eval()
    for p in backbone.parameters():p.requires_grad=False

    features={split:real_embeddings(backbone,real,indices,args.batch,device)
              for split,indices in splits.items()}
    synthetic_images=[Image.open(io.BytesIO(r["bytes"])).convert("RGB") for r in synthetic]
    synthetic_feats=extract_features(backbone,synthetic_images,transforms_for(False),
                                     args.batch,device) if synthetic else None

    encoder={label:i for i,label in enumerate(labels)}
    y={split:np.array([encoder[real[i]["weak_label"]] for i in indices])
       for split,indices in splits.items()}
    synthetic_y=np.array([encoder[r["weak_label"]] for r in synthetic],dtype=np.int64)

    scaler=StandardScaler().fit(features["train"])
    X={split:scaler.transform(values) for split,values in features.items()}
    Xs=scaler.transform(synthetic_feats) if synthetic_feats is not None else None

    baseline,val_base,c_base=select_regularization(X["train"],y["train"],X["val"],y["val"],labels)
    if synthetic:
        augmented=np.concatenate([X["train"],Xs])
        y_aug=np.concatenate([y["train"],synthetic_y])
        # Keep synthetic influence bounded per family; single rare generated
        # images must never outweigh multiple real train references.
        n_real=Counter(y["train"]);n_synth=Counter(synthetic_y)
        synthetic_weights=np.array([
            min(.75,args.max_synthetic_fraction*n_real[int(label)]/n_synth[int(label)])
            for label in synthetic_y
        ])
        weights=np.concatenate([np.ones(len(y["train"])),synthetic_weights])
        mixed,val_mixed,c_mixed=select_regularization(
            augmented,y_aug,X["val"],y["val"],labels,weights)
    else:
        mixed,val_mixed,c_mixed=baseline,val_base,c_base
        weights=np.ones(len(y["train"]))

    # Both classifiers are independently selected using only REAL validation.
    # The locked real test is evaluated once per arm and never used for selection.
    results={
        "experiment":"materialnet-synthetic-seed-ablation",
        "status":"RESEARCH_ONLY","release_eligible":False,
        "physically_verified":False,
        "synthetic_label_source":"GPT-generated prompts and visual family approvals, NOT physical substrates",
        "benchmark":"weak labels on independent CC0 catalog previews only",
        "classes":labels,"seed":args.seed,
        "real_sample_count":len(real),
        "real_splits":{split:len(indices) for split,indices in splits.items()},
        "real_split_asset_ids":{split:[real[i]["asset_id"] for i in indices] for split,indices in splits.items()},
        "gpt_synthetic_train_count":len(synthetic),
        "gpt_synthetic_train_ids":[r["asset_id"] for r in synthetic],
        "gpt_ignored_unmapped_classes":skipped,
        "synthetic_total_weight":float(weights[len(y["train"]):].sum()),
        "max_synthetic_fraction_per_class":args.max_synthetic_fraction,
        "no_synthetic_validation_or_test":True,
        "real_only":{"validation_macro_f1":val_base,"C":c_base,
                     "real_test":metrics(y["test"],baseline.predict(X["test"]),labels)},
        "real_plus_synthetic":{"validation_macro_f1":val_mixed,"C":c_mixed,
                     "real_test":metrics(y["test"],mixed.predict(X["test"]),labels)},
        "automatic_production_promotion":False
    }
    results["delta_test_macro_f1"]=(
        results["real_plus_synthetic"]["real_test"]["macro_f1_weak_catalog"]-
        results["real_only"]["real_test"]["macro_f1_weak_catalog"])
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    (output/"evaluation.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({
        "real_train":len(y["train"]),"real_val":len(y["val"]),"real_test":len(y["test"]),
        "train_only_synthetic":len(synthetic),"unmapped":skipped,
        "real_only_test_f1":results["real_only"]["real_test"]["macro_f1_weak_catalog"],
        "real_plus_synthetic_test_f1":results["real_plus_synthetic"]["real_test"]["macro_f1_weak_catalog"],
        "delta_test_macro_f1":results["delta_test_macro_f1"],
        "release_eligible":False
    },indent=2),flush=True)
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--polyhaven",default="training/external/polyhaven_cc0_staging")
    p.add_argument("--ambientcg",default="training/external/ambientcg_cc0_staging")
    p.add_argument("--synthetic-zip",required=True)
    p.add_argument("--out",default="training/out/materialnet-synthetic-ablation")
    p.add_argument("--max-synthetic-fraction",type=float,default=.15)
    p.add_argument("--batch",type=int,default=20)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if not 0<args.max_synthetic_fraction<=.30:
        p.error("Synthetic fraction cap must be (0,.30]")
    train_and_compare(args)
