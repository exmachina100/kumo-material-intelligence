"""Train a SMALL experimental material-image classifier from Poly Haven catalog hints.

Not expert-reviewed! This is a pipeline / learning experiment, NOT a production
CMF material recognizer or any claim of physical material identification.
Never publish these weights as verified MaterialNet-S.

Install: pip install numpy pillow scikit-learn
Usage: python training/experiments/train_catalog_proxy.py
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score

ALLOWED={"metal","polymer","leather","textile","wood","stone","ceramic","glass","rubber","coating","paper"}
FEATURE_NAMES=(
    [f"hist_{channel}_{i}" for channel in "rgb" for i in range(8)]
    + [f"mean_{x}" for x in "rgb"]
    + [f"std_{x}" for x in "rgb"]
    + ["grad_x","grad_y","gray_p10","gray_p50","gray_p90","gray_std"]
)

def features(path: Path) -> np.ndarray:
    # Fixed, documented 48x48 bilinear resize; no auto exposure / gamma correction.
    im=Image.open(path).convert("RGB").resize((48,48),Image.Resampling.BILINEAR)
    data=np.asarray(im,dtype=np.float64)/255.0
    out=[]
    for channel in range(3):
        counts=np.histogram(data[:,:,channel],bins=8,range=(0,1))[0]
        out.extend((counts/counts.sum()).tolist())
    out.extend(data.mean(axis=(0,1)).tolist())
    out.extend(data.std(axis=(0,1)).tolist())
    gray=(data[:,:,0]*.2126+data[:,:,1]*.7152+data[:,:,2]*.0722)
    out.extend([float(np.abs(np.diff(gray,axis=1)).mean()),
                float(np.abs(np.diff(gray,axis=0)).mean())])
    out.extend(np.quantile(gray,[.1,.5,.9]).tolist())
    out.append(float(gray.std()))
    result=np.asarray(out,dtype=np.float64)
    assert result.size==len(FEATURE_NAMES)
    return result

def run(source: Path, outdir: Path, minimum: int=8):
    records=json.loads((source/"provenance.json").read_text())
    candidates=[]
    seen_hash=set()
    for item in records:
        family=item.get("catalog_suggestion")
        if family not in ALLOWED or item.get("license")!="CC0":
            continue
        path=source/item.get("path","")
        if not path.is_file():continue
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=item.get("sha256") or sha in seen_hash:continue
        seen_hash.add(sha)
        candidates.append((path,family,item["asset_id"]))
    tally=Counter(family for _,family,_ in candidates)
    candidates=[row for row in candidates if tally[row[1]]>=minimum]
    classes=sorted({label for _,label,_ in candidates})
    if len(classes)<2 or len(candidates)<20:
        raise ValueError(f"Insufficient labeled proxy assets: {len(candidates)} from {len(classes)} families")
    x=np.asarray([features(path) for path,_,_ in candidates])
    y=np.asarray([family for _,family,_ in candidates])
    ids=[asset for _,_,asset in candidates]
    indices=np.arange(len(y))
    train,test=train_test_split(indices,test_size=.25,stratify=y,random_state=42)
    scaler=StandardScaler().fit(x[train])
    model=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)
    model.fit(scaler.transform(x[train]),y[train])
    pred=model.predict(scaler.transform(x[test]))
    result={
        "kind":"experimental-catalog-proxy",
        "release_eligible":False,
        "warning":"Weak catalog-category labels, NOT expert ground truth or a real-world accuracy estimate.",
        "source":"Poly Haven CC0 thumbnail previews",
        "seed":42,
        "split":"25% held out by distinct source asset ID",
        "class_counts":dict(Counter(y)),
        "train_count":len(train),"test_count":len(test),
        "accuracy_vs_proxy":float(accuracy_score(y[test],pred)),
        "macro_f1_vs_proxy":float(f1_score(y[test],pred,average="macro",zero_division=0)),
        "report_vs_proxy":classification_report(y[test],pred,output_dict=True,zero_division=0),
        "confusion_matrix":confusion_matrix(y[test],pred,labels=classes).tolist(),
        "heldout_asset_ids":[ids[i] for i in test],
    }
    artifact={
        "kind":"experimental-catalog-proxy",
        "version":"0.1.0",
        "approved":False,
        "classes":list(model.classes_),
        "feature_names":FEATURE_NAMES,
        "preprocessing":{"size":[48,48],"resize":"bilinear","rgb_scale":255},
        "scaler_mean":scaler.mean_.tolist(),
        "scaler_scale":scaler.scale_.tolist(),
        "coefficients":model.coef_.tolist(),
        "intercept":model.intercept_.tolist(),
        "source":"Poly Haven CC0; unreviewed catalog hints"
    }
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"catalog-proxy-report.json").write_text(json.dumps(result,indent=2)+"\n")
    (outdir/"catalog-proxy-model.json").write_text(json.dumps(artifact,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("report_vs_proxy","confusion_matrix","heldout_asset_ids")},indent=2))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("training/external/polyhaven_cc0_staging"))
    p.add_argument("--out",type=Path,default=Path("training/out/catalog-proxy"))
    p.add_argument("--min-class-samples",type=int,default=8)
    args=p.parse_args()
    run(args.source,args.out,args.min_class_samples)
