"""Build a defensible, sample-grouped CMF dataset from explicit human reviews.

Inputs:
 - training/external/polyhaven_cc0_staging/provenance.json
 - reviewer-exported review JSON from /apps/review/
No unreviewed catalog suggestions are used as training labels.

Usage:
 python training/external/build_reviewed_dataset.py --reviews path/to/kumo-cmf-material-reviews.json
Writes training/reviewed_data/{train,val,test}/{family}/*.jpg and audit metadata.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES={
    "metal","polymer","leather","synthetic-leather","textile","wood",
    "stone","ceramic","glass","rubber","composite","paper","coating"
}

def read_reviews(path: Path) -> list[dict]:
    parsed=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed,dict) or parsed.get("schemaVersion")!="0.1.0":
        raise ValueError("Unsupported reviewer export schema")
    rows=parsed.get("reviews")
    if not isinstance(rows,list):
        raise ValueError("Expected list of reviews")
    return rows

def plan_rows(source_dir: Path,reviews: list[dict],min_per_family: int=15) -> list[dict]:
    source=json.loads((source_dir/"provenance.json").read_text(encoding="utf-8"))
    assets={r["asset_id"]:r for r in source}
    seen=set()
    grouped=defaultdict(list)
    for review in reviews:
        if review.get("decision")!="include":
            continue
        asset_id=review.get("asset_id")
        if asset_id in seen:raise ValueError(f"Duplicate review: {asset_id}")
        seen.add(asset_id)
        if asset_id not in assets:raise ValueError(f"Unknown source asset: {asset_id}")
        if review.get("schemaVersion")!="0.1.0":raise ValueError("Review schema mismatch")
        family=review.get("family")
        if family not in FAMILIES:raise ValueError(f"Invalid family: {family}")
        if review.get("evidence") not in ("visual-only","source-verified"):
            raise ValueError(f"Missing evidence level: {asset_id}")
        if len(str(review.get("reviewer","")).strip())<2 or not review.get("reviewed_at"):
            raise ValueError(f"Missing reviewer and timestamp: {asset_id}")
        asset=assets[asset_id]
        if asset.get("license")!="CC0" or asset.get("source")!="Poly Haven":
            raise ValueError(f"Source license not approved: {asset_id}")
        name=asset.get("path","")
        if not name.endswith(".jpg") or Path(name).name!=name:
            raise ValueError(f"Unsafe filename: {name}")
        path=source_dir/name
        if not path.is_file():
            raise ValueError(f"Missing image: {path}")
        checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum!=asset.get("sha256"):
            raise ValueError(f"Image checksum mismatch: {asset_id}")
        grouped[family].append({
            "asset_id":asset_id,"family":family,"sha256":checksum,
            "sample_group":asset.get("sample_group",f"polyhaven:{asset_id}"),
            "image_path":str(path),"source_page":asset.get("source_page"),
            "evidence":review["evidence"],"reviewer":review["reviewer"],
            "reviewed_at":review["reviewed_at"]
        })
    if not grouped:
        raise ValueError("No approved material samples. Complete review before training.")
    output=[]
    for family,items in sorted(grouped.items()):
        if len(items)<min_per_family:
            raise ValueError(f"{family}: {len(items)} reviewed samples, need >= {min_per_family}. No weak labels will be inferred.")
        # Sort by source asset identifier digest; split at *sample group* level.
        items.sort(key=lambda r:hashlib.sha256(r["sample_group"].encode()).hexdigest())
        n=len(items)
        n_test=max(1,round(n*.15))
        n_val=max(1,round(n*.15))
        if n_test+n_val>=n:raise ValueError(f"Not enough samples in {family}")
        for i,row in enumerate(items):
            split="test" if i<n_test else "val" if i<n_test+n_val else "train"
            output.append({**row,"split":split})
    hashes=defaultdict(set);groups=defaultdict(set)
    for row in output:
        hashes[row["sha256"]].add(row["split"])
        groups[row["sample_group"]].add(row["split"])
    if any(len(v)>1 for v in [*hashes.values(),*groups.values()]):
        raise ValueError("Cross-split leakage detected")
    return output

def write_dataset(plan: list[dict],destination: Path):
    destination.mkdir(parents=True,exist_ok=True)
    public=[]
    for item in plan:
        name=item["asset_id"]+".jpg"
        dest=destination/item["split"]/item["family"]/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(item["image_path"],dest)
        public.append({k:v for k,v in item.items() if k!="image_path"})
    summary={"sample_count":len(public),"split_counts":dict(Counter(r["split"] for r in public)),
             "family_counts":dict(Counter(r["family"] for r in public)),
             "note":"Only manually reviewed assets. Still not a physical-material certification benchmark.",
             "records":public}
    (destination/"dataset-manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--reviews",type=Path,required=True)
    p.add_argument("--source",type=Path,default=Path("training/external/polyhaven_cc0_staging"))
    p.add_argument("--output",type=Path,default=Path("training/reviewed_data"))
    p.add_argument("--min-per-family",type=int,default=15)
    args=p.parse_args()
    plan=plan_rows(args.source,read_reviews(args.reviews),args.min_per_family)
    result=write_dataset(plan,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
