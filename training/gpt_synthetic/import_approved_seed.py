"""Fetch checksum-pinned GPT-generated CMF samples; TRAIN SPLIT ONLY."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request,urlopen

ALLOWED_HOST="kumolabs-cmf-atlas.floot.app"
ALLOWED_FAMILIES=set("textile leather synthetic-leather rubber metal wood ceramic glass coating composite polymer stone paper".split())

def validate(data):
    if data.get("kind")!="GPT-generated synthetic CMF macro textures" or data.get("physically_verified") is not False or data.get("allowed_dataset_split")!="train_only":
        raise ValueError("Untrusted synthetic dataset")
    records=data.get("records",[])
    if not records or len(records)!=data.get("visual_label_approvals"):
        raise ValueError("Missing sample approvals")
    seen=set()
    for r in records:
        a=urlparse(r.get("image_url",""))
        if a.scheme!="https" or a.hostname!=ALLOWED_HOST or not a.path.startswith("/_cdn/static/"):
            raise ValueError("Untrusted image URL")
        if r.get("asset_id") in seen or not r.get("asset_id"):
            raise ValueError("Duplicate or empty sample ID")
        seen.add(r["asset_id"])
        if (r.get("source")!="GPT synthetic" or r.get("synthetic_generated") is not True
            or r.get("is_real_material_capture") is not False
            or r.get("review_status")!="approved_synthetic_label"
            or r.get("family_label_approved_for")!="synthetic_train_only"
            or r.get("allowed_splits")!=["train"] or r.get("family") not in ALLOWED_FAMILIES):
            raise ValueError("Invalid material label or train-only guard")
        if r.get("training_eligible") is not True or len(r.get("sha256",""))!=64:
            raise ValueError("Unapproved or unpinned image")
    return records

def build(manifest_path:Path,out:Path):
    from PIL import Image
    data=json.loads(manifest_path.read_text())
    records=validate(data)
    out.mkdir(parents=True,exist_ok=True)
    for r in records:
        req=Request(r["image_url"],headers={"User-Agent":"Kumo-CMF-Synthetic-Training/0.1"})
        with urlopen(req,timeout=40) as response:
            if urlparse(response.url).hostname!=ALLOWED_HOST:raise ValueError("Unexpected redirect")
            binary=response.read(10_000_001)
        if len(binary)!=r["bytes"] or hashlib.sha256(binary).hexdigest()!=r["sha256"]:
            raise ValueError("Source bytes changed: "+r["asset_id"])
        image=Image.open(io.BytesIO(binary));image.load()
        if (image.width,image.height)!=(r["width"],r["height"]):
            raise ValueError("Image dimensions changed")
        dest=out/"train"/r["family"]/(r["asset_id"]+".png")
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(binary)
    assert not (out/"test").exists() and not (out/"val").exists()
    (out/"manifest.json").write_text(json.dumps(data,indent=2)+"\n")
    archive=out.parent/(data["batch_id"]+".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in out.rglob("*"):
            if p.is_file():z.write(p,p.relative_to(out))
    print(json.dumps({"images":len(records),"split":"train_only","archive":str(archive),
                      "bytes":archive.stat().st_size,"physically_verified":False}))
    return archive

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--manifest",type=Path,default=Path("training/gpt_synthetic/approved_seed_2026-11-18.json"))
    a.add_argument("--out",type=Path,default=Path("training/out/gpt-approved-seed"))
    args=a.parse_args()
    build(args.manifest,args.out)
