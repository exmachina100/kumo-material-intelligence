"""Build public, compact index for completed GPT CMF Release manifests.

Exposes ONLY explicitly synthetic imagery. No real-material labels or API keys.
"""
import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

REPOSITORY="exmachina100/kumo-material-intelligence"


def index(root:Path):
    root=Path(root)
    batches=[]
    for path in sorted((root/"manifests").glob("????-??-??.json"),reverse=True):
        date_string=path.stem
        try:date.fromisoformat(date_string)
        except ValueError:continue
        manifest=json.loads(path.read_text(encoding="utf-8"))
        records=manifest.get("records")
        if (manifest.get("batch_id")!="gpt-cmf-"+date_string or
            not isinstance(records,list) or
            len(records)!=manifest.get("count") or not records or
            manifest.get("auto_training_enabled") is not False or
            any(r.get("training_eligible") is not False or
                r.get("is_real_material_capture") is not False for r in records)):
            raise ValueError(f"Unsafe/malformed GPT batch manifest: {path}")
        preview=root/"previews"/(date_string+".jpg")
        if not preview.is_file():
            raise FileNotFoundError(f"Missing preview for {date_string}")
        families=Counter(r["prompt_intended_family"] for r in records)
        batches.append({
            "date":date_string,
            "batch_id":manifest["batch_id"],
            "count":len(records),
            "model":manifest["model"],
            "families":dict(sorted(families.items())),
            "preview":"previews/"+date_string+".jpg",
            "release_url":f"https://github.com/{REPOSITORY}/releases/tag/gpt-cmf-{date_string}",
            "kind":"AI-generated synthetic research material"
        })
    return {
        "schemaVersion":"0.1.0",
        "status":"available" if batches else "awaiting_first_billed_batch",
        "total_samples":sum(b["count"] for b in batches),
        "total_batches":len(batches),
        "batch_schedule":"Daily at 03:40 UTC; 6 images/run, API key required",
        "source":"OpenAI GPT Image API; NOT expert-verified real materials",
        "training_eligible":False,
        "batches":batches
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("training/gpt_synthetic"))
    args=p.parse_args()
    data=index(args.root)
    args.root.mkdir(parents=True,exist_ok=True)
    (args.root/"index.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({"batches":data["total_batches"],"samples":data["total_samples"],
                      "status":data["status"]},indent=2))

if __name__=="__main__":main()
