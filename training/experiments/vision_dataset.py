"""Source-aware, license-audited weak-label CMF dataset selection.

Material families are catalog discovery hints, not expert-verified ground truth.
All records are split by source asset ID and SHA. Do not report real-world accuracy.
"""
from collections import Counter
from pathlib import Path
import hashlib,json

ALLOWED={"metal","polymer","leather","textile","wood","stone","ceramic",
         "glass","rubber","coating","paper","composite"}
SOURCES={"Poly Haven","ambientCG"}

def select_multisource_samples(source_folders, min_per_class=8):
    candidates=[];seen_ids=set();seen_hashes=set()
    for folder in map(Path,source_folders):
        if not (folder/"provenance.json").is_file():
            continue
        records=json.loads((folder/"provenance.json").read_text(encoding="utf-8"))
        if not isinstance(records,list):raise ValueError("Expected source provenance array")
        for item in records:
            label=item.get("catalog_suggestion")
            if (label not in ALLOWED or item.get("license")!="CC0" or
                item.get("source") not in SOURCES or item.get("synthetic_generated") is True):
                continue
            if item.get("review_status")=="excluded":
                continue
            aid,name=item.get("asset_id"),item.get("path")
            if not isinstance(aid,str) or not aid or aid in seen_ids:continue
            if not isinstance(name,str) or not name.endswith(".jpg") or Path(name).name!=name:continue
            path=folder/name
            if not path.is_file():continue
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=item.get("sha256") or digest in seen_hashes:continue
            seen_ids.add(aid);seen_hashes.add(digest)
            candidates.append({
                "asset_id":aid,"path":str(path),"weak_label":label,
                "sample_group":item.get("sample_group") or f"{item['source']}:{aid}",
                "source":item["source"],"source_page":item.get("source_page"),
                "sha256":digest,
                "capture_technique":item.get("source_capture_technique",item.get("capture_method")),
                "is_real_material_capture":item.get("is_real_material_capture")
            })
    counts=Counter(row["weak_label"] for row in candidates)
    filtered=[row for row in candidates if counts[row["weak_label"]]>=min_per_class]
    filtered.sort(key=lambda row:row["asset_id"])
    if len({row["weak_label"] for row in filtered})<2 or len(filtered)<20:
        raise ValueError("Need at least two categories and 20 licensed source images")
    return filtered

def select_catalog_samples(source: Path,min_per_class=8):
    return select_multisource_samples([source],min_per_class)
