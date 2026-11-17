"""Auditable weak-label experiment selection. Uses only Python standard library.

Source catalog categories are weak labels, not expert-approved physical truth.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json

ALLOWED = {"metal","polymer","leather","textile","wood","stone","ceramic","glass","rubber","coating","paper"}

def select_catalog_samples(source: Path, min_per_class: int = 8):
    records = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    results, seen_id, seen_hash = [], set(), set()
    for record in records:
        label = record.get("catalog_suggestion")
        if label not in ALLOWED or record.get("license") != "CC0" or record.get("source") != "Poly Haven":
            continue
        asset_id, name = record.get("asset_id"), record.get("path")
        if not isinstance(asset_id,str) or not asset_id or asset_id in seen_id:
            continue
        if not isinstance(name,str) or not name.endswith(".jpg") or Path(name).name != name:
            continue
        p = source / name
        if not p.is_file():
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != record.get("sha256") or digest in seen_hash:
            continue
        seen_id.add(asset_id)
        seen_hash.add(digest)
        results.append({"asset_id":asset_id,"path":str(p),"weak_label":label,
                        "source_page":record.get("source_page"),"sha256":digest})
    counts = Counter(x["weak_label"] for x in results)
    filtered = [x for x in results if counts[x["weak_label"]] >= min_per_class]
    filtered.sort(key=lambda x:x["asset_id"])
    if len({x["weak_label"] for x in filtered}) < 2 or len(filtered) < 20:
        raise ValueError("At least two categories and 20 checked assets are required")
    return filtered
