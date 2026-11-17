"""Licensed Poly Haven texture-reference staging for Kumo Material Intelligence.

CC0 asset previews, respecting the Poly Haven API ToS. Catalog categories are
*discovery hints*, not proof of material composition. NEVER use this staging
folder directly as supervised training labels.

Usage:
  python training/external/ingest_polyhaven.py --limit 110 --per-family 16
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CATALOG = "https://api.polyhaven.com/assets?t=textures"
USER_AGENT = "KumoMaterialIntelligence-DatasetResearch/0.2 (https://github.com/exmachina100/kumo-material-intelligence)"
# Discovery only, never used as ground-truth training labels.
DISCOVERY = {
    "leather": ("leather", "suede", "nubuck"),
    "textile": ("fabric", "textile", "cloth", "woven", "cotton", "denim"),
    "metal": ("metal", "steel", "aluminium", "aluminum", "iron", "copper", "brass"),
    "wood": ("wood", "veneer", "timber", "plywood"),
    "stone": ("stone", "marble", "granite", "slate", "rock"),
    "ceramic": ("ceramic", "porcelain", "tile"),
    "polymer": ("plastic", "polymer", "vinyl"),
    "glass": ("glass",),
    "rubber": ("rubber",),
    "coating": ("paint", "painted", "lacquer"),
    "paper": ("paper", "cardboard"),
    "concrete": ("concrete", "cement"),
}
SEEDS = [
    "fabric_leather_01", "fabric_leather_02", "leather_white", "brown_leather",
    "bi_stretch", "white_oak_veneer", "natural_walnut_veneer",
    "walnut_veneer", "oak_veneer_01", "brushed_concrete",
    "anti_slip_concrete", "beige_wall_001", "asphalt_01", "bark_brown_01",
]
ALLOWED_HOSTS = {"api.polyhaven.com", "cdn.polyhaven.com"}
MAX_BYTES = 10_000_000


def safe_get(url: str, max_bytes: int = MAX_BYTES) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("Refusing non-Poly-Haven endpoint")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        # Reject redirects to unapproved hosts, too.
        if urlparse(response.url).hostname not in ALLOWED_HOSTS:
            raise ValueError("Unexpected download redirect host")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Asset exceeds maximum allowed size")
    return data


def discover_family(item: dict) -> str:
    """A catalog suggestion only, not a physical-material label."""
    category = str(item.get("category") or "").lower()
    tags = " ".join(str(t).lower() for t in item.get("tags", []))
    name = str(item.get("name") or "").lower()
    # Categories are more reliable than free-text names or tags.
    for text in (category, name, tags):
        for family, terms in DISCOVERY.items():
            if any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in terms):
                return family
    return "other"


def select_candidates(catalog: dict, limit: int, per_family: int) -> list[tuple[str, dict, str]]:
    if limit < 1 or per_family < 1:
        raise ValueError("Selection limits must be positive")
    groups: dict[str, list[tuple[str, dict, str]]] = {key: [] for key in DISCOVERY}
    for asset_id, meta in catalog.items():
        if not isinstance(meta, dict) or meta.get("type", 1) != 1:
            continue
        if not meta.get("thumbnail_url") or not re.fullmatch(r"[a-zA-Z0-9_-]+", asset_id):
            continue
        family = discover_family(meta)
        if family in groups:
            groups[family].append((asset_id, meta, family))
    for group in groups.values():
        group.sort(key=lambda candidate: (-int(candidate[1].get("download_count") or 0), candidate[0]))
    selected: list[tuple[str, dict, str]] = []
    seen = set()
    for asset_id in SEEDS:
        meta = catalog.get(asset_id)
        if isinstance(meta, dict) and meta.get("thumbnail_url"):
            family = discover_family(meta)
            selected.append((asset_id, meta, family))
            seen.add(asset_id)
            if len(selected) == limit:
                return selected
    # Round-robin selection limits dominance from outdoor/architecture surfaces.
    for i in range(per_family):
        for family in DISCOVERY:
            available = [x for x in groups[family] if x[0] not in seen]
            if not available:
                continue
            asset = available[0]
            selected.append(asset)
            seen.add(asset[0])
            if len(selected) >= limit:
                return selected
    return selected


def ingest(output: Path, limit: int = 110, per_family: int = 16, pause: float = 0.15) -> dict:
    from PIL import Image
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(safe_get(CATALOG, max_bytes=12_000_000))
    selected = select_candidates(catalog, limit, per_family)
    old_path = output / "provenance.json"
    old = json.loads(old_path.read_text()) if old_path.is_file() else []
    prior = {x["asset_id"]: x for x in old if isinstance(x, dict) and x.get("asset_id")}
    success, errors = 0, []
    records = {}
    for asset_id, meta, suggestion in selected:
        target = output / f"{asset_id}.jpg"
        url = meta["thumbnail_url"]
        try:
            if not target.is_file() or target.stat().st_size < 500:
                data = safe_get(url)
                image = Image.open(io.BytesIO(data))
                image.load()
                image = image.convert("RGB")
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                if min(image.size) < 120:
                    raise ValueError("Preview resolution is too low")
                image.save(target, quality=89, optimize=True)
                time.sleep(pause)
            record = {
                "asset_id": asset_id,
                "path": target.name,
                "source": "Poly Haven",
                "source_page": f"https://polyhaven.com/a/{asset_id}",
                "download_url": url,
                "license": "CC0",
                "license_page": "https://polyhaven.com/license",
                "title": meta.get("name", asset_id),
                "catalog_category": meta.get("category"),
                "catalog_tags": meta.get("tags", []),
                "catalog_suggestion": suggestion,
                "training_family": "UNREVIEWED",
                "review_status": "needs_expert_review",
                "is_real_material_capture": None,
                "capture_method": (meta.get("attributes") or {}).get("method"),
                "image_role": "thumbnail_preview",
                "sample_group": "polyhaven:" + asset_id,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "source_author": meta.get("authors", {}),
            }
            if asset_id in prior and prior[asset_id].get("review_status") == "reviewed":
                record["review_status"] = prior[asset_id]["review_status"]
                record["training_family"] = prior[asset_id].get("training_family", "UNREVIEWED")
            records[asset_id] = record
            success += 1
        except Exception as error:
            errors.append({"asset_id": asset_id, "type": type(error).__name__, "message": str(error)[:150]})
    # Preserve previous records to avoid dropping existing sample references.
    for asset_id, record in prior.items():
        records.setdefault(asset_id, record)
    records = dict(sorted(records.items()))
    old_path.write_text(json.dumps(list(records.values()), indent=2, ensure_ascii=False) + "\n")
    (output / "ingest-report.json").write_text(json.dumps({
        "requested": limit, "selected": len(selected), "available": len(records),
        "new_or_refreshed": success, "failures": errors[:40],
        "note": "All material families need human review before supervised training.",
    }, indent=2) + "\n")
    print(f"Selected {len(selected)}; available {len(records)}; processed {success}; errors {len(errors)}")
    if not success:
        raise RuntimeError("No texture previews could be processed")
    return {"selected": len(selected), "available": len(records), "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("training/external/polyhaven_cc0_staging"))
    parser.add_argument("--limit", type=int, default=110)
    parser.add_argument("--per-family", type=int, default=16)
    parser.add_argument("--pause", type=float, default=0.15)
    args = parser.parse_args()
    ingest(args.output, args.limit, args.per_family, args.pause)
