"""OpenAI GPT Image -> audited synthetic CMF staging, NOT ground-truth training.

Daily capped generator. Each request creates ONE edge-to-edge material macro image.
Images remain synthetic and unverified and are NEVER put into validation/test.

The caller supplies OPENAI_API_KEY as an environment secret. Never print it.

Examples:
  python training/gpt_synthetic/drip.py --dry-run --count 6
  python training/gpt_synthetic/drip.py --count 6 --output /tmp/kumo-gpt-cmf-20261118
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

API_URL="https://api.openai.com/v1/images/generations"
MAX_PER_RUN=12
MODEL_OPTIONS={"gpt-image-2.5-flare","gpt-image-2.6-flare","gpt-image-2","gpt-image-2.5-sunburst"}
DEFAULT_MODEL="gpt-image-2.5-flare"
FAMILIES={
    "textile":"tightly woven polyester/nylon upholstery, unmistakable interlaced individual threads and subtle irregular yarn tension",
    "synthetic-leather":"automotive polyurethane upholstery with fine micro-pebbled embossed grain, subtle folded creases but no stitched seams",
    "polymer":"moulded dark automotive polypropylene plastic with uniform fine injection-moulded micro-grain and slight variations in satin reflectivity",
    "rubber":"industrial elastomeric rubber sheet with fine raised anti-slip geometry and resilient microtexture",
    "metal":"brushed anodized aluminum sheet with controlled anisotropic sanding lines and faint natural scuffs",
    "leather":"natural tanned hide surface with irregular organic grain pores and subtle microcreases, no visible seams",
    "wood":"thin natural walnut or oak veneer with longitudinal organic grain and fine pores",
    "stone":"natural slate or basalt textured stone surface with irregular fine mineral inclusions and granular fractures",
    "ceramic":"fine-glazed architectural ceramic tile material, micro orange-peel relief and realistic muted glints without tile edges",
    "glass":"subtly frosted translucent architectural glass with fine acid-etched microtexture, softly backlit without visible objects",
    "composite":"woven carbon-fiber composite laminate under a thin clear resin layer, crisp distinct twill weave and natural specular anisotropy",
    "coating":"painted metal panel surface with fine matte powder coat texture, subtle pigment granulation and gentle microgloss",
    "paper":"uncoated fibrous kraft paper material with visible pulp flecks and nonrepeating microscopic fibers",
}
PALETTES=("graphite charcoal","warm taupe","deep indigo","mineral gray","earthy rust",
          "desaturated olive","soft silver","saddle brown","bone ivory","petrol teal")
LIGHTING=("neutral large softbox illumination","diffuse north-window daylight",
          "subtle raking light revealing genuine relief","soft cross-polarized studio style lighting")
SCALES=("close optical macro at approximately 1:1 texture scale",
        "straight-on 85mm macro at approximately 2:1 scale",
        "very close 100mm macro with edge-to-edge sharp texture",
        "orthographic perpendicular material-swatch macro composition")


def validate_model(model: str) -> str:
    if model not in MODEL_OPTIONS:
        raise ValueError("Unsupported/retired GPT image model. Choose: "+", ".join(sorted(MODEL_OPTIONS)))
    return model


def plan_batch(date_value: date, count: int=6):
    if not isinstance(date_value,date):raise TypeError("date required")
    if not isinstance(count,int) or not 1<=count<=MAX_PER_RUN:
        raise ValueError("Batch count must be 1-12, hard capped to protect API budget")
    families=list(FAMILIES)
    plans=[]
    for i in range(count):
        # Constant stride 6 yields complete, even family coverage over 13 days.
        family=families[(date_value.toordinal()*6+i)%len(families)]
        period=date_value.toordinal()//13
        palette=PALETTES[(period+i*3)%len(PALETTES)]
        lighting=LIGHTING[(period+i)%len(LIGHTING)]
        scale=SCALES[(period//2+i)%len(SCALES)]
        texture=FAMILIES[family]
        prompt=(
            "Create a SINGLE photorealistic material surface macro photograph for a "
            "carefully controlled automotive and industrial CMF texture research dataset. "
            f"The ENTIRE frame is covered by ONE continuous flat material: {texture}. "
            f"Color direction: {palette}; lighting: {lighting}; camera: {scale}. "
            "Render physically plausible microscopic surface detail, subtle natural irregularities "
            "and believable depth while keeping the image sharp across the complete frame. "
            "No collage, no grid, no multiple images, no text, no brand, no logos, "
            "no objects, no products, no people, no hands, no stitching or seams, "
            "no borders, no cards, no labels, no sample swatch edges, no watermark, "
            "no false decorative patterns. Camera perpendicular to the surface; "
            "realistic CMF studio reference photography appearance. "
            "The requested family is an image-generation intent, NOT physical verification."
        )
        plans.append({
            "batch_date":date_value.isoformat(),"index":i,
            "family":family,"palette":palette,"lighting":lighting,"scale":scale,
            "filename":f"gpt_{date_value:%Y%m%d}_{i:02d}_{family.replace('-','_')}.jpg",
            "prompt":prompt,
        })
    return plans


def make_record(plan,model,sha,width,height,file_size,quality_hints=None,dhash=None):
    return {
        "asset_id":f"gpt:{plan['batch_date']}:{plan['index']:02d}:{plan['family']}",
        "file":plan["filename"],"source":"GPT-generated synthetic",
        "generator":"OpenAI Image API","model":model,
        "prompt":plan["prompt"],"prompt_intended_family":plan["family"],
        "label_kind":"prompt_intent_not_verified",
        "material_palette_intent":plan["palette"],
        "lighting_intent":plan["lighting"],
        "surface_scale_intent":plan["scale"],
        "sha256":sha,"width":width,"height":height,"bytes":file_size,
        "dhash":dhash,
        "quality_hints":quality_hints or [],
        "sample_group":f"gpt-image:{plan['batch_date']}:{plan['index']:02d}",
        "batch_group":f"gpt-cmf-{plan['batch_date']}",
        "synthetic_generated":True,"is_real_material_capture":False,
        "review_status":"needs_expert_review","training_eligible":False,
        "permitted_splits":["synthetic_train_only_after_review"],
        "commercial_license_claim":"none",
        "warning":"AI-generated picture, not a physical substrate proof. Never use for real-material benchmark evaluation."
    }


def fetch_image(prompt,model,api_key,quality="medium",attempts=3):
    if not api_key:raise RuntimeError("OPENAI_API_KEY missing; no images generated")
    body=json.dumps({
        "model":validate_model(model),"prompt":prompt,
        "size":"1024x1024","quality":quality,
        "output_format":"jpeg","n":1
    }).encode("utf-8")
    for attempt in range(attempts):
        request=urllib.request.Request(
            API_URL,data=body,method="POST",
            headers={
                "Authorization":"Bearer "+api_key,
                "Content-Type":"application/json",
                "User-Agent":"Kumo-CMF-GPT-Drip/0.1"
            }
        )
        try:
            with urllib.request.urlopen(request,timeout=200) as response:
                raw=response.read(25_000_000)
            output=json.loads(raw)
            images=output.get("data") or []
            if len(images)!=1 or not images[0].get("b64_json"):
                raise RuntimeError("API did not return exactly one base64 image")
            return base64.b64decode(images[0]["b64_json"],validate=True)
        except urllib.error.HTTPError as error:
            status=error.code
            if status not in (429,500,502,503,504) or attempt>=attempts-1:
                raise RuntimeError(f"GPT image generation failed: HTTP {status}") from None
            retry_after=error.headers.get("Retry-After","")
            delay=min(50,max(1,int(retry_after))) if retry_after.isdigit() else min(30,2**(attempt+2))
            time.sleep(delay)
        except (TimeoutError,urllib.error.URLError) as error:
            if attempt>=attempts-1:raise RuntimeError("Image API network error after retries") from error
            time.sleep(min(30,2**(attempt+2)))
    raise RuntimeError("Image API attempts exhausted")


def fingerprint(image):
    from PIL import Image
    gray=image.convert("L").resize((9,8),Image.Resampling.LANCZOS)
    pix=list(gray.getdata())
    value=0
    for y in range(8):
        for x in range(8):
            value=(value<<1)|int(pix[y*9+x] > pix[y*9+x+1])
    return f"{value:016x}"


def hamming(a,b):return (int(a,16)^int(b,16)).bit_count()


def inspect_and_save(data,path):
    from PIL import Image,ImageStat
    with Image.open(io.BytesIO(data)) as original:
        original.load()
        if min(original.size)<768:
            raise ValueError("Model returned image too small for minimum dataset quality")
        image=original.convert("RGB")
    stats=ImageStat.Stat(image.resize((128,128)))
    hints=[]
    if min(stats.stddev)<4:hints.append("low_color_variation")
    luma=image.convert("L")
    mean=ImageStat.Stat(luma).mean[0]
    if mean<30:hints.append("underexposed_appearance")
    if mean>232:hints.append("very_bright_appearance")
    path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path,"JPEG",quality=92,subsampling=0,optimize=True)
    blob=path.read_bytes()
    return {
        "sha":hashlib.sha256(blob).hexdigest(),
        "width":image.width,"height":image.height,
        "file_size":len(blob),"hints":hints,"dhash":fingerprint(image)
    }


def execute(date_value: date,count:int,output:Path,dry_run=False,
            model=DEFAULT_MODEL,quality="medium",api_key=None):
    validate_model(model)
    if quality not in ("low","medium","high"):raise ValueError("Unsupported quality")
    plan=plan_batch(date_value,count)
    if dry_run:
        print(json.dumps({"dry_run":True,"count":count,"model":model,
                          "plan":[{"filename":x["filename"],"family":x["family"],
                                   "palette":x["palette"],"lighting":x["lighting"]} for x in plan]},indent=2))
        return plan
    key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY","")
    if not key:raise RuntimeError("Configure repository Actions secret OPENAI_API_KEY before scheduled generation")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for item in plan:
        dest=output/item["filename"]
        print(f"Generating {item['index']+1}/{count}: {item['family']}",flush=True)
        data=fetch_image(item["prompt"],model,key,quality=quality)
        metrics=inspect_and_save(data,dest)
        if any(existing["sha256"]==metrics["sha"] or
               hamming(existing["dhash"],metrics["dhash"])<=3 for existing in records):
            dest.unlink(missing_ok=True)
            raise RuntimeError("Repeated near-duplicate image detected; batch withheld for review")
        records.append(make_record(item,model,metrics["sha"],
                                   metrics["width"],metrics["height"],
                                   metrics["file_size"],metrics["hints"],metrics["dhash"]))
    manifest={
        "schema_version":"1.0","batch_id":"gpt-cmf-"+date_value.isoformat(),
        "created_utc":date.today().isoformat(),
        "source":"OpenAI GPT Image API synthetic research samples",
        "model":model,"api_calls":len(records),
        "count":len(records),"quality":quality,
        "review_status":"needs_expert_review",
        "allowed_training_usage":"synthetic train-only after human review, never val/test",
        "auto_training_enabled":False,
        "licensing_note":"Not CC0. Review platform/API terms and image redistribution rights.",
        "records":records
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    from PIL import Image,ImageDraw
    # Compact contact-sheet preview is the only raster image committed into repo;
    # full-quality image ZIP belongs in GitHub Release assets to limit git bloat.
    thumb_size=256
    cols=3
    rows=(len(records)+cols-1)//cols
    contact=Image.new("RGB",(cols*thumb_size,rows*(thumb_size+30)),(21,25,29))
    drawer=ImageDraw.Draw(contact)
    for i,record in enumerate(records):
        with Image.open(output/record["file"]) as raw:
            tile=raw.convert("RGB").resize((thumb_size,thumb_size),Image.Resampling.LANCZOS)
        x=(i%cols)*thumb_size
        y=(i//cols)*(thumb_size+30)
        contact.paste(tile,(x,y))
        drawer.text((x+8,y+thumb_size+7),record["prompt_intended_family"],fill=(228,232,229))
    contact.save(output/"preview.jpg",quality=78,optimize=True)
    archive=output.parent/(manifest["batch_id"]+".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as pkg:
        for file in output.iterdir():
            if file.is_file():pkg.write(file,file.name)
    print(json.dumps({"batch_id":manifest["batch_id"],
                      "images":len(records),"full_archive":str(archive),
                      "bytes":archive.stat().st_size,"preview":str(output/"preview.jpg")},indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--date",default=date.today().isoformat())
    p.add_argument("--count",type=int,default=6)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--quality",default="medium")
    p.add_argument("--output",type=Path,default=Path("training/gpt_synthetic/generated"))
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    try:
        day=date.fromisoformat(args.date)
        execute(day,args.count,args.output,args.dry_run,args.model,args.quality)
    except (ValueError,RuntimeError) as error:
        print("GPT CMF drip error:",error,file=sys.stderr)
        raise SystemExit(2) from None

if __name__=="__main__":
    main()
