"""Stage CC0 ambientCG material previews for expert-reviewed CMF datasets.

Official v3 API: https://ambientcg.com/api/v3/assets
Staged catalog keyword categories are NOT expert material labels.
Do not mix photo/procedural examples in an evaluation without source controls.
"""
from __future__ import annotations
import argparse,hashlib,io,json,re,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlparse,urlencode
from PIL import Image

API='https://ambientcg.com/api/v3/assets'
USER_AGENT='KumoMaterialIntelligence-Research/0.3 (https://github.com/exmachina100/kumo-material-intelligence)'
SEARCHES={'polymer':'plastic','rubber':'rubber','textile':'fabric',
          'leather':'leather','glass':'glass','composite':'carbon fiber'}
ALLOWED_HOSTS={'ambientcg.com','www.ambientcg.com','acg-media.struffelproductions.com'}

def get(url,limit=10_000_000):
    p=urlparse(url)
    if p.scheme!='https' or p.hostname not in ALLOWED_HOSTS:
        raise ValueError('Unapproved ambientCG endpoint: '+str(p.hostname))
    req=Request(url,headers={'User-Agent':USER_AGENT,'Accept':'application/json,image/png,image/jpeg,*/*'})
    with urlopen(req,timeout=25) as response:
        final=urlparse(response.url)
        if final.hostname not in ALLOWED_HOSTS:raise ValueError('Unexpected redirect')
        data=response.read(limit+1)
    if len(data)>limit:raise ValueError('Asset exceeds 10 MB limit')
    return data

def candidates(family,query,per_family):
    url=API+'?'+urlencode({'type':'material','q':query,'limit':min(150,per_family*3),
            'include':'title,tags,technique,thumbnails,url'})
    data=json.loads(get(url,2_000_000))
    if not isinstance(data,dict) or not isinstance(data.get('assets'),list):
        raise ValueError('Unsupported ambientCG API schema')
    rows=[]
    for entry in data['assets']:
        if not isinstance(entry,dict):continue
        aid=entry.get('id')
        if not isinstance(aid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{3,100}',aid):continue
        thumbs=entry.get('thumbnails') or {}
        if not isinstance(thumbs,dict):continue
        preview=thumbs.get('512-PNG') or thumbs.get('256-PNG')
        if not isinstance(preview,str) or urlparse(preview).hostname not in ALLOWED_HOSTS:continue
        rows.append((aid,entry,preview))
        if len(rows)>=per_family:break
    return rows

def ingest(output:Path,per_family=24,pause=.2):
    output.mkdir(parents=True,exist_ok=True)
    meta_path=output/'provenance.json'
    previous=json.loads(meta_path.read_text()) if meta_path.is_file() else []
    records={x['asset_id']:x for x in previous if isinstance(x,dict) and x.get('asset_id')}
    counts={};failures=[]
    for family,query in SEARCHES.items():
        try:
            selected=candidates(family,query,per_family)
        except Exception as exc:
            failures.append({'family':family,'error':str(exc)[:180]})
            continue
        success=0
        for aid,entry,preview in selected:
            key='ambientcg:'+aid
            target=output/(aid+'.jpg')
            try:
                if not target.is_file():
                    image=Image.open(io.BytesIO(get(preview))).convert('RGB')
                    image.thumbnail((512,512),Image.Resampling.LANCZOS)
                    if min(image.size)<128:raise ValueError('Preview resolution too small')
                    image.save(target,'JPEG',quality=90,optimize=True)
                    time.sleep(pause)
                technique=entry.get('technique')
                old=records.get(key,{})
                records[key]={
                    'asset_id':key,'path':target.name,'title':entry.get('title') or aid,
                    'source':'ambientCG','source_page':entry.get('url') or 'https://ambientcg.com/a/'+aid,
                    'download_url':preview,'license':'CC0',
                    'license_page':'https://docs.ambientcg.com/license/',
                    'catalog_suggestion':family,'catalog_query':query,
                    'catalog_tags':entry.get('tags',[]),'source_capture_technique':technique,
                    'is_real_material_capture':technique in ('surface-photogrammetry','surface-photometric-stereo'),
                    'image_role':'catalog_thumbnail_preview',
                    'training_family':old.get('training_family','UNREVIEWED'),
                    'review_status':old.get('review_status','needs_expert_review'),
                    'sample_group':'ambientcg:'+aid,
                    'sha256':hashlib.sha256(target.read_bytes()).hexdigest()
                }
                success+=1
            except Exception as exc:
                failures.append({'asset_id':key,'error':str(exc)[:180]})
        counts[family]=success
        print(f'{family}: {success} CC0 samples downloaded / checked',flush=True)
    meta_path.write_text(json.dumps([records[x] for x in sorted(records)],indent=2)+'\n')
    report={'total':len(records),'counts_by_query':counts,'failures':failures[:60],
            'note':'Unreviewed CC0 thumbnails. Source techniques include procedural and photographic materials. No physical substrate verification.'}
    (output/'ingest-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not records:raise RuntimeError('No licensed material images staged')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('training/external/ambientcg_cc0_staging'))
    p.add_argument('--per-family',type=int,default=24)
    a=p.parse_args()
    ingest(a.output,a.per_family)
