"""Download specifically curated Poly Haven CC0 previews for material dataset review.

This script does NOT silently reclassify real textures into unverified automotive CMF substrates.
External images are placed into an ingest/staging directory, not train/val/test.
API catalog, thumbnail URLs, and license must be verified at run time.
"""
import argparse, csv, hashlib, io, json, time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request,urlopen
from PIL import Image

CATALOG='https://api.polyhaven.com/assets?t=textures'
# Conservative source labels: these are catalog material categories, not verified vehicle materials.
CURATED={
    'fabric_leather_01':'leather','fabric_leather_02':'leather','leather_white':'leather','brown_leather':'leather',
    'bi_stretch':'textile','white_oak_veneer':'wood','natural_walnut_veneer':'wood','walnut_veneer':'wood',
    'oak_veneer_01':'wood','brushed_concrete':'concrete','anti_slip_concrete':'concrete',
    'beige_wall_001':'plaster','asphalt_01':'asphalt','bark_brown_01':'wood'
}

def get(url,max_size=10_000_000):
    if urlparse(url).scheme!='https' or urlparse(url).hostname not in ('api.polyhaven.com','cdn.polyhaven.com'):
        raise ValueError('Refusing non-Poly-Haven URL')
    with urlopen(Request(url,headers={'User-Agent':'KumoCMFDatasetResearch/0.1'}),timeout=25) as r:
        data=r.read(max_size+1)
        if len(data)>max_size:raise ValueError('Download exceeds size limit')
        return data

def ingest(output,limit=14):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    catalog=json.loads(get(CATALOG,max_size=8_000_000))
    records=[]
    for asset_id,category in list(CURATED.items())[:limit]:
        item=catalog.get(asset_id)
        if not item:
            print('SKIP missing in current catalog:',asset_id);continue
        url=item.get('thumbnail_url')
        if not url:
            print('SKIP no thumbnail:',asset_id);continue
        try:
            data=get(url)
            im=Image.open(io.BytesIO(data)).convert('RGB')
            # Catalog thumbnails are for seed data only, not full-resolution PBR maps.
            im.thumbnail((512,512),Image.Resampling.LANCZOS)
            if min(im.size)<120:raise ValueError('Preview too small')
            target=output/f'{asset_id}.jpg';im.save(target,quality=89,optimize=True)
            rec={'path':target.name,'asset_id':asset_id,'source':'Poly Haven','source_page':f'https://polyhaven.com/a/{asset_id}',
            'download_url':url,'license':'CC0','license_page':'https://polyhaven.com/license',
            'catalog_category':category,'training_family':'UNREVIEWED','is_real_material_capture':True,
            'image_role':'preview','sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'source_author':item.get('authors',{}),'review_status':'needs_expert_review'}
            records.append(rec);time.sleep(.15)
        except Exception as e:print('SKIP failed:',asset_id,type(e).__name__,str(e)[:100])
    (output/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Downloaded',len(records),'CC0 texture previews to',output)
    return len(records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('training/external/polyhaven_cc0_staging'))
    p.add_argument('--limit',type=int,default=14);a=p.parse_args()
    count=ingest(a.output,a.limit)
    if count==0:raise SystemExit('No assets could be obtained')
