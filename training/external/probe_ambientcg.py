"""Inspect live ambientCG API v3 schemas for licensed CMF material ingestion.

Prints only structural metadata, not private data or downloaded binary content.
"""
import json
from urllib.request import Request,urlopen
from urllib.parse import urlencode
from urllib.parse import urlparse

BASE='https://ambientcg.com/api/v3/assets'
QUERIES=('plastic','rubber','fabric','leather','glass','metal')
for q in QUERIES:
    url=BASE+'?'+urlencode({'type':'material','q':q,'limit':'3',
                'include':'title,tags,thumbnails,previews,url,technique'})
    try:
        with urlopen(Request(url,headers={'User-Agent':'KumoCMFResearch/0.3',
                                          'Accept':'application/json'}),timeout=25) as response:
            body=response.read(1_500_000)
        data=json.loads(body)
        print('QUERY',q,'ROOT',type(data).__name__, 'KEYS',
              list(data)[:12] if isinstance(data,dict) else 'list-size-'+str(len(data)))
        for key in ('data','assets','results','items'):
            if isinstance(data,dict) and isinstance(data.get(key),list):
                arr=data[key]
                print('CANDIDATES',key,len(arr))
                if arr:
                    item=arr[0]
                    print('ITEM KEYS',list(item.keys()))
                    for field in ('id','title','type','technique','url','thumbnails','previews'):
                        v=item.get(field)
                        print('FIELD',field,json.dumps(v)[:900])
                break
    except Exception as exc:
        print('ERROR',q,type(exc).__name__,str(exc)[:180])
