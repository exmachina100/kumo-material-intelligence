"""Generate deterministic synthetic CMF PNG samples for smoke testing only.
These are not real photos and MUST NOT be used as evidence of real-world material recognition accuracy.
"""
import argparse, csv, math, random
from pathlib import Path
from PIL import Image

FAMILIES=['metal','polymer','leather','synthetic-leather','textile','wood','stone','ceramic','glass','rubber','composite','paper','coating']
BASE={'metal':(141,148,153),'polymer':(98,111,118),'leather':(139,93,64),'synthetic-leather':(96,86,91),
'textile':(125,130,118),'wood':(155,116,75),'stone':(143,143,141),'ceramic':(196,192,187),
'glass':(120,167,169),'rubber':(61,64,67),'composite':(76,82,85),'paper':(213,202,177),'coating':(118,137,137)}

def texture(family,index,size=192):
    seed=(FAMILIES.index(family)+1)*100000+index*97
    rng=random.Random(seed);phase=rng.random()*8;freq=2.5+rng.random()*4;base=BASE[family]
    pixels=bytearray(size*size*3)
    for y in range(size):
        yy=y/size
        for x in range(size):
            xx=x/size
            noise=rng.gauss(0,1)
            if family=='metal':v=9*math.sin(y*.6+phase)+2*noise+10*(xx-.5)
            elif family=='polymer':v=5*noise+2*math.sin(x*.14+y*.22)
            elif family in ('leather','synthetic-leather'):v=7*noise+6*math.sin(x*.23+math.sin(y*.11)*2)*math.cos(y*.26)+2*math.sin((x+y)*.08)
            elif family=='textile':v=7*math.sin(x*.65)+7*math.sin(y*.62)+3*noise
            elif family=='wood':v=17*math.sin((xx*freq+.08*math.sin(yy*13+phase))*math.pi*5)+4*noise
            elif family=='stone':v=12*noise+8*math.sin(x*.05+y*.09)+3*math.sin(x*.26)
            elif family=='ceramic':v=1.7*noise+11*(1-xx)*(.4+math.sin(yy*1.3))
            elif family=='glass':v=14*(1-xx)+14*math.sin(xx*3.2+phase)+2*noise
            elif family=='rubber':v=8*noise+3*math.sin(x*.5+y*.5)
            elif family=='composite':v=12*math.sin((x+y)*.45)*math.sin((x-y)*.48)+4*noise
            elif family=='paper':v=8*noise+2*math.sin(y*.38)
            else:v=10*(1-xx)+8*math.sin((xx+yy)*4+phase)+1.5*noise
            v+=(index%4-1.5)*5;offset=(y*size+x)*3
            for ch in range(3):pixels[offset+ch]=max(0,min(255,round(base[ch]+v+(ch-1)*(index%3)*3)))
    return Image.frombytes('RGB',(size,size),bytes(pixels))

def generate(root,count=12,size=192):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'manifest.csv').open('w',newline='',encoding='utf-8') as fp:
        wr=csv.writer(fp);wr.writerow(['path','family','split','source','license','sample_group','is_real_photo'])
        for family in FAMILIES:
            for i in range(count):
                split='train' if i<count-4 else 'val' if i<count-2 else 'test'
                name=f"{family.replace('-','_')}_{i:03d}.png"
                dst=root/split/family/name;dst.parent.mkdir(parents=True,exist_ok=True)
                texture(family,i,size).save(dst,optimize=True)
                wr.writerow([str(dst.relative_to(root)),family,split,'kumo_generated','project_created',f'{family}-{i}',False])
    return count*len(FAMILIES)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('training/image_samples'))
    p.add_argument('--count',type=int,default=12);p.add_argument('--size',type=int,default=192)
    a=p.parse_args()
    print('Generated',generate(a.output,a.count,a.size),'RGB PNGs:',a.output)
