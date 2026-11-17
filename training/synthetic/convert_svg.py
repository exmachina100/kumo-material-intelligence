"""Render committed synthetic SVGs into locally ignored PNGs for pipeline smoke tests."""
import argparse
from pathlib import Path

def convert(root: Path, output: Path, size: int):
    import cairosvg
    from PIL import Image
    from io import BytesIO
import hashlib, random
    count=0
    for src in sorted(root.glob('*/*/*.svg')):
        split,family=src.relative_to(root).parts[:2]
        target=output/split/family/(src.stem+'.png')
        target.parent.mkdir(parents=True,exist_ok=True)
        data=cairosvg.svg2png(url=str(src),output_width=size,output_height=size)
        image=Image.open(BytesIO(data)).convert('RGB')
        # Stable per-sample micro-dithering prevents exact duplicate procedural exports.
        # These images are for pipeline testing only, never real-world accuracy measurement.
        random_seed=int(hashlib.sha256(str(src.relative_to(root)).encode()).hexdigest()[:12],16)
        rng=random.Random(random_seed)
        px=image.load()
        for y in range(image.height):
            for x in range(image.width):
                if (x+2*y)%7==0:
                    rgb=px[x,y]
                    delta=rng.choice((-2,-1,1,2))
                    px[x,y]=tuple(max(0,min(255,v+delta)) for v in rgb)
        image.save(target,format='PNG',optimize=True)
        count+=1
    print(f'Converted {count} synthetic images to {output}')
    if not count:raise SystemExit('No SVG samples found')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('training/synthetic/svg'))
    p.add_argument('--output',type=Path,default=Path('training/data'))
    p.add_argument('--size',type=int,default=160)
    a=p.parse_args()
    convert(a.root,a.output,a.size)
