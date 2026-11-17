"""Render committed synthetic SVGs into locally ignored PNGs for pipeline smoke tests."""
import argparse
from pathlib import Path

def convert(root: Path, output: Path, size: int):
    import cairosvg
    from PIL import Image
    from io import BytesIO
    count=0
    for src in sorted(root.glob('*/*/*.svg')):
        split,family=src.relative_to(root).parts[:2]
        target=output/split/family/(src.stem+'.png')
        target.parent.mkdir(parents=True,exist_ok=True)
        data=cairosvg.svg2png(url=str(src),output_width=size,output_height=size)
        image=Image.open(BytesIO(data)).convert('RGB')
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
