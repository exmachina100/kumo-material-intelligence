"""Offline dataset audit: detect duplicate image bytes across splits and report class balance.
Does not upload, modify, or copy any image.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse, hashlib, json

EXTENSIONS={'.jpg','.jpeg','.png','.webp','.bmp'}
SPLITS=('train','val','test')

def audit(root):
    counts={};seen=defaultdict(list);problems=[]
    for split in SPLITS:
        folder=root/split
        if not folder.is_dir():
            problems.append(f'Missing directory: {folder}')
            continue
        counter=Counter()
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
            label=path.relative_to(folder).parts[0]
            counter[label]+=1
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            seen[digest].append({'split':split,'path':str(path),'label':label})
        counts[split]=dict(counter)
        if not counter:problems.append(f'No images in {folder}')
    cross=[]
    for digest,entries in seen.items():
        if len({item['split'] for item in entries})>1:
            cross.append({'hash':digest,'occurrences':entries})
    if cross:problems.append(f'Found {len(cross)} exact duplicate images across splits')
    labels=[set(row) for row in counts.values()]
    if len(labels)==3 and not all(s==labels[0] for s in labels[1:]):
        problems.append('Class labels differ across train, val and test')
    return {'counts':counts,'exactCrossSplitDuplicates':cross,'problems':problems,'passed':not problems,
            'warning':'Perceptual near-duplicates and multiple photos of the same physical sample require additional manual/source-level checks.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('training/data'));p.add_argument('--json',type=Path)
    args=p.parse_args();report=audit(args.data)
    result=json.dumps(report,indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(result+'\n')
    print(result)
    if not report['passed']:raise SystemExit(2)
if __name__=='__main__':main()
