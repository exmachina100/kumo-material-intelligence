"""Validate source inventory and SVG XML before rasterization; standard library only."""
from collections import Counter
from pathlib import Path
import csv
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/'manifest.csv'
EXPECTED_FAMILIES={'metal','polymer','leather','synthetic-leather','textile','wood','stone','ceramic','glass','rubber','composite','paper','coating'}
EXPECTED_SPLITS={'train':30,'val':3,'test':3}

def validate():
    with MANIFEST.open(newline='',encoding='utf-8') as file:
        reader=csv.DictReader(file)
        rows=list(reader)
    errors=[]
    counter=Counter()
    seen=set()
    for row in rows:
        path=Path(row['path'])
        if '..' in path.parts or path.is_absolute():
            errors.append(f'Unsafe manifest path {path}')
            continue
        source=Path(__file__).resolve().parents[2]/path
        expected=ROOT/'svg'/row['split']/row['family']/path.name
        if source!=expected:
            errors.append(f'Path mismatch {path}')
        if str(path) in seen:errors.append(f'Duplicate row {path}')
        seen.add(str(path))
        if not source.is_file():
            errors.append(f'Missing SVG {path}')
            continue
        try:
            xml=ET.parse(source).getroot()
            if xml.tag!='{http://www.w3.org/2000/svg}svg':
                errors.append(f'Invalid SVG root {path}')
        except ET.ParseError as exc:
            errors.append(f'Bad XML {path}: {exc}')
        counter[(row['family'],row['split'])]+=1
    on_disk={str(p.relative_to(Path(__file__).resolve().parents[2])) for p in (ROOT/'svg').rglob('*.svg')}
    if seen!=on_disk:
        errors.append(f'Manifest/file mismatch: {len(seen)} listed vs {len(on_disk)} present')
    for family in EXPECTED_FAMILIES:
        for split,count in EXPECTED_SPLITS.items():
            if counter[(family,split)]!=count:
                errors.append(f'Expected {count} {family}/{split}; got {counter[(family,split)]}')
    if len(rows)!=468:
        errors.append(f'Expected 468 samples, found {len(rows)}')
    return rows,errors

if __name__=='__main__':
    rows,errors=validate()
    if errors:
        print('\n'.join(errors[:30]),file=sys.stderr)
        raise SystemExit(1)
    print(f'Validated {len(rows)} SVG samples, 13 families, 390 train / 39 val / 39 test')
