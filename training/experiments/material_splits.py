"""Deterministic CMF source-group split with separate train, validation and test.

Every shared source/sample/batch group stays entirely in a single split.
The evaluation set is never trained or used for checkpoint selection.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import random

def group_stratified_split(rows,seed=42,val_fraction=.15,test_fraction=.20,min_groups=6):
    if not 0<val_fraction<.4 or not 0<test_fraction<.4 or val_fraction+test_fraction>=.6:
        raise ValueError("Fractions must preserve at least 40% training groups")
    by_class=defaultdict(lambda:defaultdict(list))
    label_for_group={}
    for index,row in enumerate(rows):
        label=row.get("weak_label")
        group=row.get("sample_group") or f"{row.get('source','unknown')}:{row.get('asset_id',index)}"
        if not label or not group:raise ValueError("Missing material family or source group")
        if group in label_for_group and label_for_group[group]!=label:
            raise ValueError(f"Source group {group} has conflicting labels")
        label_for_group[group]=label
        by_class[label][group].append(index)
    result={"train":[],"val":[],"test":[]}
    for family,groups in sorted(by_class.items()):
        names=sorted(groups)
        if len(names)<min_groups:
            raise ValueError(f"{family}: requires {min_groups} independent source groups, got {len(names)}")
        # Isolated deterministic shuffle per family, independent of dict/input order.
        hash_seed=int.from_bytes(hashlib.sha256(f"{seed}|{family}".encode()).digest()[:8],"big")
        rng=random.Random(hash_seed)
        rng.shuffle(names)
        n=len(names)
        n_test=max(1,round(n*test_fraction))
        n_val=max(1,round(n*val_fraction))
        if n-n_test-n_val<3:
            raise ValueError(f"{family}: insufficient train groups")
        for i,group in enumerate(names):
            split="test" if i<n_test else "val" if i<n_test+n_val else "train"
            result[split].extend(groups[group])
    for split in result:
        result[split].sort()
    if sum(map(len,result.values()))!=len(rows):
        raise AssertionError("Lost or duplicated samples in source-grouped split")
    return result
