"""Prevent physical-source and synthetic-batch leakage in model evaluations."""
import importlib.util
import unittest
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'training/experiments/material_splits.py'
spec=importlib.util.spec_from_file_location('material_splits',target)
splitter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)

def fixtures(n=20):
    return [{"asset_id":f"{f}-{i}","weak_label":f,
             "sample_group":f"origin-{f}-{i//2}","source":"Test CC0"}
            for f in ('wood','textile','metal') for i in range(n)]

class MaterialGroupSplitTests(unittest.TestCase):
    def test_reproducible_nonoverlapping_splits(self):
        rows=fixtures()
        one=splitter.group_stratified_split(rows,seed=42)
        two=splitter.group_stratified_split(rows,seed=42)
        self.assertEqual(one,two)
        for split,indices in one.items():
            self.assertTrue(indices)
        groups={split:{rows[i]['sample_group'] for i in indices}
                for split,indices in one.items()}
        self.assertFalse(groups['train']&groups['val'])
        self.assertFalse(groups['train']&groups['test'])
        self.assertFalse(groups['val']&groups['test'])
    def test_repeated_sample_groups_are_not_split(self):
        rows=fixtures()
        result=splitter.group_stratified_split(rows,seed=71)
        by_group={}
        for split,indices in result.items():
            for i in indices:
                g=rows[i]['sample_group']
                self.assertEqual(by_group.setdefault(g,split),split)
        self.assertEqual(sum(map(len,result.values())),len(rows))
        for split,indices in result.items():
            self.assertEqual(set(rows[i]['weak_label'] for i in indices),
                             {'wood','textile','metal'})
    def test_conflicting_labels_in_same_source_group_are_rejected(self):
        rows=fixtures()
        rows[1]['weak_label']='metal'
        with self.assertRaises(ValueError):splitter.group_stratified_split(rows)
    def test_too_few_independent_groups_are_rejected(self):
        rows=fixtures(n=4)
        with self.assertRaises(ValueError):splitter.group_stratified_split(rows)

if __name__=='__main__':
    unittest.main()
