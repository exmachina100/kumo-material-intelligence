"""Contract tests for the scheduled GPT-generated synthetic CMF drip.

No API calls, no image generation costs, and no nondeterministic network I/O.
"""
import importlib.util
import json
import unittest
from collections import Counter
from datetime import date,timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'training/gpt_synthetic/drip.py'
spec=importlib.util.spec_from_file_location('gpt_material_drip',MODULE)
drip=importlib.util.module_from_spec(spec)
spec.loader.exec_module(drip)

class MaterialDripTests(unittest.TestCase):
    def test_daily_plan_is_deterministic_and_budget_limited(self):
        d=date(2026,11,18)
        one=drip.plan_batch(d,6)
        self.assertEqual(one,drip.plan_batch(d,6))
        self.assertEqual(len(one),6)
        self.assertEqual(len({p['filename'] for p in one}),6)
        self.assertEqual(len({p['family'] for p in one}),6)
        with self.assertRaises(ValueError):drip.plan_batch(d,0)
        with self.assertRaises(ValueError):drip.plan_batch(d,13)

    def test_rotation_visits_every_family_without_starving_categories(self):
        start=date(2026,11,18)
        families=[entry['family'] for day in range(13)
                  for entry in drip.plan_batch(start+timedelta(days=day),6)]
        counts=Counter(families)
        self.assertEqual(set(counts),set(drip.FAMILIES))
        self.assertTrue(max(counts.values())-min(counts.values())<=1)

    def test_prompts_are_individual_macro_photographs_not_contact_sheets(self):
        plan=drip.plan_batch(date(2026,11,18),6)
        for spec in plan:
            prompt=spec['prompt'].lower()
            self.assertIn('single',prompt)
            self.assertIn('surface',prompt)
            self.assertIn('macro',prompt)
            self.assertIn('no collage',prompt)
            self.assertIn('no text',prompt)
            self.assertIn('no brand',prompt)

    def test_manifest_flags_never_claim_real_capture_or_verified_labels(self):
        spec=drip.plan_batch(date(2026,11,18),1)[0]
        record=drip.make_record(spec,model='gpt-image-2.5-flare',
                                sha='a'*64,width=1024,height=1024,file_size=10000)
        self.assertFalse(record['is_real_material_capture'])
        self.assertFalse(record['training_eligible'])
        self.assertEqual(record['review_status'],'needs_expert_review')
        self.assertEqual(record['label_kind'],'prompt_intent_not_verified')
        self.assertEqual(record['source'],'GPT-generated synthetic')

    def test_dry_run_needs_no_api_key_or_output_directory(self):
        with TemporaryDirectory() as temp:
            d=date(2026,11,18)
            output=Path(temp)/'unused'
            plan=drip.execute(date_value=d,count=2,output=output,dry_run=True)
            self.assertEqual(len(plan),2)
            self.assertFalse(output.exists())

    def test_invalid_dates_and_models_are_rejected(self):
        with self.assertRaises(ValueError):drip.validate_model('../bad')
        with self.assertRaises(ValueError):drip.validate_model('gpt-image-1')
        self.assertEqual(drip.validate_model('gpt-image-2.5-flare'),'gpt-image-2.5-flare')

if __name__=='__main__':unittest.main()
