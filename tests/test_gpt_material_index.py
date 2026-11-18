import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('material_index',ROOT/'training/gpt_synthetic/build_index.py')
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)

class IndexTests(unittest.TestCase):
    def test_empty_does_not_pretend_generations_exist(self):
        with TemporaryDirectory() as d:
            data=m.index(Path(d))
            self.assertEqual(data['total_samples'],0)
            self.assertEqual(data['status'],'awaiting_first_billed_batch')

    def test_index_counts_real_delivered_batches_but_keeps_unverified(self):
        with TemporaryDirectory() as d:
            p=Path(d);(p/'manifests').mkdir();(p/'previews').mkdir()
            manifest={'batch_id':'gpt-cmf-2026-11-18','model':'gpt-image-2.5-flare',
                      'count':2,'auto_training_enabled':False,
                      'records':[{'prompt_intended_family':name,'training_eligible':False,
                                  'is_real_material_capture':False} for name in ('polymer','rubber')]}
            (p/'manifests'/'2026-11-18.json').write_text(json.dumps(manifest))
            (p/'previews'/'2026-11-18.jpg').write_bytes(b'preview')
            result=m.index(p)
            self.assertEqual(result['total_samples'],2)
            self.assertFalse(result['training_eligible'])
            self.assertEqual(result['batches'][0]['families'],{'polymer':1,'rubber':1})
            self.assertIn('releases/tag/gpt-cmf-2026-11-18',result['batches'][0]['release_url'])

    def test_ineligible_flags_cannot_be_auto_upgraded(self):
        with TemporaryDirectory() as d:
            p=Path(d);(p/'manifests').mkdir();(p/'previews').mkdir()
            a={'batch_id':'gpt-cmf-2026-11-18','model':'gpt-image-2.5-flare',
               'count':1,'auto_training_enabled':False,
               'records':[{'prompt_intended_family':'polymer','training_eligible':True,
                           'is_real_material_capture':False}]}
            (p/'manifests'/'2026-11-18.json').write_text(json.dumps(a))
            (p/'previews'/'2026-11-18.jpg').write_bytes(b'preview')
            with self.assertRaises(ValueError):m.index(p)

if __name__=='__main__':unittest.main()
