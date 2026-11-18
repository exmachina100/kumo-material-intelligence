import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("approved",ROOT/"training/gpt_synthetic/import_approved_seed.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class ApprovedGptSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/"training/gpt_synthetic/approved_seed_2026-11-18.json").read_text())
    def test_ten_unique_synthetic_label_approvals(self):
        samples=mod.validate(self.data)
        self.assertEqual(len(samples),10)
        self.assertEqual(len(set(s["family"] for s in samples)),10)
    def test_never_allow_test_or_real_labels(self):
        d=copy.deepcopy(self.data);d["records"][0]["allowed_splits"]=["test"]
        with self.assertRaises(ValueError):mod.validate(d)
        d=copy.deepcopy(self.data);d["records"][0]["synthetic_generated"]=False
        with self.assertRaises(ValueError):mod.validate(d)
    def test_reject_changed_source_domain(self):
        d=copy.deepcopy(self.data);d["records"][0]["image_url"]="https://evil.example.com/a.png"
        with self.assertRaises(ValueError):mod.validate(d)
if __name__=="__main__":unittest.main()
