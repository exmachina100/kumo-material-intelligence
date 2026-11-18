"""Security and split-policy tests for GPT-synthetic image training ablation."""
import copy,hashlib,importlib.util,io,json,sys,unittest,zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
# The lightweight general CI runner does not install PyTorch. These integration
# tests are run in the dedicated model-training workflow after dependencies.
HAS_HEAVY=all(importlib.util.find_spec(name) is not None for name in ("PIL","torch","torchvision","sklearn","numpy"))
if HAS_HEAVY:
    from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if HAS_HEAVY:
    sys.path.insert(0,str(ROOT/"training/experiments"))
    spec=importlib.util.spec_from_file_location("ablation",ROOT/"training/experiments/synthetic_ablation.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

@unittest.skipUnless(HAS_HEAVY,"Heavy ML dependencies are tested in the dedicated model-training workflow")
class SyntheticAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original=json.loads((ROOT/"training/gpt_synthetic/approved_seed_2026-11-18.json").read_text())
        cls.fixture=copy.deepcopy(original)
        cls.fixture["records"]=cls.fixture["records"][:1]
        cls.fixture["visual_label_approvals"]=1
        canvas=Image.new("RGB",(20,20),(80,60,40))
        out=io.BytesIO();canvas.save(out,format="PNG");cls.image=out.getvalue()
        row=cls.fixture["records"][0]
        row["sha256"]=hashlib.sha256(cls.image).hexdigest()
        row["bytes"]=len(cls.image)
        cls.asset_path=f'train/{row["family"]}/{row["asset_id"]}.png'

    def create_zip(self,path,extra=None,content=None):
        with zipfile.ZipFile(path,"w") as archive:
            archive.writestr("manifest.json",json.dumps(self.fixture))
            archive.writestr(self.asset_path,self.image if content is None else content)
            for k,v in (extra or {}).items():archive.writestr(k,v)

    def test_only_matching_real_model_class_is_used_and_test_never_sees_synthetic(self):
        with TemporaryDirectory() as tmp:
            archive=Path(tmp)/"samples.zip";self.create_zip(archive)
            rows=[{"sha256":"a"*64}]
            samples,ignored=module.load_approved_synthetic(archive,rows,["textile","wood"])
            self.assertEqual(len(samples),1)
            self.assertEqual(len(ignored),0)
            self.assertEqual(samples[0]["weak_label"],"textile")
            samples,ignored=module.load_approved_synthetic(archive,rows,["wood","ceramic"])
            self.assertFalse(samples)
            self.assertEqual(len(ignored),1)

    def test_rejects_synthetic_test_data_in_archive(self):
        with TemporaryDirectory() as tmp:
            archive=Path(tmp)/"samples.zip"
            self.create_zip(archive,{"test/rubber/invalid.png":self.image})
            with self.assertRaises(ValueError):
                module.load_approved_synthetic(archive,[],["textile"])

    def test_rejects_image_tampering_and_real_source_sha_overlap(self):
        with TemporaryDirectory() as tmp:
            archive=Path(tmp)/"samples.zip"
            self.create_zip(archive,content=b"not-the-pinned-image")
            with self.assertRaises(ValueError):
                module.load_approved_synthetic(archive,[],["textile"])
            self.create_zip(archive)
            with self.assertRaises(ValueError):
                module.load_approved_synthetic(archive,[{"sha256":hashlib.sha256(self.image).hexdigest()}],["textile"])

if __name__=="__main__":unittest.main()
