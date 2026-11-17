import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vision_dataset", ROOT/"training/experiments/vision_dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class VisionDataTests(unittest.TestCase):
    def test_uses_only_valid_cc0_with_checksum_and_distinct_assets(self):
        with TemporaryDirectory() as td:
            folder=Path(td)
            rows=[]
            for family in ("metal","wood"):
                for i in range(10):
                    name=f"{family}{i}.jpg"
                    data=(f"{family}-{i}").encode()
                    (folder/name).write_bytes(data)
                    rows.append({"asset_id":f"{family}-{i}","path":name,"source":"Poly Haven",
                                 "license":"CC0","catalog_suggestion":family,
                                 "sha256":hashlib.sha256(data).hexdigest()})
            rows.append({"asset_id":"bad","path":"../bad.jpg","source":"Poly Haven",
                         "license":"CC0","catalog_suggestion":"metal"})
            rows.append({"asset_id":"unlicensed","path":"metal0.jpg","source":"Poly Haven",
                         "license":"proprietary","catalog_suggestion":"metal"})
            (folder/"provenance.json").write_text(json.dumps(rows))
            got=module.select_catalog_samples(folder,8)
            self.assertEqual(len(got),20)
            self.assertEqual({x["weak_label"] for x in got},{"metal","wood"})
            self.assertEqual(len({x["asset_id"] for x in got}),20)

    def test_rejects_too_few_samples(self):
        with TemporaryDirectory() as td:
            (Path(td)/"provenance.json").write_text("[]")
            with self.assertRaises(ValueError):
                module.select_catalog_samples(Path(td),8)

if __name__=="__main__":
    unittest.main()
