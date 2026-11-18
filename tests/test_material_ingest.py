import importlib.util
import tempfile
import unittest
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

INGEST=load("cmf_ingest","training/external/ingest_polyhaven.py")
BUILD=load("cmf_review_build","training/external/build_reviewed_dataset.py")

class CandidateSelectionTests(unittest.TestCase):
    def test_category_takes_precedence_over_tag(self):
        self.assertEqual(INGEST.discover_family({
            "category":"Wood/Veneers","name":"metal color","tags":["metal"]}),"wood")
    def test_textile_plastic_rubber_category_mapping(self):
        self.assertEqual(INGEST.discover_family({"category":"Textiles & Leather/Fabrics","name":"Fabric Weave"}),"textile")
        self.assertEqual(INGEST.discover_family({"category":"Plastic & Rubber/Plastic","name":"Glossy Sheet"}),"polymer")
        self.assertEqual(INGEST.discover_family({"category":"Plastic & Rubber/Rubber","name":"Tire Texture"}),"rubber")
        self.assertEqual(INGEST.discover_family({"category":"Wood/Planks","tags":["metallic color"]}),"wood")
        self.assertEqual(INGEST.discover_family({"category":"Textiles & Leather/Leather","name":"Bovine Hide"}),"leather")
        self.assertEqual(INGEST.discover_family({"category":"Textiles & Leather/Fabric","name":"Woven Sheet"}),"textile")
        self.assertEqual(INGEST.discover_family({"category":"Plastic & Rubber/Rubber","name":"Elastomer"}),"rubber")
    def test_balanced_candidates_and_type_filter(self):
        catalog={
            "metal_plate":{"type":1,"thumbnail_url":"https://cdn.polyhaven.com/metal.png",
                            "category":"Metal/Steel","download_count":100},
            "timber":{"type":1,"thumbnail_url":"https://cdn.polyhaven.com/wood.png",
                     "category":"Wood","download_count":80},
            "not_texture":{"type":0,"thumbnail_url":"https://cdn.polyhaven.com/a.png",
                           "category":"Metal"}
        }
        rows=INGEST.select_candidates(catalog,10,4)
        self.assertEqual({r[0] for r in rows},{"metal_plate","timber"})
        self.assertEqual(len({r[0] for r in rows}),len(rows))
    def test_disallow_foreign_hosts(self):
        with self.assertRaises(ValueError):
            INGEST.safe_get("https://invalid.example/asset.jpg")

class ReviewedDatasetTests(unittest.TestCase):
    def test_review_gate_and_grouped_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"source";source.mkdir()
            registry=[];reviews=[]
            for i in range(6):
                name=f"wood_{i}.jpg";data=f"sample {i}".encode()
                (source/name).write_bytes(data)
                registry.append({"asset_id":f"wood_{i}","path":name,"source":"Poly Haven",
                                 "license":"CC0","sha256":hashlib.sha256(data).hexdigest(),
                                 "sample_group":f"polyhaven:wood_{i}"})
                reviews.append({"schemaVersion":"0.1.0","asset_id":f"wood_{i}","decision":"include",
                                "family":"wood","evidence":"visual-only","reviewer":"Test Reviewer",
                                "reviewed_at":"2026-11-18T00:00:00Z"})
            (source/"provenance.json").write_text(json.dumps(registry))
            plan=BUILD.plan_rows(source,reviews,min_per_family=6)
            self.assertEqual(len(plan),6)
            self.assertEqual({v["split"] for v in plan},{"train","val","test"})
            destination=Path(tmp)/"out"
            result=BUILD.write_dataset(plan,destination)
            self.assertEqual(result["sample_count"],6)
            self.assertTrue((destination/"dataset-manifest.json").is_file())
    def test_approved_ambientcg_reviews_join_without_catalog_label_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first=root/'polyhaven';first.mkdir()
            other=root/'ambientcg';other.mkdir()
            (first/'provenance.json').write_text('[]')
            records=[];reviews=[]
            for i in range(6):
                aid=f'ambientcg:Fabric{i:03d}'
                name=f'Fabric{i:03d}.jpg'
                blob=f'fabric sample {i}'.encode()
                (other/name).write_bytes(blob)
                records.append({'asset_id':aid,'path':name,'source':'ambientCG','license':'CC0',
                                'catalog_suggestion':'polymer',
                                'sha256':hashlib.sha256(blob).hexdigest(),
                                'sample_group':aid})
                reviews.append({'schemaVersion':'0.1.0','asset_id':aid,
                                'decision':'include','family':'textile',
                                'evidence':'visual-only','reviewer':'CMF Reviewer',
                                'reviewed_at':'2026-11-18T00:00:00Z'})
            (other/'provenance.json').write_text(json.dumps(records))
            plan=BUILD.plan_rows(first,reviews,min_per_family=6,additional_sources=[other])
            self.assertEqual(len(plan),6)
            self.assertEqual({x['family'] for x in plan},{'textile'})
            output=root/'reviewed'
            BUILD.write_dataset(plan,output)
            self.assertEqual(len(list(output.rglob('*.jpg'))),6)
            self.assertTrue(any('ambientcg_Fabric' in p.name for p in output.rglob('*.jpg')))
    def test_unreviewed_source_cannot_be_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp);(src/"provenance.json").write_text("[]")
            with self.assertRaisesRegex(ValueError,"No approved"):
                BUILD.plan_rows(src,[],min_per_family=3)

if __name__=="__main__":
    unittest.main()
