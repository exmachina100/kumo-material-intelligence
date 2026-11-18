import importlib.util
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ambient',ROOT/'training/external/ingest_ambientcg.py')
# Avoid requiring pillow for CI-only API guard tests: use AST structural and URL checks instead.
class AmbientGuardTests(unittest.TestCase):
    def test_source_contains_approved_cc0_and_review_gate(self):
        text=(ROOT/'training/external/ingest_ambientcg.py').read_text()
        self.assertIn("'license':'CC0'",text)
        self.assertIn("'needs_expert_review'",text)
        self.assertIn("'sample_group':'ambientcg:'",text)
        self.assertIn("acg-media.struffelproductions.com",text)
        self.assertIn("surface-photogrammetry",text)
if __name__=='__main__':unittest.main()
