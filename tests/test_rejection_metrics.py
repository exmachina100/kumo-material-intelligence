import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("rejection",ROOT/"training/experiments/rejection_metrics.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class RejectionTests(unittest.TestCase):
    def test_expected_acceptance_rates(self):
        r=mod.rejection_summary([.9,.6,.7],[.2,.85,.4],thresholds=(.65,.85))
        self.assertEqual(r["in_domain_count"],3)
        self.assertEqual(r["out_domain_count"],3)
        self.assertEqual(r["thresholds"][0]["in_accepted"],2)
        self.assertEqual(r["thresholds"][0]["out_accepted"],1)
        self.assertEqual(r["thresholds"][1]["in_accepted"],1)
        self.assertEqual(r["thresholds"][1]["out_accepted"],1)

    def test_empty_ood_is_explicit(self):
        r=mod.rejection_summary([.5],[])
        self.assertEqual(r["out_domain_count"],0)
        self.assertIsNone(r["thresholds"][0]["out_domain_false_accept_rate"])

    def test_bad_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            mod.rejection_summary([1.3],[])
        with self.assertRaises(ValueError):
            mod.rejection_summary([.5],[float("nan")])

if __name__=="__main__":
    unittest.main()
