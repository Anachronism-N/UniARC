import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluation", ROOT / "scripts/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_empty_generation_counts_as_error(self):
        records = [{"id": "1", "prediction": "", "reference": "happy"},
                   {"id": "2", "prediction": "happy", "reference": "happy"}]
        self.assertEqual(evaluation.score(records, "accuracy")["score"], 0.5)

    def test_multitoken_labels_are_distinguished(self):
        records = [{"id": "1", "prediction": "air conditioner", "reference": "air horn"}]
        self.assertEqual(evaluation.score(records, "accuracy")["score"], 0)
        self.assertEqual(evaluation.score(records, "accuracy", "first-token")["score"], 1)

    def test_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.jsonl"
            item = json.dumps({"id": "1", "prediction": "", "reference": "x"})
            path.write_text(item + "\n" + item + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluation.load_records(path)

    def test_xares_normalization_preserves_documented_convention(self):
        self.assertEqual(evaluation.normalize("ＦＯＯ-bar,   isn't!", "xares"), "foo bar isnt")

    def test_wer_corpus_counts_empty_prediction(self):
        try:
            import jiwer
        except ImportError:
            self.skipTest("Optional jiwer package not installed")
        records = [{"id": "1", "prediction": "", "reference": "a b"},
                   {"id": "2", "prediction": "c", "reference": "c"}]
        result = evaluation.score(records, "wer")
        self.assertAlmostEqual(result["score"], 2 / 3)
        self.assertAlmostEqual(result["iwer"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
