import json
from pathlib import Path
import tempfile
import unittest
from run_study import score_run
from lits.config import ExperimentConfig
from answer_format import extract_extended


class StudyChecks(unittest.TestCase):
    def test_format_extension_accepts_explicit_conclusion_not_intermediate_numbers(self):
        self.assertEqual(extract_extended("Reasoning: 196=4*49.\n\nSo there are 9 positive divisors.")[0], "9")
        self.assertEqual(extract_extended("Intermediate value 9; further work needed.")[0], "")
        self.assertEqual(extract_extended("Therefore 196 has 9 divisors.")[0], "")
        self.assertEqual(extract_extended("Final answer: -2.5")[0], "-2.5")
        self.assertEqual(extract_extended(r"The final value is $\boxed{9}$.")[0], "9")
    def test_cpu_loading_setting_not_forwarded_to_search(self):
        c = ExperimentConfig(dataset="math500", policy_model_name="local",
                             eval_model_name="local", search_args={"device":"cpu", "n_iters":2})
        self.assertEqual(c.create_search_config().n_iters, 2)
        self.assertEqual(c.get_search_args()["device"], "cpu")
        c.search_args["not_a_search_setting"] = 1
        with self.assertRaises(TypeError): c.create_search_config()

    def test_plurality_does_not_select_reference_and_missing_stays_in_denominator(self):
        rows = [{"unique_id":"a", "answer":"9"}, {"unique_id":"b", "answer":"2"}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "terminal_nodes").mkdir()
            nodes = [{"step":{"action":"The answer is 8."}},
                     {"step":{"action":"The answer is 9."}},
                     {"step":{"action":"The answer is 8."}}]
            (out / "terminal_nodes/terminal_nodes_0.json").write_text(json.dumps({"terminal_nodes":nodes}))
            r = score_run(out, rows, 0)
            self.assertEqual(r["examples"][0]["prediction"], "8")
            self.assertEqual(r["accuracy"], 0)
            self.assertEqual(r["planned_examples"], 2)
            self.assertEqual(r["missing_examples"], 1)

    def test_valid_answer_uses_author_numeric_checker(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "terminal_nodes").mkdir()
            (out / "terminal_nodes/terminal_nodes_0.json").write_text(json.dumps({
                "terminal_nodes":[{"step":{"action":"The answer is 9.0."}}]}))
            r = score_run(out, [{"unique_id":"a", "answer":"9"}], 0)
            self.assertEqual(r["accuracy"], 1)


if __name__ == "__main__": unittest.main()
