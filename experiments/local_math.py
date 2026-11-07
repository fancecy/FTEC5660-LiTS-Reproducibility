"""Pinned offline copy of the author's numeric MATH500 dataset.

Keep the original source order. Do not expose solutions to the search pipeline.
"""
import json
from pathlib import Path
from lits.benchmarks.registry import register_dataset
from lits_benchmark.math_qa_eval import evaluate_math_qa  # original evaluator


@register_dataset("math500", task_type="language_grounded")
def load_math500(**kwargs):
    path = Path(__file__).parent / "data" / "math500-float.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{"question": r["question"], "answer": r["answer"],
             "id": r["unique_id"]} for r in rows]
