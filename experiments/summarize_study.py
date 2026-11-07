"""Summarize measured runs, explicitly marking incomplete studies."""
import csv
import json
from pathlib import Path
import statistics
import sys


def summarize(study):
    study = Path(study)
    state = json.loads((study / "study_status.json").read_text())
    grouped = []
    for budget in state["iteration_budgets"]:
        runs = [r for r in state["completed_runs"] if r["n_iters"] == budget]
        if not runs: continue
        row = dict(study_status=state["status"], n_iters=budget, completed_trials=len(runs),
                   planned_trials=len(state["seeds"]), questions_per_trial=len(state["question_ids"]))
        for key in ["accuracy", "output_tokens", "input_tokens", "calls", "inference_seconds", "wall_seconds"]:
            values = [r[key] for r in runs]
            row[key + "_mean"] = statistics.mean(values)
            row[key + "_sample_std"] = statistics.stdev(values) if len(values)>1 else ""
        row["missing_examples_total"] = sum(r["missing_examples"] for r in runs)
        row["no_terminal_examples_total"] = sum(r["no_terminal_examples"] for r in runs)
        grouped.append(row)
    if grouped:
        with (study / "results_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f,fieldnames=list(grouped[0])); w.writeheader(); w.writerows(grouped)
    points = []
    for f in sorted(study.glob("seed*_iters*/offline_evaluation.json")):
        run = f.parent.name
        for row in json.loads(f.read_text())["examples"]:
            points.append(dict(run=run, **{k:v for k,v in row.items() if k != "parsed_answers"}))
    if points:
        with (study / "per_question_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f,fieldnames=list(points[0])); w.writeheader(); w.writerows(points)
    return grouped


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]),indent=2))
