"""Re-score completed generations with the frozen extractor and export audit rows.

This separate stage avoids regenerating outputs after an evaluation-only change.
It never changes a running generation study or marks failed generation complete.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from run_study import ROOT, REPO, score_run
from summarize_study import summarize

p = argparse.ArgumentParser()
p.add_argument("study", type=Path)
p.add_argument("--wait", action="store_true")
args = p.parse_args()
status = args.study / "study_status.json"
deadline = time.monotonic() + 6*3600
while True:
    try: state = json.loads(status.read_text())
    except json.JSONDecodeError:
        if not args.wait: raise
        time.sleep(1)
        continue
    if state["status"] == "complete": break
    if state["status"] in ["failed", "interrupted"]:
        raise RuntimeError("Generation did not complete; do not finalize results")
    if not args.wait: raise RuntimeError("Generation is still running")
    if time.monotonic() > deadline: raise TimeoutError("Study did not complete within 6 hours")
    time.sleep(15)

rows = [json.loads(s) for s in (ROOT / "data/math500-float.jsonl").read_text().splitlines()
        if s.strip()][:len(state["question_ids"])]
assert [r["unique_id"] for r in rows] == state["question_ids"]
updated, audit = [], []
for old in state["completed_runs"]:
    out = args.study / f"seed{old['seed']}_iters{old['n_iters']}"
    result = score_run(out, rows, old["exit_code"])
    new = {k:v for k,v in result.items() if k != "examples"}
    new.update(seed=old["seed"], n_iters=old["n_iters"], wall_seconds=old["wall_seconds"])
    updated.append(new)
    for ex in result["examples"]:
        if ex["prediction"] != ex["extended_prediction"]:
            audit.append(dict(seed=old["seed"], n_iters=old["n_iters"],
                              manual_review_status="pending", **ex))
state["completed_runs"] = updated
state["scoring_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
state["extractor_sha256"] = hashlib.sha256((ROOT / "answer_format.py").read_bytes()).hexdigest()
state["evaluation_finalized"] = True
state["manual_review_status"] = "pending"
(args.study / "changed_predictions_for_review.json").write_text(json.dumps(audit,indent=2))
status.write_text(json.dumps(state,indent=2))
print(json.dumps(summarize(args.study),indent=2),flush=True)
