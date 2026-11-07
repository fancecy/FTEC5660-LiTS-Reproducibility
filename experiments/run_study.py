"""Run a fixed, budget-limited LiTS study and score saved outputs offline.

One experimental variable: MCTS iteration budget (2 versus 4).
This is a partial reproduction, not a direct replication of paper Table 6.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "demos")]
from lits.components.utils import retrieve_answer_from_last_step
from lits_benchmark.math_qa_eval import evaluate_math_qa


def score_run(out, rows, exit_code):
    scored = []
    for idx, row in enumerate(rows):
        f = out / "terminal_nodes" / f"terminal_nodes_{idx}.json"
        nodes = json.loads(f.read_text())["terminal_nodes"] if f.exists() else []
        answers = []
        for node in nodes:
            step = node.get("step") or {}
            answer = retrieve_answer_from_last_step(step.get("action", ""))
            if answer:
                answers.append(answer)
        # Deterministic plurality; insertion order breaks ties. No reference
        # answer is used when selecting a prediction. No LLM judge is called.
        pred = Counter(answers).most_common(1)[0][0] if answers else ""
        scored.append(dict(index=idx, id=row["unique_id"], prediction=pred,
                           reference=str(row["answer"]),
                           correct=bool(evaluate_math_qa(pred, str(row["answer"]))),
                           terminal_count=len(nodes), parsed_answers=answers,
                           execution_missing=not f.exists()))
    usage_path = out / "inferencelogger.log"
    usage = [json.loads(s) for s in usage_path.read_text().splitlines()
             if s.strip()] if usage_path.exists() else []
    result = dict(exit_code=exit_code, planned_examples=len(rows),
                  correct=sum(s["correct"] for s in scored),
                  accuracy=sum(s["correct"] for s in scored)/len(rows),
                  missing_examples=sum(s["execution_missing"] for s in scored),
                  no_terminal_examples=sum(s["terminal_count"] == 0 for s in scored),
                  calls=len(usage), input_tokens=sum(s["input_tokens"] for s in usage),
                  output_tokens=sum(s["output_tokens"] for s in usage),
                  inference_seconds=sum(s["running_time"] or 0 for s in usage),
                  examples=scored)
    (out / "offline_evaluation.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--name", default="main")
    args = p.parse_args()
    rows = [json.loads(s) for s in (ROOT / "data/math500-float.jsonl").read_text().splitlines()
            if s.strip()][:args.limit]
    if not rows or args.limit > 316:
        raise ValueError("Use 1 to 316 examples")
    dest = ROOT / "results" / args.name
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    source_commit="25e94d862cb4514c22d9a80a635e392461769df7",
                    model="Qwen/Qwen2.5-1.5B-Instruct",
                    model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                    dataset_revision="d27dc61b1123c2fdc4eea855c802be87dfa9d536",
                    question_ids=[r["unique_id"] for r in rows],
                    dataset_sha256=hashlib.sha256((ROOT / "data/math500-float.jsonl").read_bytes()).hexdigest(),
                    seeds=args.seeds, iteration_budgets=[2,4], max_steps=2,
                    termination="binary_sampling", termination_samples=1,
                    output_selection="plurality of regex-parsed terminal answers; first tie",
                    status="running", completed_runs=[])
    state_path = dest / "study_status.json"
    if state_path.exists():
        raise FileExistsError(f"Study already exists: {dest}; use another --name")
    def save():
        state_path.write_text(json.dumps(manifest, indent=2))
        from summarize_study import summarize
        summarize(dest)
    save()
    try:
        for j, seed in enumerate(args.seeds):
            # Alternate order across trials to reduce systematic thermal bias.
            for iters in ([2,4] if j % 2 == 0 else [4,2]):
                out = dest / f"seed{seed}_iters{iters}"
                out.mkdir()
                cmd = [sys.executable, str(ROOT / "run_local.py"), "--mode", "search",
                       "--seed", str(seed), "--dataset", "math500", "--include", "local_math",
                       "--policy-model", manifest["model"], "--policy", "concat",
                       "--transition", "concat", "--reward", "generative",
                       "--search-arg", "device=cpu", f"n_iters={iters}", "n_actions=2",
                       "max_steps=2", "roll_out_steps=1", "terminate_constraints=[binary_sampling]",
                       "sample_size_terminate=1",
                       "terminate_on_first_solution=false", "force_terminating_on_depth_limit=true",
                       "enable_think_policy=false", "enable_think_eval=false",
                       "runtime_limit_before_iter=600", "--component-arg",
                       "max_new_tokens=192", "max_length=4096", "think_for_correctness=false",
                       "think_for_usefulness=false", "n_for_correctness=1", "n_for_usefulness=1",
                       "--var", f"limit={args.limit}", "-o", str(out)]
                (out / "command.json").write_text(json.dumps(cmd, indent=2))
                print(f"START seed={seed} iterations={iters}", flush=True)
                start = time.perf_counter()
                with (out / "console.log").open("w") as log:
                    proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                result = score_run(out, rows, proc.returncode)
                result.update(seed=seed, n_iters=iters, wall_seconds=time.perf_counter()-start)
                manifest["completed_runs"].append({k:v for k,v in result.items() if k != "examples"})
                save()
                print(f"END seed={seed} iterations={iters} accuracy={result['accuracy']:.3f}", flush=True)
                if proc.returncode:
                    raise RuntimeError(f"Search failed; see {out / 'console.log'}")
        manifest["status"] = "complete"
    except BaseException as e:
        manifest["status"] = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
        manifest["error"] = str(e)
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__": main()
