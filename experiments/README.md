# LiTS partial reproduction: answer-format robustness

## Study objective

The objective is a partial reproduction of the LiTS math search workflow,
followed by a controlled modification of answer extraction rules. The reported
capability being tested is the documented MCTS math reasoning pipeline.
A local extraction failure motivates the modification; that failure is an
observation from this study rather than a result reported by the original paper.
The two extraction rules receive identical saved generations. The experiment
measures evaluation sensitivity and does not establish improved reasoning or
a direct replication of the paper's benchmark accuracy.

The original plan explored search budgets of 2 and 4 iterations. Exploratory
runs exposed a concrete scoring problem: a terminal answer correctly stated
that 196 has 9 positive divisors, but the existing numeric extractor returned
an empty answer because the phrase did not match its expected marker. The
formal study therefore fixes the search budget at 2 iterations and compares
only the two extraction rules. Pilot findings are not final benchmark results.

## Sources and pinned versions

- Upstream: https://github.com/xinzhel/lits-llm
- Source commit: `25e94d862cb4514c22d9a80a635e392461769df7`.
- Paper: https://arxiv.org/html/2603.00631v2, section 3.3, Table 6.
- Paper reference: ReST-MCTS reports 37% on 100 numerical MATH500 questions
  with Llama3-8B-Instruct. This figure is contextual and is not directly
  comparable with this study's model, subset, settings, and scoring adapter.
- Local model: Qwen/Qwen2.5-1.5B-Instruct, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (about 3 GB).
- Dataset: xinzhel/math500-float, revision
  `d27dc61b1123c2fdc4eea855c802be87dfa9d536`, 316 numerical rows.
- Hardware: M4 MacBook Air, 16 GB memory. The execution environment cannot
  access Metal acceleration, so measured search runs use CPU, 4 Torch threads.

## Fixed formal protocol

Use the first 10 questions in source order; question identifiers are saved in
`planned_question_ids.json`. No selection based on success is permitted.
The first question was also used during exploratory development, so this is
not a fully held-out evaluation. `local_math.py` removes reference solutions
from model inputs. References are used only for correctness checks.

Generate outputs at seeds 42, 43 and 44. Python, NumPy and Torch are seeded at
the run level; bitwise reproducibility across machines is not guaranteed.
All formal runs use MCTS with 2 iterations, branching factor 2, maximum depth 2,
rollout length 1, policy temperature 0.7, policy top-p 0.95, and at most 192
new policy tokens. Reward scoring uses one correctness and one usefulness
sample without thinking prompts. Termination uses binary sampling with one
sample. Whole-search early stopping is disabled. A 600-second check before
iterations is not a hard timeout for individual model generations.

These resource reductions, the model replacement, and the evaluator adapter
must be disclosed. The study tests a meaningful subset and workflow; it does
not claim to replicate the paper's reported accuracy.

## One modification and evaluation rules

Baseline: the author's `retrieve_answer_from_last_step` numeric extractor,
followed by the author's `evaluate_math_qa` numeric correctness checker.

Modified extractor: retain any baseline answer. Otherwise accept a numeric
boxed answer or a final paragraph that starts with an explicit conclusion cue
and contains exactly one numeric candidate. An explicit final count statement also permits an object restatement such as "divisors of 196". Ambiguous or unsupported formats
remain unparsed. The function receives no reference answer and calls no LLM.
This is a conservative format extension, not a general semantic evaluator.

Both scorers use deterministic plurality across parsed terminal answers, with
first-occurrence tie breaking. This terminal selection is a study adapter:
the latest upstream CLI instead uses LLM voting through native tool calls,
which the local Hugging Face wrapper does not support. Both conditions use
identical generated trees and identical selection/checking rules; only answer
extraction changes. No new model calls are needed for the modified condition.

Report strict baseline extracted-answer accuracy, modified extracted-answer
accuracy, changed predictions, parser failures, missing outputs, empty terminal
sets, token usage and generation time. Keep all planned examples in the
denominator. Review every changed prediction in `extraction_evidence` before
claiming the score change reflects a valid answer being recovered. An improved
score must not be described as improved reasoning. Repeated seeds are repeated
measurements on the same 10 questions, not 30 distinct questions.

## Run in the prepared workspace

From the assignment workspace:

```bash
work/.venv/bin/python outputs/lits-reproduction/experiments/test_local_study.py
work/.venv/bin/python outputs/lits-reproduction/experiments/run_study.py --limit 10 --seeds 42 43 44 --name main
work/.venv/bin/python outputs/lits-reproduction/experiments/finalize_study.py outputs/lits-reproduction/experiments/results/main
```

Each study name must be new. For a fresh machine, create a Python 3.13 virtual
environment at the repository root, install with
`python -m pip install -e . -c experiments/environment.lock.txt`, then run
`python experiments/bootstrap_model.py`. Use that environment for the study.
On macOS, set `PIP_CERT=/etc/ssl/cert.pem` if the Python certificate bundle is
not configured. Certificate verification remains enabled.
The `models` and `hf` links in the prepared workspace are conveniences ignored
by Git; the bootstrap creates a pinned model cache on a fresh machine.

`run_local.py` invokes the original LiTS CLI, rather than reimplementing MCTS.
Every run saves `command.json`, resolved `config.json`, trees, terminal nodes,
usage logs and `offline_evaluation.json`. Study status is recorded as running,
complete, interrupted or failed. `results_summary.csv` reports means and sample
standard deviations only from completed runs, with completion status visible.
`per_question_results.csv` preserves all expected questions. After generation, `finalize_study.py` re-scores all saved outputs with the frozen extractor and exports `changed_predictions_for_review.json`. Review these cases before final interpretation. Never present a
running study as a final result. Runtime output is ignored by Git during work;
commit selected completed evidence before publishing the submission repository.

## Debug diary

1. Initial Python HTTPS validation failed. The system certificate bundle fixed
   installation without disabling verification. `pip check` found no broken
   requirements; observed versions are saved in `environment.lock.txt`.
2. The installed runtime cannot access Metal. Use CPU measurements and disclose
   this constraint; do not label them as GPU measurements.
3. Upstream forwarded `device=cpu` into `MCTSConfig`, causing a TypeError.
   A compatibility fix excludes the model-loading setting from the search
   constructor while retaining it for model loading and saved configuration.
   Tests confirm unknown search options still fail. This fix is shared by both
   evaluation conditions, not the main experimental modification.
4. The CLI list parser retains quotes in `["step_signal"]`; supported syntax
   is `[step_signal]`. The exploratory failure log is retained.
5. Free-form policy output does not set the explicit step-signal termination
   flag. Early pilots using that combination were invalid as accuracy evidence.
   The initial search-budget-only explanation was corrected after code review.
   Corrected runs use binary-sampling termination, as the upstream default does.
6. Exploratory runs overlapped in execution; their elapsed times are diagnostic,
   not fair performance benchmarks. Formal generation runs execute sequentially.
7. Correct answer content can lack the extractor's expected answer marker.
   This observation motivated the isolated extraction modification above.

## Completed submission

All 30 formal task executions and frozen re-scoring completed. The saved case
audit reviews all six changed predictions without score overrides. Final
statistics are in `results/main/report_statistics.json`; English report and
interactive slides are in `../submission/` and the GitHub release. A personal
Zoom recording remains a separate submission to be made by the student.

Timing note: inference logs use `time.time`, while the launcher's historical
`wall_seconds` field uses `perf_counter`. Summed model durations exceed the
launcher counter in the third seed. The reason is not established; retain both
raw fields and do not use them interchangeably or infer a speed improvement.
