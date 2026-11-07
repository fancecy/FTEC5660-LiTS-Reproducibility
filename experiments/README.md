# Local LiTS reproducibility study

This directory contains a resource-limited reproduction of the LiTS math-search
workflow and a controlled change to the MCTS iteration budget. Results must be
read from the saved study files; setup success is not an accuracy result.

## Sources and scope

- Assignment project: LiTS, instructor list item 3.
- Upstream: https://github.com/xinzhel/lits-llm
- Upstream commit: `25e94d862cb4514c22d9a80a635e392461769df7`.
- Paper: https://arxiv.org/html/2603.00631v2, section 3.3 and Table 6.
- Paper reference: ReST-MCTS achieves 37% on 100 numerical MATH500 questions
  using Llama3-8B-Instruct. This is a contextual reference, not an expected score
  for our model, smaller subset, or adapted settings.
- Local model: Qwen/Qwen2.5-1.5B-Instruct, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Dataset: xinzhel/math500-float, revision
  `d27dc61b1123c2fdc4eea855c802be87dfa9d536`, 316 numeric-answer rows.
  The formal study uses the first 10 rows in source order. No success-based
  selection is permitted. `local_math.py` removes solutions from model inputs.

## Research question

Does increasing MCTS iterations from 2 to 4 improve accuracy on the fixed subset,
and what happens to token use, inference calls, and elapsed time?

Both arms use CPU inference, branching factor 2, maximum depth 2, rollout length
1, policy temperature 0.7, policy top-p 0.95, maximum 192 generated policy tokens,
one correctness sample, one usefulness sample, no thinking prompts, and
binary-sampling termination with one sample. Early termination of the whole search is disabled.
There is a 600-second check before each search iteration; this does not interrupt
an individual long generation and must not be described as a hard per-task limit.
The paired pilot determines feasibility before the formal study.

Three run-level seeds are planned: 42, 43, 44. Python, NumPy and Torch are seeded.
This does not guarantee bitwise reproducibility across hardware or library versions.
Arm order alternates across seeds. Both arms share the same model and settings.
CPU compatibility fixes and the evaluator adapter apply equally to both arms;
they are not additional experimental interventions.

## Evaluation

The latest upstream evaluation CLI requires an LLM with native tool calling to
vote on language-task terminal answers. The local HF model does not provide that
interface. This study instead uses the author's numeric answer-extraction helper
and numeric correctness checker, with deterministic plurality across parsed
terminal answers and first-occurrence tie breaking. The reference is never used
to select a prediction. This evaluation deviation must be reported.

Missing outputs, empty terminal sets and unparseable answers count as incorrect
in the fixed planned denominator. They are also reported separately. A crashed
search stops the study and is not represented as a completed experiment.
No paid API, LLM judge or external answer service is used.

## Run in the prepared workspace

From the assignment workspace, using the environment already installed there:

```bash
work/.venv/bin/python outputs/lits-reproduction/experiments/test_local_study.py
work/.venv/bin/python outputs/lits-reproduction/experiments/run_study.py --limit 1 --seeds 42 --name paired_pilot_new
work/.venv/bin/python outputs/lits-reproduction/experiments/run_study.py --limit 10 --seeds 42 43 44 --name main
```

Each study name must be new to avoid mixing or overwriting runs. The original
search CLI is invoked by `run_local.py`; this is not a reimplementation of MCTS.
See each run's `command.json` and `config.json` for the exact command and resolved
settings. `study_status.json` identifies complete, running, interrupted or failed
studies. `offline_evaluation.json` records every expected question, not just
successful outputs. Do not cite a running study as a final result.

The `models` and `hf` links are local workspace conveniences and are ignored by
Git. They must be replaced by downloaded model files/cache when reproducing on
another computer. For a fresh setup, create a Python 3.13 virtual environment at the repository root, run `python -m pip install -e . -c experiments/environment.lock.txt`, then run `python experiments/bootstrap_model.py`. The bootstrap downloads the exact model revision (approximately 3 GB). Use the same environment to run the study scripts. On macOS, `PIP_CERT=/etc/ssl/cert.pem` can be set if the Python certificate bundle is not configured. Training is not required.

## Debug diary

1. Python HTTPS certificate validation initially failed. Using the macOS system
   certificate bundle fixed installation without disabling TLS verification.
2. Installed upstream dependencies in an isolated virtual environment. `pip check`
   reported no broken requirements. Exact observed versions are recorded.
3. The execution environment cannot access Metal acceleration despite the host
   being an M4 MacBook Air with 16 GB RAM. All measured search runs use CPU.
4. Explicit `device=cpu` was forwarded to `MCTSConfig`, causing a TypeError.
   Excluded this model-loading-only field from the search constructor while
   retaining it in the saved experiment/model-loading configuration. A check
   also ensures unknown search options still raise TypeError.
5. The CLI list parser retains quotes in `["step_signal"]`. The correct supported
   syntax is `[step_signal]`; the failed exploratory log is retained.
6. Exploratory pilots used step-signal termination with free-form output. This
   configuration is incompatible: the free-form policy does not set the explicit
   termination flag. Depth-limited nodes are also not automatically collected as
   terminal answers. The initial budget-only explanation was therefore incorrect.
   Retain these pilots as configuration debugging evidence, not final results.
   The corrected pilot and formal study use binary-sampling termination, as the
   upstream default does, with one sample to limit CPU cost. Both arms use depth 2.

## Submission status

The report, presentation recording, GitHub publication and final interpretation
remain pending until the formal experiment finishes. A local repository or
script is not a substitute for the required accessible submission links.
