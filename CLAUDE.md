# CLAUDE.md — MedAgentGym (lmnr-ai fork)

Fork of `wshi83/MedAgentGym`. We use it to **generate agent trajectories** (Laminar traces),
not to reproduce the paper's numbers. Optimize for "the run completes and the trajectory is
interesting", not for leaderboard fidelity.

## Hard constraint: datasets

We are **not** authorized to use MIMIC-III, MIMIC-Extract, eICU, TREQS, EHRShot, EHRCon,
EHR-SeqSQL or nPowerAI. Their task modules, configs and `data/` directories were deleted.
Do not re-add them, and do not copy code back from upstream that references them.

Only four datasets remain: `biocoder`, `biodsbench`, `medagentbench`, `medcalcbench`.

## Layout

- `main.py` — entrypoint; owns the task-name → task-class dispatch table (`TASK_CLASSES`).
  Registering a new dataset means adding a row there plus a `configs/<model>/exp-*.yaml`.
- `ehr_gym/env/task/<dataset>.py` — one class per dataset; `setup()` builds the goal string
  and the ground truth, `validate()` grades. Grading is per-dataset and ad hoc.
- `ehr_gym/env/action/function.py` — the three actions agents can take: `validate_code`,
  `debug`, `terminal`. `ACTION_SET` in `action_set.py` is the registry. Both run things
  with `sys.executable` / a PATH rooted at it, never a bare `python` — see below.
- `ehr_gym/env/task/substitution.py` — the one place that splices a solution into a task
  template. Used by `env/base.py` for the agent's submission and by `task/biocoder.py`
  for the reference. Don't reintroduce `str.replace`.
- `scripts/` — one-off maintenance: `fetch_biodsbench_data.py` downloads the study data,
  `filter_biocoder.py` / `filter_biodsbench.py` drop ungradable rows and rewrite
  `data/metadata.json`. Shared subprocess helpers live in `scripts/_common.py`.
- `ehr_gym/llm/chat_api.py` — Azure OpenAI / OpenAI clients only. No local model serving.

## Environment

- Dependencies are `uv`-managed. `uv sync --extra tasks` — the `tasks` extra is **not**
  optional in practice: `validate_code` runs agent-generated code with the *harness's own*
  interpreter, so anything the agent imports must be in the same environment. `pip` is a
  declared dependency of that extra because the `terminal` action lets agents `pip install`.
- Several pins in the `tasks` extra are load-bearing, not cosmetic. `numpy<2` (the scraped
  BioCoder snippets use `np.asfarray` / `np.float_`, removed in NumPy 2.0),
  `scikit-image<0.23` (`multichannel=` was removed), `pandas<3`, `jax<0.4.32` (later
  versions force numpy 2). Resolve the whole extra in one command — installing jax on its
  own silently upgrades numpy and defeats the pin.
- The BioCoder keep set is a **function of what is installed**. Changing the extra changes
  which references run, so re-run `scripts/filter_biocoder.py` after touching it.
- **Do not run `uv pip install` without `--python .venv/bin/python`.** `VIRTUAL_ENV` in the
  agent sandbox points at the runtime's own venv, not this project's, and uv honours it.
- `credentials.toml` is gitignored; copy `credentials.example.toml`. Every top-level key in
  it is exported verbatim as an environment variable by `set_environment_variables()`.
- No CUDA and no `torch` — we only call hosted APIs. Don't reintroduce a GPU base image.
- **Agent code executes with `cwd` = repo root** and freely writes files there
  (`test.bam`, `output/`, `STAR`, …). Run experiments from a scratch directory, or expect
  to `git clean -fd` afterwards. Anything *we* write that runs scraped task code should go
  through `scripts/_common.run_python`, which uses a temp cwd.

## Grading, per dataset (all execution-based; there is no LLM-as-a-judge anywhere)

- **BioCoder** (496 train / 149 test) — ground truth is generated at `setup()` time by
  *executing the reference program and capturing stdout*, then compared to the agent's
  stdout. If the reference crashes, the "ground truth" becomes a traceback containing a
  per-run temp filename and the task is unwinnable, so `setup()` now raises instead.
  `scripts/filter_biocoder.py` already removed every row that crashes, prints nothing
  (the empty string matches any silent program), or prints something different on each
  run. Rows whose `<<insert solution here>>` marker sits inside an expression
  (`n = <<insert solution here>>`) were also dropped: a function *definition* cannot be
  spliced into a call site, so no re-indentation can save them.
  `pysam` alone accounts for ~70 of the test tasks — never drop it.
- **BioDSBench** (42 train / 43 test) — the agent's code is spliced *between* the task's
  setup code and a block of `assert`s, so the assertions decide the reward. That means the
  agent must be shown `code_histories` (the goal string does this) and that the splice must
  actually happen (`context_pattern` drives `need_context` in `env/base.py`; upstream keyed
  off an `info` field BioDSBench doesn't have, so every non-crashing submission scored 1).
  Study data is fetched by `scripts/fetch_biodsbench_data.py` into
  `data/biodsbench/data/<study_id>/`; tasks span 11 studies that reuse the same ten
  filenames, so the per-study subdirectory matters. Both `data_*.txt` (raw cBioPortal TSV)
  and `data_*.csv` are written — a few tasks read the `.txt`.
- **MedAgentBench** (240 train / 60 test) — needs a live HAPI FHIR server on
  `http://localhost:8080/fhir/` (`bash data/medagentbench/start_eval_docker.sh`). 54/60
  test rows have an empty `sol` and are graded by programmatic `task1..task10` graders that
  query the server. Tasks 3/5/8/9/10 mutate server state, so restart between full runs.
- **MedCalcBench** (1047 test, no train split) — self-contained. 55 of the 57 calculators
  produce a number graded against `[Lower Limit, Upper Limit]`; the other two produce a
  date or a gestational age, and for those the limits are just the answer repeated.
  `MedCalBenchTask.compare()` dispatches on the shape of the *ground truth*, so a date
  question is never graded as a float. All 1047 ground truths are accepted by it.

## Gotchas

- `data/metadata.json` is what `--end_idx -1` resolves against; it is a hand-maintained
  file, not derived from the `.jsonl`s. Upstream's counts were each one short. Regenerate
  it (`wc -l` per split) whenever task files change; the filter scripts do it for you.
- Trajectories are written to `<work_dir>/<task>/<result_dir_tag>/<mode>/history_*.json`
  and existing files are skipped, so re-running the same command resumes. Bump
  `result_dir_tag` when you want a genuinely fresh run.
- `--num_rollouts K > 1` changes the filename to `history_<idx>_<rollout>.json`; the
  single-rollout name has no suffix. Don't mix the two in one `result_dir_tag`.
