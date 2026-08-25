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
  `debug`, `terminal`. `ACTION_SET` in `action_set.py` is the registry.
- `ehr_gym/llm/chat_api.py` — Azure OpenAI / OpenAI clients only. No local model serving.

## Environment

- Dependencies are `uv`-managed. `uv sync --extra tasks` — the `tasks` extra is **not**
  optional in practice: `validate_code` runs agent-generated code with the *venv's* Python,
  so anything the agent imports must be in the same environment. `pip` is a declared
  dependency of that extra because the `terminal` action lets agents `pip install`.
- `credentials.toml` is gitignored; copy `credentials.example.toml`. Every top-level key in
  it is exported verbatim as an environment variable by `set_environment_variables()`.
- No CUDA and no `torch` — we only call hosted APIs. Don't reintroduce a GPU base image.
- **Agent code and BioCoder reference programs execute with `cwd` = repo root** and freely
  write files there (`test.bam`, `output/`, `STAR`, …). Run experiments from a scratch
  directory, or expect to `git clean -fd` afterwards.

## Grading, per dataset (all execution-based; there is no LLM-as-a-judge anywhere)

- **BioCoder** — ground truth is generated at `setup()` time by *executing the reference
  program and capturing stdout*, then compared to the agent's stdout. If the reference
  program itself crashes, the "ground truth" becomes a traceback containing a per-run temp
  filename, and the task is unwinnable. Only 101/157 test and 543/981 train references run
  cleanly even with the full `tasks` extra; the rest need exotic packages (`centrosome`,
  `ensembler`, `simtk`, `jax`, `wx`, `biom`, …) or are broken by upstream's flat
  `<<insert solution here>>` substitution, which lifts class methods to module scope.
  `pysam` alone accounts for ~70 of the 157 test tasks — never drop it.
- **BioDSBench** — expects cBioPortal study CSVs at `/workdir/data_*.csv` (10 files, 11
  studies). They are **not in the repo** and are not downloadable from it; without them
  every task fails on `FileNotFoundError`.
- **MedAgentBench** — needs a live HAPI FHIR server on `http://localhost:8080/fhir/`
  (`bash data/medagentbench/start_eval_docker.sh`). 54/60 test rows have an empty `sol`
  and are graded by programmatic `task1..task10` graders that query the server. Tasks
  3/5/8/9/10 mutate server state, so restart the container between full runs.
- **MedCalcBench** — self-contained, graded by `float(pred)` against a
  `[Lower Limit, Upper Limit]` range. 60 of the 1047 rows have *date-string* answers
  (e.g. `09/23/2014`), so `float()` raises and they can never pass. No train split ships.

## Gotchas

- `data/metadata.json` is what `--end_idx -1` resolves against; it is a hand-maintained
  file, not derived from the `.jsonl`s. Upstream's counts were each one short. Regenerate
  it (`wc -l` per split) whenever task files change.
- Trajectories are written to `<work_dir>/<task>/<result_dir_tag>/<mode>/history_*.json`
  and existing files are skipped, so re-running the same command resumes. Bump
  `result_dir_tag` when you want a genuinely fresh run.
- `--num_rollouts K > 1` changes the filename to `history_<idx>_<rollout>.json`; the
  single-rollout name has no suffix. Don't mix the two in one `result_dir_tag`.
