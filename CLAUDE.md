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
  `daytona_run.py` is the odd one out: it is a *driver*, run on a dev machine, and the
  only thing in the repo that needs the `daytona` extra — see below.
- `ehr_gym/llm/chat_api.py` — one client, `gpt-5.6-luna` on Azure AI Foundry's
  `/openai/v1`. No local model serving, and no per-model parameterization: **the call shape
  is hard-coded, deliberately.** This deployment 400s on `max_tokens` (it wants
  `max_completion_tokens`) and 400s again on any `temperature` but its default, so those
  two knobs could only ever be set to a rejected call — one wasted request each, in every
  worker process, before the agent's first real turn. They are not config keys and should
  not become config keys again for a second model; add a second `ChatModel` instead.
  `make_chat_model(config)` is the single constructor and reads only `model_name` and
  `max_new_tokens`.
- `ehr_gym/tracing.py` — Laminar setup. Everything else calls `Laminar.*` unguarded;
  those are no-ops until `initialize()` runs.

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
- **Agent code runs in a per-task scratch directory, never in the repo root.** `EHREnv.reset`
  creates one (`medagentgym-rollout-*` under the system temp dir) and `EHREnv.close` deletes
  it, with `ehr_gym.env.action.function.set_workspace` pointing the actions at it. That is
  what makes it safe for one sandbox to run hundreds of tasks in a row: task code writes
  `test.bam` / `output/` / `STAR` wherever it likes and none of it reaches task N+1. Two
  consequences worth knowing:
  - `_rollout` must `env.close()` in a `finally` — the workspace is otherwise leaked, and
    with it the isolation guarantee.
  - Paths handed to agent code must be **absolute** (see `BioDSBenchTask.setup`), because
    the harness's cwd is the repo root but the agent's cwd is not.
- `terminal` installs land in `<workspace>/.packages` via `PIP_TARGET`, on `PYTHONPATH` for
  the agent's own processes only, so `pip install numpy==1.19` in one task cannot decide
  what the next task imports. Anything *we* write that runs scraped task code should go
  through `scripts/_common.run_python`, which uses a temp cwd for the same reason.

## Calling models

- **The request shape is hard-coded in `chat_api.py`, not configured and not discovered.**
  There is one deployment, gpt-5.6-luna, and it 400s on `max_tokens` by name (it wants
  `max_completion_tokens`) and 400s again on any `temperature` but its default. Those were
  the only two knobs a config could turn, and both could only be turned to a rejected call,
  so they are no longer config keys. Do not go back to sniffing model names, to learning
  from the 400s, or to declaring the shape in YAML: each cost two rejected calls in every
  joblib worker process, and a worker is created per task. A second deployment means a
  second `ChatModel`, not a re-parameterized one.
- A `BadRequestError` is raised immediately rather than retried. It means the request is
  wrong — bad parameter, or a conversation past the context window — and no amount of
  backoff fixes that; `max_retry` is reserved for 429s and outages.
- **Reasoning tokens count against `max_completion_tokens`.** A gpt-5.6-luna call with a
  600-token ceiling returns `finish_reason: "length"`, `reasoning_tokens: 600` and *empty*
  content. In the harness that surfaces as "the model cannot produce valid JSON", not as a
  budget problem, so `__call__` logs a warning on `finish_reason == "length"`. Keep
  `max_new_tokens` in the tens of thousands for reasoning models.
- gpt-5.6-luna's rate limit is tight enough that `--n_jobs 1` is the safe setting; the
  429 backoff is 10s doubling, and four exhausted retries abandon the trajectory.

## Tracing

Every rollout is a Laminar trace: root span `<task>[<mode>/<idx>]` (session =
`result_dir_tag`), and directly under it a flat alternating sequence of `openai.chat` and
`env.<action>` spans. **Trajectory consumers want that flat shape**, so do not reintroduce
per-step or per-`act` wrapper spans; the only span the harness opens by hand is the root
one in `run_single_rollout` and the `env.<action>` one in `EHREnv.step`. Two things are
easy to get wrong:

- joblib `prefer="processes"` workers do **not** inherit the parent's tracer provider, so
  `tracing.initialize()` runs inside `run_single_experiment`, not in `main()`.
- A worker can exit without running interpreter shutdown hooks, so every rollout ends in a
  `tracing.flush()`.

The root span carries the trajectory metadata (`trajectory_metadata` in `main.py`), in the
schema shared across Laminar's trajectory datasets. Two traps:

- **`gt_event_identified` is a failure flag, not a pass flag**: true when the trajectory did
  not pass, false when it did, and true for a rollout that never reached a grader. It reads
  like the opposite, and a run exported with the sense inverted is silently wrong.
- `Laminar.set_trace_metadata` may be called **at most once per trace**, which is why it is
  called from the `finally` of `run_single_rollout` and not at span start — `num_steps` and
  the verdict are not known until the rollout ends, and a crashed one still needs metadata.

- **An ambient `LMNR_PROJECT_API_KEY` in the shell will quietly capture a run.** Laminar
  keys carry the project, so a key inherited from the surrounding machine sends every
  trajectory to a project nobody is watching, and nothing in the run says so.
  `set_environment_variables()` therefore treats `credentials.toml` as exhaustive: a Laminar
  key it does not name is *removed* from the environment rather than left in place. Which
  key is in use is logged (first eight characters) at the top of every worker.

To confirm delivery, read the spans back with `LaminarClient(project_api_key=...).sql.query`
— e.g. `SELECT name, span_id, parent_span_id, path FROM spans WHERE trace_id = '...'`.
Root spans have `parent_span_id = '00000000-0000-0000-0000-000000000000'`. There is no
`metadata` column — selecting one is a ClickHouse `UNKNOWN_IDENTIFIER` 400; the trajectory
metadata is inside `attributes`, a JSON string, under `lmnr.association.properties.metadata.*`,
so read the whole column and unpack it client-side. This needs a
key with **read** scope; a write-only key 404s on `/v1/sql/query` and
`/v1/projects/current`, which looks exactly like the routes not existing. The check that
works with any key is differential: a bogus key logs `Failed to export traces to
api.lmnr.ai:8443, error code: StatusCode.UNAUTHENTICATED` and a good key logs nothing.

## Daytona

`scripts/daytona_run.py` shards a run across Daytona sandboxes; the README documents the
flags. The non-obvious parts:

- **`Dockerfile.daytona` exists because the SDK has no `.dockerignore` support.**
  `Image.from_dockerfile` parses `COPY` sources and uploads them as build context, so the
  repo `Dockerfile`'s `COPY . /home/` would ship `.venv`, `workdir/` and `credentials.toml`
  to a remote builder. `Dockerfile.daytona` copies only `pyproject.toml`, `uv.lock` and
  `.python-version`; the code arrives per sandbox via `git.clone` at a pinned ref.
  `COPY --from=` lines are skipped by that parser, so the uv stage is safe.
- The image builds the venv at `/home/.venv` with `--no-install-project`, and the clone
  lands beside it at `/home/MedAgentGym`, so the per-sandbox `uv sync` is nearly a no-op.
  Commands must run `/home/.venv/bin/python` and set `UV_PROJECT_ENVIRONMENT`, not rely on
  the image's `ENV` surviving into Daytona's exec.
- **Anything between `client.create` and returning the handle has to delete the sandbox on
  failure.** The caller's `finally` can only clean up what it was given, so a setup step
  that raises leaks a *running* sandbox with no reference to it — and it fails identically
  in every shard, so one bad clone leaked 24 at once. Both `provision` and
  `start_fhir_sandbox` wrap their setup in `except BaseException: client.delete(...); raise`.
- **A commit SHA cannot be cloned with `commit_id=`** unless it is on the default branch:
  the server resolves it there and answers `Failed to clone repository: not found: object
  not found`. `_fill_sandbox` clones the default branch and then
  `git fetch --depth 1 origin <sha> && git checkout --detach FETCH_HEAD`, which works for
  any commit GitHub will serve.
- Credentials are uploaded as a `credentials.toml`, not passed as `env_vars`, so they stay
  out of sandbox metadata. `set_environment_variables()` exports every key of that file,
  which is also how `MEDAGENTBENCH_FHIR_URL` reaches the task module.
- **Sandbox labels are the only handle a second process has on a run, and only
  `(run, shard)` is unique.** Every sandbox carries `medagentgym` (the task, or `fhir`),
  `run` (`args.run_id`, one per invocation of `daytona_run.py`) and `shard` (`shard-0`,
  `shard-1`, …, and on a FHIR sandbox it is *its worker's* name, which is what pairs the
  two). Shard numbering restarts at 0 every run, so pairing on it alone lets any finished
  `shard-0` — of any task, from any run — claim and delete another run's `shard-0` FHIR
  server while that shard is still grading against it. `daytona_collect.py` therefore keys
  its server index on `pairing()`, and a sandbox with no `run` label is left alone rather
  than guessed at.
- A shard is started with `nohup` and polled for `/tmp/exit_code` rather than run as one
  long `exec`: a multi-hour HTTP response is at the mercy of every proxy in between.
  Sandboxes are created with `auto_stop_interval=0` (the 15-minute idle timer would stop a
  silent shard) and a `ttl_minutes` backstop for a driver that dies before cleanup.
- **Starting a shard is not idempotent, and `exec` failures do not mean the command did not
  run.** Daytona answers `command execution timeout` under load whether or not it started
  what it was given, and `launch` opens with `rm -f /tmp/run.started /tmp/exit_code` — so a
  blind retry can put a second `main.py` on one shard's `workdir/` *and* delete the markers
  the first one reports through, which reads downstream as a shard that never finished.
  `launch` therefore retries only after `shard_state` says `idle`, the one answer that rules
  a live job out; `running`/`done` mean it started anyway, and `unknown` (`settled_shard_state`
  asked six times and got nothing) hands the shard to `run_detached`'s poll loop instead,
  which tolerates a silent sandbox and is bounded by `--timeout`. Marker files, not `pgrep`:
  `procps` is not in the image and a missing `pgrep` would report every running shard `idle`.
- `MEDAGENTBENCH_FHIR_URL` overrides `http://localhost:8080/fhir/` in
  `env/task/medagentbench.py`. The prompt's example URL is built from the same value —
  they must not drift, or the agent POSTs to localhost while the server is elsewhere.
- **`--with-fhir` needs `Dockerfile.fhir`; `jyxsu6/medagentbench` cannot be a sandbox on
  its own.** That image is distroless (no shell, no `sleep`, uid 65532), so Daytona's agent
  has nothing to exec into: `create()` succeeds, the sandbox reports `SandboxState.STARTED`
  with `error_reason=None`, and every `process.exec` fails with `failed to resolve container
  IP after 3 attempts`. Overriding the entrypoint does not help — there is no `sleep` binary
  either. `Dockerfile.fhir` copies its `/app` (the HAPI war), `/configs` and `/data` (a
  ~1.4 GB H2 database) onto `eclipse-temurin:17-jre-jammy` in a multi-stage build, so no
  build context is uploaded, and the driver starts the war by hand with the argv that was
  the original image's `ENTRYPOINT`. Ready in ~2 minutes; wants 8 GB of RAM, and 10 GB of
  disk is the per-sandbox ceiling on the default plan (a 30 GB request is a 400).
- **One FHIR sandbox per shard, not per run.** MedAgentBench's write tasks are graded on
  what the agent POSTed, so a server shared across shards makes grading order-dependent.
  `run_shard` starts its own and copies `creds` before writing `MEDAGENTBENCH_FHIR_URL` into
  it — the dict comes from `main()` and is shared by every thread in the pool.
- Readiness is polled from the *driver* against the public preview URL rather than with a
  `curl` inside the sandbox: that is the path the workers use, and the JRE image has no
  `curl`. `start_fhir_sandbox` deletes its own sandbox on any failure — the caller only
  gets the handle if it returns, so anything that escapes would leak a running container.

## Grading, per dataset (all execution-based; there is no LLM-as-a-judge anywhere)

- **Rewards come from stdout, feedback comes from stdout + stderr.** `validate_code`
  returns both `env_message` (combined, what the agent debugs from) and `stdout` (what
  tasks grade on). They were the same field, which made a submission that emitted a
  `UserWarning` fail even when it was behaviourally identical to the reference — and it
  contradicted `scripts/filter_biocoder.py`, which picked the keep set on stdout alone.
- **`task.validate()`'s `task_info["message"]` is the agent's only signal that a working
  program gave a wrong answer.** `env._step` appends it to `obs["env_message"]`. Without
  that the agent sees nothing but its own output and resubmits byte-identical code for the
  whole step budget — the dominant failure mode in the first smoke run.

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
- **MedAgentBench** (240 train / 60 test) — needs a live HAPI FHIR server, by default on
  `http://localhost:8080/fhir/` (`bash data/medagentbench/start_eval_docker.sh`), or
  wherever `MEDAGENTBENCH_FHIR_URL` points. 54/60
  test rows have an empty `sol` and are graded by programmatic `task1..task10` graders that
  query the server. Tasks 3/5/8/9/10 mutate server state, so each shard gets its own server
  (`--with-fhir`) rather than restarting a shared one. `task3` asserted a hardcoded
  `Patient/S2380121` — `task3_1`'s MRN and nobody else's — so its other 28 datapoints were
  unwinnable; it now compares against `case_data['eval_MRN']` like every other grader here.
  Worth re-checking that pattern before trusting a 0 on this dataset: the graders are string
  containment over `str(results)`, so a bad constant looks exactly like a bad agent.
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
- `data/smoke_indices.json` is the ten-per-dataset sample for end-to-end checks. It has no
  `medagentbench` key on purpose: that dataset needs a Docker FHIR server.
- `agent.act` returns `LLM_FAILURE` (`"error: llm"`) or `PARSE_FAILURE`
  (`"error: invalid response"`). `env._step` routes anything containing `"error"` back to
  the model as feedback; only `LLM_FAILURE` abandons a rollout, because a malformed
  response is something the model can fix next turn and an unreachable API is not. Match
  these with `startswith`, not `==` — upstream's `while action == "error"` was dead code.
  Because that retry loop is now live, `act()` has to be safe to call twice with the *same*
  observation: it picks the turn to send from `obs["type"]`, not from whether the history
  is empty, and skips the append when that turn is already the last one. Both branches are
  only reachable after an `LLM_FAILURE`, which leaves a prompt in the history with no
  assistant reply after it.
