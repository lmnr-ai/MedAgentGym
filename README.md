<p align="center">
  <img src="./assets/figure2.png" width="100%" alt="teaser">
</p>

----
<p align="center">
  <a href="https://openreview.net/forum?id=jHDZEUgS4r" target="_blank"><img src="https://img.shields.io/badge/arXiv-2506.02911-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="ICLR"></a>
  <a href="https://wshi83.github.io/MedAgentGym-Page"><img src="https://img.shields.io/badge/Doc-Documentation-4285F4?style=for-the-badge&logo=googledocs&logoColor=white" alt="Documentation"></a>
  <a href="https://huggingface.co/MedAgentGym"><img src="https://img.shields.io/badge/HuggingFace-Model&Data-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Model&Data"></a>
  <a href="mailto:medagentgym@gmail.com"><img src="https://img.shields.io/badge/Email-Question-30B980?style=for-the-badge&logo=minutemailer&logoColor=white" alt="Email Question"></a>
</p>


## MedAgentGYM
This is the official repository for the paper: "MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale". In the paper, we introduce MedAgentGYM, the first publicly available training environment designed to enhance coding-based medical reasoning capabilities in large language model (LLM) agents. 

<p align="center">
  <img src="./assets/figure1.png" width="100%" alt="teaser">
</p>

### Datasets

> **Fork note.** This fork intentionally ships **only the four datasets that require no
> credentialed data-use agreement**. Everything tied to restricted EHR corpora
> (MIMIC-III, eICU, TREQS, EHRShot, EHRCon, EHR-SeqSQL, MIMIC-Extract) and to nPowerAI
> has been removed from the task modules, configs, and `./data/`. Do not re-add them.

| Dataset | `--task` | Train | Test | External prerequisite |
| --- | --- | --- | --- | --- |
| BioCoder | `biocoder` | 496 | 149 | the `tasks` extra (reference programs are executed) |
| BioDSBench | `biodsbench` | 42 | 43 | `scripts/fetch_biodsbench_data.py` (~160 MB of study data) |
| MedAgentBench | `medagentbench` | 240 | 60 | HAPI FHIR server on `http://localhost:8080/fhir/` |
| MedCalcBench | `medcalcbench` | — | 1047 | none |

Task files (`train_tasks.jsonl` / `test_tasks.jsonl`) hold the task id, description,
question, and ground-truth answer for each datapoint. `data/metadata.json` holds the
per-split datapoint counts and is what `--end_idx -1` resolves against.

Every dataset here is graded by executing code; there is no LLM-as-a-judge anywhere.

Per-dataset notes:

- **MedAgentBench** grades against a live FHIR server: `bash data/medagentbench/start_eval_docker.sh`
  brings up `jyxsu6/medagentbench` on port 8080. Tasks 3/5/8/9/10 write to it, so restart
  the server between full runs to get a clean state.
- **BioCoder** derives its ground truth at task-setup time by *executing the reference
  program* and capturing stdout, so a task is only gradable if its reference runs. The
  counts above are what `scripts/filter_biocoder.py` verified against the `tasks` extra;
  re-run it whenever that extra changes.
- **BioDSBench** ships task definitions only. Run `scripts/fetch_biodsbench_data.py` once
  to download the eleven cBioPortal studies the tasks read; the output is gitignored.
- **MedCalcBench** has no train split in this repo; only `test_tasks.jsonl` (1047 rows).

### What this fork changes

- Only the four unrestricted datasets remain; all restricted task modules, configs and
  data directories are gone.
- Packaging moved from `requirements.txt` to `uv` (`pyproject.toml` + `uv.lock`).
- The Docker image is plain `python:3.11-slim` instead of `nvidia/cuda:*-devel`, and
  `torch` is gone — nothing here runs a local model.
- `rollout.py` was folded into `main.py` as `--num_rollouts` / `--rollout_indices_path`.
- Ray, vLLM, `request_info`, and the unused `langchain` / `transformers` / WolframAlpha
  code paths were removed. Parallelism is joblib-only.
- Upstream spliced the agent's code into a task template with a plain `str.replace`,
  which dropped the marker's indentation and quietly corrupted every submission made
  inside a class or function body. Substitution now lives in
  `ehr_gym/env/task/substitution.py` and re-indents the block.
- Ungradable datapoints were removed (`scripts/filter_*.py`): BioCoder references that
  crash, print nothing, or print something different on each run, and BioDSBench tasks
  whose own reference fails their assertions.
- MedCalcBench's grader compared everything as a float, so the 60 date and
  gestational-age answers could never pass. It now parses each answer shape.
- The grader's verdict never reached the agent: `validate()` computed "the answer is
  incorrect" and `env.step` threw it away, so a model that ran a working program with a
  wrong answer saw only its own stdout and resubmitted the same code until the step
  budget ran out. The verdict is now appended to the observation.
- Grading used stdout **and** stderr, so a submission that merely emitted a
  `UserWarning` scored 0. Rewards now come from stdout alone, which is what
  `scripts/filter_biocoder.py` picked the keep set with; the agent is still shown stderr.
- Every rollout is traced to [Laminar](https://www.lmnr.ai/) — see *Tracing* below.
- `scripts/daytona_run.py` runs a sharded experiment on Daytona sandboxes, which gives
  agent-generated code a throwaway machine and removes the Docker prerequisite for
  MedAgentBench — see *Running on Daytona* below.

### Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The `tasks` extra holds
the scientific packages that the *agent-generated* code needs; it is installed into the
same environment because `validate_code` executes that code with the venv interpreter.

```bash
uv sync --extra tasks
cp credentials.example.toml credentials.toml   # then fill in your Azure AI Foundry keys
uv run python scripts/fetch_biodsbench_data.py # only needed for --task biodsbench
```

### Maintenance scripts

| Script | Purpose |
| --- | --- |
| `scripts/fetch_biodsbench_data.py` | download + convert the cBioPortal studies BioDSBench reads |
| `scripts/filter_biocoder.py` | drop BioCoder rows whose reference program is not gradable |
| `scripts/filter_biodsbench.py` | drop BioDSBench rows whose reference fails its own assertions |

Both filters rewrite the task files in place and update `data/metadata.json`; pass
`--dry-run` to see the drop report without touching anything.

### Run an experiment

```bash
uv run python main.py --config_path configs/gpt_4_1_mini/exp-gpt_4_1_mini-biocoder.yaml --n_jobs 5
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--n_jobs N` | joblib process-level parallelism (default 1) |
| `--num_rollouts K` | sample K trajectories per task; writes `history_<idx>_<rollout>.json` |
| `--mode train\|test` | which split to run (default `test`) |
| `--start_idx` / `--end_idx` | index range; `--end_idx -1` means "all", read from `data/metadata.json` |
| `--rollout_indices_path` | JSON of `{task: {mode: [idx, ...]}}`, e.g. `data/rollout_indices.json` |

Trajectories land in `<work_dir>/<task>/<result_dir_tag>/<mode>/history_*.json`. Existing
files are skipped, so a run can be resumed by re-invoking the same command.

`data/smoke_indices.json` holds ten indices per dataset, for a cheap end-to-end check:

```bash
uv run python main.py --config_path configs/gpt_5_6_luna/exp-gpt_5_6_luna-medcalcbench.yaml \
  --rollout_indices_path data/smoke_indices.json --n_jobs 1
```

### Models

`model_type` in a config selects the client:

| `model_type` | Client | Needs |
| --- | --- | --- |
| `Foundry` | Azure AI Foundry's OpenAI-compatible `/openai/v1` | `AZURE_OPENAI_ENDPOINT`, `AZURE_API_KEY` |
| `Azure` | classic Azure OpenAI, routed by deployment name | `AZURE_OPENAI_API_KEY`, `API_VERSION` |
| `OpenAI` | OpenAI | `OPENAI_API_KEY` |

Deployments disagree about which sampling parameters they accept — GPT-5 models reject
`temperature` and renamed `max_tokens` to `max_completion_tokens`. Rather than sniffing
model names, `ehr_gym/llm/chat_api.py` sends everything, reads the 400 the API returns,
and retries without the offending parameter; the lesson is cached per model, so each
worker pays for it once.

Give reasoning models a large `max_new_tokens`. Reasoning tokens are billed against the
same ceiling as the answer, so a budget that looks generous can be spent entirely on
thinking and return truncated JSON — which reaches the parser as a formatting failure.
The `gpt_5_6_luna` configs use 32768.

### Tracing

Set `LMNR_PROJECT_API_KEY` (in `credentials.toml`) and every rollout becomes a Laminar
trace; leave it unset and the SDK is never initialized, so nothing else changes.

```
<task>[<mode>/<idx>]          root span, one per trajectory, session = result_dir_tag
  openai.chat                 from the SDK's OpenAI auto-instrumentation
  env.validate_code           the action's arguments, its reward, and the feedback
  openai.chat                 the next turn, and so on
```

The two span kinds are deliberately siblings rather than nested under per-step wrappers,
so a trajectory reads as one flat alternating sequence of "what the model said" and "what
the environment did".

Rollouts run in joblib worker *processes*, which do not inherit the parent's tracer
provider, so `ehr_gym/tracing.py` initializes inside the worker and flushes after every
rollout — a worker can be torn down without running interpreter shutdown hooks.

### Running in Docker

The image is CPU-only — this harness only calls hosted LLM APIs, so there is no CUDA
runtime and no `torch`.

```bash
bash build_docker.sh
TASK_NAME=biocoder N_JOBS=5 bash run_docker.sh
```

### Running on Daytona

`scripts/daytona_run.py` shards a run across [Daytona](https://www.daytona.io/) sandboxes.
Worth doing for three reasons: agent-generated code gets a throwaway machine instead of
your checkout, MedAgentBench's FHIR server becomes a sandbox instead of a `docker run`,
and the work is spread over `--sandboxes × --n-jobs` concurrent rollouts.

```bash
uv sync --extra daytona
export DAYTONA_API_KEY=...

uv run python scripts/daytona_run.py --task biocoder --sandboxes 4 --n-jobs 2
```

What it does per sandbox: build (or reuse) the image from `Dockerfile.daytona`, `git clone`
the repo at `--ref`, `uv sync --locked --extra tasks`, upload your `credentials.toml` and a
`shard.json` of that sandbox's task indices, run `main.py --rollout_indices_path shard.json`,
then download the `workdir/` tree and delete the sandbox.

| Flag | Meaning |
| --- | --- |
| `--sandboxes N` | how many sandboxes to shard the indices across (round-robin) |
| `--max-concurrent N` | shards in flight at once (default: all of them) |
| `--n-jobs N` | joblib workers *inside* each sandbox |
| `--ref` | branch, tag or commit to clone (default `main`) — this is what pins the run |
| `--indices-path` | run a fixed index list, e.g. `data/smoke_indices.json` |
| `--snapshot NAME` | reuse a prebuilt Daytona snapshot instead of building the Dockerfile |
| `--with-fhir` | give each shard its own MedAgentBench FHIR server sandbox |
| `--keep-sandboxes` | leave sandboxes running so a failure can be inspected |

Things worth knowing before the first run:

- **`--ref` is the version that runs**, not your working tree. Push your branch first, or
  the sandboxes will run `main`.
- **`Dockerfile.daytona` is not `Dockerfile`.** The Daytona SDK reads `COPY` sources
  directly and has no `.dockerignore` support, so the repo `Dockerfile`'s `COPY . /home/`
  would upload your `.venv`, `workdir/` and `credentials.toml` into a remote image build.
  The Daytona image carries only the dependency environment; the code arrives via `git`.
  It only needs rebuilding when `uv.lock` changes — pass `--snapshot` to skip the build
  entirely once you have one.
- **Your credentials are copied into every sandbox**, because that file is how the harness
  loads them. Use keys scoped to this work. Only the keys listed in `FORWARDED_KEYS` are
  sent, and they go in as a file rather than as sandbox environment variables.
- **`--with-fhir` gives every shard its own FHIR server**, each starting from the image's
  pristine database. A third of MedAgentBench's tasks POST to the server and are graded on
  what they wrote, so a shared server would let one shard's writes show up in another's
  reads. `--sandboxes` is therefore also the isolation knob: `--sandboxes 60` on the test
  split is one clean server per task. Each one costs ~2 minutes of startup and 8 GB of RAM,
  and is torn down with its worker, so pair a high `--sandboxes` with `--max-concurrent`:
  isolation is the shard count, cost is how many of them exist at the same time.
- **The FHIR sandboxes are public** for the life of the run, since the workers reach them
  over their preview URLs and hold no Daytona token. They serve only MedAgentBench's
  synthetic patients, but they are world-reachable while the run lasts.
- **The FHIR sandbox is built from `Dockerfile.fhir`, not from `jyxsu6/medagentbench`
  directly.** That image is distroless — no shell, no coreutils — so Daytona's agent has
  nothing to run inside it and every exec fails with `failed to resolve container IP` even
  though the sandbox reports `STARTED`. `Dockerfile.fhir` lifts its `/app`, `/configs` and
  `/data` onto a JRE base; nothing is uploaded from your checkout to build it.
- **BioDSBench study data is fetched inside each sandbox** (~160 MB per sandbox), because
  it is gitignored and so not in the clone.

## Results

### Sampled Data Helps Agent Training

Figure below highlights substantial performance gains from SFT across four OSS backbone LLMs of varying sizes.
<p align="center">
  <img src="./assets/figure4.png" width="100%" alt="teaser">
</p>

### Warmed-up DPO Works Best for Coding Agent Training
The table below compares several post-training methods, revealing that simple SFT over successful trajectories significantly boosts performance on structured coding tasks, demonstrating its effectiveness in capturing structured coding patterns. Besides, DPO is particularly beneficial for optimizing open-ended task performance. Although DPO alone slightly underperforms compared to SFT, combining an initial SFT warm-up with subsequent DPO further improves overall results by leveraging their complementary strengths.

<p align="center">
  <img src="./assets/figure5.png" width="100%" alt="teaser">
</p>

### MedAgentGym Enables Both Inference- and Training-Time Scaling

<p align="center">
  <img src="./assets/figure6.png" width="100%" alt="teaser">
</p>


**Inference-Time Scaling:** The left figure illustrates performance scaling with increased trajectory sampling. Pass@K significantly improves from 17.0% at K = 1 to 45.0% at 16, while Best@K shows steady advancement from 17.0% to 41.7%. The relatively small gap between metrics indicates that our trained verifier effectively identifies successful trajectories, unleashing its potential as a reward model for integration into advanced online RL frameworks such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO).

**Training-Time Scaling:** The right figure examines agent performance as a function of increased training data volumes (25%, 50%, 75%, and 100%) in SFT. We observe consistent performance improvements with greater training data availability, suggesting additional computational resources dedicated to sampling further trajectories are likely to yield continued performance gains.

## 📚 Citation

```bibtex
@inproceedings{
xu2026medagentgym,
title={MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science},
author={Ran Xu and Yuchen Zhuang and Yishan Zhong and Yue Yu and Zifeng Wang and Xiangru Tang and Hang Wu and May Dongmei Wang and Peifeng Ruan and Donghan Yang and Tao Wang and Guanghua Xiao and Xin Liu and Carl Yang and Yang Xie and Wenqi Shi},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=jHDZEUgS4r}
}
```
