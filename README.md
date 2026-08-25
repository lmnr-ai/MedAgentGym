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

| Dataset | `--task` | Data shipped in-repo | External prerequisite |
| --- | --- | --- | --- |
| BioCoder | `biocoder` | yes | bioinformatics Python packages for reference execution |
| BioDSBench | `biodsbench` | task definitions only | cBioPortal study CSVs under `data/biodsbench/data/` |
| MedAgentBench | `medagentbench` | yes | HAPI FHIR server on `http://localhost:8080/fhir/` |
| MedCalcBench | `medcalcbench` | yes (test split only) | none |

Task files (`train_tasks.jsonl` / `test_tasks.jsonl`) hold the task id, description,
question, and ground-truth answer for each datapoint. `data/metadata.json` holds the
per-split datapoint counts and is what `--end_idx -1` resolves against.

Per-dataset notes:

- **MedAgentBench** grades against a live FHIR server: `bash data/medagentbench/start_eval_docker.sh`
  brings up `jyxsu6/medagentbench` on port 8080. Tasks 3/5/8/9/10 write to it, so restart
  the server between full runs to get a clean state.
- **BioCoder** derives its ground truth at task-setup time by *executing the reference
  program* and capturing stdout. If a reference program cannot import a package, that
  task's "ground truth" silently becomes an error string and the task is unwinnable —
  install the `tasks` extra before running.
- **BioDSBench** ships task definitions only. The referenced study CSVs
  (`/workdir/data_*.csv`) are not in this repo and must be placed under
  `data/biodsbench/data/`.
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

### Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The `tasks` extra holds
the scientific packages that the *agent-generated* code needs; it is installed into the
same environment because `validate_code` executes that code with the venv interpreter.

```bash
uv sync --extra tasks
cp credentials.example.toml credentials.toml   # then fill in your Azure AI Foundry keys
```

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

### Running in Docker

The image is CPU-only — this harness only calls hosted LLM APIs, so there is no CUDA
runtime and no `torch`.

```bash
bash build_docker.sh
TASK_NAME=biocoder N_JOBS=5 bash run_docker.sh
```

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
