#!/bin/bash
set -e

TASK_NAME="${TASK_NAME:-biocoder}"
N_JOBS="${N_JOBS:-5}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"

python /home/main.py \
    --config_path "/home/configs/gpt_4_1_mini/exp-gpt_4_1_mini-${TASK_NAME}.yaml" \
    --n_jobs "${N_JOBS}" \
    --num_rollouts "${NUM_ROLLOUTS}"
