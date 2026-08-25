#!/bin/bash
# Mount the source tree so edits do not require a rebuild; cache/ and workdir/
# are bind-mounted so generated code and trajectories survive the container.
MEDAGENTGYM_PATH="${MEDAGENTGYM_PATH:-$(pwd)}"

mkdir -p "${MEDAGENTGYM_PATH}/cache" "${MEDAGENTGYM_PATH}/workdir"

docker run \
    --network host \
    -v "${MEDAGENTGYM_PATH}/main.py:/home/main.py" \
    -v "${MEDAGENTGYM_PATH}/ehr_gym:/home/ehr_gym" \
    -v "${MEDAGENTGYM_PATH}/configs:/home/configs" \
    -v "${MEDAGENTGYM_PATH}/data:/home/data" \
    -v "${MEDAGENTGYM_PATH}/cache:/home/cache" \
    -v "${MEDAGENTGYM_PATH}/workdir:/home/workdir" \
    -v "${MEDAGENTGYM_PATH}/entrypoint.sh:/home/entrypoint.sh" \
    -v "${MEDAGENTGYM_PATH}/credentials.toml:/home/credentials.toml" \
    -e TASK_NAME="${TASK_NAME:-biocoder}" \
    -e N_JOBS="${N_JOBS:-5}" \
    -e NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}" \
    -it ehr_gym:latest
