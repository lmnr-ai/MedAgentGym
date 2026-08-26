import argparse
import json
import logging
import os
import time

import toml
from joblib import Parallel, delayed
from lmnr import Laminar

from ehr_gym import tracing
from ehr_gym.agent.base import LLM_FAILURE, EHRAgent
from ehr_gym.env.base import EHREnv
from ehr_gym.utils.general import load_config, save_conversation_history

logging.basicConfig(level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s")
logger = logging.getLogger(__name__)

# Identifies this harness in the trace metadata. `custom/` because it is a fork
# of a benchmark runner rather than one of the named agent frameworks.
HARNESS = "custom/MedAgentGym"

TASK_CLASSES = {
    "biocoder": ("ehr_gym.env.task.biocoder", "BiocoderTask"),
    "biodsbench": ("ehr_gym.env.task.biodsbench", "BioDSBenchTask"),
    "medagentbench": ("ehr_gym.env.task.medagentbench", "MedAgentBenchTask"),
    "medcalcbench": ("ehr_gym.env.task.medcalcbench", "MedCalBenchTask"),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MedAgentGym experiments")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--task", type=str, choices=sorted(TASK_CLASSES))
    parser.add_argument("--credentials_path", type=str)
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--result_dir_tag", type=str)
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--mode", type=str, default="test", help="train/test")
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="Trajectories to sample per task. >1 writes history_<idx>_<rollout>.json",
    )
    parser.add_argument(
        "--rollout_indices_path",
        type=str,
        help="JSON file of {task: {mode: [idx, ...]}}; overrides --start_idx/--end_idx",
    )
    return parser.parse_args()


def convert_config_to_args(config, args):
    for key in (
        "task",
        "credentials_path",
        "work_dir",
        "result_dir_tag",
        "start_idx",
        "end_idx",
        "num_steps",
    ):
        if getattr(args, key) is None:
            setattr(args, key, config[key])
    return args


def set_environment_variables(credentials_path):
    for key, value in toml.load(credentials_path).items():
        os.environ[key] = value


def get_task_class(task):
    if task not in TASK_CLASSES:
        raise ValueError(f"Invalid task: {task}")
    module_name, class_name = TASK_CLASSES[task]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def history_path(save_dir, idx, rollout_idx, num_rollouts):
    name = f"history_{idx}.json" if num_rollouts == 1 else f"history_{idx}_{rollout_idx}.json"
    return os.path.join(save_dir, name)


def trajectory_metadata(args, config, idx, rollout_idx, result):
    """The trace metadata every trajectory carries, in the shared schema.

    `gt_event_identified` is a *failure* flag: true when the trajectory did not
    pass, false when it did. A rollout that never reached a grader — the LLM
    endpoint gave up, or the harness crashed — counts as a failure too, so this
    is exactly `not success` and is never absent. `outcome` is what separates
    the two kinds of failure.
    """
    return {
        "source": args.task,
        "domain": "healthcare",
        "gt_event_identified": not result["success"],
        "generated": True,
        "harness": HARNESS,
        "model": config["Agent"]["llm"]["model_name"],
        # Steps actually taken, not the budget; the budget is `step_budget`.
        "num_steps": result["steps"],
        # General metadata, flattened into the same level by convention.
        "split": args.mode,
        "task_index": idx,
        "rollout": rollout_idx,
        "score": result["score"],
        "outcome": result["reason"],
        "step_budget": args.num_steps,
        "run_tag": args.result_dir_tag,
        "debugger_model": config["Debugger"]["model_name"],
    }


def run_single_rollout(args, config, idx, rollout_idx, output_path):
    """Run one trajectory for task `idx` and persist it. Returns 1 on success."""
    # What the metadata says if `_rollout` raises. `set_trace_metadata` may be
    # called at most once per trace, so it happens in the `finally` rather than
    # at span start: `num_steps` and the verdict are not known until the end,
    # and a trajectory that crashed should still be findable.
    result = {"success": 0, "score": 0, "steps": 0, "reason": "crashed"}
    with Laminar.start_as_current_span(
        f"{args.task}[{args.mode}/{idx}]",
        input={"task": args.task, "mode": args.mode, "index": idx, "rollout": rollout_idx},
        session_id=args.result_dir_tag,
        tags=[args.task, args.mode],
    ):
        try:
            result = _rollout(args, config, idx, output_path)
            Laminar.set_span_output(result)
            return result["success"]
        finally:
            Laminar.set_trace_metadata(
                trajectory_metadata(args, config, idx, rollout_idx, result)
            )


def _rollout(args, config, idx, output_path):
    agent_config = config["Agent"]
    task_cls = get_task_class(args.task)
    env = EHREnv(
        task_entrypoint=task_cls,
        task_kwargs={
            "data_path": config["Data"]["data_path"],
            "debugger_config": config["Debugger"],
            "mode": args.mode,
        },
    )
    agent = EHRAgent(agent_config, permitted_actions=task_cls.permitted_actions)
    obs, _ = env.reset(idx)

    attempts = 0
    done = False
    reward = 0
    steps = 0
    for step in range(args.num_steps):
        steps = step + 1
        action, params = agent.act(obs)
        while action.startswith(LLM_FAILURE):
            logger.error(
                f"Task {args.task}-{idx} failure: agent action failed "
                f"for {agent_config['n_retry']} times."
            )
            attempts += 1
            if attempts >= config["Env"]["n_retry"]:
                agent.conversation_history.append({"result": "failure"})
                save_conversation_history(agent.conversation_history, output_path)
                # The API was unreachable, so nothing the agent did was ever put
                # in front of a grader. `reason` carries the LLM failure so the
                # trajectory is still distinguishable from one that was graded
                # and lost.
                return {"success": 0, "score": 0, "steps": steps, "reason": action}
            time.sleep(1)
            action, params = agent.act(obs)
        obs, reward, done, _, _ = env.step(action, **params)
        if done:
            break

    if done:
        agent.conversation_history.append({"result": "success", "score": reward})
    else:
        agent.conversation_history.append({"result": "failure"})
    save_conversation_history(agent.conversation_history, output_path)
    return {
        "success": 1 if done else 0,
        "score": reward,
        "steps": steps,
        "reason": "solved" if done else "step budget exhausted",
    }


def run_single_experiment(args, config, idx):
    """Run every requested rollout for task `idx`, skipping ones already on disk."""
    # This is the joblib entry point, so it is also the first thing that runs in
    # a worker process -- and therefore where tracing has to be set up.
    tracing.initialize()
    save_dir = os.path.join(args.work_dir, args.task, args.result_dir_tag, args.mode)
    os.makedirs(save_dir, exist_ok=True)
    successes = 0
    for rollout_idx in range(args.num_rollouts):
        output_path = history_path(save_dir, idx, rollout_idx, args.num_rollouts)
        if os.path.exists(output_path):
            logger.info(f"Trajectory {output_path} already exists. Skipping...")
            continue
        logger.info(f"Running experiment for index {idx} (rollout {rollout_idx})...")
        try:
            successes += run_single_rollout(args, config, idx, rollout_idx, output_path)
        finally:
            tracing.flush()
    return successes


def run_experiments(args, config, indices):
    if not indices:
        logger.warning("No experiments to run")
        return
    results = Parallel(n_jobs=args.n_jobs, prefer="processes")(
        delayed(run_single_experiment)(args, config, idx) for idx in indices
    )
    success_rate = sum(results) / (len(results) * args.num_rollouts)
    print("-" * 50)
    print(f"Success Rate: {success_rate}")

    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "running_records.jsonl"), "a+") as f:
        f.write(f"Experiment {args.task}: {success_rate}\n")


def resolve_indices(args, config):
    if args.rollout_indices_path:
        with open(args.rollout_indices_path, "r") as f:
            return json.load(f)[args.task][args.mode]
    end_idx = args.end_idx
    if end_idx == -1:
        with open(config["Data"]["metadata_path"], "r") as f:
            end_idx = json.load(f)[args.task][args.mode]
    return list(range(args.start_idx, end_idx))


def main():
    args = parse_arguments()
    config = load_config(args.config_path) if args.config_path else {}
    if config:
        args = convert_config_to_args(config, args)
    set_environment_variables(args.credentials_path)
    run_experiments(args, config, resolve_indices(args, config))


if __name__ == "__main__":
    main()
