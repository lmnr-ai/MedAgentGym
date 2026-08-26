import logging
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ehr_gym.llm.chat_api import make_user_message
from ehr_gym.utils.env_utils import parse_and_truncate_error

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./cache")
CODE_TIMEOUT_SECONDS = 120
# Where `pip install` puts things when the agent runs it, relative to the workspace.
SITE_DIR = ".packages"

# Scratch directory of the rollout currently running. Everything the agent does to
# the filesystem -- files its code writes, packages it installs from `terminal` --
# is confined here and thrown away when the rollout ends, so one sandbox can run
# hundreds of tasks back to back without task N leaking into task N+1. `EHREnv`
# owns the lifecycle; it is a module global rather than an action parameter because
# `BiocoderTask.setup` also runs code (the reference program, whose stdout is the
# ground truth) and has to land in the same place. Safe because rollouts are never
# concurrent inside one process: joblib parallelism here is `prefer="processes"`.
_WORKSPACE: Path | None = None


def set_workspace(path: Path | None) -> None:
    global _WORKSPACE
    _WORKSPACE = Path(path) if path is not None else None


def current_workspace() -> Path:
    """The directory agent code runs in. Falls back to the cache dir when no env owns it."""
    work = _WORKSPACE if _WORKSPACE is not None else CACHE_DIR
    work.mkdir(parents=True, exist_ok=True)
    return work


def _agent_env(work: Path) -> dict[str, str]:
    """Environment for a subprocess of the agent's making."""
    env = os.environ.copy()
    site = work / SITE_DIR
    # `pip install numpy==1.19` from one task must not decide what the next task
    # imports, so installs go to a per-rollout directory that is on the path of
    # the agent's own processes only.
    env["PIP_TARGET"] = str(site)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return env


def _run_code_file(code_file: Path, timeout: int) -> tuple[int, str, str, float]:
    start_time = time.time()
    work = code_file.parent
    try:
        # `sys.executable`, not `"python"`: the BioCoder ground truth is produced by
        # running the reference in *this* interpreter at setup time, so grading is
        # only meaningful if the agent's code sees the same site-packages. Resolving
        # `python` through PATH silently picks up whatever venv the shell activated.
        process = subprocess.run(
            [sys.executable, code_file.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work,
            env=_agent_env(work),
        )
        return process.returncode, process.stdout, process.stderr, time.time() - start_time
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return (
            -1,
            "",
            f"Process timed out after {execution_time:.2f}s (timeout limit: {timeout}s)",
            execution_time,
        )
    except Exception as e:
        return -1, "", str(e), time.time() - start_time


def validate_code(code: str) -> dict[str, Any]:
    """
    Validate the generated code.

    Examples:
        validate_code("import numpy as np\na=[1,2,3]\nanswer=np.mean(a)")
    """
    start_time = time.time()
    try:
        compile(code, "<string>", "exec")
    except Exception as e:
        error_msg = (
            f"Compilation error occurred for the code: {str(e)}\n"
            f"Full traceback:\n{traceback.format_exc()}"
        )
        logger.error("Compilation error during code validation.")
        return {
            "type": "code_execution",
            "status": "FAILED",
            "env_message": parse_and_truncate_error(error_msg),
            "execution_time": f"{time.time() - start_time:.2f}s",
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    code_file = current_workspace() / f"validation_code_{timestamp}_{uuid.uuid4()}.py"
    if "print" not in code and "answer" in code:
        code += "\nprint(answer)"
    code_file.write_text(code)

    try:
        returncode, stdout, stderr, execution_time = _run_code_file(
            code_file, timeout=CODE_TIMEOUT_SECONDS
        )
        combined_output = stdout + ("\n" + stderr if stderr else "")
        success = returncode == 0
        output = combined_output if success else parse_and_truncate_error(combined_output)
    finally:
        code_file.unlink(missing_ok=True)

    logger.info(f"Code validation completed in {execution_time:.2f}s.")
    return {
        "type": "code_execution",
        "status": "SUCCESS" if success else "FAILED",
        "env_message": output,
        # What gets *graded*, as opposed to what the agent is shown. stderr stays
        # in `env_message` because warnings and tracebacks are what the agent
        # debugs from, but it must not decide the reward: a submission that is
        # behaviourally identical to the reference still emits its own warnings,
        # and `scripts/filter_biocoder.py` already picked the keep set on stdout
        # alone. A failed run grades on the error text so an empty stdout can
        # never be mistaken for a silent correct answer.
        "stdout": stdout if success else output,
        "execution_time": f"{execution_time:.2f}s",
    }


DEBUG_PROMPT = """You are a Python debugging expert. Your task is to help the user debug their code.
The user is solving the problem:
{problem}

The user has provided the following code:
{code}

The user has encountered the following error:
{error_msg}

Please provide an explanation of the error and suggest a solution to fix it.
"""


def debug(code: str, error_msg: str, debugger, history: list[dict]) -> dict[str, Any]:
    """
    Ask a second LLM to explain the failure and propose a fix.

    `debugger` and `history` are injected by the env, not by the agent.
    """
    problem = next(
        (obs["info"]["task_goal"] for obs in history if obs["type"] == "initial_observation"),
        "",
    )
    message = make_user_message(
        DEBUG_PROMPT.format(problem=problem, code=code, error_msg=error_msg)
    )
    response, cost = debugger([message])
    return {
        "type": "debugging_info",
        "env_message": (
            response.content
            + "\nPlease then use validate_code action to validate the debugged code."
        ),
        "cost": [cost],
    }


def terminal(cmd: str) -> dict[str, Any]:
    """
    Run a command in the terminal.

    Examples:
        terminal("pip install pysam")
    """
    start_time = time.time()
    work = current_workspace()
    # `pip install ...` is the whole point of this action, so make `pip` and
    # `python` mean the interpreter that will later run the agent's code.
    env = _agent_env(work)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=work, env=env
    )
    execution_time = time.time() - start_time
    if result.returncode == 0:
        return {
            "type": "cmd_result",
            "status": "SUCCESS",
            "env_message": result.stdout,
            "execution_time": f"{execution_time:.2f}s",
        }
    return {
        "type": "cmd_result",
        "status": "FAILED",
        "env_message": f"Command '{cmd}' failed with error: {result.stderr}",
        "execution_time": f"{execution_time:.2f}s",
    }
