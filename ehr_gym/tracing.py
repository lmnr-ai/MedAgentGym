"""Laminar tracing for the harness.

Trajectories are the product of this fork, so every rollout is also a Laminar
trace:

    <task>[<mode>/<idx>]          root span, one per trajectory
      openai.chat                 from the SDK's OpenAI auto-instrumentation
      env.validate_code           the action the agent chose, and its reward
      openai.chat                 the next turn, and so on

The two span kinds are deliberately siblings under the trajectory rather than
being nested under per-step wrappers: a trajectory then reads as one flat
alternating sequence of "what the model said" and "what the environment did",
which is the shape trajectory consumers expect.

Tracing is optional: with no `LMNR_PROJECT_API_KEY` in the environment the SDK
is never initialized and every `Laminar.*` call in the harness degrades to a
no-op, so the run behaves exactly as it did before.
"""

import logging
import os

from lmnr import Laminar

logger = logging.getLogger(__name__)


def initialize() -> bool:
    """Start Laminar in this process. Safe to call repeatedly.

    Rollouts run in joblib worker *processes*, which do not inherit the parent's
    tracer provider, so this has to run inside the worker rather than once in
    `main()`.
    """
    if Laminar.is_initialized():
        return True
    api_key = os.getenv("LMNR_PROJECT_API_KEY")
    if not api_key:
        logger.info("LMNR_PROJECT_API_KEY is not set; running without tracing.")
        return False
    Laminar.initialize(project_api_key=api_key, base_url=os.getenv("LMNR_BASE_URL") or None)
    return True


def flush() -> None:
    """Push buffered spans to the backend.

    Called at the end of every rollout: a joblib worker can be torn down without
    running interpreter shutdown hooks, and a trajectory that never reaches the
    backend is the one thing this fork cannot afford to lose.
    """
    if Laminar.is_initialized():
        Laminar.flush()
