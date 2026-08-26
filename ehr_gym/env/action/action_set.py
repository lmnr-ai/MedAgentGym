from typing import Callable

from ehr_gym.env.action.function import debug, terminal, validate_code

# The action space of the gym. Task classes expose the subset they permit via
# `permitted_actions`; the env looks the callable up here by name.
ACTION_SET: dict[str, Callable] = {
    "validate_code": validate_code,
    "debug": debug,
    "terminal": terminal,
}
