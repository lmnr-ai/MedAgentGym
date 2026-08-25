import time
from typing import Any

from lmnr import Laminar

from ehr_gym.agent.parser import parse_llm_response
from ehr_gym.agent.prompt import DynamicPrompt
from ehr_gym.llm.chat_api import (
    make_assistant_message,
    make_chat_model,
    make_system_message,
    make_user_message,
)


# Both failure modes have to be visible to `env.step`, which routes anything
# containing "error" straight back to the model as feedback. Only the first is
# worth abandoning a trajectory over: an unparseable response is something the
# model can fix on the next turn, an unreachable API is not.
LLM_FAILURE = "error: llm"
PARSE_FAILURE = "error: invalid response"


class EHRAgent:
    """Agent class to interact with EHREnvironment using LLM"""

    def __init__(self, agent_config, permitted_actions):
        self.agent_config = agent_config
        self.llm_config = agent_config["llm"]

        self.llm = make_chat_model(self.llm_config)
        self.conversation_history = []
        self.prompt = DynamicPrompt()
        self.parser = parse_llm_response
        self.cost = []
        self.permitted_actions = permitted_actions

    def act(self, obs: Any) -> tuple[str, dict[str, Any]]:
        """Main interface to get action from agent."""
        if self.conversation_history == []:
            action_definitions = "\n".join(
                self.prompt.action_definition[action] for action in self.permitted_actions
            )
            action_formats = "\nor\n".join(
                self.prompt.action_format[action] for action in self.permitted_actions
            )
            system_msg = self.prompt.prompt_template.format(
                instruction=obs["info"]["instruction"],
                action_definition=action_definitions,
                format_output=action_formats,
            )
            self.conversation_history.append(make_system_message(content=system_msg))
            user_msg = make_user_message(content=obs["info"]["task_goal"])
        else:
            user_msg = make_user_message(content=obs["env_message"])
        self.conversation_history.append(user_msg)

        with Laminar.start_as_current_span("agent.act", input=user_msg["content"]):
            for _ in range(self.agent_config["n_retry"]):
                try:
                    response, cost = self.llm(self.conversation_history)
                    response = response.content
                    self.cost.append(cost)
                except Exception as e:
                    print("Error Message ", e)
                    time.sleep(self.agent_config["retry_delay"])
                    action, params = f"{LLM_FAILURE}: {e}", {}
                    continue
                try:
                    action, params = self.parser(response)
                except Exception as e:
                    print("Error Message ", e)
                    action, params = f"{PARSE_FAILURE}: {e}", {}
                # The response goes in verbatim even when it did not parse: the
                # environment feeds the parse error back as the next user turn,
                # and that only reads as a correction if the turn it corrects is
                # actually in the transcript.
                self.conversation_history.append(make_assistant_message(content=response))
                break
            Laminar.set_span_output({"action": action, "params": params})

        return action, params
