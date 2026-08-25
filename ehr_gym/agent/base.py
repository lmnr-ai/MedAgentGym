import time
from typing import Any

from ehr_gym.agent.parser import parse_llm_response
from ehr_gym.agent.prompt import DynamicPrompt
from ehr_gym.llm.chat_api import (
    AzureModelArgs,
    OpenAIModelArgs,
    make_assistant_message,
    make_system_message,
    make_user_message,
)


class EHRAgent:
    """Agent class to interact with EHREnvironment using LLM"""

    def __init__(self, agent_config, permitted_actions):
        self.agent_config = agent_config
        self.llm_config = agent_config["llm"]

        if self.llm_config["model_type"] == "OpenAI":
            self.llm = OpenAIModelArgs(
                model_name=self.llm_config["model_name"],
                max_total_tokens=self.llm_config["max_total_tokens"],
                max_input_tokens=self.llm_config["max_input_tokens"],
                max_new_tokens=self.llm_config["max_new_tokens"],
                temperature=self.llm_config["temperature"],
                vision_support=False,
            ).make_model()
        elif self.llm_config["model_type"] == "Azure":
            self.llm = AzureModelArgs(
                model_name=self.llm_config["model_name"],
                temperature=self.llm_config["temperature"],
                max_new_tokens=self.llm_config["max_new_tokens"],
                deployment_name=self.llm_config["deployment_name"],
                log_probs=self.llm_config["log_probs"],
            ).make_model()
        else:
            raise ValueError("Model type {} not supported.".format(self.llm_config["model_type"]))
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

        for _ in range(self.agent_config["n_retry"]):
            try:
                response, cost = self.llm(self.conversation_history)
                response = response.content
                self.cost.append(cost)
            except Exception as e:
                print("Error Message ", e)
                time.sleep(self.agent_config["retry_delay"])
                action, params = f"error: str({e})", {}
                continue
            try:
                action, params = self.parser(response)
            except Exception as e:
                print("Error Message ", e)
                time.sleep(self.agent_config["retry_delay"])
                action, params = f"error: str({e})", {}
                response = f"Error: {e}. Please regenerate the action."
            self.conversation_history.append(make_assistant_message(content=response))
            break

        return action, params
