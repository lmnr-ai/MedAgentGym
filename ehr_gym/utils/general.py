import json

import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_conversation_history(conversation_history, save_path):
    """
    Save the conversation history to a file.
    """
    with open(save_path, "w") as f:
        json.dump(conversation_history, f, indent=4)
