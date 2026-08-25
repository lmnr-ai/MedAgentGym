class DynamicPrompt:
    """Builds the system prompt from the subset of actions a task permits."""

    def __init__(self):
        self.prompt_template = """{instruction}
You have access to the following actions with params, and receive corresponding feedback after each action:
{action_definition}

Code requirements:
    - Use the variable 'answer' to store the answer of the code.
    - Code should be self-contained and not rely on any variables or state outside.

Response format requirements, strictly one of the following:
{format_output}
    - Must be valid JSON format
    - No additional text or formatting allowed"""
        self.action_definition = {
            "validate_code": """validate_code: Test code execution to check the intermediate results or for final answer
    - params: code (str)
    - feedback: execution result (success or failure), error message if failed, code output if success""",
            "debug": """debug: Debug the code with the execution error message to find the problem
    - params: code (str), error_msg (str)
    - feedback: debugged code output (str)""",
            "terminal": """terminal: Write terminal commands to install some mandatory packages or libraries for code execution
    - params: cmd (str)
    - feedback: execution result (success or failure), error message if failed, command output if success""",
        }
        self.action_format = {
            "validate_code": """{
    "action": "validate_code",
    "params": {
        "code": "<code>"
    }
}""",
            "debug": """{
    "action": "debug",
    "params": {
        "code": "<code>",
        "error_msg": "<error_message>"
    }
}""",
            "terminal": """{
    "action": "terminal",
    "params": {
        "cmd": "<cmd>"
    }
}""",
        }
