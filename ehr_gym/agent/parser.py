import json

VALID_ACTIONS = {"validate_code", "debug", "terminal"}
REQUIRED_PARAMS = {
    "validate_code": {"code"},
    "debug": {"code", "error_msg"},
    "terminal": {"cmd"},
}


def parse_llm_response(response: str) -> tuple[str, dict]:
    """
    Parse an agent response into an (action, params) pair.

    Args:
        response: The raw response from the LLM.
    Returns:
        The action name and its params.
    """
    try:
        # First try to find JSON block in markdown code blocks
        if "```json" in response and "```" in response:
            # Extract content between ```json and ```
            start = response.find("```json") + 7
            end = response.find("```", start)
            if start > 6 and end > start:  # Valid positions found
                response = response[start:end].strip()
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            # Extract the JSON part
            response = response[start : end + 1].strip()
        if "</think>" in response:
            # only keep the context after </think>
            response = response.split("</think>")[-1].strip()
        if '"""' in response:
            left = response.find('"""')
            right = response.rfind('"""')
            if left != -1 and right != -1 and left != right:
                # Splice the triple-quoted block back in as a JSON string
                code = response[left + 3 : right].strip()
                code = code.replace("\n", "\\n")
                response = response[:left].strip() + '"' + code + '"' + response[right + 3 :].strip()

        if not response.strip():
            raise ValueError("Empty response")

        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}\nResponse: {response}")

        if not isinstance(response_dict, dict):
            raise ValueError(f"Response must be a JSON object, got {type(response_dict)}")

        allowed_keys = {"action", "params"}
        actual_keys = set(response_dict.keys())
        if actual_keys != allowed_keys:
            raise ValueError(
                f"Response must contain exactly these keys: {allowed_keys}, got: {actual_keys}"
            )

        action = response_dict["action"]
        params = response_dict["params"]

        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Must be one of: {VALID_ACTIONS}")
        if not isinstance(params, dict):
            raise ValueError(f"'params' must be a dict, got {type(params)}")

        expected_params = REQUIRED_PARAMS[action]
        if set(params.keys()) != expected_params:
            raise ValueError(
                f"'{action}' params must contain exactly: {expected_params}, got: {set(params.keys())}"
            )
        for key, value in params.items():
            if not isinstance(value, str):
                raise ValueError(f"'{key}' must be string, got {type(value)}")

        return action, params

    except Exception as e:
        raise ValueError(f"Failed to fix parse error: {e}")
