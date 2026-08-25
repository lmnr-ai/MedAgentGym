def parse_and_truncate_error(error_msg: str) -> str:
    """
    Parse and truncate error messages to ensure complete but not too long output
    """
    return error_msg.replace("^", "")
