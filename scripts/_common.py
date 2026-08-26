"""Helpers shared by the dataset filtering scripts."""

import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

TIMEOUT_SECONDS = 180


def run_python(code: str, cwd: str | None = None) -> tuple[int, str, str]:
    """Run `code` in a scratch directory so stray output files stay out of the repo."""
    scratch = tempfile.mkdtemp(prefix="reference_")
    path = os.path.join(scratch, f"reference_{uuid.uuid4().hex}.py")
    Path(path).write_text(code)
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=cwd or scratch,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    finally:
        Path(path).unlink(missing_ok=True)


def failure_reason(stderr: str) -> str:
    """Bucket a traceback so the drop report is readable."""
    missing = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", stderr)
    if missing:
        return "missing-module:" + missing.group(1).split(".")[0]
    if stderr.strip() == "TIMEOUT":
        return "timeout"
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    if not lines:
        return "unknown"
    named = re.match(r"([A-Za-z_.]*(?:Error|Exception|Exit|Interrupt))\b", lines[-1])
    return named.group(1) if named else "other"
