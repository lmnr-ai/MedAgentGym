"""Insertion of a solution into a task's context template.

Several datasets hand the agent a program with a `<<insert solution here>>`
marker and score it by running context-with-solution-substituted. Upstream did a
plain `str.replace`, which corrupts the program whenever the marker sits inside
an indented block: the solution lands at column 0, leaving the enclosing class or
function body empty. That produces an `IndentationError`, or a class whose method
ended up at module scope, and it corrupts the *reference* program and the
*agent's* program identically -- so the task becomes unwinnable rather than hard.

Re-indenting the solution to the marker's own indentation fixes 33 of 157
BioCoder test tasks and 12 of 981 train tasks.
"""

import re
import textwrap

DEFAULT_PATTERN = "<<insert solution here>>"


def insert_solution(context: str, solution: str, pattern: str = DEFAULT_PATTERN) -> str:
    """Substitute `solution` for `pattern` in `context`, preserving indentation.

    When the marker is alone on its line, the solution is re-indented to match.
    Otherwise (the marker sits inside an expression) the solution is spliced in
    verbatim, which is all upstream ever did.
    """
    standalone = re.search(
        r"^([ \t]*)" + re.escape(pattern) + r"[ \t]*$", context, flags=re.M
    )
    if standalone is None:
        return context.replace(pattern, "\n" + solution + "\n")

    indent = standalone.group(1)
    body = textwrap.dedent(solution).strip("\n")
    if indent:
        body = "\n".join(indent + line if line.strip() else line for line in body.split("\n"))
    return context[: standalone.start()] + body + context[standalone.end() :]
