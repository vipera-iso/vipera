"""Error taxonomy with stable codes.

A code survives refactors and is searchable in issue reports, which a message
string is not. ``PlannerError`` and ``BackendError`` also subclass
``ValueError`` / ``RuntimeError`` so existing ``except`` clauses keep working
while call sites migrate to the specific class.
"""

from __future__ import annotations

import logging

#: One logger for the whole package. Libraries do not configure handlers; the
#: CLI and any embedding application attach theirs.
logger = logging.getLogger("vipera")


class ViperaError(Exception):
    """Base for every error vipera raises deliberately."""

    code = "E0000"


class PlannerError(ViperaError, ValueError):
    """A plan cannot be produced or is not executable as requested."""

    code = "E3001"


class BackendError(ViperaError, RuntimeError):
    """An engine is missing, failed to load, or stopped answering."""

    code = "E4001"


def log_failure(operation: str, exc: BaseException) -> None:
    """Record one failed engine call, with its traceback.

    Engine calls run in worker threads, so an exception never reaches uvicorn's
    own error logging; without this the only trace of a failed request is the
    child's log file. ``exc_info`` is passed explicitly so the call sites do not
    have to be inside the ``except`` block that produced the exception.
    """
    logger.error("%s failed: %s", operation, exc, exc_info=exc)
