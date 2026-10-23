"""Error taxonomy with stable codes.

A code survives refactors and is searchable in issue reports, which a message
string is not. ``PlannerError`` and ``BackendError`` also subclass
``ValueError`` / ``RuntimeError`` so existing ``except`` clauses keep working
while call sites migrate to the specific class.
"""

from __future__ import annotations


class ViperaError(Exception):
    """Base for every error vipera raises deliberately."""

    code = "E0000"


class PlannerError(ViperaError, ValueError):
    """A plan cannot be produced or is not executable as requested."""

    code = "E3001"


class BackendError(ViperaError, RuntimeError):
    """An engine is missing, failed to load, or stopped answering."""

    code = "E4001"
