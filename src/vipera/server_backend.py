"""Compatibility import path for `vipera.backends.server`.

The implementation moved there so the backend does not import `vipera.engine`
(which used to import it back, at function scope, to dodge a cycle). New code
should import from `vipera.backends.server`.
"""

from __future__ import annotations

from vipera.backends.server import (
    ServerEngine,
    server_command,
    supports_n_cpu_moe,
    supports_spec_type,
)

__all__ = [
    "ServerEngine",
    "server_command",
    "supports_n_cpu_moe",
    "supports_spec_type",
]
