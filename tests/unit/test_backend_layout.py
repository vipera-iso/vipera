"""The backends split must not break old imports or reintroduce the cycle."""

import subprocess
import sys


def test_legacy_engine_import_path_still_works():
    from vipera.engine import (
        ChatEngine,
        Engine,
        create_engine,
        load_attempts,
        resolve_backend,
    )

    assert callable(create_engine)
    assert callable(resolve_backend)
    assert callable(load_attempts)
    assert Engine is not None
    assert ChatEngine is not None


def test_legacy_server_backend_import_path_still_works():
    from vipera.server_backend import ServerEngine, probe_devices

    assert ServerEngine is not None
    assert callable(probe_devices)


def test_server_backend_does_not_import_the_engine_shim():
    # The old layout imported vipera.server_backend from inside engine.py to
    # dodge a cycle. Importing the server backend must not pull the shim back in.
    code = (
        "import sys, vipera.backends.server\n"
        "assert 'vipera.engine' not in sys.modules, sorted(sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_registry_reaches_the_server_module_dynamically():
    # resolve_backend looks attributes up on the module so a probe can be
    # substituted; a `from ... import x` binding would freeze it.
    from vipera.backends import registry, server

    assert registry.server_backend is server
