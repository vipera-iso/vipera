"""The gateway: `/health` plus whichever protocol layers are selected.

Routes live in ``vipera.protocols``: ``openai`` serves ``/v1/*`` and ``ollama``
serves ``/api/*``. This module only decides which engine answers a request (a
loaded engine, or a pool of them) and hands that to the layers.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI

from vipera import __version__
from vipera.optimize.layer import OptimizeLayer
from vipera.pool import UnknownModelError
from vipera.protocols import mount
from vipera.protocols.gateway import Gateway
from vipera.protocols.openai import ChatRequest  # noqa: F401 - re-exported for callers


def build_app(engine=None, model_name: str | None = None, *, pool=None, protocols=None) -> FastAPI:
    """Build the gateway.

    Single-model mode passes `engine` + `model_name`. Multi-model mode passes
    `pool` instead: engines are planned and mounted on first request, and
    `model` on the request selects which one to use. `protocols` selects the
    HTTP layers (default: openai).
    """
    if engine is None and pool is None:
        raise ValueError("build_app needs an engine or a pool")

    app = FastAPI(title="Vipera Gateway", version=__version__)
    # The server backend can run N requests at once (llama.cpp slots), so the
    # gateway admits N instead of serialising. One shared context still means
    # one at a time for the in-process engine and for a pool.
    slots = 1 if pool is not None else int(getattr(engine, "parallel", 1) or 1)
    lock = asyncio.Semaphore(slots) if slots > 1 else asyncio.Lock()

    if pool is not None:

        async def engine_for(name: str | None):
            # Planning and loading a GGUF can take minutes; keep it off the
            # event loop so /health and other requests stay responsive.
            active, _ = await asyncio.to_thread(pool.acquire, name)
            return active

        def release(engine) -> None:
            pool.release(engine)

        def ids() -> list[str]:
            return pool.ids()

        def default_model() -> str | None:
            return pool.default

        def loaded_ids() -> list[str]:
            return pool.loaded_ids()
    else:
        if model_name is None:
            raise ValueError("build_app needs model_name when serving a single engine")
        # Closures don't inherit the narrowing above, so bind the name here.
        single = model_name

        async def engine_for(name: str | None):
            if name and name != single:
                raise UnknownModelError(name, [single])
            return engine

        def release(engine) -> None:
            return None

        def ids() -> list[str]:
            return [single]

        def default_model() -> str | None:
            return single

        def loaded_ids() -> list[str]:
            return [single]

    optimizer = OptimizeLayer.from_config()
    gateway = Gateway(
        lock=lock,
        engine_for=engine_for,
        ids=ids,
        default_model=default_model,
        loaded_ids=loaded_ids,
        optimizer=optimizer,
        release=release,
    )

    @app.get("/health")
    def health():
        if pool is not None:
            body = {"status": "ok", "model": pool.default, "loaded": pool.loaded_ids()}
        else:
            body = {"status": "ok", "model": model_name}
        # Prefix reuse per served model: a client that keeps its system prompt
        # stable shows a cache_rate above zero here.
        if optimizer is not None:
            body["optimize"] = optimizer.stats.summary()
        return body

    app.state.protocols = mount(app, gateway, protocols)
    app.state.gateway = gateway
    return app
