"""Request-size bounds shared by the HTTP layers.

Both layers must reject the same oversized payload, and both must do it before
an engine is acquired: in pool mode a load can take minutes, so a body the
backend cannot use should never pay for one. Keeping the numbers in a single
place is what stops the two layers from drifting apart — the OpenAI layer had
these bounds while the Ollama one accepted an unbounded body.

Kept free of FastAPI and engine imports so either layer can use it without
dragging the other one in.
"""

from __future__ import annotations

#: Largest answer any single request may ask for.
MAX_TOKENS = 32_768
#: Message count per chat request (history included).
MAX_MESSAGES = 200
#: Total characters across a chat request's message contents.
MAX_PROMPT_CHARS = 500_000
#: Inputs per embedding call, and the characters across all of them.
MAX_EMBED_INPUTS = 256
MAX_EMBED_CHARS = 1_000_000
