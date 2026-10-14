"""vipera.algorithms — the optimization algorithms, as runnable code.

The `docs/reference/literature/` pack is a folder of PDFs and `algorithms-digest.md`
is prose. This package is the third thing: each algorithm that survived review,
implemented, registered, and covered by a test that asserts the property that
makes it worth using rather than that it returns a number.

Four families, one file each:

* ``kv``      — KV retention policies (StreamingLLM sink+window, H2O, recency,
                surprise) over an explicit attention stream.
* ``quant``   — round-to-nearest vs GPTQ Hessian-weighted error diffusion.
* ``select``  — lazy greedy (CELF) submodular maximization, the expert-keep-set
                problem stated as the optimization it is.
* ``spec``    — speculative decoding: exact acceptance rule and the closed-form
                expected accepted length for a draft of length k.

Everything runs on the standard library alone, without a model, a GPU or a
network — the same constraint `pyproject.toml` states for the rest of vipera.

Two of the four reach real artifacts in this repository and are labelled
``[measured]`` for it; see ``bench/algo_measure.py``, which reads real Q4_K
weights out of ``models/Qwen3-0.6B-Q4_K_M.gguf`` and the real OLMoE routing trace
in ``bench/results/``. The other two are ``[simulated]`` / ``[exact]`` and say so.
"""

from __future__ import annotations

from .registry import REGISTRY, Algorithm, get, list_algorithms, run

__all__ = [
    "Algorithm",
    "REGISTRY",
    "get",
    "list_algorithms",
    "run",
]
