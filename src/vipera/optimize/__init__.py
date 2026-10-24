"""The optimization layer: request shaping, prefix accounting, measured tuning.

Each module is useful on its own:

* :mod:`vipera.optimize.shaping` trims a request into the loaded context;
* :mod:`vipera.optimize.prefix` says whether the KV cache is being reused;
* :mod:`vipera.optimize.tuning` remembers what the machine actually ran at.
"""

from vipera.optimize.prefix import PrefixStats, cached_tokens, prefix_key
from vipera.optimize.shaping import Shaping, estimate_tokens, shape
from vipera.optimize.tuning import Tuning, TuningStore, apply_tuning, fingerprint

__all__ = [
    "PrefixStats",
    "Shaping",
    "Tuning",
    "TuningStore",
    "apply_tuning",
    "cached_tokens",
    "estimate_tokens",
    "fingerprint",
    "prefix_key",
    "shape",
]
