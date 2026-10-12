"""vipera — an optimization layer for local LLM model files.

Stage 1 (this code): read a GGUF, inventory it, and plan weight placement for a
weak hardware target. Nothing here loads tensor data, so it is cheap and safe.
"""

__version__ = "0.0.1"
